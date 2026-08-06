#!/usr/bin/env python3
"""crossfeed 幻觉校验：逐个短语-框对，裁剪后问 Qwen 该短语是否真在框内。

动机：crossfeed 把 Qwen 的 174 词 dense caption 喂给 Florence-2 grounding，
拿到 39.4 框/图（相对 F2 自驱动的 9.8 框翻了 4 倍）。但 caption 本身会写出
图里没有的东西（早先实测 F2 自己的 caption 就把岸边写成 "wooden bench"，
grounding 仍给它配了框），grounding 只负责定位、不负责否证。

做法：把每个框裁出来（带少量 padding 保留上下文），问 Qwen
"这个裁剪里有 <短语> 吗？yes/no"。这是独立于 grounding 的第二意见 ——
判定模型（Qwen 看裁剪图）与产出模型（F2 看全图+文本）依据不同，
所以它能抓到"框位错"和"概念根本不存在"两类错误。

注意这不是绝对真值：Qwen 自己也会错，且它正是 caption 的作者，
对自己写过的概念可能有确认偏误。结论应读作"精度下界估计"。
"""
import argparse, base64, io, json, os, random, re, time
from collections import defaultdict

SKIP_PHRASE = re.compile(
    r"^(this|the)?\s*(image|photo|picture|scene|view|background|foreground)$", re.I)


def b64(img, maxside=512):
    im = img.copy()
    if max(im.size) > maxside:
        s = maxside / max(im.size)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crossfeed-glob", default="crossfeed_shard*.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--sample", type=int, default=300, help="抽样图片数")
    ap.add_argument("--max-boxes", type=int, default=12, help="每图最多校验多少个短语")
    ap.add_argument("--pad", type=float, default=0.12, help="裁剪外扩比例，保留上下文")
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import glob
    import itertools
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(k, None)
    from PIL import Image
    from openai import OpenAI
    from concurrent.futures import ThreadPoolExecutor

    urls = [u.strip() for u in args.base_url.split(",") if u.strip()]
    clients = [OpenAI(api_key="x", base_url=u, timeout=600) for u in urls]
    rr = itertools.count()

    rows = []
    for f in glob.glob(os.path.join(args.out_dir, args.crossfeed_glob)):
        for l in open(f):
            r = json.loads(l)
            if "error" not in r and r.get("grounding_qwen"):
                rows.append(r)
    random.seed(args.seed)
    random.shuffle(rows)
    rows = rows[:args.sample]
    print(f"抽样 {len(rows)} 张图", flush=True)

    # 展开成 (图, 短语, 框) 任务；每图取前 max_boxes 个短语，每短语只验第一个框
    tasks = []
    for r in rows:
        n = 0
        for phrase, boxes in r["grounding_qwen"].items():
            if SKIP_PHRASE.match(phrase.strip()):
                continue
            tasks.append((r["id"], r["path"], phrase, boxes[0], len(boxes)))
            n += 1
            if n >= args.max_boxes:
                break
    print(f"待校验 短语-框 对: {len(tasks)}", flush=True)

    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try:
                d = json.loads(l)
                done.add((d["id"], d["phrase"]))
            except Exception:
                pass
    tasks = [t for t in tasks if (t[0], t[2]) not in done]
    print(f"去掉已完成，剩 {len(tasks)}", flush=True)
    if not tasks:
        return

    def work(t):
        iid, path, phrase, box, nbox = t
        rec = {"id": iid, "phrase": phrase, "box": box, "n_box_for_phrase": nbox}
        try:
            img = Image.open(path).convert("RGB")
            W, H = img.size
            x0, y0, x1, y1 = box
            pw, ph = (x1 - x0) * args.pad, (y1 - y0) * args.pad
            crop = img.crop((max(0, x0 - pw), max(0, y0 - ph),
                             min(W, x1 + pw), min(H, y1 + ph)))
            rec["crop_frac"] = round((x1 - x0) * (y1 - y0) / (W * H), 4)
            if crop.width < 8 or crop.height < 8:
                rec["verdict"] = "TOO_SMALL"
                return rec
            q = (f'Is "{phrase.strip()}" clearly visible in this image crop? '
                 f'Answer with exactly one word: yes or no.')
            c = clients[next(rr) % len(clients)]
            r = c.chat.completions.create(
                model=args.model, max_tokens=6, temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64(crop)}"}},
                    {"type": "text", "text": q}]}])
            a = (r.choices[0].message.content or "").strip().lower()
            rec["raw"] = a[:20]
            rec["verdict"] = "YES" if a.startswith("yes") else ("NO" if a.startswith("no") else "UNPARSED")
        except Exception as e:
            rec["verdict"] = "ERROR"
            rec["error"] = f"{type(e).__name__}: {e}"
        return rec

    fo = open(args.out, "a", buffering=1)
    t_all = time.perf_counter()
    n = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for rec in ex.map(work, tasks):
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 200 == 0:
                el = time.perf_counter() - t_all
                print(f"{n}/{len(tasks)} {el/n*1000:.0f}ms/pair "
                      f"eta {(len(tasks)-n)*el/n/60:.1f}min", flush=True)
    fo.close()
    print(f"DONE {n} in {(time.perf_counter()-t_all)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
