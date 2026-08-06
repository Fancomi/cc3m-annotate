#!/usr/bin/env python3
"""gemma4 vs Qwen caption 对比：同一批图、同一 prompt、逐张对照。

从 merged.jsonl 抽 N 张，对每张跑两个模型（gemma4 新起 + qwen 已有产出复用），
输出对照 JSON：short + dense 双级 caption。
"""
import argparse, base64, io, json, os, time
from PIL import Image

SHORT = "Describe this image in one concise sentence."
DENSE = ("Describe this image in one dense paragraph. Explicitly name every distinct visible "
         "object, material and body part using concrete nouns. Be exhaustive and factual.")

def b64(img, maxside=768):
    im = img.copy()
    if max(im.size) > maxside:
        s = maxside / max(im.size)
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()

def ask(client, model, img_b64, prompt, max_tokens=320):
    try:
        r = client.chat.completions.create(
            model=model, max_tokens=max_tokens, temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": prompt}]}])
        txt = r.choices[0].message.content or ""
        if "</think>" in txt:
            txt = txt.split("</think>", 1)[1]
        return txt.strip()
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gemma-url", default="http://127.0.0.1:8101/v1")
    ap.add_argument("--gemma-model", default="/dev/shm/models/gemma-4-26B-A4B-it")
    ap.add_argument("--qwen-url", default="http://127.0.0.1:8001/v1")  # qwen 已关，只用已有产出
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(k, None)
    from openai import OpenAI
    gemma = OpenAI(api_key="x", base_url=args.gemma_url, timeout=600)
    print(f"gemma endpoint: {args.gemma_url} model={args.gemma_model}", flush=True)

    # 读 merged，抽前 N 张（均匀）
    import random
    items = [json.loads(l) for l in open(args.merged)]
    random.seed(args.seed)
    random.shuffle(items)
    items = items[:args.n]
    print(f"抽样 {len(items)} 张", flush=True)

    # 并行跑 gemma（qwen 复用已有产出）
    from concurrent.futures import ThreadPoolExecutor
    results = []
    t_all = time.perf_counter()
    def work(it):
        rec = {"id": it["id"], "path": it["path"], "gt_caption": it["gt_caption"]}
        # qwen 已有产出
        q = it.get("qwen") or {}
        rec["qwen_short"] = q.get("short")
        rec["qwen_dense"] = q.get("dense")
        try:
            img = Image.open(it["path"]).convert("RGB")
            enc = b64(img)
            t0 = time.perf_counter()
            rec["gemma_short"] = ask(gemma, args.gemma_model, enc, SHORT, 160)
            rec["gemma_dense"] = ask(gemma, args.gemma_model, enc, DENSE, 320)
            rec["gemma_dt_s"] = round(time.perf_counter() - t0, 2)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        return rec
    with ThreadPoolExecutor(max_workers=8) as ex:
        for n, r in enumerate(ex.map(work, items), 1):
            results.append(r)
            if n % 10 == 0:
                el = time.perf_counter() - t_all
                print(f"{n}/{len(items)} {el/n:.1f}s/图 eta {(len(items)-n)*el/n/60:.0f}min", flush=True)
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"DONE {len(results)} in {(time.perf_counter()-t_all)/60:.1f}min -> {args.out}", flush=True)

if __name__ == "__main__":
    main()
