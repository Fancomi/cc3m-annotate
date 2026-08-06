#!/usr/bin/env python3
"""Qwen3.6-35B-A3B-FP8 caption 批处理（cc3m）。

优先走 sglang OpenAI 兼容端点（--base-url），端点不可用时回退 transformers 本地推理。
两级 prompt：短 caption + 密集列举物体的详细 caption（后者用于喂 Florence-2 grounding）。
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worklist", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", required=True,
                    help="sglang 端点，逗号分隔可给多个，如 http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    import itertools, os as _os
    # 代理会拦截 127.0.0.1，必须清掉
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        _os.environ.pop(k, None)
    from openai import OpenAI
    from concurrent.futures import ThreadPoolExecutor
    urls = [u.strip() for u in args.base_url.split(",") if u.strip()]
    clients = [OpenAI(api_key="x", base_url=u, timeout=600) for u in urls]
    print(f"[shard {args.shard}] endpoints={len(clients)}", flush=True)
    _rr = itertools.count()

    items = [json.loads(l) for l in open(args.worklist)]
    items = [it for i, it in enumerate(items) if i % args.num_shards == args.shard]
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try: done.add(json.loads(l)["id"])
            except Exception: pass
    items = [it for it in items if it["id"] not in done]
    print(f"[shard {args.shard}] todo={len(items)} skipped={len(done)}", flush=True)
    if not items: return

    def ask(img_b64, prompt, client):
        # enable_thinking=False：Qwen3.6 默认开推理链，会把 "The user wants..." 的思考
        # 过程当正文吐出来并吃掉全部 token 预算（实测本地推理 163s/图且拿不到 caption）。
        # sglang 走 extra_body 透传 chat_template_kwargs。
        r = client.chat.completions.create(
            model=args.model, max_tokens=args.max_tokens, temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": prompt}]}])
        txt = r.choices[0].message.content or ""
        if "</think>" in txt:
            txt = txt.split("</think>", 1)[1]
        return txt.strip()

    def work(it):
        rec = {"id": it["id"], "path": it["path"], "gt_caption": it["gt_caption"]}
        try:
            img = Image.open(it["path"]).convert("RGB")
            enc = b64(img)
            client = clients[next(_rr) % len(clients)]
            t0 = time.perf_counter()
            rec["qwen_short"] = ask(enc, SHORT, client)
            rec["qwen_dense"] = ask(enc, DENSE, client)
            rec["dt_s"] = round(time.perf_counter() - t0, 2)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        return rec

    fo = open(args.out, "a", buffering=1)
    t_all = time.perf_counter(); n = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for rec in ex.map(work, items):
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); n += 1
            if n % 20 == 0:
                el = time.perf_counter() - t_all
                print(f"[shard {args.shard}] {n}/{len(items)} {el/n:.2f}s/img "
                      f"eta {(len(items)-n)*el/n/60:.0f}min", flush=True)
    fo.close()
    print(f"[shard {args.shard}] DONE {n} in {(time.perf_counter()-t_all)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
