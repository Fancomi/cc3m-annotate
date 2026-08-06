#!/usr/bin/env python3
"""Qwen3.6-35B-A3B-FP8 caption 批处理（cc3m）—— transformers 本地推理版。

为什么不走 sglang：vllm_deploy/run_qwen3_6_sgl.sh 依赖的 sglang 源码编译尚未完成
（sgl-kernel 要 clone ROCm/composable_kernel 等大子模块）。caption 是离线批处理，
不需要 sglang 的高吞吐调度，transformers + FP8 kernels 直接跑即可。
端点版见 run_qwen.py（sglang 起来后可切）。

两级 prompt：短 caption + 密集列举物体的详细 caption。
后者的用途是喂 Florence-2 的 <CAPTION_TO_PHRASE_GROUNDING>——实测 grounding
的 detail 上限完全由输入文本决定（10 词 caption 只出 3 个框，79 词出 12 个）。
"""
import argparse, json, os, time
import torch
from PIL import Image

SHORT = "Describe this image in one concise sentence."
DENSE = ("Describe this image in one dense paragraph. Explicitly name every distinct visible "
         "object, material and body part using concrete nouns. Be exhaustive and factual.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worklist", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/root/paddlejob/workspace/env_run/penghaotian/models/Qwen3.6-35B-A3B-FP8")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--maxside", type=int, default=768)
    args = ap.parse_args()

    from transformers import AutoProcessor, AutoModelForImageTextToText

    items = [json.loads(l) for l in open(args.worklist)]
    items = [it for i, it in enumerate(items) if i % args.num_shards == args.shard]
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
    items = [it for it in items if it["id"] not in done]
    print(f"[shard {args.shard}] todo={len(items)} skipped={len(done)}", flush=True)
    if not items:
        return

    t = time.perf_counter()
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype="auto", device_map="cuda:0").eval()
    print(f"[shard {args.shard}] loaded in {time.perf_counter()-t:.0f}s", flush=True)

    def ask(img, prompt):
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": prompt}]}]
        # enable_thinking=False：Qwen3.6 默认开推理链，会把 "The user wants..." 的思考
        # 过程当正文吐出来，且吃掉全部 token 预算（实测 163s/图且拿不到 caption）。
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt",
                                       enable_thinking=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False)
        txt = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        # 双保险：若模板未生效，剥掉 <think>...</think>
        if "</think>" in txt:
            txt = txt.split("</think>", 1)[1]
        return txt.strip()

    fo = open(args.out, "a", buffering=1)
    t_all = time.perf_counter()
    for n, it in enumerate(items, 1):
        rec = {"id": it["id"], "path": it["path"], "gt_caption": it["gt_caption"]}
        try:
            img = Image.open(it["path"]).convert("RGB")
            if max(img.size) > args.maxside:
                s = args.maxside / max(img.size)
                img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
            t0 = time.perf_counter()
            rec["qwen_short"] = ask(img, SHORT)
            rec["qwen_dense"] = ask(img, DENSE)
            rec["dt_s"] = round(time.perf_counter() - t0, 2)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if n % 10 == 0:
            el = time.perf_counter() - t_all
            print(f"[shard {args.shard}] {n}/{len(items)} {el/n:.2f}s/img "
                  f"eta {(len(items)-n)*el/n/60:.0f}min", flush=True)
    fo.close()
    print(f"[shard {args.shard}] DONE {n} in {(time.perf_counter()-t_all)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
