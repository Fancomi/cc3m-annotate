#!/usr/bin/env python3
"""交叉实验：用 Qwen 的 dense caption 驱动 Florence-2 的 grounding。

动机（前期实测结论）：<CAPTION_TO_PHRASE_GROUNDING> 的 detail 上限完全由输入文本
决定 —— 同一张图，10 词的 <CAPTION> 只 ground 出 3 个框，79 词的
<MORE_DETAILED_CAPTION> 出 12 个。而本轮批处理里 Qwen dense 平均 174 词，
是 Florence-2 自己 more_detailed（89 词）的近 2 倍。

所以这里把 Qwen 的 dense caption 回喂给 Florence-2，检验能否拿到更密的区域标注。
每张图输出两组 grounding 供逐图对比：
  grounding_f2   —— 用 Florence-2 自己的 <MORE_DETAILED_CAPTION>（已在上一轮产出，直接复用）
  grounding_qwen —— 用 Qwen 的 dense caption（本脚本新增）

注意 Florence-2 的文本编码上限：超长 caption 会被 tokenizer 截断，
故记录 caption 词数与截断标志，便于判断"框变多"是否被截断抵消。
"""
import argparse, json, os, time
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True, help="上一轮 merge 出的 merged.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/root/paddlejob/workspace/env_run/penghaotian/models/Florence-2-large")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--num-beams", type=int, default=3)
    args = ap.parse_args()

    from transformers import AutoProcessor, AutoModelForCausalLM

    items = []
    for i, l in enumerate(open(args.merged)):
        if i % args.num_shards != args.shard:
            continue
        r = json.loads(l)
        if "florence2" in r and "qwen" in r and r["qwen"].get("dense"):
            items.append(r)

    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
    items = [r for r in items if r["id"] not in done]
    print(f"[shard {args.shard}] todo={len(items)} skipped={len(done)}", flush=True)
    if not items:
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float16).cuda().eval()
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    max_pos = model.config.text_config.max_position_embeddings

    def ground(img, text):
        task = "<CAPTION_TO_PHRASE_GROUNDING>"
        inp = proc(text=task + text, images=img, return_tensors="pt")
        n_tok = int(inp["input_ids"].shape[1])
        inp = {k: (v.cuda().half() if v.dtype == torch.float32 else v.cuda()) for k, v in inp.items()}
        with torch.no_grad():
            ids = model.generate(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"],
                                 max_new_tokens=1024, num_beams=args.num_beams, do_sample=False)
        txt = proc.batch_decode(ids, skip_special_tokens=False)[0]
        v = proc.post_process_generation(txt, task=task, image_size=(img.width, img.height))[task]
        agg = {}
        for lb, bb in zip(v["labels"], v["bboxes"]):
            agg.setdefault(lb, []).append([round(x, 1) for x in bb])
        return agg, n_tok, len(v["labels"])

    fo = open(args.out, "a", buffering=1)
    t_all = time.perf_counter()
    for n, r in enumerate(items, 1):
        rec = {"id": r["id"], "path": r["path"], "gt_caption": r["gt_caption"]}
        try:
            img = Image.open(r["path"]).convert("RGB")
            qcap = r["qwen"]["dense"]
            t0 = time.perf_counter()
            agg, ntok, nbox = ground(img, qcap)
            rec.update({
                "qwen_dense_words": len(qcap.split()),
                "qwen_dense_tokens": ntok,
                "truncated": ntok >= max_pos,
                "grounding_qwen": agg,
                "n_phrase_qwen": len(agg), "n_box_qwen": nbox,
                # 对照：上一轮用 F2 自己 more_detailed 得到的结果
                "f2_more_detailed_words": len(r["florence2"]["more_detailed"].split()),
                "n_phrase_f2": len(r["florence2"]["grounding"] or {}),
                "n_box_f2": sum(len(v) for v in (r["florence2"]["grounding"] or {}).values()),
                "dt_s": round(time.perf_counter() - t0, 2),
            })
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if n % 25 == 0:
            el = time.perf_counter() - t_all
            print(f"[shard {args.shard}] {n}/{len(items)} {el/n:.2f}s/img "
                  f"eta {(len(items)-n)*el/n/60:.0f}min", flush=True)
    fo.close()
    print(f"[shard {args.shard}] DONE {n} in {(time.perf_counter()-t_all)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
