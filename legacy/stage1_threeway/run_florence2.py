#!/usr/bin/env python3
"""Florence-2 三级 caption 批处理（cc3m）。

每张图跑 <CAPTION> / <DETAILED_CAPTION> / <MORE_DETAILED_CAPTION> 三级，
外加 <OD> 与 <DENSE_REGION_CAPTION> 两个纯图像输入的检测任务，一次性存全。

分片并行：--shard i --num-shards n，每个进程绑一张卡。
断点续跑：输出 jsonl 按行 append，重启时跳过已完成 id。
"""
import argparse, json, os, time, sys
import torch
from PIL import Image

TASKS_TEXT = ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"]
TASKS_BOX = ["<OD>", "<DENSE_REGION_CAPTION>"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worklist", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/root/paddlejob/workspace/env_run/penghaotian/models/Florence-2-large")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--num-beams", type=int, default=3)
    args = ap.parse_args()

    from transformers import AutoProcessor, AutoModelForCausalLM

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

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float16).cuda().eval()
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    def run(img, task, text=None):
        prompt = task if text is None else task + text
        inp = proc(text=prompt, images=img, return_tensors="pt")
        inp = {k: (v.cuda().half() if v.dtype == torch.float32 else v.cuda()) for k, v in inp.items()}
        with torch.no_grad():
            ids = model.generate(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"],
                                 max_new_tokens=args.max_new_tokens,
                                 num_beams=args.num_beams, do_sample=False)
        txt = proc.batch_decode(ids, skip_special_tokens=False)[0]
        return proc.post_process_generation(txt, task=task, image_size=(img.width, img.height))[task]

    fo = open(args.out, "a", buffering=1)
    t_all = time.perf_counter()
    for n, it in enumerate(items, 1):
        rec = {"id": it["id"], "path": it["path"], "gt_caption": it["gt_caption"]}
        try:
            img = Image.open(it["path"]).convert("RGB")
            rec["size"] = [img.width, img.height]
            t0 = time.perf_counter()
            for task in TASKS_TEXT:
                rec[task] = run(img, task)
            # 两段式：用最详细的 caption 回喂 grounding（detail 上限由文本决定）
            g = run(img, "<CAPTION_TO_PHRASE_GROUNDING>", rec["<MORE_DETAILED_CAPTION>"])
            agg = {}
            for lb, bb in zip(g["labels"], g["bboxes"]):
                agg.setdefault(lb, []).append([round(x, 1) for x in bb])
            rec["grounding"] = agg
            for task in TASKS_BOX:
                v = run(img, task)
                rec[task] = {"labels": v["labels"],
                             "bboxes": [[round(x, 1) for x in b] for b in v["bboxes"]]}
            rec["dt_s"] = round(time.perf_counter() - t0, 2)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if n % 20 == 0:
            el = time.perf_counter() - t_all
            print(f"[shard {args.shard}] {n}/{len(items)} {el/n:.2f}s/img eta {(len(items)-n)*el/n/60:.0f}min",
                  flush=True)
    fo.close()
    print(f"[shard {args.shard}] DONE {len(items)} in {(time.perf_counter()-t_all)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
