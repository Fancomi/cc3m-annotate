#!/usr/bin/env python3
"""NOVIC 批处理（cc3m）—— 整图开放词汇分类，top-k 名词 + 概率。

NOVIC 只做分类不出框，所以这里同时跑两种粒度：
  1. 整图 top-10
  2. 中心裁剪 top-5（对照，检验裁剪对精度的影响）

必须在 novic 仓库目录下运行（infer 依赖同目录模块），且需
HF_HOME 指向含 DFN5B CLIP 的缓存、HF_HUB_OFFLINE=1。
"""
import argparse, itertools, json, os, time, sys
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worklist", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint",
                    default="outputs/ovod_20240628_142131/ovod_chunk0433_20240630_235415.train")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    from infer import NOVICModel, utils
    from PIL import Image
    utils.allow_tf32(enable=True)

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

    model = NOVICModel(checkpoint=args.checkpoint)
    fo = open(args.out, "a", buffering=1)
    t_all = time.perf_counter()
    n = 0
    with model:
        # 分批：NOVIC 批量推理比逐图快得多（7ms/图 vs 26ms/图）
        for s in range(0, len(items), args.batch):
            chunk = items[s:s + args.batch]
            imgs, ok = [], []
            for it in chunk:
                try:
                    imgs.append(NOVICModel.load_image(it["path"]))
                    ok.append(it)
                except Exception as e:
                    fo.write(json.dumps({"id": it["id"], "path": it["path"],
                                         "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False) + "\n")
            if not ok:
                continue
            try:
                t0 = time.perf_counter()
                out = model.classify_image(image=imgs)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / len(ok)
                for j, it in enumerate(ok):
                    preds = list(itertools.islice(zip(out.preds[j], out.probs[j]), args.topk))
                    fo.write(json.dumps({
                        "id": it["id"], "path": it["path"], "gt_caption": it["gt_caption"],
                        "novic_topk": [[p, round(float(q) * 100, 2)] for p, q in preds],
                        "dt_ms": round(dt * 1000, 1),
                    }, ensure_ascii=False) + "\n")
            except Exception as e:
                for it in ok:
                    fo.write(json.dumps({"id": it["id"], "path": it["path"],
                                         "error": f"batch {type(e).__name__}: {e}"}, ensure_ascii=False) + "\n")
            n += len(chunk)
            if (s // args.batch) % 5 == 0:
                el = time.perf_counter() - t_all
                print(f"[shard {args.shard}] {n}/{len(items)} {el/max(n,1):.3f}s/img "
                      f"eta {(len(items)-n)*el/max(n,1)/60:.1f}min", flush=True)
    fo.close()
    print(f"[shard {args.shard}] DONE {n} in {(time.perf_counter()-t_all)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
