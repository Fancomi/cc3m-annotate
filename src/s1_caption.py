#!/usr/bin/env python3
"""阶段 1 · caption：用 gemma4 为 cc3m 全量图片生成两级 caption。

输入  cc3m-tsv/_shards/*.tsv       每行 `图片绝对路径 \t cc3m 原始 caption`
输出  <out>/shard{N}.jsonl         每行 {id, shard, path, gt_caption, gemma_short, gemma_dense, dt_s}
        id    = 该图在所属 tsv 内的行号
        shard = tsv 编号（0-575）；(shard, id) 才是全局主键，path 亦全局唯一

依赖  8 个 gemma4 sglang 实例（见 run/1_caption.sh 自动拉起）

为什么选 gemma4 而不是 Qwen3.6：同图同 prompt 对照 60 张，gemma4 的 caption
驱动 grounding 后精度高 5.3 个点（71.4% vs 66.1%），且更少脑补不存在的物体。
"""
import argparse, glob, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch import run_pool
from common import DENSE, SHORT, ask_vlm, b64, make_clients, round_robin, take_shard


def load_tsv(tsv_dir, limit_per_tsv=0):
    """读全部 tsv，产出待办条目。tsv 编号即 shard 字段。"""
    items = []
    for tf in sorted(glob.glob(os.path.join(tsv_dir, "cc3m-train-*.tsv"))):
        shard = int(os.path.basename(tf).split("-")[-1].split(".")[0])
        with open(tf) as f:
            for i, line in enumerate(f):
                if limit_per_tsv and i >= limit_per_tsv:
                    break
                p, _, g = line.rstrip("\n").partition("\t")
                items.append({"id": i, "shard": shard, "path": p, "gt_caption": g})
    return items


def main():
    ap = argparse.ArgumentParser(description="阶段1：gemma4 全量 caption")
    ap.add_argument("--tsv-dir", required=True, help="cc3m-tsv/_shards 目录")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--urls", required=True, help="sglang 端点，逗号分隔")
    ap.add_argument("--model", default="/dev/shm/models/gemma-4-26B-A4B-it")
    ap.add_argument("--shard", type=int, default=0, help="本进程负责的分片号")
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=16, help="每进程并发请求数")
    ap.add_argument("--limit-per-tsv", type=int, default=0, help=">0 时每个 tsv 只取前 N 行（冒烟）")
    args = ap.parse_args()

    from PIL import Image

    clients = make_clients(args.urls)
    pick = round_robin(clients)
    items = take_shard(load_tsv(args.tsv_dir, args.limit_per_tsv), args.shard, args.num_shards)
    print(f"[cap{args.shard}] endpoints={len(clients)}", flush=True)

    def work(it):
        rec = dict(it)
        try:
            enc = b64(Image.open(it["path"]).convert("RGB"))
            t0 = time.perf_counter()
            # 传整个 clients 列表：失败时 ask_vlm 会换端点重试
            rec["gemma_short"] = ask_vlm(clients, args.model, enc, SHORT, 160, pick=pick)
            rec["gemma_dense"] = ask_vlm(clients, args.model, enc, DENSE, 320, pick=pick)
            rec["dt_s"] = round(time.perf_counter() - t0, 2)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        return rec

    run_pool(items, work, os.path.join(args.out, f"shard{args.shard}.jsonl"),
             f"cap{args.shard}", workers=args.concurrency)


if __name__ == "__main__":
    main()
