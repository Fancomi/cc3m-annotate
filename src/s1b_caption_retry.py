#!/usr/bin/env python3
"""阶段 1b · 补齐：重跑 caption 中失败的条目。

sglang 实例在长时批处理中偶发重启，导致部分请求 APIConnectionError
（全量 289 万张实测约 24 万条，8%）。这些条目缺 gemma_dense，阶段 2 会跳过它们。

三步（scan 只需一次，全扫 3.7G 较慢）:
  scan   提取缺 dense 的条目 -> <cap-dir>/retry_worklist.jsonl
  run    按分片重跑          -> <cap-dir>/retry_shard{N}.jsonl
  merge  成功项按 path 覆盖回原 shard，保持行序与图片一一对应
"""
import argparse, glob, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch import run_pool
from common import (DENSE, SHORT, ask_vlm, b64, iter_jsonl, iter_shards,
                    make_clients, round_robin, take_shard, write_jsonl)

WORKLIST = "retry_worklist.jsonl"


def need_retry(r):
    return "error" in r or not r.get("gemma_dense")


def cmd_scan(args):
    wl = os.path.join(args.cap_dir, WORKLIST)
    n = 0
    keys = ("id", "shard", "path", "gt_caption")
    with open(wl, "w") as fo:
        for f in sorted(glob.glob(os.path.join(args.cap_dir, "shard*.jsonl"))):
            k = 0
            for r in iter_jsonl(f):
                if need_retry(r):
                    write_jsonl(fo, {kk: r[kk] for kk in keys if kk in r})
                    k += 1
            print(f"  {os.path.basename(f)}: {k} 条待补", flush=True)
            n += k
    print(f"-> {wl}  共 {n} 条")


def cmd_run(args):
    wl = os.path.join(args.cap_dir, WORKLIST)
    if not os.path.exists(wl):
        raise SystemExit(f"缺 {wl}，先跑 scan")
    from PIL import Image
    clients = make_clients(args.urls)
    pick = round_robin(clients)
    items = take_shard(list(iter_jsonl(wl)), args.shard, args.num_shards)

    def work(it):
        rec = dict(it)
        try:
            enc = b64(Image.open(it["path"]).convert("RGB"))
            t0 = time.perf_counter()
            rec["gemma_short"] = ask_vlm(clients, args.model, enc, SHORT, 160, pick=pick)
            rec["gemma_dense"] = ask_vlm(clients, args.model, enc, DENSE, 320, pick=pick)
            rec["dt_s"] = round(time.perf_counter() - t0, 2)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        return rec

    run_pool(items, work, os.path.join(args.cap_dir, f"retry_shard{args.shard}.jsonl"),
             f"retry{args.shard}", workers=args.concurrency)


def cmd_merge(args):
    """把补齐成功项覆盖回原 shard。按 path 对齐（path 全局唯一）。"""
    fixed = {r["path"]: r for r in iter_shards(args.cap_dir, "retry_shard*.jsonl")
             if r.get("gemma_dense") and "error" not in r}
    print(f"可用补齐结果 {len(fixed)} 条")
    if not fixed:
        return
    tf = tl = 0
    for f in sorted(glob.glob(os.path.join(args.cap_dir, "shard*.jsonl"))):
        tmp, nf, nl = f + ".tmp", 0, 0
        with open(f) as fi, open(tmp, "w") as fo:
            for line in fi:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r["path"] in fixed:
                    write_jsonl(fo, fixed[r["path"]])
                    nf += 1
                else:
                    fo.write(line)
                    nl += need_retry(r)
        os.replace(tmp, f)
        tf, tl = tf + nf, tl + nl
        print(f"  {os.path.basename(f)}: 补齐 {nf}，仍缺 {nl}")
    print(f"合计补齐 {tf}，仍缺 {tl}")


def main():
    ap = argparse.ArgumentParser(description="阶段1b：补齐 caption 失败项")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="提取待补条目")
    s.add_argument("--cap-dir", required=True)

    r = sub.add_parser("run", help="重跑待补条目")
    r.add_argument("--cap-dir", required=True)
    r.add_argument("--urls", required=True)
    r.add_argument("--model", default="/dev/shm/models/gemma-4-26B-A4B-it")
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--num-shards", type=int, default=8)
    r.add_argument("--concurrency", type=int, default=16)

    m = sub.add_parser("merge", help="合并回原 shard")
    m.add_argument("--cap-dir", required=True)

    args = ap.parse_args()
    {"scan": cmd_scan, "run": cmd_run, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    main()
