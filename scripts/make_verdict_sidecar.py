#!/usr/bin/env python3
"""把 verify_full.jsonl 摊平成与 clean 逐框对齐的边车文件。

动机：verify_full.jsonl 的主键是 (path, phrase)，只覆盖 40.8% 的 (短语,框) 对
（每图前 12 个短语 × 每短语首框）。下游拿它做过滤时无法区分「判 NO」与「没审过」，
这是唯一的歧义来源。边车为 clean 里**每一个框位**显式给出一个槽：
"YES" / "NO" / "TOO_SMALL" / "ERROR" / "UNPARSED" / null（null = 未审）。

输出与 clean 行一一对应（含 grounding 为空的图），按 path 关联，框按下标对齐。

纯 CPU，两趟：先把 verify 读成 path -> {phrase: verdict}，再流式改写 clean。
"""
import argparse, glob, json, os, sys, time
from collections import Counter


def load_verdicts(path):
    """path -> {phrase: verdict}。短语字符串做 interning，30M 条里重复率很高。"""
    d = {}
    cache = {}
    n = 0
    t0 = time.perf_counter()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            p = r["path"]
            ph = cache.setdefault(r["phrase"], r["phrase"])
            sub = d.get(p)
            if sub is None:
                sub = d[p] = {}
            sub[ph] = r["verdict"]
            n += 1
            if n % 5_000_000 == 0:
                print(f"[idx] {n} lines {(time.perf_counter()-t0)/60:.1f}min",
                      flush=True)
    uniq = sum(len(v) for v in d.values())
    print(f"[idx] 读 {n} 行 → {len(d)} 图 / {uniq} 个 (path,phrase)"
          f"{'' if uniq == n else f'  ← 有 {n-uniq} 行重复主键（续跑追加所致，后写覆盖）'}",
          flush=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", default="out/clean")
    ap.add_argument("--pattern", default="clean_shard*.jsonl")
    ap.add_argument("--verify", default="out/verify_full.jsonl")
    ap.add_argument("--out-dir", default="out/verdict")
    args = ap.parse_args()

    d = load_verdicts(args.verify)
    os.makedirs(args.out_dir, exist_ok=True)

    stat = Counter()
    k_audited = Counter()      # 被审短语的图内序号分布，用来验证「前 12 个」这条规则
    bi_audited = Counter()     # 被审框的下标分布，用来验证「只首框」这条规则
    t0 = time.perf_counter()

    for src in sorted(glob.glob(os.path.join(args.clean_dir, args.pattern))):
        dst = os.path.join(args.out_dir,
                           os.path.basename(src).replace("clean_", "verdict_"))
        with open(src) as fi, open(dst, "w") as fo:
            for line in fi:
                r = json.loads(line)
                sub = d.get(r["path"]) or {}
                out, n_pair, n_aud = {}, 0, 0
                for k, (phrase, boxes) in enumerate((r.get("grounding") or {}).items()):
                    v = sub.get(phrase)
                    slots = [None] * len(boxes)
                    if v is not None:
                        slots[0] = v
                        n_aud += 1
                        k_audited[k] += 1
                        bi_audited[0] += 1
                        stat[v] += 1
                    out[phrase] = slots
                    n_pair += len(boxes)
                stat["pair"] += n_pair
                stat["img"] += 1
                fo.write(json.dumps({"path": r["path"], "n_pair": n_pair,
                                     "n_audited": n_aud, "verdict": out},
                                    ensure_ascii=False) + "\n")
        print(f"[out] {dst} {(time.perf_counter()-t0)/60:.1f}min", flush=True)

    aud = sum(v for k, v in stat.items() if k not in ("pair", "img"))
    print(f"\n图 {stat['img']}  框位 {stat['pair']}  有 verdict {aud} "
          f"({aud/stat['pair']*100:.2f}%)  null {stat['pair']-aud}")
    for k in ("YES", "NO", "TOO_SMALL", "ERROR", "UNPARSED"):
        if stat[k]:
            print(f"  {k:10s} {stat[k]:>12,}")
    print("被审短语 k 分布:", dict(sorted(k_audited.items())))
    print("被审框 box_idx 分布:", dict(sorted(bi_audited.items())))
    miss = sum(len(v) for v in d.values()) - aud
    print(f"verify 里对不上 clean 的 (path,phrase): {miss}")


if __name__ == "__main__":
    main()
