#!/usr/bin/env python3
"""去重收口：把 caption 分片压实成「每图一行、优先成功记录」。

为什么需要：阶段1 的续传是 append 语义 —— 补跑一条 error 记录时，
成功结果追加在原 error 行之后，同一张图会出现多行。下游按 path 索引
（阶段2/3 建 dict）时后写的会覆盖先写的，行为正确但文件冗余，
且行数不再等于图数，统计会偏。

规则：同一 path 的多行里保留「成功且最后写入」的那条；若全部失败则保留最后一条
error（供下一轮重试）。原地覆写（先写 .tmp 再 rename），保持首次出现的顺序。

用法:
  python compact.py --dir out/caption              # 压实
  python compact.py --dir out/caption --dry-run    # 只报告不改文件
"""
import argparse, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import is_ok, iter_jsonl, write_jsonl


def compact_file(path, dry=False):
    order, best = [], {}
    for r in iter_jsonl(path):
        k = r["path"]
        if k not in best:
            order.append(k)
            best[k] = r
        elif is_ok(r) or not is_ok(best[k]):
            best[k] = r        # 成功覆盖失败；同为成功时取后写的
    n_line = sum(1 for _ in open(path))
    n_img = len(order)
    n_ok = sum(1 for k in order if is_ok(best[k]))
    if not dry and n_line != n_img:
        tmp = path + ".tmp"
        with open(tmp, "w") as fo:
            for k in order:
                write_jsonl(fo, best[k])
        os.replace(tmp, path)
    return n_line, n_img, n_ok


def main():
    ap = argparse.ArgumentParser(description="caption 分片去重收口")
    ap.add_argument("--dir", required=True, help="caption 输出目录")
    ap.add_argument("--pattern", default="shard*.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, a.pattern)))
    if not files:
        raise SystemExit(f"{a.dir} 下没有匹配 {a.pattern} 的文件")
    tl = ti = to = 0
    for f in files:
        nl, ni, no = compact_file(f, a.dry_run)
        tl, ti, to = tl + nl, ti + ni, to + no
        print(f"  {os.path.basename(f)}: {nl} 行 -> {ni} 图，其中成功 {no}"
              f"{'（未写盘）' if a.dry_run else ''}")
    print(f"合计 {tl} 行 -> {ti} 图，成功 {to}（{to/max(1,ti)*100:.2f}%），仍缺 {ti-to}")


if __name__ == "__main__":
    main()
