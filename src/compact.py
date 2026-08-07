#!/usr/bin/env python3
"""去重收口：把 caption 分片压实成「每图一行、优先成功记录」。

两种重复都要清：

1. **文件内重复** —— 阶段1 的续传是 append 语义：补跑一条 error 记录时，
   成功结果追加在原 error 行之后，同一张图在同一分片里出现多行。

2. **跨分片重复** —— 若两次运行用了不同的 `--num-shards`，或换过分片键，
   同一张图会落进不同的分片文件。实测踩过：首轮按 tsv 序号分片、补跑按 id
   分片，多出 16.8 万条跨分片重复。

规则：同一 path 全局只保留一条 —— 有成功记录就留最后写入的那条成功记录，
全是 error 才留最后一条 error（供下一轮重试）。

跨分片重复不能按「先出现者胜」处理：早分片里的 error 会顶掉晚分片里的成功结果，
那张图在阶段2 就没有 caption 可用，等于把已经跑出来的结果丢了。所以先全量扫一遍
定出每个 path 的胜者，再逐文件重写，胜者留在它自己所在的分片里。

原地覆写（先写 .tmp 再 rename），保持首次出现的顺序。

用法:
  python compact.py --dir out/caption              # 压实
  python compact.py --dir out/caption --dry-run    # 只报告不改文件
"""
import argparse, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import is_ok, iter_jsonl, write_jsonl


def _tag(seq, ok):
    """把「全局第几条记录 + 是否成功」压成一个整数，省下 2.9M 条元组的内存。"""
    return (seq << 1) | int(ok)


def pick_winners(files):
    """第一遍：全量扫描定胜者，返回 {path: tag}。

    seq 跨分片连续递增，所以后扫到的记录 tag 一定更大，「取最后写入」不必比大小，
    只需判断成功位：新记录成功、或旧记录失败，就换人。
    """
    win, seq = {}, 0
    for f in files:
        for r in iter_jsonl(f):
            k = r["path"]
            ok = is_ok(r)
            cur = win.get(k)
            if cur is None or ok or not (cur & 1):
                win[k] = _tag(seq, ok)
            seq += 1
    return win


def compact_file(path, seq, win, dry=False):
    """第二遍：按胜者表重写单个分片。

    seq 是本文件首条记录的全局序号，必须与 pick_winners 的计数完全对齐 ——
    坏行两遍都跳过且都不占序号。返回 (原始行数, 保留数, 其中成功数, 下一个序号)。
    """
    n_line = n_keep = n_ok = 0
    tmp = path + ".tmp"
    fo = None if dry else open(tmp, "w")
    try:
        with open(path) as fi:
            for line in fi:
                n_line += 1
                try:
                    r = json.loads(line)
                except Exception:
                    continue                      # 坏行：不占序号，直接丢掉
                ok = is_ok(r)
                if win.get(r["path"]) == _tag(seq, ok):
                    n_keep += 1
                    n_ok += int(ok)
                    if fo:
                        write_jsonl(fo, r)
                seq += 1
    finally:
        if fo:
            fo.close()
    if not dry:
        if n_keep != n_line:
            os.replace(tmp, path)
        else:
            os.remove(tmp)                        # 没变化就别动原文件的 mtime
    return n_line, n_keep, n_ok, seq


def main():
    ap = argparse.ArgumentParser(description="caption 分片去重收口")
    ap.add_argument("--dir", required=True, help="caption 输出目录")
    ap.add_argument("--pattern", default="shard*.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, a.pattern)))
    if not files:
        raise SystemExit(f"{a.dir} 下没有匹配 {a.pattern} 的文件")

    win = pick_winners(files)
    seq = 0
    tl = ti = to = 0
    for f in files:
        nl, ni, no, seq = compact_file(f, seq, win, a.dry_run)
        tl, ti, to = tl + nl, ti + ni, to + no
        print(f"  {os.path.basename(f)}: {nl} 行 -> {ni} 图，其中成功 {no}"
              f"{'（未写盘）' if a.dry_run else ''}")
    print(f"合计 {tl} 行 -> {ti} 图，成功 {to}（{to/max(1,ti)*100:.2f}%），仍缺 {ti-to}")


if __name__ == "__main__":
    main()
