#!/usr/bin/env python3
"""统计报告：产出规模、短语密度、校验精度、过滤规则权衡。

用法:
  python report.py --cap-dir out/caption --ground-dir out/ground --clean-dir out/clean \
                   --verify out/verify.jsonl --out docs/RESULT.md
各输入均可选，缺谁就少对应章节。
"""
import argparse, os, statistics, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import iter_jsonl, iter_shards

BUCKETS = [(0, 0.005, "<0.5%"), (0.005, 0.02, "0.5-2%"), (0.02, 0.1, "2-10%"),
           (0.1, 0.4, "10-40%"), (0.4, 1.01, ">=40%")]
RULES = [("无过滤", 0, 1), ("面积>=0.5%", 0.005, 1), ("面积>=2%", 0.02, 1),
         ("短语>=2 词", 0, 2), ("面积>=0.5% 且 >=2 词", 0.005, 2),
         ("面积>=2% 且 >=2 词", 0.02, 2), ("面积>=2% 且 >=4 词", 0.02, 4)]


def md_table(head, rows):
    L = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    L += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return L


def sec_caption(d, L):
    n = err = 0
    sw, dw = [], []
    for r in iter_shards(d):
        n += 1
        if "error" in r or not r.get("gemma_dense"):
            err += 1
            continue
        sw.append(len(r["gemma_short"].split()))
        dw.append(len(r["gemma_dense"].split()))
    if not n:
        return
    L.append("\n## 1. caption 产出\n")
    L += md_table(["指标", "值"], [
        ["图片数", f"{n:,}"],
        ["缺 dense（需补跑）", f"{err:,} ({err/n*100:.2f}%)"],
        ["short 词数 中位/均值", f"{statistics.median(sw):.0f} / {statistics.mean(sw):.1f}"],
        ["dense 词数 中位/均值", f"{statistics.median(dw):.0f} / {statistics.mean(dw):.1f}"],
    ])


def sec_ground(d, pat, title, L):
    ph, bx, n, err = [], [], 0, 0
    for r in iter_shards(d, pat):
        n += 1
        if "error" in r:
            err += 1
            continue
        ph.append(r.get("n_phrase", 0))
        bx.append(r.get("n_box", 0))
    if not ph:
        return
    L.append(f"\n## {title}\n")
    L += md_table(["指标", "值"], [
        ["图片数", f"{n:,}"],
        ["出错", f"{err:,}"],
        ["短语/图 中位/均值", f"{statistics.median(ph):.0f} / {statistics.mean(ph):.1f}"],
        ["框/图 中位/均值", f"{statistics.median(bx):.0f} / {statistics.mean(bx):.1f}"],
        ["零短语图占比", f"{sum(1 for x in ph if x == 0)/len(ph)*100:.1f}%"],
    ])


def sec_verify(path, L):
    rows = [r for r in iter_jsonl(path)]
    if not rows:
        return
    c = Counter(r["verdict"] for r in rows)
    y, n = c.get("YES", 0), c.get("NO", 0)
    L.append("\n## 4. 校验精度（下界估计）\n")
    L.append(f"共 {len(rows):,} 对；判定分布 " +
             "  ".join(f"{k}={v}" for k, v in c.most_common()) + "\n")
    if y + n:
        L.append(f"**精度下界 {y/(y+n)*100:.1f}%**（YES/(YES+NO)）\n")

    ok = [r for r in rows if r["verdict"] in ("YES", "NO") and r.get("crop_frac") is not None]
    if ok:
        L.append("\n### 精度 vs 框面积\n")
        rs = []
        for lo, hi, name in BUCKETS:
            g = [r for r in ok if lo <= r["crop_frac"] < hi]
            if g:
                rs.append([name, f"{sum(r['verdict']=='YES' for r in g)/len(g)*100:.1f}%", len(g)])
        L += md_table(["框面积占比", "精度", "n"], rs)

        L.append("\n### 精度 vs 短语词数\n")
        rs = []
        for lo, hi, name in [(1, 2, "1 词"), (2, 4, "2-3 词"), (4, 7, "4-6 词"), (7, 99, "7+ 词")]:
            g = [r for r in ok if lo <= len(r["phrase"].split()) < hi]
            if g:
                rs.append([name, f"{sum(r['verdict']=='YES' for r in g)/len(g)*100:.1f}%", len(g)])
        L += md_table(["短语词数", "精度", "n"], rs)

        L.append("\n### 过滤规则权衡\n")
        rs = []
        for name, area, words in RULES:
            g = [r for r in ok if r["crop_frac"] >= area and len(r["phrase"].split()) >= words]
            if g:
                rs.append([name, f"{sum(r['verdict']=='YES' for r in g)/len(g)*100:.1f}%",
                           f"{len(g)/len(ok)*100:.1f}%"])
        L += md_table(["规则", "精度", "保留率"], rs)

    bad = Counter(r["phrase"] for r in rows if r["verdict"] == "NO")
    if bad:
        L.append("\n最常被否定的短语：" +
                 ", ".join(f"`{k}`({v})" for k, v in bad.most_common(10)) + "\n")


def main():
    ap = argparse.ArgumentParser(description="生成统计报告")
    ap.add_argument("--cap-dir")
    ap.add_argument("--ground-dir")
    ap.add_argument("--clean-dir")
    ap.add_argument("--verify")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    L = ["# cc3m 区域标注 · 产出统计\n"]
    if a.cap_dir and os.path.isdir(a.cap_dir):
        sec_caption(a.cap_dir, L)
    if a.ground_dir and os.path.isdir(a.ground_dir):
        sec_ground(a.ground_dir, "ground_shard*.jsonl", "2. grounding 原始产出", L)
    if a.clean_dir and os.path.isdir(a.clean_dir):
        sec_ground(a.clean_dir, "clean_shard*.jsonl", "3. 清洗后产出", L)
    if a.verify and os.path.exists(a.verify):
        sec_verify(a.verify, L)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
