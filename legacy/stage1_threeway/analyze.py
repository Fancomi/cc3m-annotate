#!/usr/bin/env python3
"""三方 + 交叉实验的定量分析。

产出 ANALYSIS.md：各链路的产出密度、词汇多样性、与 CC3M 原标注的名词重合度、
以及 crossfeed（Qwen caption 驱动 F2 grounding）相对基线的增益。
"""
import json, glob, os, re, statistics, collections

A = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(A, "out")
STOP = set("""a an the of in on at to for with and or is are was were be been being this that these those
his her its their my your our it he she they them him us we you i as by from into over under near
image photo picture shows showing view background foreground left right center middle top bottom front back
there here which who whom whose what while during also very more most some any all both each other another
one two three several many few little large small big
""".split())


def nouns(text):
    """粗粒度取实义词，用于跨链路词汇重合度对比（非严格 POS 标注）。"""
    return {w for w in re.findall(r"[a-z][a-z\-']+", (text or "").lower())
            if len(w) > 2 and w not in STOP}


rows = [json.loads(l) for l in open(os.path.join(OUT, "merged.jsonl"))]
cf = {}
for f in glob.glob(os.path.join(OUT, "crossfeed_shard*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if "error" not in r:
            cf[r["id"]] = r

L = ["# cc3m 标注三方对照 · 定量分析\n"]
L.append(f"样本 {len(rows)} 张（cc3m 576 个 shard 均匀抽样），三方链路覆盖 "
         f"{sum(1 for r in rows if all(k in r for k in ('florence2','novic','qwen')))} 张。\n")

# ---- 1. 产出密度 ----
L.append("## 1. 产出密度\n")
L.append("| 指标 | 值 |")
L.append("|---|---|")
gt_w = [len(r["gt_caption"].split()) for r in rows]
L.append(f"| CC3M 原标注 平均词数 | {statistics.mean(gt_w):.1f} |")
for key, tag in [("caption", "F2 `<CAPTION>`"), ("detailed", "F2 `<DETAILED_CAPTION>`"),
                 ("more_detailed", "F2 `<MORE_DETAILED_CAPTION>`")]:
    w = [len(r["florence2"][key].split()) for r in rows if "florence2" in r]
    L.append(f"| {tag} 平均词数 | {statistics.mean(w):.1f} |")
qs = [len(r["qwen"]["short"].split()) for r in rows if "qwen" in r]
qd = [len(r["qwen"]["dense"].split()) for r in rows if "qwen" in r]
L.append(f"| Qwen short 平均词数 | {statistics.mean(qs):.1f} |")
L.append(f"| Qwen dense 平均词数 | {statistics.mean(qd):.1f} |")
od = [len(r["florence2"]["od"]["labels"]) for r in rows if "florence2" in r]
dn = [len(r["florence2"]["dense_region_caption"]["labels"]) for r in rows if "florence2" in r]
gr = [len(r["florence2"]["grounding"] or {}) for r in rows if "florence2" in r]
L.append(f"| F2 `<OD>` 平均框数 | {statistics.mean(od):.1f} |")
L.append(f"| F2 `<DENSE_REGION_CAPTION>` 平均区域数 | {statistics.mean(dn):.1f} |")
L.append(f"| F2 grounding 平均唯一短语数 | {statistics.mean(gr):.1f} |")
L.append("")

# ---- 2. 词汇多样性 ----
L.append("## 2. 词汇多样性（2880 张图上的去重词表规模）\n")
L.append("| 链路 | 唯一标签/名词数 | 说明 |")
L.append("|---|---|---|")
od_v = collections.Counter(l for r in rows if "florence2" in r for l in r["florence2"]["od"]["labels"])
dn_v = collections.Counter(l for r in rows if "florence2" in r for l in r["florence2"]["dense_region_caption"]["labels"])
gr_v = collections.Counter(k for r in rows if "florence2" in r for k in (r["florence2"]["grounding"] or {}))
nv1 = collections.Counter(r["novic"]["topk"][0][0] for r in rows if "novic" in r)
nv_all = collections.Counter(t[0] for r in rows if "novic" in r for t in r["novic"]["topk"])
L.append(f"| F2 `<OD>` 标签 | {len(od_v)} | 检测头，偏 COCO/O365 风格类目 |")
L.append(f"| F2 dense region caption | {len(dn_v)} | 短语级，含修饰语 |")
L.append(f"| F2 grounding 短语 | {len(gr_v)} | 由输入 caption 决定 |")
L.append(f"| NOVIC top-1 | {len(nv1)} | 生成式，42919 名词空间 |")
L.append(f"| NOVIC top-10 全体 | {len(nv_all)} | |")
L.append("")
L.append("最常见 top-1（NOVIC）：" + ", ".join(f"{k}({v})" for k, v in nv1.most_common(10)))
L.append("")
L.append("最常见标签（F2 OD）：" + ", ".join(f"{k}({v})" for k, v in od_v.most_common(10)))
L.append("")

# ---- 3. 与 CC3M 原标注的实义词重合 ----
L.append("## 3. 与 CC3M 原标注的实义词重合率\n")
L.append("衡量各链路是否覆盖到人工标注提到的概念（分母为原标注实义词数）。\n")
L.append("| 链路 | 平均召回 | 平均新增词数 |")
L.append("|---|---|---|")


def recall_stats(get_text):
    rec, extra = [], []
    for r in rows:
        g = nouns(r["gt_caption"])
        if not g:
            continue
        p = get_text(r)
        if p is None:
            continue
        p = nouns(p)
        rec.append(len(g & p) / len(g))
        extra.append(len(p - g))
    return statistics.mean(rec), statistics.mean(extra), len(rec)


for tag, fn in [
    ("F2 `<CAPTION>`", lambda r: r["florence2"]["caption"] if "florence2" in r else None),
    ("F2 `<MORE_DETAILED_CAPTION>`", lambda r: r["florence2"]["more_detailed"] if "florence2" in r else None),
    ("F2 `<OD>` 标签拼接", lambda r: " ".join(r["florence2"]["od"]["labels"]) if "florence2" in r else None),
    ("F2 dense region 拼接", lambda r: " ".join(r["florence2"]["dense_region_caption"]["labels"]) if "florence2" in r else None),
    ("NOVIC top-10 拼接", lambda r: " ".join(t[0] for t in r["novic"]["topk"]) if "novic" in r else None),
    ("Qwen short", lambda r: r["qwen"]["short"] if "qwen" in r else None),
    ("Qwen dense", lambda r: r["qwen"]["dense"] if "qwen" in r else None),
]:
    m, e, n = recall_stats(fn)
    L.append(f"| {tag} | {m*100:.1f}% | {e:.1f} |")
L.append("")

# ---- 4. crossfeed ----
if cf:
    L.append("## 4. 交叉实验：Qwen dense caption 驱动 F2 grounding\n")
    L.append(f"覆盖 {len(cf)} 张。前期已确认 grounding 的 detail 上限由输入文本决定，"
             "这里用更长的 Qwen caption（平均 174 词）替换 F2 自己的 more_detailed（89 词）。\n")
    ph_q = [r["n_phrase_qwen"] for r in cf.values()]
    ph_f = [r["n_phrase_f2"] for r in cf.values()]
    bx_q = [r["n_box_qwen"] for r in cf.values()]
    bx_f = [r["n_box_f2"] for r in cf.values()]
    L.append("| 驱动文本 | 平均词数 | 平均唯一短语 | 平均框数 |")
    L.append("|---|---|---|---|")
    L.append(f"| F2 `<MORE_DETAILED_CAPTION>` | {statistics.mean([r['f2_more_detailed_words'] for r in cf.values()]):.0f} "
             f"| {statistics.mean(ph_f):.1f} | {statistics.mean(bx_f):.1f} |")
    L.append(f"| Qwen dense | {statistics.mean([r['qwen_dense_words'] for r in cf.values()]):.0f} "
             f"| {statistics.mean(ph_q):.1f} | {statistics.mean(bx_q):.1f} |")
    L.append("")
    win = sum(1 for r in cf.values() if r["n_phrase_qwen"] > r["n_phrase_f2"])
    L.append(f"- Qwen caption 拿到更多短语的比例：**{win/len(cf)*100:.1f}%**（{win}/{len(cf)}）")
    L.append(f"- 短语数增益中位：**{statistics.median([r['n_phrase_qwen']-r['n_phrase_f2'] for r in cf.values()]):+.0f}**")
    tr = sum(1 for r in cf.values() if r["truncated"])
    L.append(f"- 因超过 Florence-2 文本位置上限而被截断：{tr}/{len(cf)}（{tr/len(cf)*100:.1f}%）")
    L.append("")

with open(os.path.join(A, "ANALYSIS.md"), "w") as f:
    f.write("\n".join(L) + "\n")
print("-> ANALYSIS.md")
