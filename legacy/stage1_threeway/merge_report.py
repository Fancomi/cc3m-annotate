#!/usr/bin/env python3
"""三方结果合并 + 对照报告。

三条链路的产出在 out/*.jsonl，按 id 对齐后生成：
  merged.jsonl   每张图一行，含 gt / florence2 五个任务 / novic topk / qwen 两级 caption
  REPORT.md      统计与抽样对照
"""
import glob, json, os, statistics, collections

A = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(A, "out")


def load(pat):
    d = {}
    for f in glob.glob(os.path.join(OUT, pat)):
        for l in open(f):
            try:
                r = json.loads(l)
            except Exception:
                continue
            d[r["id"]] = r
    return d


flo, nov, qwe = load("florence2_shard*.jsonl"), load("novic_shard*.jsonl"), load("qwen_shard*.jsonl")
wl = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(A, "worklist.jsonl"))}
print(f"florence2={len(flo)} novic={len(nov)} qwen={len(qwe)} worklist={len(wl)}")

merged = []
for i in sorted(wl):
    r = {"id": i, "path": wl[i]["path"], "gt_caption": wl[i]["gt_caption"]}
    f = flo.get(i)
    if f and "error" not in f:
        r["florence2"] = {
            "caption": f.get("<CAPTION>"), "detailed": f.get("<DETAILED_CAPTION>"),
            "more_detailed": f.get("<MORE_DETAILED_CAPTION>"),
            "od": f.get("<OD>"), "dense_region_caption": f.get("<DENSE_REGION_CAPTION>"),
            "grounding": f.get("grounding"), "dt_s": f.get("dt_s")}
    n = nov.get(i)
    if n and "error" not in n:
        r["novic"] = {"topk": n["novic_topk"], "dt_ms": n.get("dt_ms")}
    q = qwe.get(i)
    if q and "error" not in q:
        r["qwen"] = {"short": q.get("qwen_short"), "dense": q.get("qwen_dense"), "dt_s": q.get("dt_s")}
    merged.append(r)

with open(os.path.join(OUT, "merged.jsonl"), "w") as fo:
    for r in merged:
        fo.write(json.dumps(r, ensure_ascii=False) + "\n")

L = []
L.append("# cc3m 三方标注对照\n")
L.append(f"样本：worklist {len(wl)} 张（576 个 shard 均匀抽样）\n")
L.append("| 链路 | 覆盖 | 中位耗时 | 输出 |")
L.append("|---|---|---|---|")
fok = [r for r in flo.values() if "error" not in r]
nok = [r for r in nov.values() if "error" not in r]
qok = [r for r in qwe.values() if "error" not in r]
if fok:
    L.append(f"| Florence-2 (5 任务+grounding) | {len(fok)}/{len(wl)} | "
             f"{statistics.median([r['dt_s'] for r in fok]):.2f} s | 三级 caption + OD + dense + grounding |")
if nok:
    L.append(f"| NOVIC | {len(nok)}/{len(wl)} | "
             f"{statistics.median([r['dt_ms'] for r in nok]):.1f} ms | top-10 开放词汇名词 |")
if qok:
    L.append(f"| Qwen3.6-35B-FP8 | {len(qok)}/{len(wl)} | "
             f"{statistics.median([r['dt_s'] for r in qok]):.1f} s | short + dense caption |")
L.append("")
if fok:
    L.append("## Florence-2 产出规模")
    L.append(f"- `<OD>` 平均检出 {statistics.mean([len(r['<OD>']['labels']) for r in fok]):.1f} 个框")
    L.append(f"- `<DENSE_REGION_CAPTION>` 平均 {statistics.mean([len(r['<DENSE_REGION_CAPTION>']['labels']) for r in fok]):.1f} 个区域")
    L.append(f"- grounding 平均 {statistics.mean([len(r.get('grounding',{})) for r in fok]):.1f} 个唯一短语")
    for k, tag in [("<CAPTION>", "caption"), ("<DETAILED_CAPTION>", "detailed"),
                   ("<MORE_DETAILED_CAPTION>", "more_detailed")]:
        w = [len(str(r.get(k, "")).split()) for r in fok]
        L.append(f"- `{k}` 平均 {statistics.mean(w):.0f} 词")
    labs = collections.Counter(l for r in fok for l in r["<OD>"]["labels"])
    L.append(f"- `<OD>` 标签词汇量 {len(labs)}，最常见：" +
             ", ".join(f"{k}({v})" for k, v in labs.most_common(12)))
    L.append("")
if nok:
    vocab = collections.Counter(t[0] for r in nok for t in r["novic_topk"][:1])
    L.append("## NOVIC top-1 词汇")
    L.append(f"- {len(nok)} 张图产生 {len(vocab)} 个不同 top-1 名词")
    L.append("- 最常见：" + ", ".join(f"{k}({v})" for k, v in vocab.most_common(12)))
    L.append("")
L.append("## 抽样对照（前 6 张有全部三方结果的图）")
n = 0
for r in merged:
    if not all(k in r for k in ("florence2", "novic", "qwen")):
        continue
    L.append(f"\n### id={r['id']}  `{os.path.basename(r['path'])}`")
    L.append(f"- **CC3M GT**: {r['gt_caption']}")
    L.append(f"- **NOVIC top5**: " + " / ".join(f"{a}={b}%" for a, b in r["novic"]["topk"][:5]))
    L.append(f"- **F2 caption**: {r['florence2']['caption']}")
    L.append(f"- **F2 more_detailed**: {r['florence2']['more_detailed']}")
    L.append(f"- **F2 OD**: " + ", ".join(r["florence2"]["od"]["labels"]))
    L.append(f"- **F2 dense**: " + " | ".join(r["florence2"]["dense_region_caption"]["labels"]))
    L.append(f"- **F2 grounding**: " + ", ".join(f"{k}×{len(v)}" for k, v in r["florence2"]["grounding"].items()))
    L.append(f"- **Qwen short**: {r['qwen']['short']}")
    L.append(f"- **Qwen dense**: {r['qwen']['dense']}")
    n += 1
    if n >= 6:
        break
with open(os.path.join(A, "REPORT.md"), "w") as fo:
    fo.write("\n".join(L) + "\n")
print("-> out/merged.jsonl, REPORT.md")
