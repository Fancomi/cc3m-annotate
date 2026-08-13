#!/usr/bin/env python3
"""规则消融：在一次 caption 索引之上，把多套清洗配置全部跑出来。

为什么不用 run/ab_clean.sh 逐档跑：每次 s3_clean.py 启动都要重建 289 万条
caption 索引（约 2.5 分钟）。要试 7 套配置就是 18 分钟纯等待。这里索引只建一次，
配置在内存里逐套应用，且直接调用 s3_clean.clean_one —— 与生产逻辑完全一致，
不是另写一份。

产出 out/ab/<档名>/clean_shard0.jsonl，可直接喂 run/ab_verify.sh 校验。

用法:
    python3 scripts/ablate_rules.py --imgs 20000
    VARIANTS="base minpx all" python3 scripts/ablate_rules.py   # 只跑部分
"""
import argparse, glob, json, os, sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from common import iter_jsonl, norm_phrase, write_jsonl
from s3_clean import clean_one

# 所有档位共用的短语级规则（已定案）：vague + abstract + 整词 garbled + min_words 1 + dup + xdup
COMMON = dict(vague=True, abstract=True, garbled=True, word_boundary=True,
              min_words=1, dup=True, xdup=True)

# 逐框规则的消融。base 不做任何面积/像素过滤 —— 小物体全留，作为对照基线。
# 其余每档只开一条，最后 all 全开，这样每条规则的单独贡献可以直接读出来。
VARIANTS = {
    "c":       dict(min_area=0.02, min_px=0,  max_ratio=0,  max_cover=0,    max_boxes=0, edge=False),
    "base":    dict(min_area=0,    min_px=0,  max_ratio=0,  max_cover=0,    max_boxes=0, edge=False),
    "minpx":   dict(min_area=0,    min_px=48, max_ratio=0,  max_cover=0,    max_boxes=0, edge=False),
    "minpx32": dict(min_area=0,    min_px=32, max_ratio=0,  max_cover=0,    max_boxes=0, edge=False),
    "ratio":   dict(min_area=0,    min_px=0,  max_ratio=8., max_cover=0,    max_boxes=0, edge=False),
    "cover":   dict(min_area=0,    min_px=0,  max_ratio=0,  max_cover=0.95, max_boxes=0, edge=False),
    "nbox":    dict(min_area=0,    min_px=0,  max_ratio=0,  max_cover=0,    max_boxes=8, edge=False),
    "edge":    dict(min_area=0,    min_px=0,  max_ratio=0,  max_cover=0,    max_boxes=0, edge=True),
    "all":     dict(min_area=0,    min_px=48, max_ratio=8., max_cover=0.95, max_boxes=8, edge=True),
    # 消融结论：ratio(-0.7pt) 与 nbox(-0.4pt) 精度反降，被数据否掉；
    # minpx/area 精度提升全靠砍量换（有效信号 -18%~-20%），且小框判 NO 大半是
    # 判定器在 20px 尺度看不清（nose 在大框 5/5 全对、小框屡判 NO），不能归为数据错。
    # 只留 cover(+0.8pt) 与 edge(+0.1pt) 这两条几乎零代价的。
    "rec":     dict(min_area=0,    min_px=0,  max_ratio=0,  max_cover=0.95, max_boxes=0, edge=True),
}
DESC = {
    "c":       "现行 C 档（面积>=2%）",
    "base":    "基线：不做任何框大小过滤（小物体全保留）",
    "minpx":   "base + 框短边 >= 48px",
    "minpx32": "base + 框短边 >= 32px",
    "ratio":   "base + 长宽比 <= 8",
    "cover":   "base + 单框覆盖 <= 95%",
    "nbox":    "base + 单短语框数 <= 8",
    "edge":    "base + 删贴边窄条",
    "all":     "base + 上述五条全开",
    "rec":     "推荐档：base + 覆盖<=95% + 删贴边窄条（小物体全留）",
}


def main():
    R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground", default=os.path.join(R, "out", "ground", "ground_shard0.jsonl"))
    ap.add_argument("--cap-dir", default=os.path.join(R, "out", "caption"))
    ap.add_argument("--ab", default=os.path.join(R, "out", "ab"))
    ap.add_argument("--imgs", type=int, default=20000)
    ap.add_argument("--variants", default=os.environ.get("VARIANTS", " ".join(VARIANTS)))
    args = ap.parse_args()

    picked = args.variants.split()
    for v in picked:
        if v not in VARIANTS:
            sys.exit(f"未知档位 {v}，可选 {list(VARIANTS)}")

    recs = []
    for i, r in enumerate(iter_jsonl(args.ground)):
        if i >= args.imgs:
            break
        if r.get("grounding"):
            recs.append(r)
    print(f"ground 取 {len(recs)} 图", flush=True)

    # caption 与 ground 的分片口径不同（ground 按 id%8，caption 按 tsv 顺序），
    # 必须扫全部分片才能凑齐；只保留需要的 path 以免 289 万条全驻留。
    need = {r["path"] for r in recs}
    cap = {}
    for f in sorted(glob.glob(os.path.join(args.cap_dir, "shard*.jsonl"))):
        for r in iter_jsonl(f):
            if r["path"] in need and r.get("gemma_dense"):
                cap[r["path"]] = norm_phrase(r["gemma_dense"])
        print(f"  {os.path.basename(f)} -> 累计 {len(cap)}/{len(need)}", flush=True)
        if len(cap) == len(need):
            break
    print(f"caption 索引 {len(cap)}/{len(need)}", flush=True)

    results = {}
    for v in picked:
        cfg = argparse.Namespace(**COMMON, **VARIANTS[v])
        stat = Counter()
        n_in = n_out = 0
        d = os.path.join(args.ab, v)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "clean_shard0.jsonl"), "w") as fo:
            for r in recs:
                wh = tuple(r.get("img_wh") or (0, 0))
                g = clean_one(r["grounding"], cap.get(r["path"], ""), wh, cfg, stat)
                n_in += len(r["grounding"])
                n_out += len(g)
                write_jsonl(fo, {"id": r["id"], "shard": r.get("shard"), "path": r["path"],
                                 "img_wh": r.get("img_wh"), "grounding": g,
                                 "n_phrase": len(g), "n_box": sum(len(x) for x in g.values())})
        results[v] = (n_in, n_out, stat)
        print(f"[{v:8}] {n_out:>7,} / {n_in:,} 短语（保留 {n_out/n_in*100:5.1f}%）  "
              + "  ".join(f"{k}={n}" for k, n in stat.most_common() if n), flush=True)

    print(f"\n{'档位':<9}{'短语/图':>9}{'保留':>8}   说明")
    print("-" * 72)
    for v in picked:
        n_in, n_out, _ = results[v]
        print(f"{v:<9}{n_out/len(recs):>9.2f}{n_out/n_in*100:>7.1f}%   {DESC[v]}")
    print(f"\n产出 {args.ab}/<档名>/clean_shard0.jsonl —— 接着跑："
          f'\n  SAMPLE=800 VARIANTS="{" ".join(picked)}" bash run/ab_verify.sh')


if __name__ == "__main__":
    main()
