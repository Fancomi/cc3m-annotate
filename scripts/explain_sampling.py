#!/usr/bin/env python3
"""复现阶段4 的抽样与任务展开，逐层解释 7942 是怎么来的，并量化 max_boxes 截断的损失。

与 s4_verify.py 用完全相同的 seed / 排序 / 过滤逻辑，所以结论对得上那次真实运行。
"""
import glob, json, os, random, sys

sys.path.insert(0, "/root/paddlejob/workspace/env_run/penghaotian/vision_encoder/cc3m-annotate/src")
from common import VAGUE, iter_jsonl

IN_DIR = "/root/paddlejob/workspace/env_run/penghaotian/vision_encoder/cc3m-annotate/out/clean"
SAMPLE = 1000
MAX_BOXES = 12
SEED = 0

rows = []
for f in sorted(glob.glob(os.path.join(IN_DIR, "clean_shard*.jsonl"))):
    rows += [r for r in iter_jsonl(f) if r.get("grounding")]
print(f"[0] clean 里有 grounding 的图: {len(rows):,}")

random.seed(SEED)
random.shuffle(rows)
rows = rows[:SAMPLE]
print(f"[1] 抽样后: {len(rows)} 图")

n_phrase_total = sum(len(r["grounding"]) for r in rows)
print(f"[2] 这 1000 图的短语总数（未截断）: {n_phrase_total:,}"
      f"  平均 {n_phrase_total/len(rows):.2f}/图")

# 逐层扣减
cut_by_maxboxes = 0
cut_by_vague = 0
tasks = 0
per_img = []
capped_imgs = 0
for r in rows:
    items = list(r["grounding"].items())
    per_img.append(len(items))
    if len(items) > MAX_BOXES:
        capped_imgs += 1
        cut_by_maxboxes += len(items) - MAX_BOXES
    for k, (phrase, boxes) in enumerate(items):
        if k >= MAX_BOXES:
            break
        if VAGUE.match(phrase.strip()):
            cut_by_vague += 1
            continue
        tasks += 1

print(f"[3] 被 max_boxes={MAX_BOXES} 截掉: {cut_by_maxboxes:,} 个短语"
      f"（涉及 {capped_imgs} 张图，占 {capped_imgs/len(rows)*100:.1f}%）")
print(f"[4] 被 VAGUE 正则跳过: {cut_by_vague:,} 个短语")
print(f"[5] 最终任务数: {tasks:,}   ← 应等于 verify_clean.jsonl 的行数")

# 分布
per_img.sort()
def pct(p):
    return per_img[min(len(per_img) - 1, int(len(per_img) * p))]
print(f"\n每图短语数分布: p50={pct(0.5)} p75={pct(0.75)} p90={pct(0.9)} "
      f"p99={pct(0.99)} max={per_img[-1]}")
print(f"短语数 <= {MAX_BOXES} 的图占比: "
      f"{sum(1 for n in per_img if n <= MAX_BOXES)/len(per_img)*100:.1f}%")

# 实际产出核对
vf = "/root/paddlejob/workspace/env_run/penghaotian/vision_encoder/cc3m-annotate/out/verify_clean.jsonl"
if os.path.exists(vf):
    actual = sum(1 for _ in open(vf))
    print(f"\n实际 verify_clean.jsonl 行数: {actual:,}  (复现值 {tasks:,}, "
          f"{'一致 ✅' if actual == tasks else '不一致 ⚠️'})")

# 截断是否有偏：被截掉的短语 vs 保留的短语，词数分布差异
from common import content_words
kept_wl, cut_wl = [], []
for r in rows:
    for k, (phrase, _) in enumerate(r["grounding"].items()):
        (kept_wl if k < MAX_BOXES else cut_wl).append(len(content_words(phrase)))
if cut_wl:
    print(f"\n截断偏差检查（实词数均值）: 前 {MAX_BOXES} 个 = {sum(kept_wl)/len(kept_wl):.2f}, "
          f"被截掉的 = {sum(cut_wl)/len(cut_wl):.2f}")
