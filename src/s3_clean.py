#!/usr/bin/env python3
"""阶段 3 · 清洗：过滤 grounding 产出里的噪声短语与低质框。

输入  <ground-dir>/ground_shard*.jsonl + <cap-dir>/shard*.jsonl
输出  <out>/clean_shard*.jsonl       同结构，grounding 已过滤

五条规则（各自可关）:
  vague    整体指代类短语（`the entire image`）—— ground 到全图，无区域价值
  garbled  归一化后无法在源 caption 中找到 —— 说明不是忠实摘抄，是解码漂移
  words    实词数不足 —— 单个虚词/标点无定位意义
  dup      同短语下 IoU>0.9 的重复框
  area     框面积占比过小 —— 实测面积 <0.5% 时精度仅 43.8%，>=10% 时 84.6%

默认只开前四条（无损清洗）。要做训练数据再加 --min-area 0.02 --min-words 2，
该档实测精度 80.3%、保留 66%，折合约 16 个有效短语/图。
"""
import argparse, glob, os, re, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import iter_jsonl, iter_shards, write_jsonl

VAGUE = re.compile(
    r"^(the|this|a|an)?\s*(entire|whole|overall)?\s*"
    r"(image|photo|photograph|picture|scene|view|frame|composition|background|foreground)s?$",
    re.I)
STOP = {"a", "an", "the", "of", "in", "on", "at", "to", "for", "with", "and", "or",
        "its", "his", "her", "their", "this", "that", "these", "those"}


def norm(s):
    """小写、去标点连字符、压空格 —— 使 `black-framed` 能匹配 caption 里的 `black framed`。"""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def clean_one(grounding, cap_norm, wh, args, stat):
    W, H = wh
    total = W * H
    out = {}
    for phrase, boxes in grounding.items():
        p = phrase.strip()
        if args.vague and VAGUE.match(p):
            stat["vague"] += 1
            continue
        if args.garbled and cap_norm and norm(p) and norm(p) not in cap_norm:
            stat["garbled"] += 1
            continue
        if len([w for w in norm(p).split() if w not in STOP]) < args.min_words:
            stat["words"] += 1
            continue
        kept = boxes
        if args.dup:
            uniq = []
            for b in kept:
                if all(iou(b, k) <= 0.9 for k in uniq):
                    uniq.append(b)
            stat["dup"] += len(kept) - len(uniq)
            kept = uniq
        if args.min_area and total:
            n0 = len(kept)
            kept = [b for b in kept if (b[2] - b[0]) * (b[3] - b[1]) / total >= args.min_area]
            stat["area"] += n0 - len(kept)
        if kept:
            out[p] = kept
        else:
            stat["empty"] += 1
    return out


def img_wh(rec):
    """优先取记录里的 img_wh；缺失则读图头（PIL 惰性打开，不解码像素）。"""
    wh = rec.get("img_wh")
    if wh and wh[0] and wh[1]:
        return tuple(wh)
    try:
        from PIL import Image
        with Image.open(rec["path"]) as im:
            return im.size
    except Exception:
        return (0, 0)


def main():
    ap = argparse.ArgumentParser(description="阶段3：短语清洗")
    ap.add_argument("--ground-dir", required=True)
    ap.add_argument("--cap-dir", required=True, help="用于判定短语是否忠实摘抄")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-area", type=float, default=0.0, help="框面积占比下限，训练数据推荐 0.02")
    ap.add_argument("--min-words", type=int, default=1, help="实词数下限，训练数据推荐 2")
    ap.add_argument("--no-vague", dest="vague", action="store_false")
    ap.add_argument("--no-garbled", dest="garbled", action="store_false")
    ap.add_argument("--no-dup", dest="dup", action="store_false")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只处理前 N 图（冒烟）")
    args = ap.parse_args()

    # caption 索引以 path 为键（path 全局唯一，id 会在 tsv 之间撞号）
    cap = {r["path"]: norm(r["gemma_dense"])
           for r in iter_shards(args.cap_dir) if r.get("gemma_dense")}
    print(f"caption 索引 {len(cap)} 条", flush=True)

    os.makedirs(args.out, exist_ok=True)
    stat = Counter()
    n_img = n_in = n_out = 0
    for f in sorted(glob.glob(os.path.join(args.ground_dir, "ground_shard*.jsonl"))):
        out_f = os.path.join(args.out, os.path.basename(f).replace("ground_", "clean_"))
        with open(out_f, "w") as fo:
            for r in iter_jsonl(f):
                if "error" in r or not r.get("grounding"):
                    continue
                g = clean_one(r["grounding"], cap.get(r["path"], ""), img_wh(r), args, stat)
                n_img += 1
                n_in += len(r["grounding"])
                n_out += len(g)
                write_jsonl(fo, {"id": r["id"], "shard": r.get("shard"), "path": r["path"],
                                 "img_wh": r.get("img_wh"), "grounding": g,
                                 "n_phrase": len(g), "n_box": sum(len(v) for v in g.values())})
                if args.limit and n_img >= args.limit:
                    break
        print(f"  {os.path.basename(f)} -> {os.path.basename(out_f)}", flush=True)
        if args.limit and n_img >= args.limit:
            break

    print(f"\n{n_img} 图：短语 {n_in} -> {n_out}（保留 {n_out/max(1,n_in)*100:.1f}%）")
    print("过滤明细 " + "  ".join(f"{k}={v}" for k, v in stat.most_common()))


if __name__ == "__main__":
    main()
