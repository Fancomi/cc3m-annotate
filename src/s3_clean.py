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

默认只开前四条（无损清洗）。训练数据档见 run/3_clean.sh 的 TRAIN=1 分支（rec 档：
--min-words 1 --word-boundary --abstract --xdup --max-cover 0.95 --edge），全量实测
精度 70.1%、19.4 短语/图。旧的 --min-area 0.02 --min-words 2 档已被消融否掉：
它把有效信号砍掉一半（8.7 短语/图）只换来 1.2pt 精度。

--word-boundary 的由来（重要）：
  garbled 默认用裸子串匹配，于是 `ers`（截自 loafers）、`eyebrow`（截自 eyebrows）
  这类解码碎片能在 caption 里匹配上而漏网。它们的共同特征是只剩一个实词，所以
  过去靠 --min-words 2 兜住 —— 代价是把 moon / sky / face / dog 这类**合法单名词**
  一起删掉。实测（30000 图）：实词数==1 的短语占全部 45.3%，其中仅 0.5% 是碎片。
  改成词边界匹配后，碎片由 garbled 直接拦下（实测能分出 98% 合法 / 2% 碎片），
  --min-words 就不必再背这个锅。
"""
import argparse, glob, os, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (VAGUE, content_words, is_abstract, iter_jsonl, iter_shards,
                    norm_phrase, write_jsonl)


def in_caption(p_norm, cap_norm, word_boundary):
    """短语是否出现在源 caption 里。

    word_boundary=True 时要求整词命中。norm_phrase 已把所有非字母数字压成单空格，
    所以两端补空格做子串判断与正则 \\b 完全等价，但快得多（全量 6680 万短语要过这条）。
    """
    if not word_boundary:
        return p_norm in cap_norm
    return f" {p_norm} " in f" {cap_norm} "


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def box_filters(boxes, wh, args, stat):
    """逐框过滤。返回保留的框列表；被删的框按规则名计入 stat。

    顺序：dup → area/minpx → ratio → fullimg → edge。先去重再判几何，
    避免同一个重复框被几条几何规则重复计数。
    """
    W, H = wh
    total = (W or 0) * (H or 0)
    kept = boxes

    if args.dup:
        uniq = []
        for b in kept:
            if all(iou(b, k) <= 0.9 for k in uniq):
                uniq.append(b)
        stat["dup"] += len(kept) - len(uniq)
        kept = uniq

    def side_px(b):
        return min(abs(b[2] - b[0]), abs(b[3] - b[1]))

    # area 用「占全图比例」，min_px 用「绝对像素」。后者对小图友好：450x450 的图
    # 2% 面积只有 64px 边长，本来看得清却被 area 砍掉；而 1920x1080 的 2% 有 200px。
    # 判定器与训练模型看到的是裁剪后的绝对像素，所以 min_px 才是对齐能力边界的判据。
    if args.min_area and total:
        n0 = len(kept)
        kept = [b for b in kept if (b[2] - b[0]) * (b[3] - b[1]) / total >= args.min_area]
        stat["area"] += n0 - len(kept)
    if args.min_px:
        n0 = len(kept)
        kept = [b for b in kept if side_px(b) >= args.min_px]
        stat["minpx"] += n0 - len(kept)

    # 细长条：F2 把整行文字或图像边缘误框时的典型形状
    if args.max_ratio:
        n0 = len(kept)
        def ratio_ok(b):
            w, h = abs(b[2] - b[0]), abs(b[3] - b[1])
            s = min(w, h)
            return s > 0 and max(w, h) / s <= args.max_ratio
        kept = [b for b in kept if ratio_ok(b)]
        stat["ratio"] += n0 - len(kept)

    # 贴满整图：短语本身不空泛（否则 vague 已拦），但框住全图 → 无区域价值。
    # 与 vague 互补：vague 看文本，这条看几何。
    if args.max_cover and total:
        n0 = len(kept)
        kept = [b for b in kept
                if (b[2] - b[0]) * (b[3] - b[1]) / total <= args.max_cover]
        stat["fullimg"] += n0 - len(kept)

    # 只压着图像边缘的窄条：解码位置漂移的表现。要求「贴边」且「另一维很窄」，
    # 否则会误伤天空/地面这类合法的贴边大区域。
    if args.edge and total:
        n0 = len(kept)
        def not_edge_sliver(b):
            w, h = abs(b[2] - b[0]), abs(b[3] - b[1])
            touch = (b[0] <= 2 or b[1] <= 2 or b[2] >= W - 2 or b[3] >= H - 2)
            thin = (w <= W * 0.06) or (h <= H * 0.06)
            return not (touch and thin)
        kept = [b for b in kept if not_edge_sliver(b)]
        stat["edge"] += n0 - len(kept)

    return kept


def clean_one(grounding, cap_norm, wh, args, stat):
    out = {}
    for phrase, boxes in grounding.items():
        p = phrase.strip()
        if args.vague and VAGUE.match(p):
            stat["vague"] += 1
            continue
        if args.abstract and is_abstract(p):
            stat["abstract"] += 1
            continue
        if args.garbled and cap_norm and norm_phrase(p) and not in_caption(
                norm_phrase(p), cap_norm, args.word_boundary):
            stat["garbled"] += 1
            continue
        if len(content_words(p)) < args.min_words:
            stat["words"] += 1
            continue
        # 一个短语配太多框：F2 对复数名词会撒网式出框，逐个都不准。
        # 放在框过滤之前判，用原始框数 —— 撒网的特征是原始就多。
        if args.max_boxes and len(boxes) > args.max_boxes:
            stat["nbox"] += 1
            continue
        kept = box_filters(boxes, wh, args, stat)
        if kept:
            out[p] = kept
        else:
            stat["empty"] += 1
    if args.xdup:
        out = dedup_across(out, stat)
    return out


def dedup_across(grounding, stat):
    """跨短语去重：同实词集合 + 框 IoU>0.9 的，只留表述最完整的那个。

    起因：放宽 min_words 后 `moon` / `The moon` / `the moon` 会同时保留，
    框几乎重合，是纯冗余。按实词集合（无序）分组而非字符串，才能把
    `dark hair` 与 `hair dark` 之类归到一起；组内保留实词数最多、
    其次字面最长的表述 —— 信息量最大。
    """
    groups = {}
    for ph, boxes in grounding.items():
        key = frozenset(content_words(ph))
        groups.setdefault(key, []).append(ph)
    out = {}
    for key, phrases in groups.items():
        if len(phrases) == 1:
            out[phrases[0]] = grounding[phrases[0]]
            continue
        # 先按信息量排序，逐个尝试加入；框与已保留项高度重合则丢弃
        phrases.sort(key=lambda s: (-len(content_words(s)), -len(s)))
        kept_boxes = []
        for ph in phrases:
            boxes = grounding[ph]
            if any(iou(b, k) > 0.9 for b in boxes for k in kept_boxes):
                stat["xdup"] += 1
                continue
            out[ph] = boxes
            kept_boxes += boxes
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
    ap.add_argument("--min-area", type=float, default=0.0,
                    help="框面积占全图比例下限。对小图不友好（450px 图的 2% 只有 64px 边长），"
                         "优先考虑 --min-px")
    ap.add_argument("--min-px", type=int, default=0,
                    help="框短边绝对像素下限，与判定器/训练模型的实际能力边界对齐")
    ap.add_argument("--max-ratio", type=float, default=0.0,
                    help="长宽比上限，>0 时删细长条框（文字行/图像边缘误框）")
    ap.add_argument("--max-cover", type=float, default=0.0,
                    help="单框面积占比上限，>0 时删贴满整图的框（与 vague 互补：那条看文本，这条看几何）")
    ap.add_argument("--max-boxes", type=int, default=0,
                    help="单短语框数上限，>0 时删撒网式出框的短语（F2 对复数名词的典型失败）")
    ap.add_argument("--edge", action="store_true",
                    help="删只压着图像边缘的窄条框（解码位置漂移）")
    ap.add_argument("--min-words", type=int, default=1, help="实词数下限，训练数据推荐 2")
    ap.add_argument("--word-boundary", action="store_true",
                    help="garbled 改用整词匹配（能拦下解码碎片，配合 --min-words 1 使用）")
    ap.add_argument("--no-vague", dest="vague", action="store_false")
    ap.add_argument("--no-garbled", dest="garbled", action="store_false")
    ap.add_argument("--no-dup", dest="dup", action="store_false")
    ap.add_argument("--abstract", action="store_true",
                    help="删抽象属性/取景类短语（the lighting、studio shot）—— 实测这类精度仅 25.9%")
    ap.add_argument("--xdup", action="store_true",
                    help="跨短语去重：同实词集合且框重合的只留最完整表述（moon / The moon）")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只处理前 N 图（冒烟）")
    args = ap.parse_args()

    # caption 索引以 path 为键（path 全局唯一，id 会在 tsv 之间撞号）
    cap = {r["path"]: norm_phrase(r["gemma_dense"])
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
