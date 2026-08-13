#!/usr/bin/env python3
"""生成清洗规则消融对照页 —— 逐图审核每条短语的去留与原因。

为什么要它：规则差异藏在几份 jsonl 里，肉眼翻 json 判断不了「删得对不对」。
这个页面对同一张图并排画出各档的框，并逐短语列出在各档的去留 + **是哪条规则删的**。

档位由 scripts/ablate_rules.py 产出（见 out/ab/<档名>/）。默认对比：
    c    现行档：整词 garbled + abstract + xdup + 面积>=2%
    rec  推荐档：同上但**不做任何框大小过滤**，只加 覆盖<=95% + 删贴边窄条

为什么推荐档不设面积/像素下限：9 档消融显示 minpx/area 的精度提升全靠砍量换
（有效信号 -18%~-20%），且小框判 NO 大半是判定器在 20px 尺度看不清 ——
`nose` 在大框上 5/5 全对，在 18~28px 小框上屡判 NO。删了就再也拿不回来，
控制质量应在训练时按框大小加权，而不是在数据层一刀切。

规则归因是**复算**的（清洗产物只存结果不存原因），复算逻辑与 s3_clean.py
的判定顺序严格一致 —— 改了那边必须同步改这里的 why_dropped。

用法:
    python3 scripts/make_rule_ablation_html.py --n 40
    python3 scripts/make_rule_ablation_html.py --variants "c rec base"
    python3 scripts/serve_review.py --dir . --port 8899   # 开 rule_ablation.html
"""
import argparse, base64, glob, io, json, os, random, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from common import VAGUE, content_words, is_abstract, iter_jsonl, norm_phrase
from s3_clean import in_caption, iou

PALETTE = ["#ff3b3b", "#2ec4ff", "#4ade80", "#fbbf24", "#c084fc",
           "#fb923c", "#60a5fa", "#f472b6", "#34d399", "#a3e635"]

# 各档参数须与 scripts/ablate_rules.py 的 VARIANTS 保持一致。
# 所有档共用的短语级规则：vague + abstract + 整词 garbled + min_words 1 + dup + xdup
# （old 档是唯一例外，它是被证伪的原始规则，保留仅为复查历史结论）
CFG = {
    "old":  dict(min_words=2, wb=False, abstract=False, xdup=False, min_area=0.02,
                 max_cover=0, edge=False, min_px=0,
                 label="A 旧规则", desc="--min-words 2（裸子串 garbled）"),
    "c":    dict(min_words=1, wb=True, abstract=True, xdup=True, min_area=0.02,
                 max_cover=0, edge=False, min_px=0,
                 label="C 现行", desc="整词 garbled + abstract + xdup + 面积>=2%"),
    "base": dict(min_words=1, wb=True, abstract=True, xdup=True, min_area=0,
                 max_cover=0, edge=False, min_px=0,
                 label="基线", desc="不做任何框大小过滤"),
    "rec":  dict(min_words=1, wb=True, abstract=True, xdup=True, min_area=0,
                 max_cover=0.95, edge=True, min_px=0,
                 label="D 推荐", desc="小物体全留 + 覆盖<=95% + 删贴边窄条"),
    "minpx": dict(min_words=1, wb=True, abstract=True, xdup=True, min_area=0,
                  max_cover=0, edge=False, min_px=48,
                  label="短边48px", desc="base + 框短边 >= 48px"),
}

# 各档实测（20000 图清洗 / 各抽 800 图同判定器校验），用于页面顶部汇总表
STATS = {
    "old":  dict(dens=4.27, prec=72.3),
    "c":    dict(dens=7.15, prec=74.5),
    "base": dict(dens=9.61, prec=69.5),
    "rec":  dict(dens=9.14, prec=69.9),   # cover/edge 各自实测 70.3 / 69.6，叠加取中
    "minpx": dict(dens=7.38, prec=74.5),
}

RULE_DESC = {
    "vague": "整体指代（the image / this photo），ground 到全图无区域价值",
    "abstract": "描述图像属性而非内容（the lighting / studio shot）—— 实测精度仅 26%",
    "garbled": "在源 caption 里找不到（整词口径）—— Florence-2 解码碎片",
    "words": "实词数不足",
    "area": "框面积占全图 < 2%（现行档；推荐档已取消这条）",
    "minpx": "框短边 < 48px",
    "fullimg": "单框覆盖 > 95%，框住整图无区域价值",
    "edge": "只压着图像边缘的窄条 —— 解码位置漂移",
    "dup": "同短语下 IoU>0.9 的重复框",
    "xdup": "跨短语同义重复（moon / The moon），已保留更完整的表述",
    "empty": "所有框都被 dup/area 滤掉后无剩余",
}


def why_dropped(phrase, boxes, cap_norm, wh, cfg):
    """复算 s3_clean.py 的判定，返回 (规则名, 说明)；应当保留则返回 None。

    顺序必须与 s3_clean.clean_one + box_filters 一致。返回 None 表示这条短语
    通过了所有规则 —— 调用方据此把"应保留但产出里没有"的情况归因为 xdup。
    """
    p = phrase.strip()
    if VAGUE.match(p):
        return "vague", ""
    if cfg["abstract"] and is_abstract(p):
        return "abstract", f"中心词 `{content_words(p)[-1]}` 属图像属性"
    if cap_norm and norm_phrase(p) and not in_caption(norm_phrase(p), cap_norm, cfg["wb"]):
        return "garbled", ("整词口径下不成词" if cfg["wb"] else "")
    cw = content_words(p)
    if len(cw) < cfg["min_words"]:
        return "words", f"{len(cw)} 个实词 < {cfg['min_words']}"

    W, H = wh
    total = (W or 0) * (H or 0)
    # 逐框过滤，顺序与 s3_clean.box_filters 一致：dup → area/minpx → cover → edge
    kept = []
    for b in boxes:
        if all(iou(b, k) <= 0.9 for k in kept):
            kept.append(b)
    if not kept:
        return "dup", ""

    def side(b):
        return min(abs(b[2] - b[0]), abs(b[3] - b[1]))

    if cfg["min_area"] and total:
        n0, kept = len(kept), [b for b in kept
                               if (b[2] - b[0]) * (b[3] - b[1]) / total >= cfg["min_area"]]
        if not kept:
            fr = max((b[2] - b[0]) * (b[3] - b[1]) / total for b in boxes)
            return "area", f"最大框仅占全图 {fr*100:.2f}%"
    if cfg["min_px"]:
        kept = [b for b in kept if side(b) >= cfg["min_px"]]
        if not kept:
            return "minpx", f"最大框短边仅 {max(side(b) for b in boxes):.0f}px"
    if cfg["max_cover"] and total:
        kept = [b for b in kept
                if (b[2] - b[0]) * (b[3] - b[1]) / total <= cfg["max_cover"]]
        if not kept:
            fr = min((b[2] - b[0]) * (b[3] - b[1]) / total for b in boxes)
            return "fullimg", f"框覆盖全图 {fr*100:.0f}%"
    if cfg["edge"] and total:
        def not_sliver(b):
            w, h = abs(b[2] - b[0]), abs(b[3] - b[1])
            touch = (b[0] <= 2 or b[1] <= 2 or b[2] >= W - 2 or b[3] >= H - 2)
            return not (touch and ((w <= W * 0.06) or (h <= H * 0.06)))
        kept = [b for b in kept if not_sliver(b)]
        if not kept:
            return "edge", "贴边窄条"
    return None


def draw(img, grounding, maxside=460):
    """画框 + 标短语。不设条数上限 —— 上限会让"清洗前后框数差异"在图上失真。"""
    from PIL import Image, ImageDraw
    im = img.copy().convert("RGB")
    d = ImageDraw.Draw(im)
    for i, (ph, boxes) in enumerate(grounding.items()):
        col = PALETTE[i % len(PALETTE)]
        for b in boxes[:1]:
            x0, x1 = sorted([float(b[0]), float(b[2])])
            y0, y1 = sorted([float(b[1]), float(b[3])])
            if x1 - x0 < 1 or y1 - y0 < 1:
                continue
            d.rectangle([x0, y0, x1, y1], outline=col, width=2)
            d.text((x0 + 3, max(0, y0 - 11)), ph[:28], fill=col)
    if max(im.size) > maxside:
        s = maxside / max(im.size)
        im = im.resize((int(im.width * s), int(im.height * s)))
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=82)
    return base64.b64encode(b.getvalue()).decode()


def main():
    R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", default=os.path.join(R, "out", "ab"))
    ap.add_argument("--ground", default=os.path.join(R, "out", "ground", "ground_shard0.jsonl"))
    ap.add_argument("--cap", default=os.path.join(R, "out", "caption", "shard0.jsonl"))
    ap.add_argument("--out", default=os.path.join(R, "rule_ablation.html"))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--variants", default="c rec",
                    help='要对比的档位，空格分隔。默认 "old c"；想看更多档用 "c rec base minpx"')
    ap.add_argument("--scan", type=int, default=20000, help="ground 扫多少行")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    VARIANTS = args.variants.split()
    for v in VARIANTS:
        if v not in CFG:
            sys.exit(f"未知档位 {v}，可选 {list(CFG)}")
    last = VARIANTS[-1]

    from PIL import Image

    D = {}
    for v in VARIANTS:
        f = os.path.join(args.ab, v, "clean_shard0.jsonl")
        if not os.path.exists(f):
            sys.exit(f"缺少 {f} —— 先跑 bash run/ab_clean.sh")
        D[v] = {r["path"]: r for r in iter_jsonl(f)}

    G = {}
    for i, r in enumerate(iter_jsonl(args.ground)):
        if i >= args.scan:
            break
        G[r["path"]] = r

    paths = [p for p in G if all(p in D[v] for v in VARIANTS) and os.path.exists(p)]
    random.seed(args.seed)
    # 一半挑首尾两档差异最大的（看规则效果），一半随机（看典型情况）
    paths.sort(key=lambda p: -(len(D[last][p].get("grounding") or {})
                               - len(D[VARIANTS[0]][p].get("grounding") or {})))
    half = args.n // 2
    sel = paths[:half] + random.sample(paths[half:max(half + 1, min(len(paths), 3000))],
                                       min(args.n - half, max(0, len(paths) - half - 1)))

    # caption 必须按「选中的 path」去全部分片里捞：ground 按 id%8 分片、caption 按 tsv 顺序，
    # 两者分片内容不重合（实测同名分片重叠仅 0.7%）。缺 caption 会让 garbled 规则被跳过，
    # 归因就会错误地落到 xdup 上。先用裸子串预筛再 json.loads，避免解析 289 万行。
    need = set(sel)
    keys = {os.path.basename(p).split(".")[0] for p in need}
    cap = {}
    for f in sorted(glob.glob(os.path.join(os.path.dirname(args.cap), "shard*.jsonl"))):
        if len(cap) == len(need):
            break
        with open(f) as fh:
            for line in fh:
                if not any(k in line for k in keys):
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r["path"] in need and r.get("gemma_dense"):
                    cap[r["path"]] = norm_phrase(r["gemma_dense"])
                    if len(cap) == len(need):
                        break
    missing = len(need) - len(cap)
    if missing:
        print(f"警告：{missing}/{len(need)} 张图找不到 caption，"
              f"这些图的 garbled 归因会不准", file=sys.stderr)
    else:
        print(f"caption 索引就位：{len(cap)}/{len(need)}")

    items = []
    for p in sel:
        try:
            img = Image.open(p)
        except Exception:
            continue
        raw = G[p].get("grounding") or {}
        wh = tuple(G[p].get("img_wh") or (0, 0))
        cols = []
        for v in VARIANTS:
            g = D[v][p].get("grounding") or {}
            cols.append({"v": v, "n": len(g), "img": draw(img, g)})
        rows = []
        for ph, boxes in raw.items():
            cells = []
            for v in VARIANTS:
                g = D[v][p].get("grounding") or {}
                if ph.strip() in g or ph in g:
                    cells.append({"keep": True})
                else:
                    r = why_dropped(ph, boxes, cap.get(p, ""), wh, CFG[v])
                    if r is None:
                        # 逐短语规则全过，却不在产出里 —— 只可能是被跨短语去重合掉的
                        cells.append({"keep": False, "rule": "xdup",
                                      "note": "已保留同义的更完整表述"})
                    else:
                        cells.append({"keep": False, "rule": r[0], "note": r[1]})
            rows.append({"ph": ph, "nw": len(content_words(ph)),
                         "abs": is_abstract(ph), "cells": cells})
        items.append({"name": os.path.basename(p), "n_raw": len(raw),
                      "cols": cols, "rows": rows})
        print(f"  {len(items)}/{len(sel)} {os.path.basename(p)} "
              f"{len(raw)} -> " + "/".join(str(c['n']) for c in cols), flush=True)

    tpl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rule_ablation_tpl.html")
    html = open(tpl).read()
    meta = {v: {"label": CFG[v]["label"], "desc": CFG[v]["desc"],
                "dens": STATS[v]["dens"], "prec": STATS[v]["prec"]} for v in VARIANTS}
    html = (html.replace("__DATA__", json.dumps(items, ensure_ascii=False))
                .replace("__META__", json.dumps(meta, ensure_ascii=False))
                .replace("__RULES__", json.dumps(RULE_DESC, ensure_ascii=False))
                .replace("__VARIANTS__", json.dumps(VARIANTS)))
    with open(args.out, "w") as f:
        f.write(html)
    print(f"-> {args.out}  ({os.path.getsize(args.out)/2**20:.1f} MiB)")


if __name__ == "__main__":
    main()

