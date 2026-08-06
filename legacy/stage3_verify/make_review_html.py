#!/usr/bin/env python3
"""生成自包含 HTML 抽查页：图片 + 框叠加 + 三方标注 + 校验裁决。

设计目标是让人**能否证**我给出的结论，而不只是看结果：
  - 每张图画出 crossfeed 的框（可切换显示基线框），标签带编号
  - 并列三方文本输出，CC3M 原标注置顶作参照
  - 逐条列出校验裁决（YES/NO），可人工判断 Qwen 的判断本身对不对
    —— 这是关键，因为 74.5% 那个数字是 Qwen 判 Qwen，需要人核验判定器可靠性

图片以 base64 内嵌，单文件可直接浏览器打开，无需起服务。
"""
import argparse, base64, glob, html, io, json, os, random

PAL = ["#ff3b3b", "#2ec4ff", "#4ade80", "#fbbf24", "#e879f9", "#fb923c",
       "#60a5fa", "#f472b6", "#22d3ee", "#a3e635", "#f87171", "#5eead4",
       "#c084fc", "#fdba74", "#86efac", "#fda4af", "#93c5fd"]
SKIP = {"this image", "the image", "image", "photo", "picture"}


def img_b64(path, maxside=560):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    ow, oh = im.size
    if max(im.size) > maxside:
        s = maxside / max(im.size)
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=86)
    return base64.b64encode(buf.getvalue()).decode(), ow, oh, im.width, im.height


def boxes_svg(groups, ow, oh, dw, dh, verdicts=None):
    """groups: [(phrase, [box,...])]，坐标是原图像素，按显示尺寸缩放。"""
    sx, sy = dw / ow, dh / oh
    out = [f'<svg class="ov" viewBox="0 0 {dw} {dh}" preserveAspectRatio="none">']
    for i, (ph, bs) in enumerate(groups):
        c = PAL[i % len(PAL)]
        v = (verdicts or {}).get(ph)
        dash = ' stroke-dasharray="5,4"' if v == "NO" else ""
        for b in bs:
            x0, y0, x1, y1 = b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy
            out.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{max(1,x1-x0):.1f}" '
                       f'height="{max(1,y1-y0):.1f}" fill="none" stroke="{c}" '
                       f'stroke-width="2"{dash}/>')
        x0, y0 = bs[0][0] * sx, bs[0][1] * sy
        out.append(f'<text x="{x0+3:.1f}" y="{max(11,y0-3):.1f}" fill="{c}" '
                   f'font-size="12" font-weight="bold" '
                   f'style="paint-order:stroke;stroke:#000;stroke-width:3px">{i+1}</text>')
    out.append("</svg>")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--only-verified", action="store_true",
                    help="只抽有校验裁决的图（便于核验判定器）")
    args = ap.parse_args()

    D = args.dir
    merged = {}
    for l in open(os.path.join(D, "out", "merged.jsonl")):
        r = json.loads(l)
        merged[r["id"]] = r
    cf = {}
    for f in glob.glob(os.path.join(D, "out", "crossfeed_shard*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if "error" not in r:
                cf[r["id"]] = r
    vf = {}
    for f in ["verify_crossfeed.jsonl"]:
        p = os.path.join(D, "out", f)
        if os.path.exists(p):
            for l in open(p):
                r = json.loads(l)
                vf.setdefault(r["id"], {})[r["phrase"]] = r
    vb = {}
    p = os.path.join(D, "out", "verify_baseline.jsonl")
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l)
            vb.setdefault(r["id"], {})[r["phrase"]] = r

    ids = sorted(vf) if args.only_verified else sorted(merged)
    random.seed(args.seed)
    random.shuffle(ids)
    ids = ids[:args.n]

    H = []
    H.append("""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>cc3m 标注抽查</title><style>
:root{--bg:#0f1115;--fg:#e6e6e6;--dim:#9aa0a6;--card:#181b21;--line:#2a2f38;
      --ok:#4ade80;--bad:#ff6b6b;--acc:#2ec4ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:14px/1.55 -apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif}
header{padding:18px 24px;border-bottom:1px solid var(--line);position:sticky;top:0;
       background:var(--bg);z-index:9}
h1{margin:0 0 6px;font-size:17px;font-weight:600}
.warn{background:#3a2a12;border-left:3px solid #fbbf24;padding:10px 14px;margin:12px 24px;
      border-radius:4px;font-size:13px;color:#fde68a}
.warn b{color:#fbbf24}
.ctl{display:flex;gap:16px;align-items:center;flex-wrap:wrap;font-size:13px;color:var(--dim)}
.ctl label{cursor:pointer;user-select:none}
.card{margin:20px 24px;background:var(--card);border:1px solid var(--line);border-radius:8px;
      padding:16px;display:grid;grid-template-columns:580px 1fr;gap:20px}
@media(max-width:1200px){.card{grid-template-columns:1fr}}
.imgwrap{position:relative;width:fit-content}
.imgwrap img{display:block;border-radius:4px}
.ov{position:absolute;inset:0;pointer-events:none}
.gt{background:#1e2530;border-left:3px solid var(--acc);padding:8px 12px;border-radius:4px;
    margin-bottom:12px}
.gt .lb{color:var(--acc);font-size:11px;letter-spacing:.06em;text-transform:uppercase}
.row{margin:9px 0;padding-bottom:9px;border-bottom:1px dotted var(--line)}
.row:last-child{border:0}
.lb{color:var(--dim);font-size:11px;letter-spacing:.06em;text-transform:uppercase;
    display:block;margin-bottom:3px}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.tags span{display:inline-block;background:#232833;border-radius:3px;padding:1px 7px;
           margin:2px 3px 2px 0;font-size:12px}
ol.ph{margin:6px 0 0;padding-left:0;list-style:none;
      max-height:340px;overflow:auto;font-size:12.5px}
ol.ph li{padding:3px 6px;border-radius:3px;display:flex;gap:8px;align-items:baseline}
ol.ph li:nth-child(odd){background:#1c2029}
.num{font-weight:700;min-width:22px;text-align:right}
.v{font-size:10.5px;font-weight:700;padding:1px 5px;border-radius:3px;flex-shrink:0}
.v.YES{background:#14351f;color:var(--ok)} .v.NO{background:#3a1518;color:var(--bad)}
.v.NA{background:#26282d;color:var(--dim)}
.nb{color:var(--dim);font-size:11px}
.hide{display:none}
.meta{color:var(--dim);font-size:12px;margin-bottom:8px}
a{color:var(--acc)}
</style></head><body>""")
    H.append(f"""<header><h1>cc3m 标注抽查 · 随机 {len(ids)} 张（seed={args.seed}）</h1>
<div class="ctl">
  <label><input type="checkbox" id="tb" checked> 显示 crossfeed 框</label>
  <label><input type="checkbox" id="tv" checked> 显示校验裁决</label>
  <span>虚线框 = 校验判为 NO</span>
</div></header>
<div class="warn">
<b>关于这份数据的范围与可信度，务必先读：</b><br>
① <b>这不是全量运行。</b>cc3m 本地共 <b>2,894,191</b> 张，本轮只处理了 <b>2,880 张（0.0995%）</b>，
按 576 个 shard 均匀抽样，目的是验证链路可行性与相对优劣，不是生产数据。<br>
② <b>没有 ground truth。</b>ANALYSIS.md 里的"精度 74.5%"是<b>用 Qwen 判 Qwen 自己驱动产出的结果</b>
（裁剪后问它"这个短语在图里吗"），既非人工标注、也非独立第三方模型。它只能当<b>下界估计</b>，
且存在确认偏误。本页逐条列出裁决，就是为了让你核验<b>判定器本身</b>是否可靠。<br>
③ CC3M 原标注是网页 alt-text，含大量视觉不可见信息（人名、事件、地点），<b>不能当 GT 用</b>。
</div>""")

    for iid in ids:
        r = merged.get(iid)
        if not r:
            continue
        try:
            b64s, ow, oh, dw, dh = img_b64(r["path"])
        except Exception as e:
            continue
        c = cf.get(iid)
        vmap = vf.get(iid, {})
        bmap = vb.get(iid, {})
        groups = []
        if c:
            for ph, bs in c["grounding_qwen"].items():
                if ph.strip().lower() in SKIP:
                    continue
                groups.append((ph, bs))
        vd = {ph: vmap[ph]["verdict"] for ph in vmap}
        f2 = r.get("florence2", {})
        H.append('<div class="card">')
        H.append(f'<div><div class="meta">id={iid} · {html.escape(os.path.basename(r["path"]))} · {ow}×{oh}</div>')
        H.append(f'<div class="imgwrap"><img src="data:image/jpeg;base64,{b64s}" width="{dw}" height="{dh}">')
        H.append(f'<div class="bx">{boxes_svg(groups, ow, oh, dw, dh, vd)}</div></div></div>')
        H.append("<div>")
        H.append(f'<div class="gt"><span class="lb">CC3M 原标注（alt-text，非 GT）</span>'
                 f'{html.escape(r["gt_caption"])}</div>')
        if "novic" in r:
            t = " · ".join(f"{html.escape(a)} <span class='nb'>{b}%</span>"
                           for a, b in r["novic"]["topk"][:6])
            H.append(f'<div class="row"><span class="lb">NOVIC top-6（开放词汇分类，42919 名词）</span>{t}</div>')
        if f2:
            H.append(f'<div class="row"><span class="lb">F2 &lt;CAPTION&gt;</span>{html.escape(f2["caption"] or "")}</div>')
            H.append(f'<div class="row"><span class="lb">F2 &lt;MORE_DETAILED_CAPTION&gt;</span>'
                     f'{html.escape(f2["more_detailed"] or "")}</div>')
            od = f2.get("od", {}).get("labels", [])
            H.append('<div class="row tags"><span class="lb">F2 &lt;OD&gt;（检测头，484 类词表）</span>'
                     + "".join(f"<span>{html.escape(x)}</span>" for x in od) + "</div>")
            dn = f2.get("dense_region_caption", {}).get("labels", [])
            H.append('<div class="row tags"><span class="lb">F2 &lt;DENSE_REGION_CAPTION&gt;</span>'
                     + "".join(f"<span>{html.escape(x)}</span>" for x in dn) + "</div>")
        if "qwen" in r:
            H.append(f'<div class="row"><span class="lb">Qwen short</span>{html.escape(r["qwen"]["short"] or "")}</div>')
            H.append(f'<div class="row"><span class="lb">Qwen dense（→ 用作 crossfeed 的驱动文本）</span>'
                     f'{html.escape(r["qwen"]["dense"] or "")}</div>')
        if groups:
            nb = sum(len(b) for _, b in groups)
            H.append(f'<div class="row"><span class="lb">crossfeed grounding · {len(groups)} 短语 / {nb} 框'
                     f'（基线 F2 自驱动：{len(f2.get("grounding") or {})} 短语）</span><ol class="ph">')
            for i, (ph, bs) in enumerate(groups):
                v = vd.get(ph, "NA")
                cf_ = PAL[i % len(PAL)]
                H.append(f'<li><span class="num" style="color:{cf_}">{i+1}.</span>'
                         f'<span class="v {v} vtag">{v if v!="NA" else "未校验"}</span>'
                         f'<span>{html.escape(ph)}</span>'
                         f'<span class="nb">{len(bs)} 框</span></li>')
            H.append("</ol></div>")
        H.append("</div></div>")

    H.append("""<script>
const tb=document.getElementById('tb'), tv=document.getElementById('tv');
tb.onchange=()=>document.querySelectorAll('.bx').forEach(e=>e.classList.toggle('hide',!tb.checked));
tv.onchange=()=>document.querySelectorAll('.vtag').forEach(e=>e.classList.toggle('hide',!tv.checked));
</script></body></html>""")

    with open(args.out, "w") as f:
        f.write("\n".join(H))
    print(f"-> {args.out}  ({os.path.getsize(args.out)/2**20:.1f} MiB, {len(ids)} 张)")


if __name__ == "__main__":
    main()
