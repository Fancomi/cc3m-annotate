#!/usr/bin/env python3
"""阶段 4 · 校验：抽样估计 (短语, 框) 对的正确率。

把每个框裁出来（外扩 pad 保留上下文），问 VLM「这个裁剪里能清楚看到 <短语> 吗」。
判定模型只看裁剪图，产出模型看全图+文本，依据不同，所以能同时抓到
「框位错」和「概念根本不存在」两类错误。

输入  <in-dir>/{ground,clean}_shard*.jsonl
输出  <out>  每行 {path, phrase, box, crop_frac, verdict}
        verdict ∈ YES / NO / UNPARSED / TOO_SMALL / ERROR

结果读作精度下界，不是绝对真值 —— 判定器与 caption 出自同一模型族时存在确认偏误。
可比的是「同判定器、同样本、同 prompt」下的相对差异。
"""
import argparse, glob, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch import run_pool
from common import ask_vlm, b64, iter_jsonl, make_clients, round_robin

# 整体指代类短语跳过：ground 到全图，裁剪后问"能否看到"没有意义
SKIP = re.compile(r"^(this|the)?\s*(image|photo|picture|scene|view|background|foreground)$", re.I)
PROMPT = ('Is "{p}" clearly visible in this image crop? '
          'Answer with exactly one word: yes or no.')


def main():
    ap = argparse.ArgumentParser(description="阶段4：grounding 抽样校验")
    ap.add_argument("--in-dir", required=True, help="阶段2 或阶段3 的输出目录")
    ap.add_argument("--pattern", default="clean_shard*.jsonl", help="输入文件名模式")
    ap.add_argument("--out", required=True, help="输出 jsonl 路径")
    ap.add_argument("--urls", required=True)
    ap.add_argument("--model", default="/dev/shm/models/gemma-4-26B-A4B-it")
    ap.add_argument("--sample", type=int, default=400, help="抽多少张图")
    ap.add_argument("--max-boxes", type=int, default=12, help="每图最多校验多少短语")
    ap.add_argument("--pad", type=float, default=0.12, help="裁剪外扩比例，保留上下文")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from PIL import Image

    rows = []
    for f in sorted(glob.glob(os.path.join(args.in_dir, args.pattern))):
        rows += [r for r in iter_jsonl(f) if r.get("grounding")]
    random.seed(args.seed)
    random.shuffle(rows)
    rows = rows[:args.sample]

    # 展开成 (图, 短语, 框) 任务：每短语只验第一个框，避免同短语多框重复计数
    tasks = []
    for r in rows:
        for k, (phrase, boxes) in enumerate(r["grounding"].items()):
            if k >= args.max_boxes:
                break
            if SKIP.match(phrase.strip()):
                continue
            tasks.append({"path": r["path"], "phrase": phrase, "box": boxes[0],
                          "n_box": len(boxes)})
    print(f"抽 {len(rows)} 图，待校验 {len(tasks)} 对", flush=True)

    clients = make_clients(args.urls)
    pick = round_robin(clients)

    def work(t):
        rec = dict(t)
        try:
            img = Image.open(t["path"]).convert("RGB")
            W, H = img.size
            x0, y0, x1, y1 = t["box"]
            pw, ph = (x1 - x0) * args.pad, (y1 - y0) * args.pad
            crop = img.crop((max(0, x0 - pw), max(0, y0 - ph), min(W, x1 + pw), min(H, y1 + ph)))
            rec["crop_frac"] = round((x1 - x0) * (y1 - y0) / (W * H), 4)
            if crop.width < 8 or crop.height < 8:
                rec["verdict"] = "TOO_SMALL"
                return rec
            a = ask_vlm(pick(), args.model, b64(crop, maxside=512, quality=88),
                        PROMPT.format(p=t["phrase"].strip()), max_tokens=6).lower()
            rec["verdict"] = "YES" if a.startswith("yes") else "NO" if a.startswith("no") else "UNPARSED"
        except Exception as e:
            rec["verdict"] = "ERROR"
            rec["error"] = f"{type(e).__name__}: {e}"
        return rec

    # 主键是 (path, phrase)：同一张图的不同短语是不同任务
    run_pool(tasks, work, args.out, "verify", workers=args.concurrency, every=200,
             key=lambda r: (r["path"], r["phrase"]))


if __name__ == "__main__":
    main()
