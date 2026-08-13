#!/usr/bin/env python3
"""清洗规则对照汇总：把各档的短语密度与校验精度并排列出。

单看精度会误导 —— 严格的规则总能把精度做高，代价是数据变少。所以这里同时给
「每图有效短语数」和「精度」，再乘出「每图正确短语数」作为综合指标：
它才是训练能拿到的监督信号量。
"""
import argparse, glob, json, os


def clean_stats(d):
    n_img = n_ph = 0
    for f in sorted(glob.glob(os.path.join(d, "clean_shard*.jsonl"))):
        for l in open(f):
            try:
                r = json.loads(l)
            except Exception:
                continue
            n_img += 1
            n_ph += len(r.get("grounding") or {})
    return n_img, n_ph


def verify_stats(p):
    if not os.path.exists(p):
        return None
    rows = [json.loads(l) for l in open(p)]
    ok = [r for r in rows if r["verdict"] in ("YES", "NO")]
    if not ok:
        return None
    y = sum(1 for r in ok if r["verdict"] == "YES")
    return len(ok), y, y / len(ok) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", required=True)
    ap.add_argument("--clean-dirs", default="old new c", help="档位名，空格分隔")
    args = ap.parse_args()

    variants = args.clean_dirs.split()
    print(f"\n{'档位':<6}{'图数':>8}{'短语':>10}{'短语/图':>9}"
          f"{'校验对':>8}{'精度':>8}{'正确短语/图':>12}")
    print("-" * 64)
    base = None
    for v in variants:
        n_img, n_ph = clean_stats(os.path.join(args.ab, v))
        vs = verify_stats(os.path.join(args.ab, f"verify_{v}.jsonl"))
        dens = n_ph / n_img if n_img else 0
        if vs:
            n_ok, y, prec = vs
            eff = dens * prec / 100
            print(f"{v:<6}{n_img:>8,}{n_ph:>10,}{dens:>9.2f}"
                  f"{n_ok:>8,}{prec:>7.1f}%{eff:>12.2f}")
            if base is None:
                base = (prec, eff)
        else:
            print(f"{v:<6}{n_img:>8,}{n_ph:>10,}{dens:>9.2f}{'—':>8}{'—':>8}{'—':>12}")

    if base:
        print("\n相对第一档的变化：")
        for v in variants[1:]:
            n_img, n_ph = clean_stats(os.path.join(args.ab, v))
            vs = verify_stats(os.path.join(args.ab, f"verify_{v}.jsonl"))
            if not vs:
                continue
            prec = vs[2]
            eff = (n_ph / n_img) * prec / 100
            print(f"  {v:<6} 精度 {prec - base[0]:+.1f} pt   "
                  f"正确短语/图 {eff - base[1]:+.2f}（{(eff/base[1]-1)*100:+.0f}%）")
    print("\n注：「正确短语/图」= 短语密度 × 精度，是训练能拿到的有效监督信号量。"
          "\n    精度是同判定器下的相对值，不是绝对真值（判定器与产出模型同族，有确认偏误）。")


if __name__ == "__main__":
    main()
