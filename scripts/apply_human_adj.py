#!/usr/bin/env python3
"""把 adjudicate.html 导出的人工裁决写回 docs/RESULT.md。

与 legacy/stage3_verify/ 版本的区别：写回目标从 ANALYSIS.md 改为 docs/RESULT.md
（当前 pipeline 的报告文档），判定模型文案从 Qwen 改为 gemma4。

分层抽样还原总体精度：
    真实精度 = (N_YES·a + N_NO·b) / (N_YES + N_NO)
其中 a = P(真存在 | 自动判 YES)，b = P(真存在 | 自动判 NO)。
同时给 Wilson 95% 区间，避免拿 100 个样本的点估计当定论。

用法: python apply_human_adj.py --json human_adjudication.json --dir <cc3m_annotate>
      --auto-precision <RESULT.md 里那个下界，百分数，默认用 JSON 内 N_YES 比例>
"""
import argparse, json, math, os


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--auto-precision", type=float, default=None,
                    help="自动判定给出的精度下界（%），默认取 JSON 内 N_YES 比例")
    args = ap.parse_args()

    d = json.load(open(args.json))
    NY, NN = d["N_YES"], d["N_NO"]
    if args.auto_precision is None:
        args.auto_precision = NY / (NY + NN) * 100 if (NY + NN) else 0
    ay = [0, 0]
    an = [0, 0]
    unsure = 0
    for r in d["answers"]:
        h = r.get("human")
        if not h or h == "UNSURE":
            unsure += (h == "UNSURE")
            continue
        t = 1 if h == "YES" else 0
        if r["auto"] == "YES":
            ay[0] += t; ay[1] += 1
        else:
            an[0] += t; an[1] += 1
    if not ay[1] or not an[1]:
        raise SystemExit("两个分层都需要有已判样本")

    a, b = ay[0] / ay[1], an[0] / an[1]
    true_p = (NY * a + NN * b) / (NY + NN)
    acc = (NY * a + NN * (1 - b)) / (NY + NN)
    la, ua = wilson(*ay)
    lb, ub = wilson(*an)
    lo = (NY * la + NN * lb) / (NY + NN)
    hi = (NY * ua + NN * ub) / (NY + NN)

    L = ["\n## 5. 人工裁决校准（adjudicate.html 结果）\n"]
    L.append(f"对第 4 节的自动判定做人工复核。分层抽样：自动判 YES 的 {NY} 对里抽 {ay[1]} 个，"
             f"判 NO 的 {NN} 对里抽 {an[1]} 个（NO 只占总体 "
             f"{NN/(NY+NN)*100:.1f}%，随机抽会导致 NO 样本不足）。"
             + (f"另有 {unsure} 个标为「说不清」，不计入。\n" if unsure else "\n"))
    L.append("| 量 | 值 | 95% 区间 | 含义 |")
    L.append("|---|---|---|---|")
    L.append(f"| a = P(真存在 \\| 判 YES) | {a*100:.1f}% | {la*100:.1f}–{ua*100:.1f}% | n={ay[1]}，判定器说有、确实有 |")
    L.append(f"| b = P(真存在 \\| 判 NO) | {b*100:.1f}% | {lb*100:.1f}–{ub*100:.1f}% | n={an[1]}，判定器误否 |")
    L.append(f"| **还原真实精度** | **{true_p*100:.1f}%** | {lo*100:.1f}–{hi*100:.1f}% | ({NY}·a + {NN}·b) / {NY+NN} |")
    L.append(f"| 自动判定给出的值 | {args.auto_precision:.1f}% | — | 第 4 节那个下界 |")
    L.append(f"| 判定器自身准确率 | {acc*100:.1f}% | — | 与人工一致比例（总体加权） |")
    L.append("")
    delta = true_p * 100 - args.auto_precision
    if delta > 2:
        L.append(f"人工复核后精度比自动判定**高 {delta:.1f} 个百分点** —— 自动判定偏保守，"
                 "确认偏误的影响小于预期，第 4 节的结论可以按下界使用。")
    elif delta < -2:
        L.append(f"人工复核后精度比自动判定**低 {-delta:.1f} 个百分点** —— gemma4 在给自己产出的短语放水，"
                 f"第 4 节那个 {args.auto_precision:.1f}% **不可作为下界引用**，需改用本节数值。")
    else:
        L.append(f"人工复核与自动判定相差 {abs(delta):.1f} 个百分点（区间内），"
                 "说明该判定器在这个任务上基本可用。")
    L.append("")
    with open(os.path.join(args.dir, "docs", "RESULT.md"), "a") as f:
        f.write("\n".join(L) + "\n")
    print(f"a={a*100:.1f}%  b={b*100:.1f}%  真实精度={true_p*100:.1f}% "
          f"[{lo*100:.1f}, {hi*100:.1f}]  判定器准确率={acc*100:.1f}%")
    print("-> 已追加到 docs/RESULT.md 第 5 节")


if __name__ == "__main__":
    main()
