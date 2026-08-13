#!/usr/bin/env bash
# 清洗规则对照校验：对 run/ab_clean.sh 的各档产出做同判定器、同抽样的精度对照。
#
# 关键：--max-boxes 0 表示不限制每图短语数（默认 12 会漏掉约 11% 的短语，
#       且放宽规则后每图短语更多，若仍限 12 则各档抽样口径不一致，无法比较）。
#
#   bash run/ab_verify.sh                    各抽 800 图
#   SAMPLE=1500 bash run/ab_verify.sh
#   VARIANTS="c" bash run/ab_verify.sh       只校验某档
#
# 依赖 gemma4 sglang（自动拉起）。产出 out/ab/verify_<档>.jsonl 与对照汇总。
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

SAMPLE="${SAMPLE:-800}"
VARIANTS="${VARIANTS:-old new c}"
AB="$OUT/ab"

for v in $VARIANTS; do
  [ -d "$AB/$v" ] || die "缺少 $AB/$v，先跑 bash run/ab_clean.sh"
done

if ! curl -s --noproxy '*' -m 3 "http://127.0.0.1:$PORT_BASE/health" >/dev/null 2>&1; then
  bash "$(dirname "${BASH_SOURCE[0]}")/sgl.sh" up || die "sglang 启动失败"
fi

for v in $VARIANTS; do
  log "校验 [$v]（抽 $SAMPLE 图，不限每图短语数）"
  "$PY_SGL" "$SRC/s4_verify.py" --in-dir "$AB/$v" --pattern "clean_shard*.jsonl" \
    --out "$AB/verify_$v.jsonl" --urls "$URLS" --model "$GEMMA" \
    --sample "$SAMPLE" --max-boxes 0 2>&1 | tee "$LOGS/ab_verify_$v.log"
done

log "对照汇总"
"$PY_ANY" "$SRC/ab_report.py" --ab "$AB" --clean-dirs "$VARIANTS"
