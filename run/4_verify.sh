#!/usr/bin/env bash
# 阶段 4 · 校验 + 报告：抽样估精度，产出统计文档
#
#   bash run/4_verify.sh               抽 400 图校验清洗后产出
#   SAMPLE=1000 bash run/4_verify.sh   加大样本
#   IN=ground bash run/4_verify.sh     校验清洗前的原始产出（做对照）
#
# 产出 out/verify.jsonl + docs/RESULT.md
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

SAMPLE="${SAMPLE:-400}"
IN="${IN:-clean}"          # clean | ground
case "$IN" in
clean)  IN_DIR="$OUT/clean";  PAT="clean_shard*.jsonl" ;;
ground) IN_DIR="$OUT/ground"; PAT="ground_shard*.jsonl" ;;
*)      die "IN 只能是 clean 或 ground" ;;
esac
VERIFY="$OUT/verify_$IN.jsonl"

need_file "$IN_DIR" "阶段$([ "$IN" = clean ] && echo 3 || echo 2) 未产出"

if ! curl -s --noproxy '*' -m 3 "http://127.0.0.1:$PORT_BASE/health" >/dev/null 2>&1; then
  bash "$(dirname "${BASH_SOURCE[0]}")/sgl.sh" up || die "sglang 启动失败"
fi

log "阶段4 校验 $IN（抽 $SAMPLE 图）-> $VERIFY"
"$PY_SGL" "$SRC/s4_verify.py" --in-dir "$IN_DIR" --pattern "$PAT" \
  --out "$VERIFY" --urls "$URLS" --model "$GEMMA" --sample "$SAMPLE" 2>&1 | tee "$LOGS/verify.log"

log "生成报告 -> $REPO/docs/RESULT.md"
"$PY_ANY" "$SRC/report.py" --cap-dir "$OUT/caption" --ground-dir "$OUT/ground" \
  --clean-dir "$OUT/clean" --verify "$VERIFY" --out "$REPO/docs/RESULT.md"
