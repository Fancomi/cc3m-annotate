#!/usr/bin/env bash
# 全流程一键执行：caption -> 补齐 -> grounding -> 清洗 -> 校验 -> 报告
#
#   bash run/run_all.sh              全量（约 60 小时：caption 30h + grounding 29h）
#   bash run/run_all.sh --smoke      冒烟（每 tsv 取 2 行 ≈ 1152 张，约 20 分钟）
#
# 每个阶段都会等前一阶段跑完。阶段内部断点续传，中断后重跑本脚本即从断点继续。
# 想单独跑某阶段，直接执行 run/{1,1b,2,3,4}_*.sh。
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
HERE="$(dirname "${BASH_SOURCE[0]}")"

SMOKE=0
[ "${1:-}" = "--smoke" ] && SMOKE=1

wait_for() {   # $1=进程特征  $2=阶段名
  log "等待「$2」结束..."
  sleep 20
  while pgrep -f "$1" >/dev/null; do sleep 60; done
  log "「$2」结束"
}

count() { cat "$1" 2>/dev/null | wc -l; }

if [ "$SMOKE" = "1" ]; then
  log "===== 冒烟模式：每个 tsv 取 2 行 ====="
  LIMIT=2 bash "$HERE/1_caption.sh"
  wait_for "s1_caption.py" "阶段1 caption"
  log "caption 条数 $(count "$OUT/caption/shard*.jsonl")"

  bash "$HERE/2_grounding.sh"
  wait_for "s2_grounding.py" "阶段2 grounding"
  log "grounding 条数 $(count "$OUT/ground/ground_shard*.jsonl")"

  bash "$HERE/3_clean.sh"
  SAMPLE=60 bash "$HERE/4_verify.sh"
  log "===== 冒烟完成，看 docs/RESULT.md ====="
  exit 0
fi

log "===== 阶段1/5 caption（全量 289 万张，约 30 小时）====="
bash "$HERE/1_caption.sh"
wait_for "s1_caption.py" "阶段1 caption"
log "caption 条数 $(count "$OUT/caption/shard*.jsonl")"

log "===== 阶段2/5 补齐失败项 ====="
bash "$HERE/1b_retry.sh" all

log "===== 阶段3/5 grounding（约 29 小时）====="
bash "$HERE/sgl.sh" down          # 释放显存给 F2
bash "$HERE/2_grounding.sh"
wait_for "s2_grounding.py" "阶段3 grounding"
log "grounding 条数 $(count "$OUT/ground/ground_shard*.jsonl")"

log "===== 阶段4/5 清洗 ====="
bash "$HERE/3_clean.sh"

log "===== 阶段5/5 校验 + 报告 ====="
bash "$HERE/4_verify.sh"

log "===== 全流程完成 ====="
echo "  产出   $OUT/{caption,ground,clean}/"
echo "  报告   $REPO/docs/RESULT.md"
