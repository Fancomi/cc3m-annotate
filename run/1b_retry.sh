#!/usr/bin/env bash
# 阶段 1b · 补齐 caption 失败项（sglang 偶发重启导致约 8% 请求失败）
#
#   bash run/1b_retry.sh          scan + run + 等待 + merge 全流程
#   bash run/1b_retry.sh scan     只扫描待补条目
#   bash run/1b_retry.sh merge    只合并（run 全部结束后执行）
#
# merge 会原地改写 out/caption/shard*.jsonl（先写 .tmp 再 rename，中断不会留半截文件）。
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

CAP_DIR="$OUT/caption"
STEP="${1:-all}"

need_file "$CAP_DIR" "阶段1 未产出"

do_scan() {
  log "扫描待补条目（全扫 3.7G，约 2 分钟）"
  "$PY_SGL" "$SRC/s1b_caption_retry.py" scan --cap-dir "$CAP_DIR"
}

do_run() {
  if ! curl -s --noproxy '*' -m 3 "http://127.0.0.1:$PORT_BASE/health" >/dev/null 2>&1; then
    bash "$(dirname "${BASH_SOURCE[0]}")/sgl.sh" up || die "sglang 启动失败"
  fi
  log "并行补跑"
  for s in $(seq 0 $((NUM_GPU - 1))); do
    spawn "$LOGS/retry_$s.log" "$PY_SGL" "$SRC/s1b_caption_retry.py" run \
      --cap-dir "$CAP_DIR" --urls "$URLS" --model "$GEMMA" \
      --shard "$s" --num-shards "$NUM_GPU" --concurrency "$CAP_CONCURRENCY"
  done
}

do_wait() {
  log "等待补跑结束"
  while pgrep -f "s1b_caption_retry.py run" >/dev/null; do sleep 60; done
  log "补跑结束"
}

do_merge() {
  log "合并回原 shard"
  "$PY_SGL" "$SRC/s1b_caption_retry.py" merge --cap-dir "$CAP_DIR"
}

case "$STEP" in
scan)  do_scan ;;
run)   do_run ;;
merge) do_merge ;;
all)   do_scan; do_run; do_wait; do_merge ;;
*)     die "用法: bash run/1b_retry.sh {all|scan|run|merge}" ;;
esac
