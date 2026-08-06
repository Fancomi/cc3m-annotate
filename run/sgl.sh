#!/usr/bin/env bash
# 拉起 / 停止 gemma4 sglang 服务（阶段 1、1b、4 依赖它）
#
#   bash run/sgl.sh up      拉起 NUM_GPU 个实例并等待就绪
#   bash run/sgl.sh down    全部停止
#   bash run/sgl.sh status  查看就绪情况
#
# 与阶段 2 共卡时用 MEM_FRACTION=0.72 bash run/sgl.sh up 留出 F2 的显存。
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

ready_count() {
  local ok=0
  for i in $(seq 0 $((NUM_GPU - 1))); do
    curl -s --noproxy '*' -m 3 "http://127.0.0.1:$((PORT_BASE + i))/health" >/dev/null 2>&1 && ok=$((ok + 1))
  done
  echo "$ok"
}

case "${1:-up}" in
up)
  need_file "$PY_SGL" "sglang 环境未装，见 docs/INSTALL.md"
  # 首次把权重拷到 /dev/shm：49G，只需一次，后续启动直接命中
  if [ ! -d "$GEMMA" ]; then
    need_file "$GEMMA_SRC" "gemma4 权重缺失"
    log "首次拷贝权重到 /dev/shm（约 49G，仅一次）..."
    mkdir -p "$SHM_MODELS" && cp -r "$GEMMA_SRC" "$GEMMA"
  fi
  log "拉起 $NUM_GPU 个 gemma4 实例（mem-fraction=$MEM_FRACTION）"
  for i in $(seq 0 $((NUM_GPU - 1))); do
    spawn "$LOGS/sgl_$i.log" CUDA_VISIBLE_DEVICES="$i" "$PY_SGL" -m sglang.launch_server \
      --model-path "$GEMMA" --port $((PORT_BASE + i)) \
      --dist-init-addr "127.0.0.1:$((29600 + i))" --tp-size 1 \
      --mem-fraction-static "$MEM_FRACTION" --context-length 32768 \
      --watchdog-timeout 3600 --reasoning-parser gemma4 \
      --skip-server-warmup --trust-remote-code
  done
  log "等待就绪（首次加载约 3 分钟）..."
  for _ in $(seq 60); do
    n=$(ready_count)
    [ "$n" -ge "$NUM_GPU" ] && { log "全部 $n/$NUM_GPU 就绪"; exit 0; }
    sleep 10
  done
  warn "超时，仅 $(ready_count)/$NUM_GPU 就绪；查看 $LOGS/sgl_*.log"
  exit 1
  ;;
down)
  pkill -f "sglang.launch_server" 2>/dev/null || true
  sleep 5
  pkill -9 -f "sglang.launch_server" 2>/dev/null || true
  log "已停止全部 sglang 实例"
  ;;
status)
  log "就绪 $(ready_count)/$NUM_GPU  进程 $(pgrep -fc 'sglang.launch_server' || echo 0)"
  ;;
*)
  die "用法: bash run/sgl.sh {up|down|status}"
  ;;
esac
