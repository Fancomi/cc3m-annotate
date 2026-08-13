#!/usr/bin/env bash
# 阶段4 自动接力守护 —— 等阶段3 完成且 GPU 释放后，自动拉起 gemma4 sglang 并跑校验+报告。
#
# 用法：
#   bash run/auto_verify.sh           前台跑（推荐 nohup 后台）
#   bash run/auto_verify.sh --check   只检查一次就退出，不做等待
#
# 守护逻辑（轮询）：
#   1) 阶段3 是否完成：out/clean/clean_shard{0..7}.jsonl 全部存在且非空
#   2) GPU 是否释放：8 张卡每卡空闲 >= NEED_FREE_MIB（默认 52000，gemma4 单实例 ~52G）
#   条件都满足 → 跑 run/4_verify.sh，成功后退出；否则每 SLEEP 秒再查。
# 日志：logs/auto_verify.log

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
source run/env.sh

SLEEP="${SLEEP:-120}"            # 轮询间隔（秒）
NEED_FREE_MIB="${NEED_FREE_MIB:-52000}"
MAX_TRY="${MAX_TRY:-600}"        # 最多轮询次数（120s * 600 ≈ 20h）

clean_ready() {
  local ok=1
  for i in $(seq 0 $((NUM_GPU - 1))); do
    [ -s "$OUT/clean/clean_shard$i.jsonl" ] || ok=0
  done
  echo "$ok"
}

gpu_ready() {
  # 8 张卡每张空闲显存都要 >= NEED_FREE_MIB
  local free minfree=999999999
  for i in $(seq 0 $((NUM_GPU - 1))); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$i" 2>/dev/null | tr -d ' ')
    [ -n "$free" ] && [ "$free" -lt "$minfree" ] && minfree="$free"
  done
  [ "$minfree" -ge "$NEED_FREE_MIB" ] && echo "$minfree" || echo 0
}

check_once() {
  local c g
  c=$(clean_ready)
  g=$(gpu_ready)
  echo "[$(date '+%F %T')] clean_ready=$c  gpu_min_free=${g}MiB (need >= ${NEED_FREE_MIB})"
  [ "$c" = "1" ] && [ "$g" -ge "$NEED_FREE_MIB" ]
}

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/auto_verify.log"; }

if [ "${1:-}" = "--check" ]; then
  if check_once; then echo "READY"; else echo "NOT_READY"; fi
  exit 0
fi

log "自动接力守护启动：等待阶段3完成 + GPU 释放（空闲>=${NEED_FREE_MIB}MiB/卡）"

for ((t = 0; t < MAX_TRY; t++)); do
  if check_once; then
    log "条件满足，开始阶段4 校验"
    bash run/4_verify.sh >>"$LOGS/auto_verify.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
      log "阶段4 完成 ✅ 产出 out/verify_clean.jsonl + docs/RESULT.md"
    else
      log "阶段4 失败 rc=$rc（见 logs/verify.log）；重试窗口已到，退出"
    fi
    exit $rc
  fi
  log "等待中… ($((t + 1))/$MAX_TRY, sleep ${SLEEP}s)"
  sleep "$SLEEP"
done

log "超过 $MAX_TRY 次轮询仍未满足条件，放弃。"
exit 1
