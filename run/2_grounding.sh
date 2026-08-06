#!/usr/bin/env bash
# 阶段 2 · grounding：Florence-2 把 caption 短语定位成框
#
#   bash run/2_grounding.sh            全量（8 卡约 29 小时，约 10 万图/时）
#   LIMIT=64 bash run/2_grounding.sh   每分片只跑 64 张（冒烟）
#
# 产出 out/ground/ground_shard{0..7}.jsonl，断点续传按 path 去重。
# 只用本地 GPU，不依赖 sglang；若 sglang 还在跑会抢显存，建议先 bash run/sgl.sh down。
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

CAP_DIR="$OUT/caption"
GND_DIR="$OUT/ground"
LIMIT="${LIMIT:-0}"

need_file "$CAP_DIR" "阶段1 未产出"
need_file "$PY_F2" "F2 环境未装，见 docs/INSTALL.md"
need_file "$F2_MODEL" "Florence-2 权重缺失"

if pgrep -f "sglang.launch_server" >/dev/null; then
  warn "sglang 仍在运行，会与 F2 抢显存。建议先 bash run/sgl.sh down"
fi

log "阶段2 grounding -> $GND_DIR  (batch=$GND_BATCH limit=$LIMIT)"
for s in $(seq 0 $((NUM_GPU - 1))); do
  spawn "$LOGS/gnd_$s.log" CUDA_VISIBLE_DEVICES="$s" "$PY_F2" "$SRC/s2_grounding.py" \
    --cap-dir "$CAP_DIR" --out "$GND_DIR" --model "$F2_MODEL" \
    --shard "$s" --num-shards "$NUM_GPU" --batch "$GND_BATCH" --limit "$LIMIT"
done

log "已启动 $NUM_GPU 个进程。查看进度："
echo "  tail -f $LOGS/gnd_0.log"
echo "  cat $GND_DIR/ground_shard*.jsonl | wc -l"
