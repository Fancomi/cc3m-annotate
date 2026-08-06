#!/usr/bin/env bash
# 阶段 1 · caption：gemma4 为全量图片生成两级 caption
#
#   bash run/1_caption.sh              全量（289 万张，8 卡约 30 小时）
#   LIMIT=20 bash run/1_caption.sh     每个 tsv 只取 20 行（冒烟）
#
# 产出 out/caption/shard{0..7}.jsonl，断点续传：重跑同一命令即继续。
# 依赖 sglang 服务，脚本会自动检查并按需拉起。
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

CAP_DIR="$OUT/caption"
LIMIT="${LIMIT:-0}"

need_file "$CC3M_TSV" "cc3m tsv 数据集"
need_file "$PY_SGL"

# 服务未就绪则拉起
if ! curl -s --noproxy '*' -m 3 "http://127.0.0.1:$PORT_BASE/health" >/dev/null 2>&1; then
  log "sglang 未就绪，先拉起"
  bash "$(dirname "${BASH_SOURCE[0]}")/sgl.sh" up || die "sglang 启动失败"
fi

log "阶段1 caption -> $CAP_DIR  (limit-per-tsv=$LIMIT)"
for s in $(seq 0 $((NUM_GPU - 1))); do
  spawn "$LOGS/cap_$s.log" "$PY_SGL" "$SRC/s1_caption.py" \
    --tsv-dir "$CC3M_TSV" --out "$CAP_DIR" --urls "$URLS" --model "$GEMMA" \
    --shard "$s" --num-shards "$NUM_GPU" --concurrency "$CAP_CONCURRENCY" \
    --limit-per-tsv "$LIMIT"
done

log "已启动 $NUM_GPU 个进程。查看进度："
echo "  tail -f $LOGS/cap_0.log"
echo "  cat $CAP_DIR/shard*.jsonl | wc -l        # 已完成条数（全量目标 2894191）"
