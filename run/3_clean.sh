#!/usr/bin/env bash
# 阶段 3 · 清洗：过滤噪声短语与低质框（纯 CPU，单进程，约 20 分钟）
#
#   bash run/3_clean.sh                 无损清洗（去空泛/抄坏/重复框）
#   TRAIN=1 bash run/3_clean.sh         训练数据档，额外加 面积>=2% 且 >=2 词
#
# 产出 out/clean/clean_shard*.jsonl
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

CAP_DIR="$OUT/caption"
GND_DIR="$OUT/ground"
CLN_DIR="$OUT/clean"

need_file "$GND_DIR" "阶段2 未产出"
need_file "$CAP_DIR" "阶段1 未产出"

EXTRA=()
if [ "${TRAIN:-0}" = "1" ]; then
  EXTRA=(--min-area 0.02 --min-words 2)
  log "训练数据档：面积>=2% 且 短语>=2 词（实测精度 80.3%，保留约 66%）"
fi

log "阶段3 清洗 -> $CLN_DIR"
"$PY_ANY" "$SRC/s3_clean.py" --ground-dir "$GND_DIR" --cap-dir "$CAP_DIR" \
  --out "$CLN_DIR" "${EXTRA[@]}" 2>&1 | tee "$LOGS/clean.log"
