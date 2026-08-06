#!/usr/bin/env bash
# 阶段 1b · 收口：压实分片并报告仍缺的条目
#
#   bash run/1b_compact.sh            压实 + 报告
#   bash run/1b_compact.sh check      只报告不改文件
#
# 阶段1 的续传是 append 语义：补跑一条 error 记录时成功结果追加在原行之后，
# 同一张图会出现多行。这里把每图压成一行（优先保成功记录）。
#
# 若仍有缺失，直接重跑 bash run/1_caption.sh —— 它会自动只重试 error 条目。
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

CAP_DIR="$OUT/caption"
need_file "$CAP_DIR" "阶段1 未产出"

if [ "${1:-}" = "check" ]; then
  "$PY_ANY" "$SRC/compact.py" --dir "$CAP_DIR" --dry-run
  exit 0
fi

log "压实 caption 分片"
"$PY_ANY" "$SRC/compact.py" --dir "$CAP_DIR" | tee "$LOGS/compact.log"

MISS=$(grep -oE '仍缺 [0-9]+' "$LOGS/compact.log" | tail -1 | grep -oE '[0-9]+' || echo 0)
if [ "${MISS:-0}" -gt 0 ]; then
  warn "仍缺 $MISS 条。重跑 bash run/1_caption.sh 会自动只补这些（error 条目不计入已完成）"
else
  log "caption 完整，可进入阶段2"
fi
