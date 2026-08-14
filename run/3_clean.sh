#!/usr/bin/env bash
# 阶段 3 · 清洗：过滤噪声短语与低质框（纯 CPU，单进程，约 20 分钟）
#
#   bash run/3_clean.sh                 无损清洗（去空泛/抄坏/重复框）
#   TRAIN=1 bash run/3_clean.sh         训练数据档 = rec 档（消融推荐，见 scripts/ablate_rules.py）
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
  # rec 档：整词 garbled + min-words 1（不再用 >=2 词砍掉合法单名词）、删抽象短语、
  # 跨短语去重、删贴满整图的框、删贴边窄条。不做任何面积/像素过滤——小物体全留。
  # 面积过滤把精度推到 76.3% 但只剩 69% 保留率（有效信号 -25%），ratio(-0.7pt)、
  # nbox(-0.4pt) 精度反降，三条都被消融否掉。
  EXTRA=(--min-words 1 --word-boundary --abstract --xdup --max-cover 0.95 --edge)
  log "训练数据档 rec：整词+abstract+xdup+cover<=0.95+edge，小物体全留（全量实测精度 70.1%，19.4 短语/图）"
fi

log "阶段3 清洗 -> $CLN_DIR"
"$PY_ANY" "$SRC/s3_clean.py" --ground-dir "$GND_DIR" --cap-dir "$CAP_DIR" \
  --out "$CLN_DIR" "${EXTRA[@]}" 2>&1 | tee "$LOGS/clean.log"
