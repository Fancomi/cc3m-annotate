#!/usr/bin/env bash
# A/B/C 清洗规则对照。
#
#   A 旧规则  --min-words 2               裸子串 garbled —— 删了 45% 短语只为滤 0.5% 碎片
#   B 词边界  --min-words 1 --word-boundary   碎片交给 garbled 拦，合法单名词放行
#   C B+精修  再加 --abstract --xdup      删取景/属性类短语，跨短语去重
#
#   bash run/ab_clean.sh              默认对照 20000 图
#   IMGS=50000 bash run/ab_clean.sh   加大规模
#   VARIANTS="new c" bash run/ab_clean.sh   只跑部分档位
#
# 产出 out/ab/{old,new,c}/clean_shard*.jsonl，随后 run/ab_verify.sh 做同判定器校验。
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

IMGS="${IMGS:-20000}"
VARIANTS="${VARIANTS:-old new c}"
AB="$OUT/ab"
mkdir -p "$AB"

args_for() {
  case "$1" in
  old) echo "--min-area 0.02 --min-words 2" ;;
  new) echo "--min-area 0.02 --min-words 1 --word-boundary" ;;
  c)   echo "--min-area 0.02 --min-words 1 --word-boundary --abstract --xdup" ;;
  *)   die "未知档位 $1" ;;
  esac
}

log "清洗规则对照：档位 [$VARIANTS]，各 $IMGS 图"
for v in $VARIANTS; do
  A=$(args_for "$v")
  log "[$v] $A"
  "$PY_ANY" "$SRC/s3_clean.py" --ground-dir "$OUT/ground" --cap-dir "$OUT/caption" \
    --out "$AB/$v" $A --limit "$IMGS" 2>&1 | tee "$LOGS/ab_clean_$v.log"
done

log "完成。产出："
for v in $VARIANTS; do
  n=$(cat "$AB/$v"/clean_shard*.jsonl 2>/dev/null | wc -l)
  echo "  $v: $n 行  $AB/$v"
done
