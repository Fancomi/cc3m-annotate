#!/usr/bin/env bash
# 环境变量与路径 —— 所有 run/*.sh 都 source 它。
# 换机器时只改这个文件。
set -euo pipefail

# ---------------- 根路径 ----------------
: "${WORK_ROOT:=/root/paddlejob/workspace/env_run/penghaotian}"
: "${ENVS_DIR:=$WORK_ROOT/envs}"
: "${MODELS_DIR:=$WORK_ROOT/models}"
: "${SHM_MODELS:=/dev/shm/models}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO/src"
OUT="${OUT:-$REPO/out}"
LOGS="${LOGS:-$REPO/logs}"
mkdir -p "$OUT" "$LOGS"

# ---------------- 数据集 ----------------
# _shards/*.tsv 每行 `图片绝对路径 \t cc3m 原始 caption`，576 个 tsv 共 2894191 行
: "${CC3M_TSV:=$WORK_ROOT/datas/cc3m-tsv/_shards}"

# ---------------- python 解释器 ----------------
# 两套环境不能混用：F2 依赖 transformers 4.46（新版删了它需要的私有 API），
# sglang 依赖 transformers 5.x。详见 docs/INSTALL.md
PY_SGL="${PY_SGL:-$ENVS_DIR/sglang__0.5.12/bin/python}"   # 阶段 1/1b/4：调 VLM 端点
PY_F2="${PY_F2:-$ENVS_DIR/dam/bin/python}"                # 阶段 2：本地跑 Florence-2
PY_ANY="${PY_ANY:-python3}"                               # 阶段 3、report：纯标准库

# ---------------- 模型 ----------------
GEMMA_SRC="${GEMMA_SRC:-$MODELS_DIR/gemma-4-26B-A4B-it}"
GEMMA="${GEMMA:-$SHM_MODELS/gemma-4-26B-A4B-it}"          # 跑在 /dev/shm 上，加载更快
F2_MODEL="${F2_MODEL:-$MODELS_DIR/Florence-2-large}"

# ---------------- 并行度 ----------------
: "${NUM_GPU:=8}"              # 卡数 = 分片数 = sglang 实例数
: "${PORT_BASE:=8101}"         # sglang 端口 8101..8108
: "${CAP_CONCURRENCY:=16}"     # 阶段 1 每进程并发请求数
: "${GND_BATCH:=8}"            # 阶段 2 batch 大小
: "${MEM_FRACTION:=0.80}"      # sglang 显存占比；与 F2 共卡时降到 0.72

# sglang 端点列表，逗号分隔
URLS=""
for i in $(seq 0 $((NUM_GPU - 1))); do
  URLS="${URLS}http://127.0.0.1:$((PORT_BASE + i))/v1,"
done
URLS="${URLS%,}"

log()  { echo -e "\033[1;36m[$(date +%H:%M:%S)]\033[0m $*"; }
warn() { echo -e "\033[1;33m[warn]\033[0m $*" >&2; }
die()  { echo -e "\033[1;31m[fail]\033[0m $*" >&2; exit 1; }

need_file() { [ -e "$1" ] || die "缺少 $1${2:+  ($2)}"; }

# 后台启动一个进程并 disown，避免 shell 退出时被 SIGHUP 杀掉
spawn() {
  local logf="$1"; shift
  setsid nohup env "$@" >"$logf" 2>&1 </dev/null &
  disown
}
