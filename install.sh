#!/usr/bin/env bash
# 环境一键配置：建两个 venv、装依赖、下权重、自检。
#
#   bash install.sh              全部步骤（幂等，已完成的跳过）
#   bash install.sh env_f2        只装某一步
#   bash install.sh --list        列出所有步骤
#
# 目标硬件：8×H800 80G / CUDA 12.9 驱动 550+ / Ubuntu 22.04 / uv 已装
# 磁盘需求：venv 约 25G，权重约 50G，/dev/shm 需 50G 空闲
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/root/paddlejob/workspace/env_run/penghaotian}"
ENVS_DIR="$WORK_ROOT/envs"
MODELS_DIR="$WORK_ROOT/models"
STAMP="$WORK_ROOT/.install_cc3m"      # 步骤完成标记
UV="${UV:-$(command -v uv || echo /root/.local/bin/uv)}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

mkdir -p "$ENVS_DIR" "$MODELS_DIR" "$STAMP"

log()  { echo -e "\033[1;36m[install]\033[0m $*"; }
die()  { echo -e "\033[1;31m[fail]\033[0m $*" >&2; exit 1; }
done_p() { [ -f "$STAMP/$1.done" ]; }
mark()   { touch "$STAMP/$1.done"; }

uvpip() { "$UV" pip install --python "$1" --index-url "$PIP_INDEX" "${@:2}"; }

# ---------------- step: 系统依赖 ----------------
step_system() {
  log "apt 依赖"
  apt-get update -qq
  apt-get install -y -qq aria2 curl git jq unzip build-essential \
    libgl1 libglib2.0-0 >/dev/null
  [ -x "$UV" ] || die "缺 uv：curl -LsSf https://astral.sh/uv/install.sh | sh"
}

# ---------------- step: F2 推理环境 ----------------
# 关键约束：transformers 必须 4.46.x。Florence-2 是 trust_remote_code 自定义架构，
# 引用了 transformers 4.46 的私有 API（_prepare_4d_attention_mask 等），
# 5.x 已删除，升级会直接 ImportError。torch 锁 cu124 是因为驱动 550 不支持 cu129。
step_env_f2() {
  local V="$ENVS_DIR/dam"
  log "建 F2 环境 $V (python 3.10)"
  "$UV" venv --python 3.10 "$V"
  uvpip "$V/bin/python" torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124
  uvpip "$V/bin/python" \
    "transformers==4.46.3" "tokenizers==0.20.3" "numpy==1.26.4" \
    timm einops pillow accelerate safetensors sentencepiece protobuf
  "$V/bin/python" -c "
import torch, transformers
assert transformers.__version__.startswith('4.46'), transformers.__version__
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('transformers', transformers.__version__)"
}

# ---------------- step: sglang 服务环境 ----------------
# 与 F2 环境隔离：sglang 要 transformers 5.x + torch cu129，与 F2 的 4.46/cu124 互斥。
step_env_sgl() {
  local V="$ENVS_DIR/sglang__0.5.12"
  log "建 sglang 环境 $V (python 3.12)"
  "$UV" venv --python 3.12 "$V"
  uvpip "$V/bin/python" "sglang[all]==0.5.12" || {
    echo "  pip 装 sglang 失败（多为网络问题）。源码编译见 docs/INSTALL.md" >&2
    return 1
  }
  uvpip "$V/bin/python" openai pillow
  "$V/bin/python" -c "import sglang, openai; print('sglang', sglang.__version__)"
}

# ---------------- step: 模型权重 ----------------
step_weights() {
  command -v huggingface-cli >/dev/null || "$UV" tool install -q huggingface_hub[cli] || true
  local HF="${HF_ENDPOINT:-https://hf-mirror.com}"
  fetch() {  # $1=repo  $2=本地目录名
    local d="$MODELS_DIR/$2"
    [ -f "$d/config.json" ] && { log "$2 已存在，跳过"; return; }
    log "下载 $1 -> $d"
    HF_ENDPOINT="$HF" huggingface-cli download "$1" --local-dir "$d" --quiet \
      || die "下载 $1 失败；可手动放到 $d"
  }
  fetch microsoft/Florence-2-large Florence-2-large
  # gemma4 权重需授权，若下载失败请手动放置到 $MODELS_DIR/gemma-4-26B-A4B-it
  fetch google/gemma-4-26b-a4b-it gemma-4-26B-A4B-it || true
}

# ---------------- step: 自检 ----------------
step_check() {
  log "自检"
  local F2="$ENVS_DIR/dam/bin/python" SGL="$ENVS_DIR/sglang__0.5.12/bin/python"
  "$F2" - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import os
M = os.environ["MODELS_DIR"] + "/Florence-2-large"
m = AutoModelForCausalLM.from_pretrained(M, trust_remote_code=True, torch_dtype=torch.float16).cuda().eval()
p = AutoProcessor.from_pretrained(M, trust_remote_code=True)
from PIL import Image
img = Image.new("RGB", (640, 480), (120, 120, 120))
inp = p(text="<CAPTION_TO_PHRASE_GROUNDING>a gray background", images=img, return_tensors="pt")
inp = {k: (v.cuda().half() if v.dtype == torch.float32 else v.cuda()) for k, v in inp.items()}
with torch.no_grad():
    out = m.generate(**inp, max_new_tokens=32, num_beams=1, do_sample=False, eos_token_id=-1)
print("F2 推理 OK, 输出", out.shape)
PY
  "$SGL" -c "import sglang, openai; print('sglang 导入 OK')"
  log "自检通过。下一步：bash run/run_all.sh --smoke"
}

STEPS=(system env_f2 env_sgl weights check)

[ "${1:-}" = "--list" ] && { printf '%s\n' "${STEPS[@]}"; exit 0; }

export MODELS_DIR
run_step() {
  done_p "$1" && { log "$1 已完成，跳过"; return; }
  "step_$1" && mark "$1"
}

if [ $# -gt 0 ]; then
  for s in "$@"; do run_step "$s"; done
else
  for s in "${STEPS[@]}"; do run_step "$s"; done
  log "全部完成。标记目录 $STAMP（删除对应 .done 可重跑某步）"
fi
