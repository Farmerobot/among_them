#!/bin/bash
#SBATCH --job-name=sft_1.5B_att_r64
#SBATCH --partition=hgx              # or ngpu if you prefer 4080/4090
#SBATCH --account=inz_atllm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1                 # 1 GPU (A100-80GB on hgx)
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sft_uv_%j.out
#SBATCH --error=logs/sft_uv_%j.err

set -euo pipefail

echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================="

echo "GPU Information:"
nvidia-smi || echo "nvidia-smi not available"
echo "========================================="

# -----------------------------------------------------------------------------
# Paths & directories
# -----------------------------------------------------------------------------
WORK_DIR="${HOME}/among_them"
cd "${WORK_DIR}"

mkdir -p logs
mkdir -p data/alpaca
# SFT training paths (read by sft_slurm.py via environment variables)
export SFT_DATA_DIR="${WORK_DIR}/data/alpaca"
export SFT_OUTPUT_DIR="${WORK_DIR}/outputs/1.5B_attention_r64/checkpoints"
export SFT_MERGED_MODEL_DIR="${WORK_DIR}/outputs/1.5B_attention_r64/merged"
mkdir -p "${SFT_OUTPUT_DIR}"
mkdir -p "${SFT_MERGED_MODEL_DIR}"

# Training hyperparameters (all required - no defaults in sft_slurm.py)
export SFT_BASE_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
export SFT_MAX_SEQ_LENGTH="9000"
export SFT_LORA_RANK="64"
export SFT_TARGET_MODULES="attention"  # attention (4 modules) or full (7 modules)
export SFT_NUM_EPOCHS="5"
export SFT_BATCH_SIZE="8"
export SFT_GRAD_ACCUM_STEPS="4"        # Effective batch = 32
export SFT_LEARNING_RATE="2e-4"
export SFT_LOAD_IN_4BIT="false"
export SFT_DEBUG="false"

# WandB configuration
export SFT_WANDB_PROJECT="among-them-sft"
export SFT_WANDB_RUN_NAME="1.5B-attention-r64-${SLURM_JOB_ID:-local}"
# WANDB_API_KEY should be set in your environment or ~/.bashrc

# -----------------------------------------------------------------------------
# NCCL & CUDA Environment Variables
# -----------------------------------------------------------------------------
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=120
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "Working Directory: ${WORK_DIR}"
echo "Data Directory: ${SFT_DATA_DIR}"
echo "Output Directory: ${SFT_OUTPUT_DIR}"
echo "Merged Model Directory: ${SFT_MERGED_MODEL_DIR}"
echo "========================================="

# -----------------------------------------------------------------------------
# Python + uv bootstrap
# -----------------------------------------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: python3 not found on PATH."
    exit 1
fi

echo "Using Python: $(${PYTHON_BIN} --version)"

export PATH="${HOME}/.local/bin:${PATH}"

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv into ~/.local/bin ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "uv version: $(uv --version)"

UV_VENV_DIR="${HOME}/.uv-venvs/among_them-sft"
mkdir -p "$(dirname "${UV_VENV_DIR}")"

echo "Creating / updating uv environment at ${UV_VENV_DIR} ..."
uv venv "${UV_VENV_DIR}" --python 3.11

# shellcheck source=/dev/null
source "${UV_VENV_DIR}/bin/activate"

echo "Python inside venv: $(python --version)"

# -----------------------------------------------------------------------------
# Install dependencies with uv pip
# -----------------------------------------------------------------------------
echo "Installing core dependencies with uv..."

uv pip install --upgrade numpy pillow

# PyTorch with CUDA 11.8 (matches cluster toolkit)
uv pip install --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu118 \
    "torch==2.7.1" \
    "torchvision==0.22.1" \
    "torchaudio==2.7.1"

echo "Installing flash-attn (compiles from source)..."
uv pip install setuptools wheel packaging ninja hatchling
uv pip install flash-attn==2.7.3 --no-build-isolation

# Transformers / TRL stack
uv pip install transformers==4.56.2 tokenizers trl==0.22.2

# xformers for memory efficiency
uv pip install xformers --index-url https://download.pytorch.org/whl/cu118

# Unsloth
uv pip install "git+https://github.com/unslothai/unsloth.git"

# Project-local package
uv pip install -e "${WORK_DIR}"

echo "Installed packages:"
python - << 'EOF'
import torch, sys
print("Python:", sys.version)
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))
EOF

echo "========================================="

# -----------------------------------------------------------------------------
# Run training
# -----------------------------------------------------------------------------
echo "Starting SFT training..."
echo "========================================="

python scripts/sft_slurm/sft_slurm.py

EXIT_CODE=$?

echo "========================================="
if [ "${EXIT_CODE}" -eq 0 ]; then
    echo "SFT training completed successfully!"
else
    echo "SFT training failed with exit code ${EXIT_CODE}"
fi
echo "End Time: $(date)"
echo "========================================="

exit "${EXIT_CODE}"
