#!/bin/bash
#SBATCH --job-name=among_them_uv
#SBATCH --partition=hgx              # or ngpu if you prefer 4080/4090
#SBATCH --account=ai155842           # CHANGE to your account if needed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20           # 16 workers + overhead
#SBATCH --gres=gpu:1                 # 1 GPU (A100-80GB on hgx, 4080/4090 on ngpu)
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/mappo_uv_%j.out
#SBATCH --error=logs/mappo_uv_%j.err

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
mkdir -p data
mkdir -p outputs/mappo_training
mkdir -p outputs/mappo_checkpoints
mkdir -p outputs/mappo_checkpoints/pretrain

export DATA_DIR="${WORK_DIR}/data"
export OUTPUT_DIR="${WORK_DIR}/outputs/mappo_training"
export CHECKPOINT_DIR="${WORK_DIR}/outputs/mappo_checkpoints"

# MAPPO-specific paths consumed by mappo_among_them_unsloth.py (main())
export MAPPO_DATA_DIR="${DATA_DIR}"
export MAPPO_OUTPUT_DIR="${OUTPUT_DIR}"
export MAPPO_CHECKPOINT_DIR="${CHECKPOINT_DIR}"

# Pretrained LoRA and value head paths on the cluster
export MAPPO_LOAD_MODEL="${WORK_DIR}/sft_sampled"
export MAPPO_LOAD_HEAD="${WORK_DIR}/outputs/mappo_checkpoints/pretrain/value_head_epoch_10.pt"

echo "Working Directory: ${WORK_DIR}"
echo "Data Directory: ${DATA_DIR}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Checkpoint Directory: ${CHECKPOINT_DIR}"
echo "MAPPO_LOAD_MODEL: ${MAPPO_LOAD_MODEL}"
echo "MAPPO_LOAD_HEAD: ${MAPPO_LOAD_HEAD}"
echo "========================================="

# -----------------------------------------------------------------------------
# Python + uv bootstrap
#   - Use system Python 3 if available
#   - Install uv into ~/.local/bin if missing
#   - Create per-job virtualenv managed by uv
# -----------------------------------------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: python3 not found on PATH."
    exit 1
fi

echo "Using Python: $(${PYTHON_BIN} --version)"

# Ensure ~/.local/bin on PATH for uv
export PATH="${HOME}/.local/bin:${PATH}"

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv into ~/.local/bin ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "uv version: $(uv --version)"

# Create a project-specific environment under ~/.cache/uv/venvs or similar
# We use an explicit venv path to keep it stable between jobs.
UV_VENV_DIR="${HOME}/.uv-venvs/among_them-mappo"
mkdir -p "$(dirname "${UV_VENV_DIR}")"

echo "Creating / updating uv environment at ${UV_VENV_DIR} ..."
uv venv "${UV_VENV_DIR}"

# Activate the environment (POSIX shell)
# uv venv creates a standard virtualenv layout
# shellcheck source=/dev/null
source "${UV_VENV_DIR}/bin/activate"

echo "Python inside venv: $(python --version)"
echo "Pip inside venv:    $(pip --version)"

# -----------------------------------------------------------------------------
# Install dependencies with uv pip
#   We target a CUDA-enabled PyTorch wheel that works across:
#   - A100 (SM 8.0, CUDA 11.x/12.x)
#   - RTX 4080/4090 (SM 8.9, CUDA 12.x)
#
#   The safest choice is to let pip pick the correct prebuilt wheel from
#   the official index (no conda). On modern drivers, recent torch 2.x
#   wheels with CUDA 12.x will work on both A100 and 40xx cards.
#
#   If your cluster has older CUDA drivers, we might need to pin to a
#   specific +cu11x wheel, but we start with generic torch>=2.1.
# -----------------------------------------------------------------------------

echo "Installing core dependencies with uv (this may take a while) ..."
# Base scientific stack first (lets pip reuse if already present in cache)
uv pip install --upgrade \
    numpy \
    pillow

# Torch & GPU stack
# On Ubuntu 22.04 + recent NVIDIA drivers, this pulls a CUDA-enabled wheel.
uv pip install --upgrade \
    "torch>=2.3.0" \
    "torchvision>=0.18.0" \
    "torchaudio>=2.3.0"

# Optional: bitsandbytes (GPU-accelerated quantization)
# If it fails on this system, training will still work without it.
# uv pip install --quiet "bitsandbytes>=0.43.0" || echo "bitsandbytes install failed; continuing without it."

# # Transformers / TRL / Unsloth stack
# uv pip install --upgrade --quiet \
#     "transformers==4.56.2" \
#     "tokenizers" \
#     "trl==0.22.2"

# # Unsloth + zoo from GitHub (as you used in Colab)
# uv pip install --quiet \
#     "unsloth[base] @ git+https://github.com/unslothai/unsloth" \
#     "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo"
uv pip install unsloth
uv pip install --upgrade --no-deps transformers==4.56.2 tokenizers trl==0.22.2 unsloth unsloth_zoo

# Project-local package install (among_them as a package)
# If your repo is structured as a Python package, this makes imports work.
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
# Environment variables for the training script
# -----------------------------------------------------------------------------
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Uncomment for debugging CUDA issues:
# export CUDA_LAUNCH_BLOCKING="1"

# Override Colab paths inside the script by environment variables the script
# can pick up, if you later modify it to read from env. For now we rely
# on the cluster-adapted version or your own edits.
# Example (if you add support in the script):
#   os.environ.get("MAPPO_DATA_DIR", default_value)
export MAPPO_DATA_DIR="${DATA_DIR}"
export MAPPO_OUTPUT_DIR="${OUTPUT_DIR}"
export MAPPO_CHECKPOINT_DIR="${CHECKPOINT_DIR}"

# -----------------------------------------------------------------------------
# Run training
# -----------------------------------------------------------------------------
echo "Starting MAPPO training with uv-managed environment..."
echo "========================================="

python scripts/mappo_among_them_unsloth.py

EXIT_CODE=$?

echo "========================================="
if [ "${EXIT_CODE}" -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code ${EXIT_CODE}"
fi
echo "End Time: $(date)"
echo "========================================="

exit "${EXIT_CODE}"