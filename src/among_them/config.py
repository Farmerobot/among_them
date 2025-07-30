import os
from enum import Enum

from dotenv import load_dotenv

# Always load from .env first, this will override any existing environment variables
load_dotenv(override=True)

class LLMBackend(Enum):
    OLLAMA = "ollama"
    MLX = "mlx"
    OPENROUTER = "openrouter"
    HUGGINGFACE = "huggingface"

# LLM Backend Selection
LLM_BACKEND = LLMBackend(os.getenv("LLM_BACKEND", "ollama").lower())

# Model configurations
OLLAMA_LLM_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "deepseek-r1:14b")
MLX_LLM_MODEL_NAME = os.getenv("MLX_LLM_MODEL_NAME", "mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL_NAME = os.getenv("OPENROUTER_MODEL_NAME")
HUGGINGFACE_MODEL_NAME = os.getenv("HUGGINGFACE_MODEL_NAME", "Farmerobot/deepseek-r1-among-them")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Other configurations
STATE_FILE = os.getenv("STATE_FILE", "data/game_state.json")

# Backward compatibility helpers
USE_MLX = LLM_BACKEND == LLMBackend.MLX
RUN_LOCALLY = LLM_BACKEND in [LLMBackend.OLLAMA, LLMBackend.MLX]
USE_OPENROUTER = LLM_BACKEND == LLMBackend.OPENROUTER
USE_HUGGINGFACE = LLM_BACKEND == LLMBackend.HUGGINGFACE

# Load MLX model if needed
if LLM_BACKEND == LLMBackend.MLX:
    from mlx_lm import load
    MLX_MODEL, MLX_TOKENIZER = load(MLX_LLM_MODEL_NAME)