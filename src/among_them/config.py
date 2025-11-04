import os
from enum import Enum

from dotenv import load_dotenv

# Always load from .env first, this will override any existing environment variables
load_dotenv(override=True)

class LLMBackend(Enum):
    OLLAMA = "ollama"
    MLX = "mlx"
    OPENROUTER = "openrouter"
    LOCAL_PROBABILITY = "local_probability"

# LLM Backend Selection
LLM_BACKEND = LLMBackend(os.getenv("LLM_BACKEND", "ollama").lower())

# Model configurations
# Default model names per backend
DEFAULT_MODELS = {
    LLMBackend.OLLAMA: "deepseek-r1:14b",
    LLMBackend.MLX: "mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit",
    LLMBackend.OPENROUTER: "deepseek/deepseek-r1:free",
    LLMBackend.LOCAL_PROBABILITY: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
}

# Single model name configuration with backend-specific default
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODELS[LLM_BACKEND])

# API keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Other configurations
STATE_FILE = os.getenv("STATE_FILE", "data/game_state.json")

# Backward compatibility helpers
USE_MLX = LLM_BACKEND == LLMBackend.MLX
RUN_LOCALLY = LLM_BACKEND in [LLMBackend.OLLAMA, LLMBackend.MLX, LLMBackend.LOCAL_PROBABILITY]
USE_OPENROUTER = LLM_BACKEND == LLMBackend.OPENROUTER

# Load MLX model if needed
if LLM_BACKEND == LLMBackend.MLX:
    from mlx_lm import load
    MLX_MODEL, MLX_TOKENIZER = load(MODEL_NAME)

LLM_DEBUG = os.getenv("LLM_DEBUG", "false").lower() == "true"