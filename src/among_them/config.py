import os

from dotenv import load_dotenv

# Always load from .env first, this will override any existing environment variables
load_dotenv(override=True)

OLLAMA_LLM_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "deepseek-r1:1.5b")
MLX_LLM_MODEL_NAME = os.getenv("MLX_LLM_MODEL_NAME", "mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit")
USE_MLX = os.getenv("USE_MLX", "false").lower() in ("true", "1", "yes")
STATE_FILE = os.getenv("STATE_FILE", "data/game_state.json")


if USE_MLX:
    from mlx_lm import load
    MLX_MODEL, MLX_TOKENIZER = load(MLX_LLM_MODEL_NAME)