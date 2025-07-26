import os

from dotenv import load_dotenv

# Always load from .env first, this will override any existing environment variables
load_dotenv(override=True)

RUN_LOCALLY = os.getenv("RUN_LOCALLY", "false").lower() in ("true", "1", "yes")
OLLAMA_LLM_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "deepseek-r1:14b")
MLX_LLM_MODEL_NAME = os.getenv("MLX_LLM_MODEL_NAME", "mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit")
USE_MLX = os.getenv("USE_MLX", "false").lower() in ("true", "1", "yes")
STATE_FILE = os.getenv("STATE_FILE", "data/game_state.json")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL_NAME = os.getenv("OPENROUTER_MODEL_NAME")


if USE_MLX:
    from mlx_lm import load
    MLX_MODEL, MLX_TOKENIZER = load(MLX_LLM_MODEL_NAME)