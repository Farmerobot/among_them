import os

from dotenv import load_dotenv

# Always load from .env first, this will override any existing environment variables
load_dotenv(override=True)

OLLAMA_LLM_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "deepseek-r1:1.5b")
STATE_FILE = os.getenv("STATE_FILE", "data/game_state.json")