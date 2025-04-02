import os

from dotenv import load_dotenv

# Always load from .env first, this will override any existing environment variables
load_dotenv(override=True)

RUN_LOCALLY = os.getenv("RUN_LOCALLY", "True").lower() in ["true", "1", "t"]
OLLAMA_LLM_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "deepseek-r1:1.5b")
PUT_JWT = os.getenv("PUT_JWT")

# Retrieve API keys and raise an error if they are missing
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = "None"
    # raise ValueError(
    #     "API key is missing. Please set OPENROUTER_API_KEY "
    #     "in your environment or in a .env file in the project root."
    # )