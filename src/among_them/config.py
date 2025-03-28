import os

from dotenv import load_dotenv

# Always load from .env first, this will override any existing environment variables
load_dotenv(override=True)

# Retrieve API keys and raise an error if they are missing
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
RUN_LOCALLY = os.getenv("RUN_LOCALLY", "True").lower() in ["true", "1", "t"]

if not OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = "None"
    # raise ValueError(
    #     "API key is missing. Please set OPENROUTER_API_KEY "
    #     "in your environment or in a .env file in the project root."
    # )

# if src/among_them/game/dummy.py does not exist, create it.
# This is for abusing streamlit refresh when game_state.json changes
if not os.path.exists("src/among_them/game/dummy.py"):
    with open("src/among_them/game/dummy.py", "w") as f:
        f.write("timestamp = '2024-11-15 00:20:17.946790'")
