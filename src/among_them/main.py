import os

from among_them.config import OLLAMA_LLM_MODEL_NAME, STATE_FILE
from among_them.game_config import GameConfig
from among_them.game_engine import GameEngine
from among_them.models.player import Player

# To run this script, you need to
# `poetry install`
# and then run the following command:
# `poetry run main`

# Console output color guide:
# Green - prompt
# Red - system prompt
# Blue - llm response

def main():
    game_config = GameConfig(
        num_tasks=4,
        num_players=6,
        num_impostors=2,
        map_size=0,
        num_task_phase_actions_per_player=10,
        num_discuss_phase_actions_per_player=2,
        impostor_cooldown=0
    )
    game_engine = GameEngine(game_config)
    if os.path.exists(STATE_FILE):
        game_engine.load_state()
        print(f"Game loaded from state file with {len(game_engine.history)} history entries")

    for history_item in game_engine.history:
        print(history_item)
    game_ended = False
    once = False
    retry_count = 0
    while not game_ended:
        try:
            game_ended, reason = game_engine.perform_step()
            print(game_engine.history[-1])
            if once:
                break
        except Exception as e:
            if "LLM did" in str(e):
                print(f"Error: {e}")
                print(f"main.py: Model failed. Retry count: {retry_count}")
                retry_count += 1
                continue
            else:
                raise e
    if game_ended:
        print("Game ended with reason:", reason)
        # Archive the state file with timestamp and start new game
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = f"data/{timestamp}.json"
        if os.path.exists(STATE_FILE):
            os.makedirs(os.path.dirname(archive_path), exist_ok=True)
            os.rename(STATE_FILE, archive_path)
            print(f"Archived game state to {archive_path}")
            main()  # Start a new game
    else:
        print("Game is still ongoing")

if __name__ == "__main__":
    main()
