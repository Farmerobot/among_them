import os
from among_them.consts import STATE_FILE
from among_them.game_engine import GameEngine
from among_them.models.player import AIPlayer
from among_them.agents.unified_agent import UnifiedAgent

# To run this script, you need to
# `poetry install`
# and then run the following command:
# `poetry run main`


def main():
    agent = UnifiedAgent("deepseek-r1:1.5b")
    players = [
        AIPlayer(name="Alice", agent=agent),
        AIPlayer(name="Bob", agent=agent),
        AIPlayer(name="Charlie", agent=agent),
        AIPlayer(name="David", agent=agent),
        AIPlayer(name="Eve", agent=agent),
    ]
    game_engine = GameEngine(players, 1)
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
                print(f"main.py: Model failed at {action_type.name}. Retry count: {retry_count}")
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
