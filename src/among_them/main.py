import os
from among_them.consts import STATE_FILE
from among_them.game_engine import GameEngine
from among_them.players.ai import AIPlayer
from among_them.agents.unified_agent import UnifiedAgent

# To run this script, you need to
# `poetry install`
# and then run the following command:
# `poetry run main`


def main():
    agent = UnifiedAgent()
    players = [
        AIPlayer(name="Alice", agent=agent),
        AIPlayer(name="Bob", agent=agent),
        AIPlayer(name="Charlie", agent=agent),
        AIPlayer(name="David", agent=agent),
        AIPlayer(name="Eve", agent=agent),
    ]
    game_engine = GameEngine(players, 2)
    if os.path.exists(STATE_FILE):
        game_engine.load_state()
        print(f"Game loaded from state file with {len(game_engine.history)} history entries")

    for history_item in game_engine.history:
        print(history_item)
    game_ended, reason = game_engine.perform_step()
    print(game_engine.history[-1])
    if game_ended:
        print("Game ended with reason:", reason)
    else:
        print("Game is still in progress")

if __name__ == "__main__":
    main()
