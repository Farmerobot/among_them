import os
import json
import sys
from typing import Dict, Any

# Add the project root to the path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from among_them.game.game_engine import GameEngine
from among_them.game.models.engine import GamePhase
from among_them.game.game_state import GameState


def reset_to_discussion_start(input_file: str, output_file: str) -> None:
    """
    Reset the game state to the start of the first discussion phase.
    
    Args:
        input_file: Path to the input game state JSON file
        output_file: Path to save the reset game state
    """
    # Create a game engine to load the state
    engine = GameEngine()
    success = engine.load_state(input_file)
    
    if not success:
        print(f"Failed to load game state from {input_file}")
        return
    
    game_state = engine.state
    
    # Find the first discussion phase in the history
    first_discussion_round = None
    for i, player in enumerate(game_state.players):
        for j, round_data in enumerate(player.history.rounds):
            if round_data.stage == GamePhase.DISCUSS:
                if first_discussion_round is None or j < first_discussion_round:
                    first_discussion_round = j
                break
    
    if first_discussion_round is None:
        print("No discussion phase found in game history")
        return
    
    print(f"Found first discussion at round {first_discussion_round}")
    first_discussion_round -= 1
    # Reset each player's state to the corresponding history round
    for player in game_state.players:
        if first_discussion_round < len(player.history.rounds):
            # Copy state from history
            player.state = player.history.rounds[first_discussion_round].model_copy(deep=True)
            
            # Truncate history to keep only rounds before the discussion
            player.history.rounds = player.history.rounds[:first_discussion_round]
    
    # Update game state
    game_state.round_number = first_discussion_round
    game_state.player_to_act_next = 0
    game_state.game_stage = GamePhase.DISCUSS
    
    # Truncate playthrough history if needed
    if hasattr(game_state, 'playthrough') and len(game_state.playthrough) > 0:
        # This is a simplified approach - ideally we would find the exact point in playthrough
        # that corresponds to the start of the discussion round
        cut_index = 0
        for i, entry in enumerate(game_state.playthrough):
            if "reported dead body" in entry:
                cut_index = i
                break
        
        if cut_index > 0:
            game_state.playthrough = game_state.playthrough[:cut_index + 1]
    
    # Save the reset game state
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert to dict and save
    engine.save_state()
    
    print(f"Reset game state saved to {output_file}")


if __name__ == "__main__":
    input_file = os.path.join(project_root, "data", "tournament", "claude-3-5-sonnet_vs_llama-3-1-405b-instruct_7.json")
    output_file = os.path.join(project_root, "data", "game_state.json")
    
    reset_to_discussion_start(input_file, output_file)
