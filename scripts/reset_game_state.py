import os
import json
import sys
import argparse
from typing import Dict, Any, List, Tuple, Optional

# Add the project root to the path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from among_them.game.game_engine import GameEngine
from among_them.game.models.engine import GamePhase
from among_them.game.game_state import GameState
from among_them.game.players.base_player import Player


def find_discussion_rounds(game_state: GameState) -> List[int]:
    """
    Find all discussion rounds in the game history.
    
    Args:
        game_state: The game state to analyze
        
    Returns:
        List of round indices that are discussion rounds
    """
    discussion_rounds = []
    
    # Use the first player's history as reference since all players share the same rounds
    if game_state.players and game_state.players[0].history.rounds:
        for i, round_data in enumerate(game_state.players[0].history.rounds):
            if round_data.stage == GamePhase.DISCUSS:
                discussion_rounds.append(i)
    
    return discussion_rounds


def get_player_by_name(game_state: GameState, player_name: str) -> Optional[Player]:
    """Find a player by name in the game state."""
    for player in game_state.players:
        if player.name.lower() == player_name.lower():
            return player
    return None


def reset_game_state(
    input_file: str,
    output_file: str,
    player_name: Optional[str] = None,
    use_previous_state: bool = True,
    discussion_round: int = 0
) -> None:
    """
    Reset the game state to a specific point in a discussion round.
    
    Args:
        input_file: Path to the input game state JSON file
        output_file: Path to save the reset game state
        player_name: Optional name of a player to focus on
        use_previous_state: If True, use the state before the player's round; if False, use current state
        discussion_round: Discussion round number (0-indexed from first discussion)
    """
    # Create a game engine to load the state
    engine = GameEngine()
    success = engine.load_state(input_file)
    
    if not success:
        print(f"Failed to load game state from {input_file}")
        return
    
    game_state = engine.state
    
    # Find all discussion rounds
    discussion_rounds = find_discussion_rounds(game_state)
    
    if not discussion_rounds:
        print("No discussion phase found in game history")
        return
    
    print(f"Found {len(discussion_rounds)} discussion rounds: {discussion_rounds}")
    
    # Determine which round to use
    if discussion_round < 0 or discussion_round >= len(discussion_rounds):
        print(f"Invalid discussion round number. Available rounds: 0-{len(discussion_rounds)-1}")
        return
    
    target_round_idx = discussion_rounds[discussion_round]
    reset_round_idx = target_round_idx
    print(f"Using discussion round {discussion_round} (game round {target_round_idx})")
    
    # If a player is specified, show their available states
    if player_name:
        target_player = get_player_by_name(game_state, player_name)
        if not target_player:
            print(f"Player '{player_name}' not found. Available players: {[p.name for p in game_state.players]}")
            return
            
        # If we're using the previous state and we're at the first discussion round or earlier
        if use_previous_state:
            if target_round_idx > 0:
                reset_round_idx = target_round_idx - 1
                print(f"Using state BEFORE round {target_round_idx} (game round {reset_round_idx})")
            else:
                print("Cannot use previous state for the first round. Using the current state instead.")
                reset_round_idx = target_round_idx
        else:
            reset_round_idx = target_round_idx
            print(f"Using the CURRENT state (game round {reset_round_idx})")
    else:
        # If no player specified, just use the specified round
        if use_previous_state:
            if target_round_idx > 0:
                reset_round_idx = target_round_idx - 1
                print(f"Using state BEFORE round {target_round_idx} (game round {reset_round_idx})")
            else:
                print("Cannot use previous state for the first round. Using the current state instead.")
                reset_round_idx = target_round_idx
    
    # Update each player's state
    for player in game_state.players:
        if reset_round_idx < len(player.history.rounds):
            # Copy state from history
            player.state = player.history.rounds[reset_round_idx].model_copy(deep=True)
            
            # Truncate history
            player.history.rounds = player.history.rounds[:reset_round_idx + 1]
    
    # Update game state
    game_state.round_number = reset_round_idx
    game_state.player_to_act_next = 0
    game_state.game_stage = GamePhase.DISCUSS
    
    # Truncate playthrough history if needed
    if hasattr(game_state, 'playthrough') and len(game_state.playthrough) > 0:
        # Find appropriate cutoff point in playthrough
        cut_index = 0
        for i, entry in enumerate(game_state.playthrough):
            # This is a more general approach for finding round transitions
            if f"round: {reset_round_idx + 1}" in entry:
                cut_index = i - 1
                break
        
        if cut_index > 0:
            game_state.playthrough = game_state.playthrough[:cut_index + 1]
    
    # Save the reset game state
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the state
    engine.save_state()
    
    print(f"Reset game state saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reset the game state to a specific point in a discussion round.')
    
    parser.add_argument('--input', '-i', 
                        default=os.path.join(project_root, "data", "tournament", "claude-3-5-sonnet_vs_llama-3-1-405b-instruct_7.json"),
                        help='Path to the input game state JSON file')
    
    parser.add_argument('--output', '-o', 
                        default=os.path.join(project_root, "data", "game_state.json"),
                        help='Path to save the reset game state')
    
    parser.add_argument('--player', '-p', 
                        help='Name of a player to focus on')
    
    parser.add_argument('--round', '-r', type=int, default=0,
                        help='Discussion round number (0-indexed from first discussion)')
    
    parser.add_argument('--after', '-a', action='store_true',
                        help='Use the current state instead of the previous state (default is previous)')
    
    args = parser.parse_args()
    
    reset_game_state(
        args.input, 
        args.output, 
        args.player, 
        not args.after,  # If --after is specified, use_previous_state should be False
        args.round
    )
