#!/usr/bin/env python3
import os
import json
import argparse
import tempfile
import sys
from typing import Dict, List, Any

# Add the project root to the Python path to import the game engine
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.among_them.game.game_engine import GameEngine
from src.among_them.game.models.engine import GamePhase

# Import the reset script functionality
from reset_game_state_raw import reset_game_state, find_discussion_rounds


def load_game_state(file_path: str) -> Dict[str, Any]:
    """Load game state from a file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load game state from {file_path}: {e}")
        return None


def reset_and_vote(input_file: str, temp_file: str, round_idx: int, player_name: str) -> None:
    """
    Reset the game state to a specific round and player, then call go_to_voting().
    """
    # Reset the game state
    reset_game_state(
        input_file=input_file,
        output_file=temp_file,
        discussion_round=round_idx,
        player_name=player_name,
        use_previous_state=True  # Use the state just after the player's message
    )
    
    # Load the reset state and run voting
    engine = GameEngine()
    engine.load_state(temp_file)
    
    # If not in discussion phase, can't vote
    if engine.state.game_stage != GamePhase.DISCUSS:
        print(f"Game is not in discussion phase after reset (round {round_idx}, player {player_name}): {engine.state.game_stage}")
        return
    
    # Simply call go_to_voting() - no need to capture the votes
    print(f"Calling go_to_voting() for round {round_idx}, player {player_name}")
    engine.go_to_voting()


def process_game_file(file_path: str) -> None:
    """
    Process a single game file by resetting to each round and player combination.
    """
    # Load the original game state
    original_game_state = load_game_state(file_path)
    if not original_game_state:
        print(f"Could not load game state from {file_path}")
        return
    
    # Find all discussion rounds
    discussion_rounds = find_discussion_rounds(original_game_state)
    if not discussion_rounds:
        print(f"No discussion rounds found in {file_path}")
        return
    
    print(f"Found {len(discussion_rounds)} discussion rounds")
    
    # Get the list of players
    players = []
    if 'players' in original_game_state:
        players = [p['name'] for p in original_game_state['players']]
    
    print(f"Players: {', '.join(players)}")
    num_players = len(players)
    
    # Process each combination of round and player
    max_round = min(4, len(discussion_rounds) - 1)  # Limit to rounds 0-4 or available rounds
    
    for round_idx in range(max_round + 1):
        for player_idx in range(num_players):
            print(f"Processing round {round_idx}, player {player_idx} ({players[player_idx]})")
            
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp:
                temp_file = temp.name
            
            try:
                # Reset and vote
                reset_and_vote(file_path, temp_file, round_idx, players[player_idx])
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_file):
                    os.remove(temp_file)


def find_tournament_files(tournament_dir: str) -> List[str]:
    """Find all JSON files in the tournament directory."""
    tournament_files = []
    for root, _, files in os.walk(tournament_dir):
        for file in files:
            if file.endswith('.json'):
                tournament_files.append(os.path.join(root, file))
    return tournament_files


def main():
    parser = argparse.ArgumentParser(description='Reset game state to each round and player, then call go_to_voting().')
    
    parser.add_argument('--input', '-i',
                      help='Path to a single input game file to analyze')
    parser.add_argument('--tournament-dir', '-t',
                      help='Path to the tournament directory containing multiple game files', default="data/tournament")
    
    parser.add_argument('--limit', '-l', type=int, default=0,
                      help='Maximum number of files to process when using --tournament-dir (0 for all)')
    
    args = parser.parse_args()
    
    if args.input:
        # Process a single file
        input_file = args.input
        if not os.path.exists(input_file):
            print(f"Input file {input_file} does not exist.")
            return
        
        print(f"Processing single game file: {input_file}")
        process_game_file(input_file)
    else:
        # Process all files in tournament directory
        tournament_dir = args.tournament_dir
        if not os.path.exists(tournament_dir):
            print(f"Tournament directory {tournament_dir} does not exist.")
            return
        
        tournament_files = find_tournament_files(tournament_dir)
        
        # Limit the number of files if requested
        if args.limit > 0 and args.limit < len(tournament_files):
            print(f"Limiting to {args.limit} of {len(tournament_files)} files")
            tournament_files = tournament_files[:args.limit]
        
        print(f"Found {len(tournament_files)} game files to process in {tournament_dir}")
        
        for i, file_path in enumerate(tournament_files):
            print(f"\nProcessing file {i+1}/{len(tournament_files)}: {os.path.basename(file_path)}")
            process_game_file(file_path)
    
    print("Done!")


if __name__ == "__main__":
    main()
