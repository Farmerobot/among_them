#!/usr/bin/env python3
import os
import json
import argparse
import csv
import tempfile
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set

# Add the project root to the Python path to import the game engine
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.among_them.game.game_engine import GameEngine
from src.among_them.game.models.engine import GamePhase
from src.among_them.game.game_state import GameState

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


def extract_original_votes(game_state: Dict[str, Any]) -> Dict[str, str]:
    """Extract original votes from the playthrough history."""
    votes = {}
    if 'playthrough' in game_state:
        for entry in game_state['playthrough']:
            if "voted for" in entry and not entry.startswith("vote "):
                parts = entry.strip().split()
                if len(parts) >= 4 and parts[-2] == "for":
                    voter = parts[0]
                    target = parts[-1]
                    votes[voter] = target
    return votes


def reset_and_get_new_votes(input_file: str, temp_file: str) -> Dict[str, str]:
    """Reset the game state and run voting to get new votes."""
    # Find the last discussion round
    game_state = load_game_state(input_file)
    if not game_state:
        return {}
    
    discussion_rounds = find_discussion_rounds(game_state)
    if not discussion_rounds:
        print(f"No discussion rounds found in {input_file}")
        return {}
    
    # Reset to after the last discussion round
    last_discussion_index = len(discussion_rounds) - 1
    reset_game_state(
        input_file=input_file,
        output_file=temp_file,
        discussion_round=last_discussion_index,
        use_previous_state=False  # After the round
    )
    
    # Load the reset state and run voting
    engine = GameEngine()
    engine.load_state(temp_file)
    
    # If not in discussion phase, can't vote
    if engine.state.game_stage != GamePhase.DISCUSS:
        print(f"Game is not in discussion phase after reset: {engine.state.game_stage}")
        return {}
    
    # Capture current votes before voting
    new_votes = {}
    
    # Run the voting
    engine.go_to_voting()
    
    # Extract votes from the playthrough
    if hasattr(engine.state, 'playthrough'):
        for entry in engine.state.playthrough:
            if "voted for" in entry and not entry.startswith("vote "):
                parts = entry.strip().split()
                if len(parts) >= 4 and parts[-2] == "for":
                    voter = parts[0]
                    target = parts[-1]
                    new_votes[voter] = target
    
    return new_votes


def calculate_accuracy(original_votes: Dict[str, str], new_votes: Dict[str, str]) -> Tuple[float, int, int]:
    """Calculate the accuracy of the new votes compared to the original votes."""
    if not original_votes or not new_votes:
        return 0.0, 0, 0
    
    matches = 0
    total = 0
    
    # Get common voters
    common_voters = set(original_votes.keys()) & set(new_votes.keys())
    total = len(common_voters)
    
    if total == 0:
        return 0.0, 0, 0
    
    for voter in common_voters:
        if original_votes[voter] == new_votes[voter]:
            matches += 1
    
    accuracy = matches / total if total > 0 else 0.0
    return accuracy, matches, total


def process_game_file(file_path: str) -> Tuple[Dict[str, str], Dict[str, str], float, int, int]:
    """Process a single game file and return voting results and accuracy."""
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp:
        temp_file = temp.name
    
    try:
        original_game_state = load_game_state(file_path)
        if not original_game_state:
            return {}, {}, 0.0, 0, 0
        
        # Extract original votes
        original_votes = extract_original_votes(original_game_state)
        
        # Reset and get new votes
        new_votes = reset_and_get_new_votes(file_path, temp_file)
        
        # Calculate accuracy
        accuracy, matches, total = calculate_accuracy(original_votes, new_votes)
        
        return original_votes, new_votes, accuracy, matches, total
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)


def main():
    parser = argparse.ArgumentParser(description='Compare original voting results with recalculated votes.')
    
    parser.add_argument('--tournament-dir', 
                      default=os.path.join(project_root, "data", "tournament"),
                      help='Path to the tournament directory containing game files')
    
    parser.add_argument('--output', '-o',
                      default=os.path.join(project_root, "data", "voting_comparison.csv"),
                      help='Path to save the CSV output')
    
    parser.add_argument('--limit', '-l', type=int, default=0,
                      help='Maximum number of games to process (0 for all)')
    
    args = parser.parse_args()
    
    # Find all JSON files in the tournament directory
    for root, _, files in os.walk(args.tournament_dir):
        tournament_files = [os.path.join(root, "llama-3-1-8b-instruct_vs_gemini-pro-1-5_1.json")]
    # for root, _, files in os.walk(args.tournament_dir):
    #     for file in files:
    #         if file.endswith('.json'):
    #             tournament_files.append(os.path.join(root, file))
    
    # Limit the number of files if requested
    if args.limit > 0:
        tournament_files = tournament_files[:args.limit]
    
    print(f"Found {len(tournament_files)} game files to process")
    
    # Process each file and collect results
    results = []
    for i, file_path in enumerate(tournament_files):
        print(f"Processing file {i+1}/{len(tournament_files)}: {os.path.basename(file_path)}")
        
        original_votes, new_votes, accuracy, matches, total = process_game_file(file_path)
        
        if not original_votes and not new_votes:
            print(f"  No votes found in {file_path}")
            continue
        
        results.append({
            'file': os.path.basename(file_path),
            'original_votes': original_votes,
            'new_votes': new_votes,
            'accuracy': accuracy,
            'matches': matches,
            'total': total
        })
        
        print(f"  Accuracy: {accuracy:.2f} ({matches}/{total})")
    
    # Write results to CSV
    with open(args.output, 'w', newline='') as csvfile:
        fieldnames = ['file', 'accuracy', 'matches', 'total', 'original_votes', 'new_votes', 'differences']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        overall_matches = 0
        overall_total = 0
        
        for result in results:
            differences = []
            for voter in set(result['original_votes'].keys()) & set(result['new_votes'].keys()):
                if result['original_votes'][voter] != result['new_votes'][voter]:
                    differences.append(f"{voter}: {result['original_votes'][voter]} -> {result['new_votes'][voter]}")
            
            writer.writerow({
                'file': result['file'],
                'accuracy': f"{result['accuracy']:.2f}",
                'matches': result['matches'],
                'total': result['total'],
                'original_votes': ", ".join([f"{k}:{v}" for k, v in result['original_votes'].items()]),
                'new_votes': ", ".join([f"{k}:{v}" for k, v in result['new_votes'].items()]),
                'differences': "; ".join(differences)
            })
            
            overall_matches += result['matches']
            overall_total += result['total']
        
        # Write summary row
        overall_accuracy = overall_matches / overall_total if overall_total > 0 else 0
        writer.writerow({
            'file': 'OVERALL',
            'accuracy': f"{overall_accuracy:.2f}",
            'matches': overall_matches,
            'total': overall_total,
            'original_votes': '',
            'new_votes': '',
            'differences': ''
        })
    
    print(f"\nResults saved to {args.output}")
    print(f"Overall accuracy: {overall_accuracy:.2f} ({overall_matches}/{overall_total})")


if __name__ == "__main__":
    main()
