#!/usr/bin/env python3
"""
Script to visualize the largest LLM CoT (Chain of Thought) by token count across all games.
"""

import json
import os
from pathlib import Path
import tiktoken
import matplotlib.pyplot as plt
import numpy as np


def count_tokens(text, model="gpt-4o"):
    """Count tokens in text using tiktoken."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def find_all_cot_token_counts(json_file_path):
    """Find all CoT token counts in a single game file."""
    try:
        with open(json_file_path, 'r') as f:
            game_data = json.load(f)
        
        # Game data structure: [history_list, player_list]
        history_list = game_data[0] if isinstance(game_data, list) and len(game_data) > 0 else []
        
        token_counts = []
        max_tokens = 0
        max_cot = ""
        max_cot_info = {}
        
        for i, history_item in enumerate(history_list):
            if isinstance(history_item, dict) and 'llm_cot' in history_item:
                cot = history_item.get('llm_cot', "")
                if cot and isinstance(cot, str):
                    token_count = count_tokens(cot)
                    token_counts.append(token_count)
                    if token_count > max_tokens:
                        max_tokens = token_count
                        max_cot = cot
                        max_cot_info = {
                            'file': json_file_path.name,
                            'history_index': i,
                            'token_count': token_count,
                            'player_name': history_item.get('action_taken', {}).get('player_name', 'Unknown'),
                            'action_type': history_item.get('action_taken', {}).get('type', {}).get('__enum__', 'Unknown')
                        }
        
        return token_counts, max_cot, max_tokens, max_cot_info
    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
        return [], "", 0, {}


def analyze_all_games(data_folder="data"):
    """Analyze all games in the data folder and find all CoT token counts."""
    data_path = Path(data_folder)
    json_files = list(data_path.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {data_folder}")
        return [], "", {}
    
    print(f"Found {len(json_files)} JSON files to analyze")
    
    # Store all CoT token counts for visualization
    all_cot_token_counts = []
    cot_token_counts_per_game = []
    largest_cot_overall = ""
    max_tokens_overall = 0
    max_cot_info_overall = {}
    
    # Process each file
    for json_file in json_files:
        token_counts, cot, token_count, cot_info = find_all_cot_token_counts(json_file)
        if token_counts:
            all_cot_token_counts.extend(token_counts)
            cot_token_counts_per_game.append(token_counts)
            if token_count > max_tokens_overall:
                max_tokens_overall = token_count
                largest_cot_overall = cot
                max_cot_info_overall = cot_info
    
    # Print summary
    print(f"\nAnalyzed {len(json_files)} games")
    print(f"Found {len(all_cot_token_counts)} CoTs with token counts")
    if all_cot_token_counts:
        print(f"Average CoT token count: {np.mean(all_cot_token_counts):.2f}")
        print(f"Median CoT token count: {np.median(all_cot_token_counts):.2f}")
        print(f"Max CoT token count: {max(all_cot_token_counts)}")
        print(f"Min CoT token count: {min(all_cot_token_counts)}")
        
        # Calculate frequency distribution
        unique_counts, frequencies = np.unique(all_cot_token_counts, return_counts=True)
        print(f"\nToken count frequency distribution:")
        for count, freq in sorted(zip(unique_counts, frequencies), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  Token count {count}: {freq} occurrences")
    
    if max_tokens_overall > 0:
        print(f"\nLargest CoT found in {max_cot_info_overall['file']} at history index {max_cot_info_overall['history_index']}")
        print(f"Player: {max_cot_info_overall['player_name']}")
        print(f"Action: {max_cot_info_overall['action_type']}")
        print(f"Token count: {max_cot_info_overall['token_count']}")
        print(f"\nLargest CoT content:\n{largest_cot_overall}")
    
    return all_cot_token_counts, largest_cot_overall, max_cot_info_overall


def visualize_cot_distribution(cot_token_counts):
    """Create a visualization of the CoT token count distribution."""
    if not cot_token_counts:
        print("No data to visualize")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(cot_token_counts, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Token Count')
    plt.ylabel('Frequency')
    plt.title('Distribution of CoT Token Counts')
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(1, 2, 2)
    plt.boxplot(cot_token_counts, vert=False)
    plt.xlabel('Token Count')
    plt.title('CoT Token Count Distribution (Box Plot)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('generated/cot_token_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nVisualization saved to generated/cot_token_distribution.png")


def main():
    """Main function to run the analysis."""
    # Create generated directory if it doesn't exist
    Path("generated").mkdir(exist_ok=True)
    
    # Analyze all games
    all_cot_token_counts, largest_cot, max_cot_info = analyze_all_games()
    
    # Visualize the distribution
    visualize_cot_distribution(all_cot_token_counts)
    
    # Save the largest CoT to a file
    if largest_cot:
        with open('generated/largest_cot.txt', 'w') as f:
            f.write(f"File: {max_cot_info.get('file', 'Unknown')}\n")
            f.write(f"Player: {max_cot_info.get('player_name', 'Unknown')}\n")
            f.write(f"Action: {max_cot_info.get('action_type', 'Unknown')}\n")
            f.write(f"Token count: {max_cot_info.get('token_count', 0)}\n")
            f.write(f"History index: {max_cot_info.get('history_index', 'Unknown')}\n\n")
            f.write("CoT content:\n")
            f.write(largest_cot)
        
        print("\nLargest CoT saved to generated/largest_cot.txt")


if __name__ == "__main__":
    main()
