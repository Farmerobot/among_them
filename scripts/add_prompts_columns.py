#!/usr/bin/env python3
"""
Script to add plan_prompt, plan_response, discussion_prompt, discussion_response
columns to the combined_annotations_v2.csv file.

This script analyzes player history to find matches between CSV responses and
discussion responses, then adds the associated prompts and responses.
"""

import os
import sys
import pandas as pd
import json
from pathlib import Path

# Add the src directory to the Python path so we can import the game engine
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

from among_them.game.game_engine import GameEngine
from among_them.game.models.engine import GamePhase

def extract_text_without_name(text):
    """Extract text without the name prefix like [Name]: """
    if text.startswith("[") and "]: " in text:
        return text.split("]: ", 1)[1].strip()
    return text.strip()

def main():
    # Initialize the game engine
    game_engine = GameEngine()
    
    # Define input and output file paths
    input_csv = project_root / "data" / "combined_annotations_v2.csv"
    output_csv = project_root / "data" / "combined_annotations_v3.csv"
    
    print(f"Processing {input_csv}...")
    
    # Load the CSV file
    df = pd.read_csv(input_csv)
    
    # Add new columns for prompts and responses
    df['plan_prompt'] = None
    df['plan_response'] = None
    df['discussion_prompt'] = None
    df['discussion_response'] = None
    
    # Track processed files to avoid redundant loading
    processed_files = {}
    
    # Process each row
    for idx, row in df.iterrows():
        source_file = row['source_file']
        speaker_name = row['speaker']
        csv_text = row['text']
        csv_text_clean = extract_text_without_name(csv_text)
        
        # Only process each unique source file once
        if source_file not in processed_files:
            # Construct the full path to the file
            file_path = project_root / "data" / "tournament" / source_file
            print(f"Processing {file_path}...")
            
            # Load the game state
            if game_engine.load_state(str(file_path)):
                # Store all players' history for this file
                processed_files[source_file] = {}
                
                # For each player, analyze their history
                for player in game_engine.state.players:
                    player_history = []
                    
                    # Look for discussion phases in player history
                    for round_idx, round_data in enumerate(player.history.rounds):
                        if round_data.stage == GamePhase.DISCUSS:
                            # Extract information for this discussion round
                            round_info = {
                                'player_name': player.name,
                                'round_idx': round_idx,
                                'prompts': round_data.prompts.copy() if round_data.prompts else [],
                                'llm_responses': round_data.llm_responses.copy() if round_data.llm_responses else [],
                                'clean_responses': [extract_text_without_name(resp) for resp in round_data.llm_responses] 
                                                  if round_data.llm_responses else []
                            }
                            player_history.append(round_info)
                    
                    processed_files[source_file][player.name] = player_history
            else:
                print(f"Warning: Could not load state from {file_path}")
                processed_files[source_file] = {}
        
        # Get player history from processed files
        player_history = processed_files.get(source_file, {}).get(speaker_name, [])
        
        # Search for matching response in player history
        for round_info in player_history:
            for i, clean_response in enumerate(round_info['clean_responses']):
                if clean_response == csv_text_clean and len(round_info['llm_responses']) >= 2 and len(round_info['prompts']) >= 2:
                    # We found a match! The last two responses are plan and discussion
                    if len(round_info['llm_responses']) >= 2:
                        # Get the last two responses (plan and discussion)
                        df.at[idx, 'plan_response'] = round_info['llm_responses'][-2] if len(round_info['llm_responses']) > 1 else ""
                        df.at[idx, 'discussion_response'] = round_info['llm_responses'][-1]
                    
                    # Get the last two prompts (plan and discussion)
                    if len(round_info['prompts']) >= 2:
                        df.at[idx, 'plan_prompt'] = round_info['prompts'][-2] if len(round_info['prompts']) > 1 else ""
                        df.at[idx, 'discussion_prompt'] = round_info['prompts'][-1]
                    
                    # Found the match, no need to continue searching
                    break
            
            # If we've already found a match, no need to check other rounds
            if df.at[idx, 'discussion_response'] is not None:
                break
    
    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")
    
    # Print some statistics
    plan_prompt_count = df['plan_prompt'].notna().sum()
    discussion_prompt_count = df['discussion_prompt'].notna().sum()
    print(f"Added plan prompts: {plan_prompt_count}/{len(df)}")
    print(f"Added discussion prompts: {discussion_prompt_count}/{len(df)}")

if __name__ == "__main__":
    main()
