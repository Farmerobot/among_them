#!/usr/bin/env python3
"""
Script to add a 'who_won' column to the combined_annotations.csv file based on
game outcome.
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

def main():
    # Initialize the game engine
    game_engine = GameEngine()
    
    # Define input and output file paths
    input_csv = project_root / "data" / "combined_annotations.csv"
    output_csv = project_root / "data" / "combined_annotations_v2.csv"
    
    print(f"Processing {input_csv}...")
    
    # Load the CSV file
    df = pd.read_csv(input_csv)
    
    # Add a new column for game winner
    df['who_won'] = None
    
    # Track processed files to avoid redundant loading
    processed_files = {}
    
    # Process each row
    for idx, row in df.iterrows():
        source_file = row['source_file']
        
        # Only process each unique source file once
        if source_file not in processed_files:
            # Construct the full path to the file
            file_path = project_root / "data" / "tournament" / source_file
            
            # Load the game state
            if game_engine.load_state(str(file_path)):
                # Check if impostors won
                impostors_win = game_engine.check_impostors_win()
                processed_files[source_file] = "impostors" if impostors_win else "crewmates"
                print(f"File {source_file}: {'Impostors' if impostors_win else 'Crewmates'} won")
            else:
                print(f"Warning: Could not load state from {file_path}")
                processed_files[source_file] = "unknown"
        
        # Update the who_won column for this row
        df.at[idx, 'who_won'] = processed_files[source_file]
    
    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")
    print(f"Winner stats: {df['who_won'].value_counts().to_dict()}")

if __name__ == "__main__":
    main()
