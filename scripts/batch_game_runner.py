import os
import json
import shutil
import subprocess
import sys
from pathlib import Path
from among_them.game_config import GameConfig
from among_them.config import STATE_FILE

def get_next_available_file_number(data_dir: Path) -> int:
    """Find the next available file number for game_state_X.json files."""
    existing_files = list(data_dir.glob("game_state_*.json"))
    if not existing_files:
        return 1
    
    numbers = []
    for file in existing_files:
        try:
            # Extract number from filename like "game_state_1.json"
            number = int(file.stem.split("_")[-1])
            numbers.append(number)
        except (ValueError, IndexError):
            continue
    
    return max(numbers) + 1 if numbers else 1

def run_single_game(game_config: GameConfig, state_file_path: str) -> bool:
    """Runs a single game by calling manual_llm_game.py with the given configuration."""
    
    # Create a temporary script that imports and calls the main function with our config
    temp_script_content = f'''# -*- coding: utf-8 -*-
import sys
sys.path.append(".")

from scripts.manual_llm_game import main
from among_them.game_config import GameConfig

# Create the game config
game_config = GameConfig(
    num_tasks={game_config.num_tasks},
    num_players={game_config.num_players},
    num_impostors={game_config.num_impostors},
    map_size={game_config.map_size},
    num_task_phase_actions_per_player={game_config.num_task_phase_actions_per_player},
    num_discuss_phase_actions_per_player={game_config.num_discuss_phase_actions_per_player},
    impostor_cooldown={game_config.impostor_cooldown}
)

# Run the game with the specific state file path
main(game_config, "{state_file_path}")
'''
    
    # Write temporary script with UTF-8 encoding
    temp_script_path = "temp_game_runner.py"
    with open(temp_script_path, 'w', encoding='utf-8') as f:
        f.write(temp_script_content)
    
    try:
        # Run the temporary script with real-time output
        result = subprocess.run([sys.executable, temp_script_path], 
                              text=True, timeout=10800)  # 3 hour timeout
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("Game timed out after 3 hours")
        return False
    except Exception as e:
        print(f"Error running game: {e}")
        return False
    finally:
        # Clean up temporary script
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)

def copy_game_state_to_numbered_file(source_file_path: str, data_dir: Path, file_number: int) -> bool:
    """Copy the current game state to a numbered file."""
    if not os.path.exists(source_file_path):
        print(f"Warning: {source_file_path} does not exist")
        return False
    
    filename = f"game_state_{file_number}.json"
    filepath = data_dir / filename
    
    try:
        shutil.copy2(source_file_path, filepath)
        print(f"Game state copied to: {filepath}")
        return True
    except Exception as e:
        print(f"Error copying game state: {e}")
        return False

def main():
    """Run multiple games with different player configurations."""
    
    # Configuration parameters
    GAMES_PER_PLAYER_COUNT = 5
    PLAYER_COUNTS = [7, 6, 5]
    
    # Paths
    data_dir = Path("src/among_them/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the next available file number
    next_file_number = get_next_available_file_number(data_dir)
    
    total_games = len(PLAYER_COUNTS) * GAMES_PER_PLAYER_COUNT
    current_game = 16
    
    print(f"Starting batch run: {total_games} games total")
    print(f"Player counts: {PLAYER_COUNTS}")
    print(f"Games per player count: {GAMES_PER_PLAYER_COUNT}")
    print(f"Starting file number: {next_file_number}")
    print("-" * 50)
    
    for player_count in PLAYER_COUNTS:
        for game_num in range(GAMES_PER_PLAYER_COUNT):
            current_game += 1
            print(f"\n{'='*60}")
            print(f"GAME {current_game}/{total_games}")
            print(f"Player count: {player_count}, Game {game_num + 1}/{GAMES_PER_PLAYER_COUNT}")
            print(f"{'='*60}")
            
            # Create game configuration
            game_config = GameConfig(
                num_tasks=5,
                num_players=player_count,
                num_impostors=2 if player_count >= 6 else 1,
                map_size=1,
                num_task_phase_actions_per_player=10,
                num_discuss_phase_actions_per_player=2,
                impostor_cooldown=1
            )
            
            try:
                # Create a unique state file path for this game
                state_file_path = f"src/among_them/data/game_state_temp_{current_game}.json"
                
                # Run the game by calling manual_llm_game.py
                success = run_single_game(game_config, state_file_path)
                
                # Check if the state file exists and has content
                if os.path.exists(state_file_path):
                    file_size = os.path.getsize(state_file_path)
                    print(f"State file exists: {state_file_path} (size: {file_size} bytes)")
                else:
                    print(f"WARNING: State file does not exist: {state_file_path}")
                
                if success:
                    # Copy the game state to numbered file
                    if copy_game_state_to_numbered_file(state_file_path, data_dir, next_file_number):
                        next_file_number += 1
                        print(f"Game {current_game} completed successfully!")
                    else:
                        print(f"Game {current_game} completed but failed to save state")
                else:
                    print(f"Game {current_game} failed to complete")
                    # Even if game failed, save the partial state
                    if os.path.exists(state_file_path):
                        failed_file_path = f"src/among_them/data/game_state_failed_{current_game}.json"
                        shutil.copy2(state_file_path, failed_file_path)
                        print(f"Partial game state saved to: {failed_file_path}")
                
                # DON'T DELETE THE FILES - keep them for debugging
                print(f"Game state file preserved at: {state_file_path}")
                
            except Exception as e:
                print(f"Error in game {current_game}: {e}")
                print("Continuing with next game...")
                continue
    
    print(f"\n{'='*60}")
    print(f"BATCH RUN COMPLETED!")
    print(f"Total games attempted: {total_games}")
    print(f"Files saved to: {data_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 