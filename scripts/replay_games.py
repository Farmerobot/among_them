#!/usr/bin/env python3
"""
Game Replay Script

Replays Among Them JSON game files using the current game engine workflow,
but using recorded LLM responses instead of calling the LLM again.
"""

import json
import os
from pathlib import Path

from among_them.config import STATE_FILE
from among_them.game_engine import GameEngine
from among_them.models.action import Action
from among_them.models.action_type import ActionType
from among_them.models.history import History
from among_them.utils.end_utils import get_end_game_reason
from among_them.utils.llm_utils import parse_llm_response_to_action
from among_them.game_config import GameConfig


def replay_game(json_file: Path, output_folder: Path):
    """Replay a single game from JSON file using only GameEngine.load_state for consistency."""
    # print(f"\n--- Replaying game from {json_file.name} ---")
    
    # Create a unique state file for this replay
    state_file = str(output_folder / f"{json_file.stem}.json")
    
    # Copy the JSON file to the state file location
    try:
        import shutil
        shutil.copyfile(json_file, state_file)
    except Exception as e:
        print(f"Error copying game file to state file: {e}")
        return 0
    
    # Initialize game engine with the state file (same as manual_llm_game.py)
    engine = GameEngine(file_path=state_file)
    
    # Load state using ONLY the engine's built-in method (no manual JSON loading)
    try:
        engine.load_state()
        # print(f"Game loaded from state file with {len(engine.history)} history entries")
        # Get the history data from the loaded engine (this is the only source of data)
        history_data = engine.history
        
        # Apply backward compatibility fixes for older game data
        for history_item in history_data:
            # Ensure action_taken has all required attributes
            action_taken = getattr(history_item, 'action_taken', None)
            if action_taken:
                # Add missing attributes if they don't exist
                if not hasattr(action_taken, 'observer_perspective'):
                    action_taken.observer_perspective = ""
                if not hasattr(action_taken, 'agent_perspective'):
                    action_taken.agent_perspective = ""
                if not hasattr(action_taken, 'global_perspective'):
                    action_taken.global_perspective = ""
    except Exception as e:
        print(f"Error loading state: {e}")
        return 0
    
    # Get the game configuration from the loaded game
    game_config = engine.game_config
    
    # Create a new game engine with the same configuration but without loading state
    replay_engine = GameEngine(game_config=game_config, file_path=state_file)
    replay_engine.players = engine.players
    # replay_engine.game_config.impostor_cooldown = 0
    
    # Reconstruct tasks_left_to_do for replay_engine's initial history by iterating through original history
    # Start with the tasks from the first history item of original engine
    if history_data and hasattr(history_data[0], 'tasks_left_to_do'):
        tasks_dict = {player: tasks[:] for player, tasks in history_data[0].tasks_left_to_do.items()}
        
        # Iterate through history to remove completed tasks
        for history_item in history_data[1:]:  # Skip the first (initialization) item
            action_taken = getattr(history_item, 'action_taken', None)
            if action_taken and action_taken.type.name == 'TASK':
                player_name = action_taken.player_name
                target_task = action_taken.target_task
                if player_name in tasks_dict and target_task not in tasks_dict[player_name]:
                    tasks_dict[player_name].append(target_task)
        
        # Assign this dict to first history item of replay engine
        if replay_engine.history and hasattr(replay_engine.history[0], 'tasks_left_to_do'):
            replay_engine.history[0].tasks_left_to_do = tasks_dict

    # print(replay_engine.history[0])
    
    # Process each history entry (skip the first which is initialization)
    action_count = 0
    for i, history_item in enumerate(history_data[1:], 1):
        if history_item.action_taken.player_name == "System":
            action_count += 1
            continue
        
        # print(f"Processing action {i}/{len(history_data)-1}")
        
        # Get current turn context (same as manual_llm_game.py)
        turn_context_result = replay_engine.get_turn_context(history_item.action_taken.player_name)
        
        # If no turn context, game is over
        if not turn_context_result or turn_context_result[0] is None:
            print("Game over - no turn context")
            break
            
        turn_context_history, actions_player_can_take, system_prompt, user_prompt, pre_discussion_vote_prompts = turn_context_result
        
        # Ensure available tasks match between replay and original game
        # The tasks are stored in the history item's tasks_left_to_do attribute
        if hasattr(history_item, 'tasks_left_to_do'):
            # Verify that the tasks in the current context match those in the original game
            # This ensures task consistency between the original game and the replay
            current_tasks = turn_context_history.tasks_left_to_do
            original_tasks = history_item.tasks_left_to_do
            
            # Print a warning if tasks don't match (this shouldn't happen with proper state loading)
            # if current_tasks != original_tasks:
            #     print(f"Warning: Task mismatch at action {i}")
            #     print(f"Current tasks: {current_tasks}")
            #     print(f"Original tasks: {original_tasks}")
        
        # Ensure available actions match between replay and original game
        # This ensures action consistency between the original game and the replay
        original_actions = history_item.actions_agent_could_take
        original_texts = sorted([a for a in original_actions])
        can_take_texts = sorted([a.set_stories().command_perspective for a in actions_player_can_take])
        # if can_take_texts != original_texts and without_wait_texts != original_texts:
        #     if "wait" in original_texts:
        #         print(f"Warning: Action mismatch at action {i}")
        #         print(f"Current actions: {can_take_texts}")
        #         print(f"Original actions: {original_texts}")
        #     else:
        #         print(f"Warning: Action mismatch at action {i}")
        #         print(f"Current actions: {without_wait_texts}")
        #         print(f"Original actions: {original_texts}")
        if "wait" not in original_texts:
            turn_context_history.actions_agent_could_take = [a.set_stories().command_perspective for a in actions_player_can_take if a.set_stories().command_perspective != "wait"]
            # print(turn_context_history.actions_agent_could_take)
        elif "wait" in original_texts and "wait" not in can_take_texts:
            actions_player_can_take.append(Action(type=ActionType.WAIT, player_name=history_item.action_taken.player_name))
            turn_context_history.actions_agent_could_take = [a.set_stories().command_perspective for a in actions_player_can_take]
        
        # For pre-discussion votes, we use the recorded data directly (as requested)
        # These are already dicts and don't need to be gathered again
        pre_discussion_votes = history_item.votes_before_this_discussion_message
        
        # Extract LLM data from history
        llm_response = history_item.llm_response
        if not getattr(history_item, "llm_cot", None) and json_file.name == "test_game.json":
            history_item.llm_cot = "cot"
        llm_cot = history_item.llm_cot
        token_usage = history_item.token_usage
            
        # Parse the action using the same function as manual_llm_game.py
        # current_player_name = turn_context_history.action_taken.player_name
        # action_idx, response_text = parse_llm_response_to_action(
        #     actions_player_can_take, llm_response, current_player_name
        # )
        if actions_player_can_take[0].type == ActionType.SPEAK and history_item.action_taken.type == ActionType.SPEAK and getattr(history_item.action_taken, "target_message", None):
            actions_player_can_take[0].target_message = history_item.action_taken.target_message
        try:
            action_taken = [a for a in actions_player_can_take if a == history_item.action_taken][0]
        except Exception as e:
            print(e)
            print(history_item.action_taken)
            print(actions_player_can_take)

        # print(f"Action parsed: {action_taken.type.name} by {current_player_name}")
            
        # Step the environment with recorded LLM data (same as manual_llm_game.py)
        try:
            old_history_length = len(replay_engine.history)
            game_over, end_reason = replay_engine.step(
                turn_context_history, 
                action_taken, 
                llm_response, 
                llm_cot, 
                token_usage, 
                pre_discussion_votes
            )
            action_count += 1
            while old_history_length < len(replay_engine.history):
                # print(replay_engine.history[old_history_length])
                old_history_length += 1
            
            if game_over:
                print(f"Game over! Reason: {end_reason} {json_file}")
                # old_reason = get_end_game_reason_old(engine.history, engine.players)
                # print(f"Old reason: {old_reason}")
                # assert end_reason == old_reason
                break
                
        except Exception as e:
            print(f"Error during step: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # print(f"Replayed {action_count} actions from {json_file.name}")
    return action_count


def replay_all_games(data_folder: str, output_folder: str):
    """Replay all JSON files in the data folder."""
    data_path = Path(data_folder)
    output_path = Path(output_folder)
    
    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all JSON files in data folder
    json_files = list(data_path.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {data_folder}")
        return
        
    print(f"Found {len(json_files)} JSON files to replay")
    
    total_replayed = 0
    successful_replays = 0
    for json_file in json_files:
        try:
            action_count = replay_game(json_file, output_path)
            total_replayed += action_count
            if action_count > 0:
                successful_replays += 1
            # print(f"Completed replay of {json_file.name}")
        except Exception as e:
            print(f"Failed to replay {json_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nSuccessfully replayed {successful_replays}/{len(json_files)} games")
    print(f"Total actions replayed: {total_replayed}")


def main():
    """Main function to replay all games."""
    data_folder = "data"  # Default data folder
    output_folder = "replay_output"  # Default output folder
    
    # Allow overriding with environment variables
    data_folder = os.environ.get("DATA_FOLDER", data_folder)
    output_folder = os.environ.get("OUTPUT_FOLDER", output_folder)
    
    print(f"Replaying games from {data_folder} to {output_folder}")
    replay_all_games(data_folder, output_folder)


if __name__ == "__main__":
    main()
