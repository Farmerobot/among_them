#!/usr/bin/env python3
"""
Game Replay Script

Replays legacy Among Them JSON game files using the current game engine workflow.
Extracts actions and LLM outputs from legacy files and replays them through the engine
using the proper get_turn_context() -> step() workflow.
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Import Among Them modules
from among_them.game_engine import GameEngine
from among_them.game_config import GameConfig
from among_them.game_jsonencoder import game_object_hook, GameJSONEncoder
from among_them.models.action import Action
from among_them.models.action_type import ActionType


class GameReplay:
    """Replay legacy games using the actual game engine workflow."""
    
    def __init__(self):
        self.replay_stats = {
            "files_processed": 0,
            "successful_replays": 0,
            "failed_replays": 0,
            "actions_replayed": 0,
            "errors": []
        }
        
    def replay_game_files(self, data_folder: str, output_folder: str, dry_run: bool = False, 
                         no_backup: bool = False) -> None:
        """Replay all JSON files in the data folder."""
        
        data_path = Path(data_folder)
        output_path = Path(output_folder)
        
        if not data_path.exists():
            print(f"❌ Data folder does not exist: {data_folder}")
            return
            
        # Create output folder if it doesn't exist
        if not dry_run:
            output_path.mkdir(parents=True, exist_ok=True)
            
        # Find all JSON files
        json_files = list(data_path.glob("*.json"))
        if not json_files:
            print(f"❌ No JSON files found in {data_folder}")
            return
            
        print(f"🚀 Among Them Game Replay Script")
        print(f"======================================================================")
        print(f"Input folder: {data_folder}")
        print(f"Output folder: {output_folder}")
        print(f"🎮 Found {len(json_files)} JSON files to replay")
        print(f"🔍 {'DRY RUN MODE' if dry_run else 'LIVE MODE'}")
        print(f"======================================================================")
        
        for json_file in sorted(json_files):
            self.replay_stats["files_processed"] += 1
            
            print(f"\n📄 Replaying: {json_file.name}")
            print(f"--------------------------------------------------")
            
            try:
                # Extract action sequence from legacy game
                action_sequence = self._extract_action_sequence(json_file)
                if not action_sequence:
                    continue
                    
                # Replay using proper engine workflow
                replayed_data = self._replay_with_engine_workflow(action_sequence)
                if not replayed_data:
                    continue
                    
                # Save the replayed game
                output_file = output_path / json_file.name
                if not dry_run:
                    success = self._save_replayed_game(replayed_data, output_file, no_backup)
                    if success:
                        print(f"✅ Replayed and saved: {json_file.name}")
                        self.replay_stats["successful_replays"] += 1
                    else:
                        print(f"❌ Failed to save: {json_file.name}")
                        self.replay_stats["failed_replays"] += 1
                else:
                    print(f"✅ Would replay: {json_file.name} (dry run)")
                    self.replay_stats["successful_replays"] += 1
                    
            except Exception as e:
                error_msg = f"Error replaying {json_file.name}: {str(e)}"
                print(f"❌ {error_msg}")
                self.replay_stats["errors"].append(error_msg)
                self.replay_stats["failed_replays"] += 1
                
        # Print summary
        self._print_summary(dry_run)
        
    def _extract_action_sequence(self, json_file: Path) -> Optional[Dict]:
        """Extract action sequence and game setup from legacy file."""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f, object_hook=game_object_hook)
                
            # Extract components based on legacy format
            if isinstance(data, list) and len(data) >= 2:
                history = data[0]
                players = data[1]
                game_config = data[2] if len(data) > 2 else GameConfig()
                
                # Extract non-system actions with their LLM outputs
                actions = []
                for entry in history:
                    # Skip system messages - engine will generate these
                    # After game_object_hook, entry is a History object, not dict
                    if entry.action_taken.player_name == "System":
                        continue
                        
                    actions.append({
                        "action": entry.action_taken,
                        "llm_cot": entry.llm_cot,
                        "llm_response": entry.llm_response,
                        "token_usage": entry.token_usage,
                        "votes_before": entry.votes_before_this_discussion_message
                    })
                
                # Extract original task assignments from first history entry
                original_tasks = None
                if history:
                    first_entry = history[0]
                    original_tasks = getattr(first_entry, 'tasks_left_to_do', None)
                
                print(f"   🎯 Extracted {len(actions)} actions from {len(history)} history entries")
                if original_tasks:
                    print(f"   📋 Extracted original task assignments for {len(original_tasks)} players")
                
                return {
                    "players": players,
                    "game_config": game_config,
                    "actions": actions,
                    "original_tasks": original_tasks
                }
            else:
                print(f"   ❌ Invalid legacy format: expected list with 2-3 elements")
                return None
                
        except Exception as e:
            print(f"   ❌ Failed to extract actions: {str(e)}")
            return None
            
    def _replay_with_engine_workflow(self, action_sequence: Dict) -> Optional[List]:
        """Replay the game using the actual GameEngine workflow: get_turn_context() -> step()."""
        try:
            players = action_sequence["players"]
            game_config = action_sequence["game_config"]
            actions = action_sequence["actions"]
            
            if not actions:
                print(f"   ❌ No actions to replay")
                return None
                
            print(f"   🎮 Starting replay with {len(actions)} actions...")
                
            # Initialize GameEngine following the manual_llm_game.py pattern
            engine = GameEngine(game_config)
            engine.players = players.copy()
            
            # Extract original task assignments from legacy game
            original_tasks = action_sequence.get("original_tasks")
            
            # After engine initialization, replace tasks with original ones
            if engine.history and original_tasks:
                engine.history[0].tasks_left_to_do = original_tasks
                print(f"   🎯 Replaced auto-generated tasks with original task assignments")
            
            # Replay each action using the correct workflow
            for i, action_data in enumerate(actions):
                try:
                    # Step 1: Get turn context from engine (like manual_llm_game.py)
                    turn_context, actions_player_can_take, system_prompt, user_prompt, pre_discussion_vote_prompts = engine.get_turn_context()
                    
                    if not turn_context:
                        print(f"   🏁 Game over at action {i}")
                        break
                        
                    # Step 2: Use official parse_llm_response_to_action to select the correct action
                    from among_them.utils.llm_utils import parse_llm_response_to_action
                    
                    legacy_action = action_data["action"]
                    llm_response = action_data["llm_response"]
                    player_name = legacy_action.player_name
                    
                    try:
                        action_idx, parsed_response = parse_llm_response_to_action(
                            actions_player_can_take, llm_response, player_name
                        )
                        action_taken = actions_player_can_take[action_idx]
                        print(f"     🎯 Selected action {action_idx}: {action_taken.type.name} by {player_name}")
                    except Exception as e:
                        print(f"   ⚠️  Could not parse LLM response for {player_name}: {str(e)}")
                        break
                        
                    # Step 3: Apply through engine using correct workflow
                    llm_response = action_data["llm_response"]
                    llm_cot = action_data["llm_cot"] 
                    token_usage = action_data["token_usage"]
                    pre_discussion_votes = action_data["votes_before"]
                    
                    print(f"     🎯 Replaying {action_taken.type.name} by {action_taken.player_name}")
                    
                    # Debug task removal issue if it's a TASK action
                    if action_taken.type.name == "TASK":
                        player_tasks = engine.history[-1].tasks_left_to_do.get(action_taken.player_name, [])
                        target_task = action_taken.target_task
                        print(f"       📋 Attempting to complete task: {target_task.name} at {target_task.location}")
                        print(f"       📋 Player has {len(player_tasks)} tasks remaining")
                        
                        # Check if target task is in the list
                        found_match = False
                        for i, task in enumerate(player_tasks):
                            if task == target_task:
                                found_match = True
                                print(f"       ✅ Found matching task at index {i}: {task.name} at {task.location}")
                                break
                            else:
                                print(f"       ❌ Task {i} mismatch: '{task.name}'@{task.location} vs '{target_task.name}'@{target_task.location}")
                        
                        if not found_match:
                            print(f"       ⚠️ WARNING: Target task not found in player's task list!")
                    
                    # Step 4: Call engine.step() with proper parameters (like manual_llm_game.py)
                    game_over, end_reason = engine.step(
                        turn_context=turn_context,
                        action_taken=action_taken,
                        llm_response=llm_response,
                        llm_cot=llm_cot,
                        token_usage=token_usage,
                        pre_discussion_votes=pre_discussion_votes
                    )
                    
                    self.replay_stats["actions_replayed"] += 1
                    
                    if game_over:
                        print(f"   🏁 Game ended: {end_reason}")
                        break
                        
                except Exception as e:
                    print(f"   ❌ Failed to replay action {i}: {str(e)}")
                    continue
                    
            print(f"   ✅ Replayed {self.replay_stats['actions_replayed']} actions, generated {len(engine.history)} total entries")
            
            # Return in the standard format
            return [engine.history, engine.players, engine.game_config]
            
        except Exception as e:
            print(f"   ❌ Replay failed: {str(e)}")
            return None
            

    def _save_replayed_game(self, replayed_data: List, output_file: Path, no_backup: bool) -> bool:
        """Save the replayed game data to output file."""
        try:
            # Create backup if file exists and backup not disabled
            if output_file.exists() and not no_backup:
                backup_name = f"{output_file.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                backup_path = output_file.parent / backup_name
                shutil.copy2(output_file, backup_path)
                print(f"   💾 Created backup: {backup_name}")
                
            # Save replayed data
            with open(output_file, 'w') as f:
                json.dump(replayed_data, f, cls=GameJSONEncoder, indent=2)
                
            return True
            
        except Exception as e:
            print(f"   ❌ Save failed: {str(e)}")
            return False
            
    def _print_summary(self, dry_run: bool) -> None:
        """Print summary of replay results."""
        print(f"\n{'='*80}")
        print(f"📊 REPLAY SUMMARY")
        print(f"{'='*80}")
        print(f"📁 Files processed: {self.replay_stats['files_processed']}")
        print(f"✅ Successful replays: {self.replay_stats['successful_replays']}")
        print(f"❌ Failed replays: {self.replay_stats['failed_replays']}")
        print(f"🎯 Total actions replayed: {self.replay_stats['actions_replayed']}")
        
        if self.replay_stats["errors"]:
            print(f"\n❌ ERRORS ({len(self.replay_stats['errors'])}):")
            for error in self.replay_stats["errors"]:
                print(f"   • {error}")
                
        if dry_run:
            print(f"\n🔍 This was a DRY RUN - no files were actually modified")


def main():
    parser = argparse.ArgumentParser(description="Replay legacy Among Them games using current engine")
    parser.add_argument("--data-folder", default="data", help="Input folder containing legacy JSON files")
    parser.add_argument("--output-folder", default="data_replayed", help="Output folder for replayed files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backup files")
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    data_folder = os.path.abspath(args.data_folder)
    output_folder = os.path.abspath(args.output_folder)
    
    replay = GameReplay()
    replay.replay_game_files(data_folder, output_folder, args.dry_run, args.no_backup)


if __name__ == "__main__":
    main()
