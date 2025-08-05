#!/usr/bin/env python3
"""
Integrity checker script for Among Them game JSON files.

This script validates all JSON files in the data folder by checking:
1. Whether games load correctly (detecting legacy classes and enums)
2. Game structure validation (system messages, votes_before field, correct phase actions)

Only game_state.json should pass all integrity checks.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from among_them.game_engine import GameEngine
from among_them.game_jsonencoder import game_object_hook
from among_them.game_config import GameConfig
from among_them.models.history import History
from among_them.models.action_type import ActionType
from among_them.models.phase import GamePhase
from among_them.models.player_role import PlayerRole


class IntegrityChecker:
    """Validates game JSON files for loading and structural integrity."""
    
    def __init__(self, data_folder: str = "data"):
        self.data_folder = Path(data_folder)
        self.passed_files = []
        self.failed_files = []
        self.debug_mode = True
    
    def check_all_files(self) -> None:
        """Check all JSON files in the data folder."""
        json_files = list(self.data_folder.glob("*.json"))
        # Exclude alpaca folder files and other non-game files
        game_files = [f for f in json_files if not str(f).startswith(str(self.data_folder / "alpaca"))]
        
        print(f"🔍 Found {len(game_files)} JSON files to check in {self.data_folder}")
        print("=" * 70)
        
        for json_file in sorted(game_files):
            self._check_single_file(json_file)
        
        self._print_summary()
    
    def _check_single_file(self, file_path: Path) -> None:
        """Check a single JSON file for integrity."""
        file_name = file_path.name
        print(f"\n📄 Checking: {file_name}")
        print("-" * 50)
        
        # Step 1: Try to load the file
        load_success, load_errors = self._check_loading(file_path)
        
        if not load_success:
            print(f"❌ {file_name}: FAILED loading check")
            for error in load_errors:
                print(f"   ⚠️  {error}")
            self.failed_files.append((file_name, load_errors))
            return
        
        # Step 2: Try to structure validate
        structure_success, structure_errors = self._check_structure(file_path)
        
        if structure_success:
            print(f"✅ {file_name}: PASSED all integrity checks")
            self.passed_files.append(file_name)
        else:
            print(f"❌ {file_name}: FAILED structure validation")
            # Group and count similar errors instead of printing each one
            error_summary = self._summarize_errors(structure_errors)
            for error_type, count, sample in error_summary:
                if count == 1:
                    print(f"   ⚠️  {sample}")
                else:
                    print(f"   ⚠️  {error_type}: {count} instances (e.g., {sample})")
            self.failed_files.append((file_name, ["Loading passed"] + structure_errors))
    
    def _summarize_errors(self, errors: List[str]) -> List[Tuple[str, int, str]]:
        """Group similar errors and return (error_type, count, sample_error) tuples."""
        error_groups = {}
        
        for error in errors:
            # Group similar error types
            if "phase as dict instead of GamePhase enum" in error:
                key = "Phase as dict errors"
            elif "phase as wrong enum type" in error:
                key = "Phase as wrong enum type errors"
            elif "actions_until_phase_ends is not int" in error:
                key = "Action count type errors"
            elif "missing" in error.lower():
                key = "Missing field errors"
            elif "actions should count down" in error:
                key = "Action countdown errors"
            elif "last action in phase should have 0" in error:
                key = "Final action count errors"
            else:
                key = "Other errors"
            
            if key not in error_groups:
                error_groups[key] = []
            error_groups[key].append(error)
        
        # Convert to summary format
        summary = []
        for error_type, error_list in error_groups.items():
            summary.append((error_type, len(error_list), error_list[0]))
        
        return summary
    
    def _check_loading(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Check if the game file loads correctly."""
        errors = []
        
        try:
            # Try loading with game_object_hook (same as manual_llm_game.py)
            with open(file_path, 'r') as f:
                json_str = f.read()
                loaded_data = json.loads(json_str, object_hook=game_object_hook)
            
            # Validate basic structure
            if not isinstance(loaded_data, (list, tuple)) or len(loaded_data) < 2:
                errors.append("Invalid save file format - expected [history, players, config?]")
                return False, errors
            
            if len(loaded_data) == 2:
                # Legacy format without GameConfig - this could be acceptable
                history, players = loaded_data
                game_config = GameConfig()  # Use defaults
            elif len(loaded_data) == 3:
                history, players, game_config = loaded_data
            else:
                errors.append(f"Unexpected data structure length: {len(loaded_data)}")
                return False, errors
            
            # Validate basic types
            if not isinstance(history, list):
                errors.append(f"History is not a list, got: {type(history)}")
            
            if not isinstance(players, list):
                errors.append(f"Players is not a list, got: {type(players)}")
                
            if not hasattr(game_config, 'num_players'):
                errors.append(f"GameConfig missing required attributes, got: {type(game_config)}")
            
            return len(errors) == 0, errors
                
        except json.JSONDecodeError as e:
            errors.append(f"JSON decode error: {e}")
        except Exception as e:
            errors.append(f"Loading error: {e}")
            
        return False, errors
    
    def _validate_strict_types(self, history: List[History], players: List, game_config: GameConfig, errors: List[str]) -> None:
        """Validate that all objects have correct types - this is the most important check."""
        
        # Check that history entries are proper History objects with correct types
        for i, entry in enumerate(history):
            if not hasattr(entry, 'phase'):
                errors.append(f"History entry {i} missing phase field")
                continue
                
            # Most critical: phase must be a proper GamePhase enum, not a dict
            if isinstance(entry.phase, dict):
                errors.append(f"History entry {i} has phase as dict instead of GamePhase enum: {entry.phase}")
                continue
            elif not isinstance(entry.phase, GamePhase):
                # This catches cases where phases might be loaded as ActionType or other enums
                errors.append(f"History entry {i} has phase as wrong enum type {type(entry.phase).__name__}: {entry.phase}")
                continue
                
            # Check other critical fields have correct types
            if not isinstance(entry.actions_until_phase_ends, int):
                errors.append(f"History entry {i} actions_until_phase_ends is not int: {type(entry.actions_until_phase_ends)}")
                
            if not hasattr(entry, 'action_taken') or not hasattr(entry.action_taken, 'type'):
                errors.append(f"History entry {i} missing or invalid action_taken")
                continue
                
            if not hasattr(entry, 'alive_player_names') or not isinstance(entry.alive_player_names, list):
                errors.append(f"History entry {i} missing or invalid alive_player_names")
                
            if not hasattr(entry, 'votes_before_this_discussion_message'):
                errors.append(f"History entry {i} missing votes_before_this_discussion_message")
        
        # Check that players are proper Player objects
        for i, player in enumerate(players):
            if not hasattr(player, 'name') or not hasattr(player, 'role'):
                errors.append(f"Player {i} missing required fields")
        
        # Check that game_config is proper GameConfig object
        if not hasattr(game_config, 'num_players') or not hasattr(game_config, 'num_impostors'):
            errors.append("GameConfig missing required fields")
    
    def _check_structure(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Check the structural integrity of the loaded game data."""
        errors = []
        
        try:
            # Create a temporary GameEngine with default config, then load the state
            temp_config = GameConfig()
            engine = GameEngine(temp_config, file_path=str(file_path))
            
            # Try to load state from the file
            success = engine.load_state()
            if not success:
                errors.append("Failed to load state with GameEngine")
                return False, errors
            
            # Validate strict types first - most important check
            self._validate_strict_types(engine.history, engine.players, engine.game_config, errors)
            
            # Only continue with other checks if types are correct
            if len(errors) == 0:
                # Validate history structure
                self._validate_history_structure(engine.history, engine.players, engine.game_config, errors)
                
                # Validate system messages
                self._validate_system_messages(engine.history, errors)
                
                # Validate votes_before field
                self._validate_votes_before_field(engine.history, errors)
                
                # Validate phase action counts
                self._validate_phase_actions(engine.history, engine.game_config, errors)
            
        except Exception as e:
            errors.append(f"Structure validation error: {e}")
            
        return len(errors) == 0, errors
    
    def _validate_history_structure(self, history: List[History], players: List, game_config: GameConfig, errors: List[str]) -> None:
        """Validate basic history structure."""
        if not history:
            errors.append("History is empty")
            return
        
        # Check that first entry is system message
        first_entry = history[0]
        if not hasattr(first_entry, 'action_taken') or first_entry.action_taken.player_name != "System":
            errors.append("First history entry should be system message")
        
        # Check all history entries have required fields
        required_fields = [
            'player_names_to_play_next', 'phase', 'actions_until_phase_ends',
            'location', 'impostor_cooldown', 'action_taken', 'alive_player_names'
        ]
        
        for i, entry in enumerate(history):
            for field in required_fields:
                if not hasattr(entry, field):
                    errors.append(f"History entry {i} missing required field: {field}")
    
    def _validate_system_messages(self, history: List[History], errors: List[str]) -> None:
        """Validate that system messages are properly placed for message-driven architecture."""
        if not history:
            return
            
        system_messages = [entry for entry in history if entry.action_taken.player_name == "System"]
        
        if not system_messages:
            errors.append("No system messages found")
            return
        
        # Check for game start message
        game_start_found = any(
            "game started" in entry.action_taken.spectator.lower() 
            for entry in system_messages
        )
        if not game_start_found:
            errors.append("Missing 'game started' system message")
        
        # Validate phase transition system messages
        self._validate_phase_transition_messages(history, errors)
        
    def _validate_phase_transition_messages(self, history: List[History], errors: List[str]) -> None:
        """Validate that proper system messages exist for phase transitions."""
        for i in range(len(history) - 1):
            current_entry = history[i]
            next_entry = history[i + 1]
            
            current_phase = current_entry.phase
            next_phase = next_entry.phase
            
            # Check for REPORT -> DISCUSS transition
            if (current_phase == GamePhase.TASKS and next_phase == GamePhase.DISCUSS and
                current_entry.action_taken.type == ActionType.REPORT):
                # Should have a system message between them or at the next entry
                if not self._has_discuss_system_message_nearby(history, i):
                    errors.append(f"Missing DISCUSS phase system message after REPORT action at entry {i}")
            
            # Check for DISCUSS -> VOTING transition
            elif (current_phase == GamePhase.DISCUSS and next_phase == GamePhase.VOTING and
                  current_entry.actions_until_phase_ends == 0):
                if not self._has_voting_system_message_nearby(history, i):
                    errors.append(f"Missing VOTING phase system message after DISCUSS phase end at entry {i}")
            
            # Check for VOTING -> TASKS transition
            elif (current_phase == GamePhase.VOTING and next_phase == GamePhase.TASKS and
                  current_entry.actions_until_phase_ends == 0):
                if not self._has_vote_result_and_tasks_messages_nearby(history, i):
                    errors.append(f"Missing vote result and TASKS phase system messages after VOTING end at entry {i}")
                    
    def _has_discuss_system_message_nearby(self, history: List[History], index: int) -> bool:
        """Check if there's a DISCUSS phase system message near the given index."""
        # Look in next few entries for discuss system message
        for i in range(index, min(index + 3, len(history))):
            entry = history[i]
            if (entry.action_taken.player_name == "System" and
                entry.phase == GamePhase.DISCUSS and
                "discussion phase" in entry.action_taken.spectator.lower()):
                return True
        return False
        
    def _has_voting_system_message_nearby(self, history: List[History], index: int) -> bool:
        """Check if there's a VOTING phase system message near the given index."""
        for i in range(index, min(index + 3, len(history))):
            entry = history[i]
            if (entry.action_taken.player_name == "System" and
                entry.phase == GamePhase.VOTING and
                "voting phase" in entry.action_taken.spectator.lower()):
                return True
        return False
        
    def _has_vote_result_and_tasks_messages_nearby(self, history: List[History], index: int) -> bool:
        """Check if there are proper vote result and tasks system messages near the given index."""
        vote_result_found = False
        tasks_message_found = False
        
        for i in range(index, min(index + 5, len(history))):
            entry = history[i]
            if entry.action_taken.player_name == "System":
                spectator = entry.action_taken.spectator.lower()
                
                # Vote result message (still in VOTING phase)
                if (entry.phase == GamePhase.VOTING and
                    ("voted out" in spectator or "ejected" in spectator)):
                    vote_result_found = True
                    
                # Back to tasks message (TASKS phase)
                elif (entry.phase == GamePhase.TASKS and
                      "cafeteria" in spectator and "task phase" in spectator):
                    tasks_message_found = True
        
        return vote_result_found and tasks_message_found
    
    def _validate_votes_before_field(self, history: List[History], errors: List[str]) -> None:
        """Validate that all discussion history items have votes_before field."""
        for i, entry in enumerate(history):
            if not hasattr(entry, 'votes_before_this_discussion_message'):
                errors.append(f"History entry {i} missing votes_before_this_discussion_message field")
            elif entry.votes_before_this_discussion_message is None:
                errors.append(f"History entry {i} has null votes_before_this_discussion_message")
    
    def _validate_phase_actions(self, history: List[History], game_config: GameConfig, errors: List[str]) -> None:
        """Validate that actions until phase ends counts down correctly within each phase."""
        current_phase = None
        last_actions_count = None
        phase_start_index = 0
        
        for i, entry in enumerate(history):
            # Handle legacy phase dicts
            entry_phase = entry.phase
            if isinstance(entry_phase, dict):
                entry_phase = entry_phase.get('__enum__', 'UNKNOWN')
            
            # Check if we're starting a new phase
            if entry_phase != current_phase:
                current_phase = entry_phase
                phase_start_index = i
                
                # Validate starting actions count for new phase
                expected_start_actions = self._calculate_expected_actions(entry, game_config)
                if expected_start_actions is not None:
                    if entry.actions_until_phase_ends > expected_start_actions:
                        errors.append(
                            f"History entry {i}: phase start has too many actions until phase ends: "
                            f"got {entry.actions_until_phase_ends}, max expected {expected_start_actions} "
                            f"(phase: {entry_phase})"
                        )
                
                last_actions_count = entry.actions_until_phase_ends
            else:
                # Same phase - validate countdown
                if last_actions_count is not None:
                    expected_countdown = last_actions_count - 1
                    if entry.actions_until_phase_ends != expected_countdown:
                        errors.append(
                            f"History entry {i}: actions should count down from {last_actions_count} to {expected_countdown}, "
                            f"got {entry.actions_until_phase_ends} (phase: {entry_phase})"
                        )
                
                last_actions_count = entry.actions_until_phase_ends
            
            # Validate that final action in phase has 0 actions remaining
            if i < len(history) - 1:  # Not the last entry
                next_entry = history[i + 1]
                next_phase = next_entry.phase
                if isinstance(next_phase, dict):
                    next_phase = next_phase.get('__enum__', 'UNKNOWN')
                
                # If next entry is different phase, current should have 0 actions left
                if next_phase != entry_phase and entry.actions_until_phase_ends != 0:
                    errors.append(
                        f"History entry {i}: last action in phase should have 0 actions until phase ends, "
                        f"got {entry.actions_until_phase_ends} (phase: {entry_phase})"
                    )
    
    def _calculate_expected_actions(self, entry: History, game_config: GameConfig) -> Optional[int]:
        """Calculate maximum expected actions at phase start based on phase and alive players."""
        alive_count = len(entry.alive_player_names) if hasattr(entry, 'alive_player_names') else 0
        
        # Handle legacy phase dicts and names
        phase = entry.phase
        if isinstance(phase, dict):
            phase_str = phase.get('__enum__', 'UNKNOWN')
        else:
            phase_str = str(phase)
        
        if phase_str in ['Tasks', 'Task']:  # Handle both current and legacy naming
            return game_config.num_task_phase_actions_per_player * alive_count
        elif phase_str == 'Discuss':
            return game_config.num_discuss_phase_actions_per_player * alive_count
        elif phase_str == 'Voting':
            return alive_count  # One vote per alive player
        else:
            return None  # Don't validate unknown phases
    
    def _print_summary(self) -> None:
        """Print summary of integrity check results."""
        print("\n" + "=" * 70)
        print("📊 INTEGRITY CHECK SUMMARY")
        print("=" * 70)
        
        print(f"✅ PASSED: {len(self.passed_files)} files")
        for file_name in self.passed_files:
            print(f"   • {file_name}")
        
        print(f"\n❌ FAILED: {len(self.failed_files)} files")
        
        # Count different types of errors
        error_counts = {
            "legacy_phase_name": 0,
            "legacy_phase_dict": 0,
            "legacy_enum": 0,
            "action_countdown": 0,
            "missing_field": 0,
            "other": 0
        }
        
        type_errors = 0
        for file_name, errors in self.failed_files:
            for error in errors:
                if "phase as dict" in error or "invalid phase type" in error or "missing" in error.lower() or "not int" in error:
                    type_errors += 1
                    break
        
        for file_name, errors in self.failed_files:
            for error in errors:
                if "actions should count down" in error or "last action in phase should have 0" in error:
                    error_counts["action_countdown"] += 1
                else:
                    error_counts["other"] += 1
        
        print(f"\n📋 ERROR TYPE SUMMARY:")
        if type_errors > 0:
            print(f"   🔍 Type validation errors (phase dicts, missing fields, wrong types): {type_errors} files")
        if error_counts["action_countdown"] > 0:
            print(f"   🔢 Action countdown validation errors: {error_counts['action_countdown']}")
        if error_counts["other"] > 0:
            print(f"   ⚠️  Other structural issues: {error_counts['other']}")
        
        # Show sample of files with detailed errors (first 5)
        print("\n📄 SAMPLE DETAILED ERRORS (first 5 files):")
        for file_name, errors in self.failed_files[:5]:
            print(f"   • {file_name}")
            for error in errors[:2]:  # Show first 2 errors
                print(f"     - {error}")
            if len(errors) > 2:
                print(f"     ... and {len(errors) - 2} more errors")


def main():
    """Main entry point."""
    # Change to project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    print("🚀 Among Them Game JSON Integrity Checker")
    print("=" * 70)
    print(f"Working directory: {os.getcwd()}")
    print(f"Checking data folder: {project_root / 'data'}")
    
    checker = IntegrityChecker()
    checker.check_all_files()


if __name__ == "__main__":
    main()
