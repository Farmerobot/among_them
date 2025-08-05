#!/usr/bin/env python3
"""
Legacy game state migrator for Among Them JSON files.

This script performs comprehensive architectural migration of legacy JSON files to match
the current game engine requirements. It handles:

1. Legacy phase dict conversion ({"__enum__": "Game Start"} -> GamePhase.TASKS)
2. Wrong enum type fixes (ActionType.TASK -> GamePhase.TASKS) 
3. Obsolete phase removal (GAME_START, VOTE_RESULTS, GAME_END)
4. System message reconstruction for message-driven phase transitions
5. Missing field additions and action countdown fixes

Only use this on legacy files - current files should already pass integrity checks.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import argparse
import shutil
from datetime import datetime

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from among_them.models.phase import GamePhase
from among_them.models.action_type import ActionType
from among_them.models.player_role import PlayerRole
from among_them.models.location import Location
from among_them.game_config import GameConfig
from among_them.utils.history_utils import create_system_message


class LegacyMigrator:
    """Migrates legacy Among Them game state files to current architecture."""
    
    def __init__(self, data_folder: str = "data", dry_run: bool = False, backup: bool = True):
        self.data_folder = Path(data_folder)
        self.dry_run = dry_run
        self.backup = backup
        self.migration_stats = {
            "files_processed": 0,
            "files_migrated": 0,
            "files_skipped": 0,
            "errors": 0,
            "phase_conversions": 0,
            "system_messages_added": 0,
            "enum_fixes": 0,
            "field_additions": 0
        }
        
    def migrate_all_files(self):
        """Migrate all JSON files in the data folder."""
        print(f"🚀 Starting legacy migration {'(DRY RUN)' if self.dry_run else ''}")
        print(f"📁 Scanning: {self.data_folder}")
        print("=" * 80)
        
        json_files = list(self.data_folder.glob("*.json"))
        if not json_files:
            print("❌ No JSON files found in data folder")
            return
            
        for file_path in json_files:
            self._migrate_file(file_path)
            
        self._print_summary()
        
    def _migrate_file(self, file_path: Path):
        """Migrate a single JSON file."""
        file_name = file_path.name
        print(f"\n📄 Processing: {file_name}")
        print("-" * 50)
        
        self.migration_stats["files_processed"] += 1
        
        try:
            # Load the JSON file
            with open(file_path, 'r') as f:
                json_str = f.read()
                
            # Parse as raw JSON first to analyze structure
            raw_data = json.loads(json_str)
            
            # Check if file needs migration
            if not self._needs_migration(raw_data):
                print("✅ File appears to be current format - skipping")
                self.migration_stats["files_skipped"] += 1
                return
                
            # Perform migration
            migrated_data = self._perform_migration(raw_data)
            
            if migrated_data is None:
                print("❌ Migration failed - could not convert file")
                self.migration_stats["errors"] += 1
                return
                
            # Create backup if requested
            if self.backup and not self.dry_run:
                backup_path = file_path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
                shutil.copy2(file_path, backup_path)
                print(f"💾 Backup created: {backup_path.name}")
                
            # Write migrated file
            if not self.dry_run:
                migrated_json = json.dumps(migrated_data, indent=2)
                with open(file_path, 'w') as f:
                    f.write(migrated_json)
                print(f"✅ Successfully migrated: {file_name}")
            else:
                print(f"✅ Would migrate: {file_name} (dry run)")
                
            self.migration_stats["files_migrated"] += 1
            
        except Exception as e:
            print(f"❌ Error processing {file_name}: {str(e)}")
            self.migration_stats["errors"] += 1
            
    def _needs_migration(self, data: Any) -> bool:
        """Check if the file needs migration by looking for legacy patterns."""
        if not isinstance(data, list) or len(data) < 2:
            return False
            
        history = data[0] if isinstance(data[0], list) else None
        if not history:
            return False
            
        # Check for legacy phase dicts or wrong enum types
        for entry in history[:5]:  # Check first 5 entries for efficiency
            if not isinstance(entry, dict):
                continue
                
            phase = entry.get("phase")
            if isinstance(phase, dict) and "__enum__" in phase:
                # Legacy phase dict found
                return True
                
            # Look for wrong enum patterns in phase field
            if isinstance(phase, dict) and phase.get("__enum__") in ["TASK", "Task", "Game Start", "Vote Results", "Game End"]:
                return True
                
        return False
        
    def _perform_migration(self, data: Any) -> Optional[List]:
        """Perform the actual migration of legacy data."""
        if not isinstance(data, list) or len(data) < 2:
            return None
            
        # Extract components
        if len(data) == 2:
            history, players = data
            game_config = None
        elif len(data) == 3:
            history, players, game_config = data
        else:
            return None
            
        # Provide default game_config if missing
        if game_config is None:
            game_config = self._create_default_game_config()
            self.migration_stats["field_additions"] += 1
            
        # Migrate history entries
        migrated_history = self._migrate_history(history, players, game_config)
        if migrated_history is None:
            return None
            
        # Return migrated data with game_config
        return [migrated_history, players, game_config]
        
    def _migrate_history(self, history: List[Dict], players: List, game_config: Dict) -> Optional[List[Dict]]:
        """Migrate history entries to current format."""
        if not isinstance(history, list):
            return None
            
        migrated_history = []
        
        # Process each history entry
        for i, entry in enumerate(history):
            if not isinstance(entry, dict):
                continue
                
            # Migrate individual history entry
            migrated_entry = self._migrate_history_entry(entry, i, players, game_config)
            if migrated_entry is not None:
                migrated_history.append(migrated_entry)
                
        # Reconstruct system messages for phase transitions
        final_history = self._reconstruct_system_messages(migrated_history, players, game_config)
        
        return final_history
        
    def _migrate_history_entry(self, entry: Dict, index: int, players: List, game_config: Dict) -> Optional[Dict]:
        """Migrate a single history entry."""
        migrated_entry = entry.copy()
        
        # Fix phase field
        phase = entry.get("phase")
        if isinstance(phase, dict) and "__enum__" in phase:
            # Convert legacy phase dict to current GamePhase
            legacy_phase = phase["__enum__"]
            current_phase = self._convert_legacy_phase(legacy_phase)
            
            if current_phase is not None:
                migrated_entry["phase"] = {
                    "__enum__": current_phase.value
                }
                self.migration_stats["phase_conversions"] += 1
                # print(f"  🔄 Phase conversion: {legacy_phase} -> {current_phase.value}")
            else:
                # Phase should be removed (GAME_END, VOTE_RESULTS)
                return None  # Skip this entry
                
        # Fix wrong enum types (e.g., ActionType.TASK -> GamePhase.TASKS)
        elif isinstance(phase, dict) and phase.get("__enum__") == "Task":
            # This catches cases where phase was loaded as ActionType.TASK
            migrated_entry["phase"] = {
                "__enum__": "Tasks"
            }
            self.migration_stats["enum_fixes"] += 1
            print(f"  🔧 Fixed wrong enum: ActionType.Task -> GamePhase.Tasks")
                
        # Add missing required fields
        if "votes_before_this_discussion_message" not in migrated_entry:
            migrated_entry["votes_before_this_discussion_message"] = {}
            self.migration_stats["field_additions"] += 1
            
        # alive_player_names will be reconstructed properly in _reconstruct_system_messages
        # by tracking KILL actions throughout the game history
            
        # Fix action_taken if it has wrong system action types
        action_taken = migrated_entry.get("action_taken")
        if isinstance(action_taken, dict) and action_taken.get("player_name") == "System":
            action_type = action_taken.get("type", {})
            if isinstance(action_type, dict):
                # Convert old system action types to SPEAK
                old_types = ["WAIT", "MOVE"]
                if action_type.get("__enum__") in old_types:
                    action_taken["type"] = {
                        "__enum__": "Speak"
                    }
                    self.migration_stats["enum_fixes"] += 1
                    
        return migrated_entry
        
    def _convert_legacy_phase(self, legacy_phase: str) -> Optional[GamePhase]:
        """Convert legacy phase names to current GamePhase enums."""
        phase_mapping = {
            # Legacy -> Current
            "Game Start": GamePhase.TASKS,  # Game starts directly in TASKS now
            "Task": GamePhase.TASKS,        # Singular -> plural
            "Tasks": GamePhase.TASKS,       # Already current
            "Discuss": GamePhase.DISCUSS,   # Already current
            "Voting": GamePhase.VOTING,     # Already current
            
            # Removed phases (return None to skip these entries)
            "Vote Results": None,           # Handled by system messages now
            "Game End": None,               # Handled by end game detection now
        }
        
        return phase_mapping.get(legacy_phase)
        
    def _reconstruct_system_messages(self, history: List[Dict], players: List, game_config: Dict) -> List[Dict]:
        """Reconstruct missing system messages for phase transitions and properly track alive_player_names."""
        if not history:
            return history
        
        # Initialize alive players from the player list
        initial_alive_players = []
        if isinstance(players, list):
            for player in players:
                if isinstance(player, dict) and "name" in player:
                    initial_alive_players.append(player["name"])
        
        # Track alive players throughout history by following KILL actions
        current_alive_players = initial_alive_players.copy()
        reconstructed_history = []
        i = 0
        
        while i < len(history):
            current_entry = history[i].copy()
            
            # Update alive_player_names based on KILL actions
            action_taken = current_entry.get("action_taken", {})
            if isinstance(action_taken, dict):
                action_type = action_taken.get("type", {})
                target_player = action_taken.get("target_player_name")
                
                # Track KILL actions (including ejections)
                if (isinstance(action_type, dict) and 
                    action_type.get("__enum__") == "Kill" and 
                    target_player and target_player in current_alive_players):
                    current_alive_players.remove(target_player)
                    print(f"    💲 Tracked KILL: {target_player} removed from alive players")
            
            # Ensure alive_player_names is correctly set
            current_entry["alive_player_names"] = current_alive_players.copy()
            reconstructed_history.append(current_entry)
            
            # Check if we need to insert system messages after this entry
            next_entry = history[i + 1] if i + 1 < len(history) else None
            
            if next_entry:
                current_phase = self._get_phase_name(current_entry.get("phase", {}))
                next_phase = self._get_phase_name(next_entry.get("phase", {}))
                
                # Detect phase transitions and insert system messages
                system_messages = self._create_transition_messages(
                    current_entry, next_entry, current_phase, next_phase, current_alive_players, game_config
                )
                
                for msg in system_messages:
                    # System messages may also update alive players if they involve ejections
                    msg_action = msg.get("action_taken", {})
                    if isinstance(msg_action, dict):
                        msg_target = msg_action.get("target_player_name")
                        msg_type = msg_action.get("type", {})
                        if (isinstance(msg_type, dict) and 
                            msg_type.get("__enum__") == "Kill" and 
                            msg_target and msg_target in current_alive_players):
                            current_alive_players.remove(msg_target)
                    
                    msg["alive_player_names"] = current_alive_players.copy()
                    reconstructed_history.append(msg)
                    self.migration_stats["system_messages_added"] += 1
                    
            i += 1
            
        # Handle game end if needed
        if reconstructed_history:
            last_entry = reconstructed_history[-1]
            if self._should_add_game_end_message(last_entry, players):
                end_msg = self._create_game_end_message(last_entry, current_alive_players, game_config)
                if end_msg:
                    reconstructed_history.append(end_msg)
                    self.migration_stats["system_messages_added"] += 1
                    
        if self.migration_stats.get("system_messages_added", 0) > 0:
            print(f"  📝 Added {self.migration_stats['system_messages_added']} system messages")
            print(f"  💲 Final alive players: {current_alive_players}")
        else:
            print(f"  📝 No system messages needed")
            
        return reconstructed_history
        
    def _dict_to_game_config(self, game_config_dict: Dict) -> GameConfig:
        """Convert dict game config to GameConfig object."""
        config = GameConfig()
        config.num_players = game_config_dict.get("num_players", 5)
        config.num_impostors = game_config_dict.get("num_impostors", 1) 
        config.num_task_phase_actions_per_player = game_config_dict.get("num_task_phase_actions_per_player", 3)
        config.num_discuss_phase_actions_per_player = game_config_dict.get("num_discuss_phase_actions_per_player", 2)
        config.impostor_cooldown = game_config_dict.get("impostor_cooldown", 1)
        return config
        
    def _history_to_dict(self, history_obj) -> Dict:
        """Convert History object back to dict format for JSON serialization."""
        # Use the game's JSON encoder to convert History object to dict
        import json
        from among_them.game_jsonencoder import GameJSONEncoder
        
        # Serialize to JSON string and parse back to get dict format
        json_str = json.dumps(history_obj, cls=GameJSONEncoder)
        return json.loads(json_str)
        
    def _dict_to_history_obj(self, entry_dict: Dict):
        """Convert dict entry to History object using game_object_hook."""
        import json
        from among_them.game_jsonencoder import game_object_hook
        
        # Serialize to JSON and use game_object_hook to reconstruct
        json_str = json.dumps(entry_dict)
        return json.loads(json_str, object_hook=game_object_hook)
        
    def _get_phase_name(self, phase_dict: Dict) -> str:
        """Extract phase name from phase dict."""
        if isinstance(phase_dict, dict):
            return phase_dict.get("__enum__", "")
        return ""
        
    def _create_transition_messages(self, current_entry: Dict, next_entry: Dict, 
                                  current_phase: str, next_phase: str, current_alive_players: List[str], game_config: Dict) -> List[Dict]:
        """Create system messages for phase transitions based on current game engine patterns."""
        messages = []
        alive_players = current_alive_players.copy()
        
        # REPORT -> DISCUSS transition
        if (current_phase == "Tasks" and next_phase == "Discuss" and 
            self._has_report_action(current_entry)):
            # Need to provide history with at least current entry for tasks_left_to_do
            current_history = [self._dict_to_history_obj(current_entry)]
            msg_dict = create_system_message(
                history=current_history,
                phase=GamePhase.DISCUSS,
                text="It is discussion phase now. Discuss who to eject from the game.",
                game_config=self._dict_to_game_config(game_config),
                alive_player_names=alive_players
            )
            messages.append(self._history_to_dict(msg_dict))
            print(f"    ↳ Added REPORT->DISCUSS transition message")
            
        # DISCUSS -> VOTING transition
        elif (current_phase == "Discuss" and next_phase == "Voting" and 
             self._is_phase_ending(current_entry)):
            current_history = [self._dict_to_history_obj(current_entry)]
            msg_dict = create_system_message(
                history=current_history,
                phase=GamePhase.VOTING,
                text="Discussion ended. Voting phase started. Vote who to eject from the game.",
                game_config=self._dict_to_game_config(game_config),
                alive_player_names=alive_players
            )
            messages.append(self._history_to_dict(msg_dict))
            print(f"    ↳ Added DISCUSS->VOTING transition message")
            
        # VOTING -> TASKS transition (with vote results)
        elif (current_phase == "Voting" and next_phase == "Tasks" and 
             self._is_phase_ending(current_entry)):
            # First: vote result message (still VOTING phase)
            ejected_player = self._determine_ejected_player(current_entry, next_entry)
            updated_alive = [p for p in alive_players if p != ejected_player] if ejected_player != "nobody" else alive_players
            
            current_history = [self._dict_to_history_obj(current_entry)]
            vote_msg_dict = create_system_message(
                history=current_history,
                phase=GamePhase.VOTING,
                text=f"{ejected_player} was voted out.",
                game_config=self._dict_to_game_config(game_config),
                alive_player_names=updated_alive,
                ejected_player=ejected_player,
                no_more_actions=True
            )
            messages.append(self._history_to_dict(vote_msg_dict))
            
            # Second: back to tasks message
            tasks_msg_dict = create_system_message(
                history=current_history,
                phase=GamePhase.TASKS,
                text="Everyone is in the cafeteria and start from there. It is task phase now.",
                game_config=self._dict_to_game_config(game_config),
                alive_player_names=updated_alive
            )
            messages.append(self._history_to_dict(tasks_msg_dict))
            print(f"    ↳ Added VOTING->TASKS transition messages (ejected: {ejected_player})")
            
        return messages
        
    def _has_report_action(self, entry: Dict) -> bool:
        """Check if entry contains a REPORT action."""
        action_taken = entry.get("action_taken", {})
        if isinstance(action_taken, dict):
            action_type = action_taken.get("type", {})
            if isinstance(action_type, dict):
                return action_type.get("__enum__") == "Report"
        return False
        
    def _is_phase_ending(self, entry: Dict) -> bool:
        """Check if this entry represents the end of a phase."""
        return entry.get("actions_until_phase_ends", 1) == 0
        
    def _determine_ejected_player(self, voting_entry: Dict, next_entry: Dict) -> str:
        """Determine who was ejected based on alive player changes."""
        current_alive = set(voting_entry.get("alive_player_names", []))
        next_alive = set(next_entry.get("alive_player_names", []))
        
        ejected = current_alive - next_alive
        if ejected:
            return list(ejected)[0]
        return "nobody"
        
    def _should_add_game_end_message(self, last_entry: Dict, players: List) -> bool:
        """Check if we should add a game end message."""
        # Simple heuristic: if very few players are alive or specific end conditions
        alive_count = len(last_entry.get("alive_player_names", []))
        total_players = len(players) if isinstance(players, list) else 5
        
        # Game might have ended if very few players left
        return alive_count <= max(1, total_players // 3)
        
    def _create_game_end_message(self, last_entry: Dict, current_alive_players: List[str], game_config: Dict) -> Optional[Dict]:
        """Create a game end system message."""
        alive_players = current_alive_players.copy()
        
        # Determine end reason based on remaining players
        if len(alive_players) <= 1:
            reason = "TOO_SMALL_NUMBER_OF_CREWMATES_LEFT"
        else:
            reason = "UNKNOWN"  # Would need more sophisticated detection
            
        # Use last entry for history context
        last_history = [self._dict_to_history_obj(last_entry)]
        msg_dict = create_system_message(
            history=last_history,
            phase=GamePhase.TASKS,
            text=f"The game ended ({reason})",
            game_config=self._dict_to_game_config(game_config),
            alive_player_names=alive_players,
            no_more_actions=True
        )
        return self._history_to_dict(msg_dict)
        
    def _create_default_game_config(self) -> Dict:
        """Create a default GameConfig dict for legacy files missing it."""
        return {
            "__enum__": "GameConfig",
            "num_players": 6,
            "num_impostors": 2,
            "num_task_phase_actions_per_player": 3,
            "num_discuss_phase_actions_per_player": 2,
            "impostor_cooldown": 3
        }
        
    def _print_summary(self):
        """Print migration summary statistics."""
        stats = self.migration_stats
        
        print("\n" + "=" * 80)
        print("📊 MIGRATION SUMMARY")
        print("=" * 80)
        print(f"📁 Files processed: {stats['files_processed']}")
        print(f"✅ Files migrated: {stats['files_migrated']}")
        print(f"⏭️  Files skipped: {stats['files_skipped']}")
        print(f"❌ Errors: {stats['errors']}")
        print()
        print("🔄 Migration Details:")
        print(f"   Phase conversions: {stats['phase_conversions']}")
        print(f"   Enum type fixes: {stats['enum_fixes']}")
        print(f"   Field additions: {stats['field_additions']}")
        print(f"   System messages added: {stats['system_messages_added']}")
        
        if self.dry_run:
            print("\n🔍 This was a DRY RUN - no files were actually modified")
        else:
            print(f"\n💾 Modified {stats['files_migrated']} files")
            if self.backup:
                print("📋 Backup files created for safety")


def main():
    parser = argparse.ArgumentParser(description="Migrate legacy Among Them game state files")
    parser.add_argument("--data-folder", default="data", help="Folder containing JSON files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without modifying files")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating backup files")
    
    args = parser.parse_args()
    
    migrator = LegacyMigrator(
        data_folder=args.data_folder,
        dry_run=args.dry_run,
        backup=not args.no_backup
    )
    
    migrator.migrate_all_files()


if __name__ == "__main__":
    main()
