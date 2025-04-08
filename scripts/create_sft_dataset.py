# Script to extract discussion messages from game_state.json files and create a CSV dataset
import json
import csv
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Import GameEngine and related modules
from among_them.game_engine import GameEngine
from among_them.models.player import Player
from among_them.models.player_role import PlayerRole
from among_them.models.history import get_action_history_str
from among_them.models.location import Location
from among_them.models.phase import GamePhase


def safe_get(obj, key, default=None):
    """
    Safely get a value from either a dictionary or an object with attributes
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    elif hasattr(obj, key):
        return getattr(obj, key, default)
    # Special handling for Action objects
    elif hasattr(obj, "spectator") and key == "spectator":
        return obj.spectator
    elif hasattr(obj, "type") and key == "type":
        return obj.type
    return default


def extract_player_info(players: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Extract player name to role mapping from the game state
    """
    player_roles = {}
    for player in players:
        # Handle different ways the role might be stored
        role = safe_get(player, "role", {})
        if isinstance(role, dict) and "__enum__" in role:
            player_roles[player["name"]] = role["__enum__"]
        elif isinstance(role, str):
            player_roles[player["name"]] = role
        else:
            player_roles[player["name"]] = str(role)
    return player_roles


def get_enum_value(obj: Any) -> str:
    """Extract value from enum objects"""
    if isinstance(obj, dict) and "__enum__" in obj:
        return obj["__enum__"]
    return str(obj)


def extract_vote_from_spectator(spectator: str) -> Tuple[str, str]:
    """Extract voter and votee from spectator message like 'Bob voted for Charlie'"""
    match = re.match(r"(.*?) voted for (.*?)$", spectator)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", ""


def find_votes_after_message(history: List[Any], current_idx: int) -> Dict[str, str]:
    """
    Find votes after a discussion message by looking at subsequent history items
    """
    votes_after = {}
    
    # Start from the next history item
    idx = current_idx + 1
    
    # Find next Vote phase
    vote_phase_found = False
    
    while idx < len(history):
        next_item = history[idx]
        
        # Get the phase of the next item
        phase = get_enum_value(safe_get(next_item, "phase", {}))
        
        if "Vote" in phase:
            vote_phase_found = True
            # This is a vote phase, try to extract votes
            
            # Check if there are direct vote records
            votes_dict = safe_get(next_item, "votes", {})
            if votes_dict:
                for voter, vote_info in votes_dict.items():
                    if isinstance(vote_info, dict):
                        votee = safe_get(vote_info, "voted_player", "")
                    else:
                        votee = str(vote_info)
                    if votee and votee != "nobody":
                        votes_after[voter] = votee
            
            # Also check spectator field for vote information
            action_taken = safe_get(next_item, "action_taken", {})
            spectator = safe_get(action_taken, "spectator", "")
            
            # Check if this is a vote action
            action_type = get_enum_value(safe_get(action_taken, "type", {}))
            if "VOTE" in action_type:
                voter = safe_get(action_taken, "player_name", "")
                # Try to extract votee from spectator text like "X voted for Y"
                if " voted for " in spectator:
                    parts = spectator.split(" voted for ")
                    if len(parts) > 1:
                        voter_from_spectator = parts[0].strip()
                        votee = parts[1].strip().rstrip('.')
                        if voter:
                            votes_after[voter] = votee
                        elif voter_from_spectator:
                            votes_after[voter_from_spectator] = votee
                # Try to extract from alternative format like "X votes Y"
                elif " votes " in spectator:
                    parts = spectator.split(" votes ")
                    if len(parts) > 1:
                        voter_from_spectator = parts[0].strip()
                        votee = parts[1].strip().rstrip('.')
                        if voter:
                            votes_after[voter] = votee
                        elif voter_from_spectator:
                            votes_after[voter_from_spectator] = votee
            
        # If we found a vote phase and then hit another discuss or task phase, stop looking
        elif vote_phase_found and ("Discuss" in phase or "Task" in phase):
            break
            
        idx += 1
            
    return votes_after


def escape_json_for_csv(json_obj: Dict[str, str]) -> str:
    """Escape JSON for CSV output to prevent quote issues"""
    # Convert to JSON string
    json_str = json.dumps(json_obj)
    # Replace double quotes with single quotes to avoid CSV quoting issues
    return json_str.replace('"', "'")


def get_players_in_room(history_item, players, location):
    """Get players who are in the same room"""
    # This is a simplified version - the actual game might have more complex logic
    return [p for p in players if p.name in safe_get(history_item, "spectators_who_saw", [])]


def get_alive_players(history, players):
    """Get players who are alive (not ejected)"""
    # In a real game, this would check who has been ejected
    # For now, we'll assume all players are alive
    return players


def get_game_engine_from_file(file_path: str) -> GameEngine:
    """
    Creates and initializes a GameEngine from a game state file
    """
    # Create placeholder players to satisfy minimum player requirement
    placeholder_players = [
        Player(name="Alice", role=PlayerRole.CREWMATE, manual_human_control=False),
        Player(name="Bob", role=PlayerRole.CREWMATE, manual_human_control=False),
        Player(name="Charlie", role=PlayerRole.IMPOSTOR, manual_human_control=False),
        Player(name="David", role=PlayerRole.CREWMATE, manual_human_control=False),
        Player(name="Eve", role=PlayerRole.CREWMATE, manual_human_control=False),
    ]
    
    # Initialize GameEngine with placeholder players
    game_engine = GameEngine(placeholder_players, 1)
    game_engine.file_path = file_path  # Set the file path
    
    # Load state (this will overwrite the placeholder players)
    game_engine.load_state()
    
    return game_engine


def get_system_prompt_for_message(game_engine: GameEngine, history_idx: int, player_name: str) -> str:
    """
    Get the system prompt that would be shown to a player at a specific point in history
    """
    # Get history up to this point
    history_up_to_now = game_engine.history[:history_idx+1]
    
    # Find the player object
    player = None
    for p in game_engine.players:
        if p.name == player_name:
            player = p
            break
    
    if not player:
        return "Player not found"
    
    # Get current location from the history item
    location = history_up_to_now[-1].location
    
    # Get all players in the room
    players_in_room = [p for p in game_engine.players if p.name in safe_get(history_up_to_now[-1], "spectators_who_saw", [])]
    
    # Get alive players (for simplicity, assume all are alive)
    alive_players = game_engine.players
    
    # Get the phase
    phase = history_up_to_now[-1].phase
    
    # Generate the system prompt
    try:
        system_prompt = get_action_history_str(
            history_up_to_now, 
            player, 
            players_in_room, 
            alive_players, 
            location, 
            phase
        )
        return system_prompt
    except Exception as e:
        print(f"Error generating system prompt: {e}")
        return f"Error: {str(e)}"


def generate_system_prompt(history_item, player_name, history_up_to_now, all_players):
    """
    Generate a simplified system prompt based on the available data
    """
    # Find the player
    player = None
    for p in all_players:
        # Handle both dictionary and Player object cases
        if isinstance(p, dict) and p.get("name") == player_name:
            player = p
            break
        elif hasattr(p, "name") and p.name == player_name:
            # Convert Player object to dict for consistent access
            player = {"name": p.name, "role": str(p.role)}
            break
    
    if not player:
        return f"Could not find player: {player_name}"
    
    # Get player role
    role = safe_get(player, "role", {})
    if isinstance(role, dict):
        role_value = role.get("__enum__", "Unknown")
    else:
        role_value = str(role)
    
    # Current location
    location = safe_get(history_item, "location", {})
    if isinstance(location, dict):
        location_value = location.get("__enum__", "Unknown")
    else:
        location_value = str(location)
    
    # Get phase
    phase = safe_get(history_item, "phase", {})
    if isinstance(phase, dict):
        phase_value = phase.get("__enum__", "Unknown")
    else:
        phase_value = str(phase)
    
    # Generate a simplified system prompt
    system_prompt = f"<player_info>\n"
    system_prompt += f"You are {player_name} in a text-based social deduction game.\n"
    system_prompt += f"You ({player_name}) are assigned the role of {role_value}.\n"
    
    # If player is impostor, identify crewmates
    if "Impostor" in role_value:
        crewmates = []
        for p in all_players:
            other_name = safe_get(p, "name", "")
            other_role = safe_get(p, "role", "")
            
            if other_name != player_name and "Crewmate" in str(other_role):
                crewmates.append(other_name)
                
        system_prompt += f"You are the impostor. Your task is to eliminate all crewmates.\n"
        if crewmates:
            system_prompt += f"{', '.join(crewmates)} are crewmates. Vote out or kill them to win.\n"
    else:
        # If player is crewmate, indicate there are impostors
        impostors_count = 0
        other_players = []
        
        for p in all_players:
            other_name = safe_get(p, "name", "")
            other_role = safe_get(p, "role", "")
            
            if other_name != player_name:
                other_players.append(other_name)
                if "Impostor" in str(other_role):
                    impostors_count += 1
        
        system_prompt += f"You are a crewmate. Vote out all impostors to win. "
        system_prompt += f"There is {impostors_count} impostor(s) among ({', '.join(other_players)})\n"
    
    # Add tasks if available
    tasks_left = safe_get(history_item, "tasks_left_to_do", {})
    if tasks_left and player_name in tasks_left:
        tasks = tasks_left[player_name]
        system_prompt += f"You ({player_name}) have {len(tasks)} tasks left:\n"
        for task in tasks:
            task_name = safe_get(task, "name", "Unknown task")
            system_prompt += f"{task_name}\n"
    
    system_prompt += "</player_info>\n\n"
    
    # Add simplified history
    system_prompt += "<history>\n"
    for i, item in enumerate(history_up_to_now):
        action = safe_get(item, "action_taken", {})
        spectator = safe_get(action, "spectator", "")
        action_player = safe_get(action, "player_name", "")
        
        if action_player == player_name:
            # This is the player's own action
            cot = safe_get(item, "llm_cot", "")
            result = safe_get(action, "result", "")
            system_prompt += f"At this point, you thought to yourself: {cot}\nAnd after thinking\n{result}\n"
        else:
            # Action by another player that was visible to this player
            spectators = safe_get(item, "spectators_who_saw", [])
            if player_name in spectators:
                if "Speak" in str(safe_get(action, "type", "")):
                    system_prompt += f"Discussion: {spectator}\n"
                else:
                    system_prompt += f"You saw: {spectator}\n"
    
    system_prompt += "</history>\n\n"
    
    # Add current location and phase info
    spectators = safe_get(history_item, "spectators_who_saw", [])
    other_players = [p for p in spectators if p != player_name]
    
    if "Task" in phase_value:
        if other_players:
            system_prompt += f"You ({player_name}) are currently with {', '.join(other_players)} in {location_value}\n"
        else:
            system_prompt += f"You ({player_name}) are currently alone in {location_value}\n"
        system_prompt += "Shhh... You can not speak now. It is against the rules\n"
    elif "Vote" in phase_value:
        system_prompt += "It is voting phase now.\n"
    else:
        system_prompt += "It is discussion phase now. You can speak now. Respond to the crewmates.\n"
    
    return system_prompt


def extract_discussion_messages(history: List[Any], player_roles: Dict[str, str], 
                             json_file_name: str, all_history: List[Any], all_players: List[Any]) -> List[Dict[str, Any]]:
    """
    Extract discussion messages from the game history
    """
    discussion_messages = []
    
    # Print debug info for first 20 items
    print("\nDebug first 20 history items:")
    for idx, item in enumerate(history[:min(20, len(history))]):
        debug_item(item, idx)
    
    # Track discussion phases to count total
    discussion_count = 0
    
    # First pass: Identify all discussion items and their indices
    discussion_indices = []
    for idx, item in enumerate(history):
        phase = get_enum_value(safe_get(item, "phase", {}))
        if phase == "Discuss" or phase == "GamePhase.DISCUSS":
            action_type = get_enum_value(safe_get(safe_get(item, "action_taken", {}), "type", {}))
            if action_type == "SPEAK" or action_type == "ActionType.SPEAK":
                discussion_indices.append(idx)
    
    # Also identify all voting phase indices
    vote_indices = []
    for idx, item in enumerate(history):
        phase = get_enum_value(safe_get(item, "phase", {}))
        if phase == "Vote" or phase == "GamePhase.VOTE":
            vote_indices.append(idx)
    
    print(f"Found {len(discussion_indices)} discussion messages at indices: {discussion_indices}")
    print(f"Found {len(vote_indices)} voting phases at indices: {vote_indices}")
    
    # Process each discussion message
    for i, idx in enumerate(discussion_indices):
        discussion_count += 1
        item = history[idx]
        
        # Get the action and spectator message
        action_taken = safe_get(item, "action_taken", {})
        spectator = safe_get(action_taken, "spectator", "")
        
        # Try different regex patterns to extract player name and message
        player_name = None
        message = None
        
        # Pattern 1: [Player]: Message
        match1 = re.match(r"\[(.*?)\]:(.*)", spectator)
        if match1:
            player_name = match1.group(1).strip()
            message = match1.group(2).strip()
        
        # Pattern 2: [Player]: Player: Message
        match2 = re.match(r"\[(.*?)\]:\s*\1:\s*(.*)", spectator)
        if match2:
            player_name = match2.group(1).strip()
            message = match2.group(2).strip()
        
        # Pattern 3: Just try to extract anything in brackets
        if not player_name and "[" in spectator and "]" in spectator:
            bracket_match = re.search(r"\[(.*?)\]", spectator)
            if bracket_match:
                player_name = bracket_match.group(1).strip()
                # The message is everything after the player name
                name_pos = spectator.find(']')
                if name_pos > 0:
                    message = spectator[name_pos+1:].strip()
                    if message.startswith(':'):
                        message = message[1:].strip()
        
        # Debug message extraction
        print(f"Discussion item {idx}: Player={player_name}, Message={message[:30]}..." if player_name and message else f"Failed to parse discussion item {idx}: {spectator[:50]}...")
        
        if player_name:
            # Get the player's role
            player_role = player_roles.get(player_name, "Unknown")
            
            # Get the LLM's chain of thought and action taken
            llm_cot = safe_get(item, "llm_cot", "")
            
            # Create the model_response by combining llm_cot and spectator with a newline
            model_cot_and_cleaned_output = f"{llm_cot}\n{spectator}"
            
            # Get votes before this discussion message
            votes_before = safe_get(item, "votes_before_this_discussion_message", {})
            
            # Get votes after this discussion message
            votes_after = {}
            
            # CASE 1: If the next item is another discussion phase
            if i < len(discussion_indices) - 1:
                next_idx = discussion_indices[i + 1]
                next_item = history[next_idx]
                # Get votes_before from the next discussion message
                next_votes_before = safe_get(next_item, "votes_before_this_discussion_message", {})
                if next_votes_before:
                    votes_after = next_votes_before
            
            # CASE 2: Check for consecutive voting phases after this discussion
            else:
                # Find all voting phases that occur after this discussion message
                following_votes = [v_idx for v_idx in vote_indices if v_idx > idx]
                
                # Check if we have voting phases after this discussion
                if following_votes:
                    # Examine each voting phase to collect votes
                    for vote_idx in following_votes:
                        vote_item = history[vote_idx]
                        
                        # If we hit the next discussion phase, stop
                        vote_phase = get_enum_value(safe_get(vote_item, "phase", {}))
                        if vote_phase != "Vote" and vote_phase != "GamePhase.VOTE":
                            break
                        
                        # Extract votes from this voting phase
                        action_taken = safe_get(vote_item, "action_taken", {})
                        action_type = get_enum_value(safe_get(action_taken, "type", {}))
                        
                        if action_type == "VOTE" or action_type == "ActionType.VOTE":
                            voter = safe_get(action_taken, "player_name", "")
                            target = safe_get(action_taken, "target_player_name", "")
                            
                            if voter and target and target != "nobody":
                                votes_after[voter] = target
                        
                        # Also check votes_before_this_discussion_message in the last voting phase
                        # as it may contain accumulated votes
                        if vote_idx == following_votes[-1]:
                            next_phase_idx = vote_idx + 1
                            if next_phase_idx < len(history):
                                next_phase_item = history[next_phase_idx]
                                phase_votes = safe_get(next_phase_item, "votes_before_this_discussion_message", {})
                                if not phase_votes:
                                    # Try direct votes field if available
                                    phase_votes = safe_get(next_phase_item, "votes", {})
                                
                                if phase_votes:
                                    for voter, vote_info in phase_votes.items():
                                        if isinstance(vote_info, dict):
                                            votee = safe_get(vote_info, "voted_player", "")
                                        else:
                                            votee = str(vote_info)
                                        
                                        if voter and votee and votee != "nobody":
                                            votes_after[voter] = votee
            
            # Check if we found any votes - if not, raise a warning but continue
            if not votes_after:
                print(f"WARNING: No votes found after discussion message by {player_name} at index {idx}")
            
            # Escape votes JSON to prevent CSV issues
            votes_before_str = escape_json_for_csv(votes_before)
            votes_after_str = escape_json_for_csv(votes_after)
            
            # Generate the system prompt for this message
            history_up_to_now = all_history[:idx+1]
            system_prompt = generate_system_prompt(item, player_name, history_up_to_now, all_players)
            
            discussion_messages.append({
                "json_file_name": json_file_name,
                "player_name": player_name,
                "player_role": player_role,
                "votes_before": votes_before_str,
                "votes_after": votes_after_str,
                "system_prompt": system_prompt,
                "model_cot_and_cleaned_output": model_cot_and_cleaned_output
            })
            
            print(f"Found discussion message from {player_name} ({player_role}) with {len(votes_after)} votes after")
    
    print(f"Total discussion phases found: {discussion_count}")
    return discussion_messages


def load_json_state(file_path: str) -> Tuple[List[Dict], List[Dict]]:
    """Load the history and players from a game state file"""
    try:
        # First try loading using GameEngine
        game_engine = get_game_engine_from_file(file_path)
        return game_engine.history, game_engine.players
    except Exception as e:
        print(f"Failed to load with GameEngine: {e}, trying direct JSON loading...")
        
        # Fallback to direct JSON loading
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        if isinstance(json_data, list) and len(json_data) == 2:
            return json_data[0], json_data[1]  # history, players
        else:
            # Try different formats
            if isinstance(json_data, dict):
                history = json_data.get("history")
                players = json_data.get("players")
                if history and players:
                    return history, players
        
        raise ValueError(f"Could not parse game state file: {file_path}")


def process_game_state_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Process a single game_state.json file and extract discussion messages
    """
    try:
        # Load the game state JSON
        history, players = load_json_state(file_path)
        
        # Print the structure of the data for debugging
        print(f"Found {len(history)} history items and {len(players)} players")
        
        # Extract player roles
        player_roles = {}
        for player in players:
            role = safe_get(player, "role", {})
            if isinstance(role, dict) and "__enum__" in role:
                player_roles[safe_get(player, "name", "")] = role["__enum__"]
            else:
                player_roles[safe_get(player, "name", "")] = str(role)
        
        print(f"Player roles: {player_roles}")
        
        # Count discussion phases for debugging
        discuss_phases = sum(1 for item in history if get_enum_value(safe_get(item, "phase", {})) == "Discuss")
        
        # Look for any messages in the spectator field that match discussion format
        message_count = 0
        for item in history:
            phase = get_enum_value(safe_get(item, "phase", {}))
            if phase == "Discuss" or phase == "GamePhase.DISCUSS":
                action_taken = safe_get(item, "action_taken", None)
                if action_taken:
                    spectator = safe_get(action_taken, "spectator", "")
                    if "[" in spectator and ("]" in spectator or "]:" in spectator):
                        message_count += 1
        
        print(f"Found {discuss_phases} discussion phases with {message_count} potential messages")
        
        json_file_name = os.path.basename(file_path)
        
        # Convert any object history to dict history for easier processing
        history_dicts = []
        for item in history:
            if isinstance(item, dict):
                history_dicts.append(item)
            else:
                # Convert History object to dict representation
                history_dict = {}
                if hasattr(item, "phase"):
                    history_dict["phase"] = item.phase
                if hasattr(item, "action_taken"):
                    action = item.action_taken
                    action_dict = {}
                    if hasattr(action, "type"):
                        action_dict["type"] = action.type
                    if hasattr(action, "spectator"):
                        action_dict["spectator"] = action.spectator
                    if hasattr(action, "player_name"):
                        action_dict["player_name"] = action.player_name
                    if hasattr(action, "result"):
                        action_dict["result"] = action.result
                    history_dict["action_taken"] = action_dict
                if hasattr(item, "spectators_who_saw"):
                    history_dict["spectators_who_saw"] = item.spectators_who_saw
                if hasattr(item, "location"):
                    history_dict["location"] = item.location
                if hasattr(item, "llm_cot"):
                    history_dict["llm_cot"] = item.llm_cot
                if hasattr(item, "votes_before_this_discussion_message"):
                    history_dict["votes_before_this_discussion_message"] = item.votes_before_this_discussion_message
                if hasattr(item, "tasks_left_to_do"):
                    history_dict["tasks_left_to_do"] = item.tasks_left_to_do
                history_dicts.append(history_dict)
        
        # Extract discussion messages
        return extract_discussion_messages(history_dicts, player_roles, json_file_name, history_dicts, players)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return []


def debug_item(item, idx):
    """Print debug information about a history item"""
    phase = safe_get(item, "phase", {})
    phase_str = get_enum_value(phase)
    
    action_taken = safe_get(item, "action_taken", {})
    action_type = safe_get(action_taken, "type", {})
    action_type_str = get_enum_value(action_type)
    
    spectator = safe_get(action_taken, "spectator", "")
    
    print(f"Item {idx}: Phase={phase_str}, ActionType={action_type_str}, Spectator={spectator[:50]}...")
    
    # Is this a discussion message?
    is_discuss = phase_str == "Discuss" or phase_str == "GamePhase.DISCUSS"
    is_speak = action_type_str == "SPEAK" or action_type_str == "ActionType.SPEAK"
    
    if is_discuss and is_speak:
        print(f"  Discussion phase found! Action type: {action_type_str}")
        
        # Check for spectator message format
        if "[" in spectator and "]" in spectator:
            print(f"  Looks like a discussion message: {spectator[:50]}...")


def main():
    # Path to the data directory containing game_state.json files
    data_dir = Path(__file__).parent.parent / "data"
    
    # Path to output CSV file
    output_csv = Path(__file__).parent.parent / "data" / "sft_dataset.csv"
    
    # Find all JSON files in the data directory
    game_state_files = []
    
    # Include game_state.json file
    specific_file = data_dir / "game_state.json"
    if specific_file.exists():
        game_state_files.append(specific_file)
    
    # Look for all JSON files
    for file in data_dir.glob("*.json"):
        if file != specific_file:  # Avoid duplicate entries
            game_state_files.append(file)
    
    if not game_state_files:
        print("No JSON files found in the data directory.")
        return
    
    print(f"Processing {len(game_state_files)} JSON files: {[str(f) for f in game_state_files]}")
    
    # Process each game state file and collect all discussion messages
    all_discussion_messages = []
    for file_path in game_state_files:
        print(f"\nProcessing {file_path}...")
        messages = process_game_state_file(str(file_path))
        all_discussion_messages.extend(messages)
    
    # Write to CSV - use proper quoting to handle multiline text and commas
    if all_discussion_messages:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ["json_file_name", "player_name", "player_role", "votes_before", "votes_after", "system_prompt", "model_cot_and_cleaned_output"]
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for message in all_discussion_messages:
                writer.writerow(message)
        
        print(f"Successfully created CSV file with {len(all_discussion_messages)} discussion messages at {output_csv}")
    else:
        print("No discussion messages found in the game state files.")


if __name__ == "__main__":
    main()