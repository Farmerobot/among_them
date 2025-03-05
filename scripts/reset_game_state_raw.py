#!/usr/bin/env python3
import os
import json
import argparse
import copy
from typing import Dict, Any, List, Optional


def find_discussion_rounds(game_state: Dict[str, Any]) -> List[int]:
    """
    Find all discussion rounds in the game history.
    
    Args:
        game_state: The game state JSON to analyze
        
    Returns:
        List of round indices that are discussion rounds
    """
    discussion_rounds = []
    
    # Use the first player's history as reference since all players share the same rounds
    if game_state.get('players') and len(game_state['players']) > 0:
        player = game_state['players'][0]
        if 'history' in player and 'rounds' in player['history'] and player['history']['rounds']:
            for i, round_data in enumerate(player['history']['rounds']):
                if round_data.get('stage') == 'Discuss':
                    discussion_rounds.append(i)
    
    return discussion_rounds


def get_player_by_name(game_state: Dict[str, Any], player_name: str) -> Optional[Dict[str, Any]]:
    """Find a player by name in the game state."""
    if 'players' in game_state:
        for player in game_state['players']:
            if player.get('name', '').lower() == player_name.lower():
                return player
    return None


def reset_game_state(
    input_file: str,
    output_file: str,
    player_name: Optional[str] = None,
    use_previous_state: bool = True,
    discussion_round: int = 0
) -> None:
    """
    Reset the game state to a specific point in a discussion round.
    
    Args:
        input_file: Path to the input game state JSON file
        output_file: Path to save the reset game state
        player_name: Optional name of a player to focus on
        use_previous_state: If True, use the state before the player's round; if False, use current state
        discussion_round: Discussion round number (0-indexed from first discussion)
    """
    # Load the game state JSON
    try:
        with open(input_file, 'r') as f:
            game_state = json.load(f)
    except Exception as e:
        print(f"Failed to load game state from {input_file}: {e}")
        return
    
    # Find all discussion rounds
    discussion_rounds = find_discussion_rounds(game_state)
    
    if not discussion_rounds:
        print("No discussion phase found in game history")
        return
    
    print(f"Found {len(discussion_rounds)} discussion rounds: {discussion_rounds}")
    
    # Determine which round to use
    if discussion_round < 0 or discussion_round >= len(discussion_rounds):
        print(f"Invalid discussion round number. Available rounds: 0-{len(discussion_rounds)-1}")
        return
    
    target_round_idx = discussion_rounds[discussion_round]
    reset_round_idx = target_round_idx
    print(f"Using discussion round {discussion_round} (game round {target_round_idx})")
    
    # If we're using the previous state and we're at the first discussion round or earlier
    if use_previous_state and target_round_idx <= 0:
        print("Cannot use previous state for the first round. Using the current state instead.")
        use_previous_state = False
    
    # Adjust reset_round_idx based on use_previous_state
    if use_previous_state:
        if target_round_idx > 0:
            reset_round_idx = target_round_idx - 1
            print(f"Using state BEFORE round {target_round_idx} (game round {reset_round_idx})")
        else:
            print(f"Using the CURRENT state for first round (game round {reset_round_idx})")
    else:
        if not use_previous_state:
            print(f"Using the CURRENT state for round {target_round_idx} (game round {reset_round_idx})")
        else:
            reset_round_idx = target_round_idx - 1
            print(f"Using state BEFORE round {target_round_idx} (game round {reset_round_idx})")
    
    # Player-specific logic
    player_index_to_reset_up_to = len(game_state.get('players', [])) - 1  # Default to all players
    
    if player_name:
        target_player = get_player_by_name(game_state, player_name)
        if not target_player:
            player_names = [p.get('name', 'Unknown') for p in game_state.get('players', [])]
            print(f"Player '{player_name}' not found. Available players: {player_names}")
            return
        
        # Find the index of the target player in the players array
        player_names = [p.get('name', '') for p in game_state.get('players', [])]
        target_player_index = -1
        for i, name in enumerate(player_names):
            if name.lower() == player_name.lower():
                target_player_index = i
                break
        
        if target_player_index == -1:
            print(f"Could not determine the order of player '{player_name}'. Aborting.")
            return
        
        print(f"Player '{player_name}' is at position {target_player_index} in the player order.")
        
        # If not using previous state (--after), include the target player
        # Otherwise (default, before) exclude the target player
        if not use_previous_state:
            player_index_to_reset_up_to = target_player_index
            print(f"Resetting state AFTER player '{player_name}' (including their state)")
        else:
            # If this is the first player (index 0) and we're in "before" mode (use_previous_state=True)
            # treat it the same as if no player was specified - reset all player states
            if target_player_index == 0:
                player_index_to_reset_up_to = len(game_state.get('players', [])) - 1  # All players
                print(f"Player '{player_name}' is the first player in order. Treating as if no player was specified.")
            else:
                player_index_to_reset_up_to = target_player_index - 1
                if player_index_to_reset_up_to < 0:
                    print(f"No players before '{player_name}'. Using initial state.")
                    player_index_to_reset_up_to = -1
                else:
                    prev_player = player_names[player_index_to_reset_up_to]
                    print(f"Resetting state BEFORE player '{player_name}' (after player '{prev_player}')")
    else:
        if not use_previous_state:
            print(f"Using the CURRENT state for round {target_round_idx} (game round {reset_round_idx})")
        else:
            print(f"Using BEFORE state (same as if no player was specified)")
    
    # Update each player's state based on position
    if 'players' in game_state:
        for i, player in enumerate(game_state['players']):
            if 'history' in player and 'rounds' in player['history']:
                if reset_round_idx < len(player['history']['rounds']):
                    # Only update players up to the specified index
                    if i <= player_index_to_reset_up_to:
                        # Copy state from history
                        player['state'] = copy.deepcopy(player['history']['rounds'][reset_round_idx])
                        
                        # Truncate history
                        player['history']['rounds'] = player['history']['rounds'][:reset_round_idx + 1]
                    else:
                        # For players after the specified player, reset to initial state or previous round
                        if reset_round_idx > 0:
                            # Use the state from the previous round
                            player['state'] = copy.deepcopy(player['history']['rounds'][reset_round_idx - 1])
                            player['history']['rounds'] = player['history']['rounds'][:reset_round_idx]
                        else:
                            # For the first round, we can only keep the initial state
                            print(f"For player at position {i}, keeping initial state (no previous round available)")
    
    # Update game state
    game_state['round_number'] = reset_round_idx
    game_state['player_to_act_next'] = min(player_index_to_reset_up_to + 1, len(game_state.get('players', [])) - 1)
    game_state['game_stage'] = 'Discuss'
    
    # Truncate playthrough history if needed
    if 'playthrough' in game_state and len(game_state['playthrough']) > 0:
        # First find the index where the target round starts
        round_start_idx = 0
        for i, entry in enumerate(game_state['playthrough']):
            if f"round: {reset_round_idx}" in entry:
                round_start_idx = i
                break
        
        # Then find where the next round would start
        next_round_idx = len(game_state['playthrough'])
        dead_players_idx = -1
        for i, entry in enumerate(game_state['playthrough'][round_start_idx:], round_start_idx):
            # Check for dead_players marker
            if "reported dead body" in entry and dead_players_idx == -1:
                dead_players_idx = i
                continue
            
            if f"round: {reset_round_idx + 1}" in entry and (not dead_players_idx == -1 or discussion_round > 0):
                next_round_idx = i
                break
        
        # If a player is specified, find their message in the current round
        if player_name and player_index_to_reset_up_to >= 0:
            # Find the last message from the specified player or the player before them
            player_cutoff_idx = next_round_idx
            
            # Player names already determined earlier
            players_to_include = [p.get('name', '') for p in game_state.get('players', [])][:player_index_to_reset_up_to + 1]
            
            # Find the last message from any player we want to include
            for i in range(next_round_idx - 1, round_start_idx - 1, -1):
                entry = game_state['playthrough'][i]
                # Check if this entry contains a message from a player we want to include
                player_found = False
                for player_to_include in players_to_include:
                    if f"[{player_to_include}]" in entry:
                        player_cutoff_idx = i + 1  # Include this message
                        player_found = True
                        break
                if player_found:
                    break
            
            # Truncate playthrough at the appropriate point
            game_state['playthrough'] = game_state['playthrough'][:player_cutoff_idx]
            print(f"Truncated playthrough to include messages up to player at position {player_index_to_reset_up_to}")
            
            # Process chat_messages ONLY for players up to player_index_to_reset_up_to
            for i, player in enumerate(game_state.get('players', [])):
                if i <= player_index_to_reset_up_to:  # Only process players up to the reset index
                    if 'state' in player and 'chat_messages' in player['state'] and player['state'].get('chat_messages'):
                        chat_messages = player['state']['chat_messages']
                        last_included_message_index = -1
                        
                        # Reverse iterate through chat messages to find last message from included players
                        for j in range(len(chat_messages) - 1, -1, -1):
                            msg = chat_messages[j]
                            # Check each player name to see if it's in the message
                            for player_to_include in players_to_include:
                                # Look for [PlayerName] pattern in the message string
                                if f"[{player_to_include}]" in msg:
                                    last_included_message_index = j
                                    break
                            if last_included_message_index != -1:
                                break
                        
                        # Truncate chat messages based on last included message
                        if last_included_message_index != -1:
                            player['state']['chat_messages'] = chat_messages[:last_included_message_index + 1]
                            print(f"Truncated chat messages for player '{player.get('name')}' to {last_included_message_index + 1} messages")
                        else:
                            print(f"No messages from relevant players found in {player.get('name')}'s chat history")
        else:
            # If no player specified, truncate at the end of the round
            game_state['playthrough'] = game_state['playthrough'][:next_round_idx]
            print(f"Truncated playthrough at the end of round {reset_round_idx}")
    
    # for player in game_state.get('players', []):
    #     player['llm_model_name'] = 'deepseek/deepseek-chat:free'

    # Save the reset game state
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the state
    try:
        with open(output_file, 'w') as f:
            json.dump(game_state, f, indent=2)
        print(f"Reset game state saved to {output_file}")
    except Exception as e:
        print(f"Failed to save game state to {output_file}: {e}")


if __name__ == "__main__":
    # Determine the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    parser = argparse.ArgumentParser(description='Reset the game state to a specific point in a discussion round.')
    
    parser.add_argument('--input', '-i', 
                        default=os.path.join(project_root, "data", "tournament", "llama-3-1-8b-instruct_vs_llama-3-1-8b-instruct_1.json"),
                        help='Path to the input game state JSON file')
    
    parser.add_argument('--output', '-o', 
                        default=os.path.join(project_root, "data", "game_state.json"),
                        help='Path to save the reset game state')
    
    parser.add_argument('--player', '-p', 
                        help='Name of a player to focus on')
    
    parser.add_argument('--round', '-r', type=int, default=0,
                        help='Discussion round number (0-indexed from first discussion)')
    
    parser.add_argument('--after', '-a', action='store_true',
                        help='Use the current state instead of the previous state (default is previous)')
    
    args = parser.parse_args()
    
    reset_game_state(
        args.input, 
        args.output, 
        args.player, 
        not args.after,  # If --after is specified, use_previous_state should be False
        args.round
    )
