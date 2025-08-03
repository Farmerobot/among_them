"""
Environment-like prompt building utilities for Among Them game.

This module implements the new prompt system that follows research principles:
- Cohesive observation-action concatenation
- Proper use of articles in action prompts  
- Environment context includes available moves and visible players
- LLM generations are appended unchanged to maintain context
"""

from typing import List
from among_them.models.action_type import ActionType
from among_them.models.player import Player
from among_them.models.history import History
from among_them.models.location import get_map
from among_them.models.action import Action
from among_them.models.player_role import PlayerRole
from among_them.models.phase import GamePhase
from among_them.game_config import GameConfig
from among_them.models.tasks import get_impostor_pretend_tasks_at_location
from among_them.utils.player_utils import get_players_in_room, get_last_player_action
from among_them.llm_prompts import UNIVERSAL_SYSTEM_PROMPT


def get_player_context(player: Player, history: List[History], all_players: List[Player], game_config: GameConfig) -> str:
    """
    Generate player context section including role-specific information.
    
    Returns:
        Player context as a string
    """
    context = []
    context.append(f"You are {player.name}.")
    
    if not history:
        return "\n".join(context)
    
    if player.role == PlayerRole.IMPOSTOR:
        context.append("You are an impostor. Your goal is to eliminate crewmates without being detected.")
        context.append(f"Your kill cooldown is {game_config.impostor_cooldown} actions.")
        
        # Other impostors
        other_impostors = [p for p in all_players if p.role == PlayerRole.IMPOSTOR and p.name != player.name]
        if other_impostors:
            impostor_names = [imp.name for imp in other_impostors]
            context.append(f"Your fellow impostors: {', '.join(impostor_names)}.")
        else:
            context.append("You are the only impostor.")
        
        # Crewmates to eliminate
        other_crewmates = [p for p in all_players if p.role == PlayerRole.CREWMATE]
        if other_crewmates:
            crewmate_names = [crew.name for crew in other_crewmates]
            context.append(f"Crewmates to eliminate: {', '.join(crewmate_names)}.")
    else:
        context.append("You are a crewmate. Your goal is to complete tasks and identify impostors.")
        
        # Impostor count and potential impostors
        other_impostors = [p for p in all_players if p.role == PlayerRole.IMPOSTOR]
        impostor_count = len(other_impostors)
        context.append(f"There are {impostor_count} impostors among the players.")
        
        other_players = [p.name for p in all_players if p.name != player.name]
        if other_players:
            context.append(f"Potential impostors: {', '.join(other_players)}.")
        
        # Tasks remaining
        if (history and history[0].tasks_left_to_do and 
            player.name in history[0].tasks_left_to_do and 
            history[0].tasks_left_to_do[player.name]):
            tasks = history[0].tasks_left_to_do[player.name]
            if tasks:
                task_descriptions = [f"{task.name}" for task in tasks]
                context.append(f"Tasks: {', '.join(task_descriptions)}.")
    
    return "\n".join(context)


def generate_action_observations(
    action: Action, 
    observing_player_name: str,
    spectators_who_saw: List[str]
) -> str:
    """
    Generate observations for a player based on an action that occurred.
    
    Args:
        action: The action that was taken
        observing_player_name: The name of the player who is observing
        spectators_who_saw: List of players who saw the action
        
    Returns:
        Observation string for the observing player
    """
    if observing_player_name == action.player_name:
        # Player took the action themselves - Add their action result
        if action.type == ActionType.SPEAK:
            # For SPEAK actions, the spectator field contains the actual speech content
            return f"You said: {action.spectator}"
        else:
            return action.agent_perspective or action.result
    else:
        # Someone else took the action
        can_observe = (observing_player_name in spectators_who_saw)
        
        if can_observe:
            if action.type == ActionType.SPEAK and not action.player_name == "System":
                # For SPEAK actions, the spectator field contains the actual speech content
                return f"Discussion: {action.spectator}"
            elif action.type == ActionType.VOTE:
                # Players cannot see others voting
                return ""
            else:
                # Observed action - use observer_perspective if available, otherwise fallback to spectator
                return action.observer_perspective or action.spectator
        else:
            # Player didn't see this action
            return f"You did not see what {action.player_name} did during his next turn."


def get_observations_at_history_point(
    player: Player,
    history: List[History],
    all_players: List[Player], 
    game_config: GameConfig,
    history_index: int,
    phase: GamePhase
) -> str:
    """Get environmental observations for a player at a specific point in game history."""
    if history_index >= len(history):
        return ""
    
    state_at_point = history[history_index]
    
    if phase == GamePhase.DISCUSS:
        observations = ["Given the situation, the best message to send is:"]
        return " ".join(observations)
    elif phase == GamePhase.VOTING:
        observations = ["Given the situation, the best action to take is:"]
        return " ".join(observations)
    
    observations = []

    history_slice = history[:history_index + 1]
    last_action = get_last_player_action(history_slice, player)
    
    player_location = last_action.location
    
    # Current location
    observations.append(f"You are in {player_location.value}.")
    
    # Available doors
    doors_map, _, _ = get_map(game_config.map_size)
    connected_locations = doors_map.get(player_location, [])
    doors = [loc.value.lower() for loc in connected_locations]
    if doors:
        door_list = ", ".join(doors)
        observations.append(f"You see doors to: {door_list}.")
    
    # Visible players in same room
    players_in_room = get_players_in_room(
        history[:history_index+1], all_players, player, player_location
    )
    if players_in_room:
        if len(players_in_room) == 1:
            observations.append(f"You see {players_in_room[0].name} here.")
        else:
            player_names = [p.name for p in players_in_room]
            player_list = ", ".join(player_names[:-1]) + f" and {player_names[-1]}"
            observations.append(f"You see {player_list} here.")
    else:
        observations.append("You are alone here.")
    
    # Task-related observations
    if player.role == PlayerRole.CREWMATE and state_at_point.tasks_left_to_do:
        player_tasks = state_at_point.tasks_left_to_do.get(player.name, [])
        available_tasks = [task for task in player_tasks 
                          if task.location == player_location]
        if available_tasks:
            if len(available_tasks) == 1:
                observations.append(f"You can do the {available_tasks[0].name} task here.")
            else:
                task_names = [task.name for task in available_tasks]
                task_list = ", ".join(task_names[:-1]) + f" and {task_names[-1]}"
                observations.append(f"You can do these tasks here: {task_list}.")
    elif player.role == PlayerRole.IMPOSTOR:
        pretend_tasks = get_impostor_pretend_tasks_at_location(player_location, game_config)
        if pretend_tasks:
            observations.append(f"You can pretend to do the {pretend_tasks[0].name} task here.")
    
    # Impostor-specific observations
    if player.role == PlayerRole.IMPOSTOR:
        cooldown = last_action.impostor_cooldown
        if cooldown > 0:
            observations.append(f"Your kill cooldown: wait {cooldown} more actions before you can kill.")
        else:
            observations.append("You can kill now.")
    
    observations.append("Given the situation, the best action to take is:")
    
    return " ".join(observations)


def reconstruct_environment_prompt_from_history(
    player: Player,
    history: List[History], 
    all_players: List[Player],
    game_config: GameConfig
) -> str:
    """
    Reconstruct the complete environment prompt for a player from game history.
    This includes all previous LLM generations and observations in chronological order.
    
    Args:
        player: The player this prompt is for
        history: Game history entries  
        all_players: All players in the game
        game_config: Game configuration
        
    Returns:
        Complete environment prompt including all previous context
    """
    prompt_parts = []
    
    # System context (static)
    prompt_parts.append(UNIVERSAL_SYSTEM_PROMPT)
    prompt_parts.append("")
    
    # Player context (with role-specific information)
    prompt_parts.append(get_player_context(player, history, all_players, game_config))
    prompt_parts.append("")

    # Reconstruct the environment flow from history
    for i, hist_entry in enumerate(history):
        # Check if this was the player's turn
        if hist_entry.action_taken.player_name == player.name:
            prompt_parts.append(get_observations_at_history_point(
                player, history, all_players, game_config, i-1, hist_entry.phase
            ))
            
            # Add LLM generation if this was the player's turn and we have the response
            if hist_entry.llm_response and hist_entry.llm_cot:
                # Add the player's LLM output (think + action tags)
                llm_output = hist_entry.llm_cot + hist_entry.llm_response
                prompt_parts.append(llm_output)

        # Add observations about what happened (for everyone)
        prompt_parts.append(generate_action_observations(
            hist_entry.action_taken,
            player.name,
            hist_entry.spectators_who_saw
        ))
    
    # Add final prompt for the next turn (consistent for all entries)
    prompt_parts.append(get_observations_at_history_point(
        player, history, all_players, game_config, len(history) - 1, history[-1].phase
    ))
    
    return "\n".join(prompt_parts)
