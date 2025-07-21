"""
Environment-like prompt building utilities for Among Them game.

This module implements the new prompt system that follows research principles:
- Cohesive observation-action concatenation
- Proper use of articles in action prompts  
- Environment context includes available moves and visible players
- LLM generations are appended unchanged to maintain context
"""

from typing import List, Optional, Dict, Set
from among_them.models.player import Player
from among_them.models.history import History
from among_them.models.location import Location, get_map
from among_them.models.action import Action
from among_them.models.player_role import PlayerRole
from among_them.models.phase import GamePhase
from among_them.game_config import GameConfig
from among_them.utils.player_utils import get_players_in_room, get_last_player_action
from among_them.utils.phase_utils import get_phase_and_when_it_ends
from among_them.llm_prompts import UNIVERSAL_SYSTEM_PROMPT


def get_available_doors(current_location: Location, game_config: GameConfig) -> List[str]:
    """Get list of locations accessible from current location using proper map."""
    doors_map, _, _ = get_map(game_config.map_size)
    connected_locations = doors_map.get(current_location, [])
    return [loc.value.lower() for loc in connected_locations]


def get_current_observations(
    player: Player, 
    history: List[History], 
    all_players: List[Player],
    game_config: GameConfig,
    phase: Optional[GamePhase] = None
) -> str:
    """Generate current environmental observations for a player."""
    if not history:
        return ""
    
    # Simply call get_observations_at_history_point with the latest index
    return get_observations_at_history_point(
        player, history, all_players, game_config, len(history) - 1, phase
    )


def get_player_location_at_history_point(
    player: Player,
    history: List[History],
    history_index: int
) -> Location:
    """Get a player's location at a specific point in history."""
    if history_index < 0 or not history:
        return Location.CAFETERIA
    
    # Use the existing function to get player's last action up to this point
    history_slice = history[:history_index + 1]
    last_action = get_last_player_action(history_slice, player)
    
    return last_action.location


def get_observations_at_history_point(
    player: Player,
    history: List[History],
    all_players: List[Player], 
    game_config: GameConfig,
    history_index: int,
    phase: Optional[GamePhase] = None
) -> str:
    """Get environmental observations for a player at a specific point in game history."""
    if history_index >= len(history):
        return ""
    
    state_at_point = history[history_index]
    observations = []
    
    # Get current phase if not provided
    if phase is None:
        try:
            phase, _ = get_phase_and_when_it_ends(history[:history_index+1], game_config, all_players)
        except:
            phase = GamePhase.TASK
    
    # Get the player's actual location (not the action location!)
    player_location = get_player_location_at_history_point(player, history, history_index)
    
    # Current location
    observations.append(f"You are in {player_location.value}.")
    
    # Available doors
    doors = get_available_doors(player_location, game_config)
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
    
    # Impostor-specific observations
    if player.role == PlayerRole.IMPOSTOR:
        cooldown = state_at_point.impostor_cooldown
        if cooldown > 0:
            observations.append(f"Your kill cooldown: wait {cooldown} more actions before you can kill.")
        else:
            observations.append("You can kill now.")
    
    # Task-related observations for crewmates
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
    
    # Phase-specific information and rules
    if phase == GamePhase.TASK:
        observations.append("Phase: Task phase. You cannot speak during this phase.")
    elif phase == GamePhase.DISCUSS:
        observations.append("Phase: Discussion phase. You can speak now. Respond to the crewmates.")
    elif phase == GamePhase.VOTING:
        observations.append("Phase: Voting phase. Choose who to vote for.")
    
    return "\n".join(observations)


def get_player_context(player: Player, history: List[History], all_players: List[Player], game_config: GameConfig) -> str:
    """Generate player context section including role-specific information."""
    context = []
    context.append(f"You are {player.name}.")
    
    if not history:
        return "\n".join(context)
    
    alive_players = [p for p in all_players if p.name in history[-1].alive_player_names]
    
    if player.role == PlayerRole.IMPOSTOR:
        context.append("You are an impostor. Your goal is to eliminate crewmates without being detected.")
        context.append(f"Your kill cooldown is {game_config.impostor_cooldown} actions.")
        
        # Other impostors
        other_impostors = [p for p in alive_players if p.role == PlayerRole.IMPOSTOR and p.name != player.name]
        if other_impostors:
            impostor_names = [imp.name for imp in other_impostors]
            context.append(f"Your fellow impostors: {', '.join(impostor_names)}.")
        else:
            context.append("You are the only impostor.")
        
        # Crewmates to eliminate
        other_crewmates = [p for p in alive_players if p.role == PlayerRole.CREWMATE]
        if other_crewmates:
            crewmate_names = [crew.name for crew in other_crewmates]
            context.append(f"Crewmates to eliminate: {', '.join(crewmate_names)}.")
    else:
        context.append("You are a crewmate. Your goal is to complete tasks and identify impostors.")
        
        # Impostor count and potential impostors
        other_impostors = [p for p in alive_players if p.role == PlayerRole.IMPOSTOR]
        impostor_count = len(other_impostors)
        context.append(f"There are {impostor_count} impostors among the players.")
        
        other_players = [p.name for p in alive_players if p.name != player.name]
        if other_players:
            context.append(f"Potential impostors: {', '.join(other_players)}.")
        
        # Tasks remaining
        if history[-1].tasks_left_to_do and player.name in history[-1].tasks_left_to_do:
            tasks = history[-1].tasks_left_to_do[player.name]
            if tasks:
                task_descriptions = [f"{task.name} (in {task.location.value})" for task in tasks]
                context.append(f"Tasks remaining: {', '.join(task_descriptions)}.")
            else:
                context.append("All tasks completed!")
    
    return "\n".join(context)


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
    
    # Game initialization
    prompt_parts.append("Game started.")
    
    # Reconstruct the environment flow from history
    for i, hist_entry in enumerate(history):
        # Skip the initial game start entry
        if i == 0:
            continue
        
        # Check if this was the player's turn
        if hist_entry.action_taken.player_name == player.name:
            # Before the player's action, add current observations and action prompt
            # (This recreates the environment context that was presented to the player)
            current_obs_at_turn = get_observations_at_history_point(player, history, all_players, game_config, i-1)
            if current_obs_at_turn:
                prompt_parts.append(current_obs_at_turn)
            prompt_parts.append("Given the situation, the best action to take is:")
            
            # Add LLM generation if this was the player's turn and we have the response
            if hist_entry.llm_response and hist_entry.llm_cot:
                # Add the player's LLM output (think + action tags)
                llm_output = hist_entry.llm_cot + hist_entry.llm_response
                prompt_parts.append(llm_output)
        
        # Add observations about what happened (for everyone)
        llm_cot_for_action = hist_entry.llm_cot if hist_entry.action_taken.player_name == player.name else ""
        action_obs = generate_action_observations(
            hist_entry.action_taken,
            player,
            next(p for p in all_players if p.name == hist_entry.action_taken.player_name),
            hist_entry.spectators_who_saw,
            llm_cot_for_action
        )
        if action_obs:
            prompt_parts.append(action_obs)
    
    # Add final current observations and action prompt for the next turn
    current_obs = get_current_observations(player, history, all_players, game_config)
    if current_obs:
        prompt_parts.append(current_obs)
    
    # Final action prompt for the next turn
    prompt_parts.append("Given the situation, the best action to take is:")
    
    return "\n".join(prompt_parts)


def build_environment_prompt(
    player: Player,
    history: List[History], 
    all_players: List[Player],
    game_config: GameConfig,
    previous_llm_generations: List[str] = None
) -> str:
    """
    Build the complete environment-like prompt for a player.
    
    Args:
        player: The player this prompt is for
        history: Game history entries
        all_players: All players in the game
        game_config: Game configuration
        previous_llm_generations: List of previous LLM outputs to include (deprecated)
        
    Returns:
        Complete prompt string ready for LLM
    """
    # Use the history-based reconstruction for full context
    return reconstruct_environment_prompt_from_history(
        player, history, all_players, game_config
    )


def generate_action_observations(
    action: Action, 
    observing_player: Player, 
    acting_player: Player,
    spectators_who_saw: List[str],
    llm_cot: str = ""
) -> str:
    """
    Generate observations for a player based on an action that occurred.
    
    Args:
        action: The action that was taken
        observing_player: The player who is observing
        acting_player: The player who took the action  
        spectators_who_saw: List of players who saw the action
        llm_cot: The LLM chain of thought if this was the observing player's action
        
    Returns:
        Observation string for the observing player
    """
    observations = []
    
    if observing_player.name == acting_player.name:
        # Player took the action themselves
        # Add their thought process if available (convert think tags to thought tags)
        if llm_cot:
            cot_with_thought_tags = llm_cot.replace("<think>", "<thought>").replace("</think>", "</thought>")
            observations.append(f"Your thoughts: {cot_with_thought_tags}")
        
        # Add their action result
        if action.type == ActionType.SPEAK:
            observations.append(f"You said: {action.spectator}")  # Use spectator for speech content
        else:
            observations.append(action.agent_perspective or action.result)
    else:
        # Someone else took the action
        can_observe = (observing_player.name in spectators_who_saw)
        
        if can_observe:
            if action.type == ActionType.SPEAK:
                # Discussion - use spectator field for speech content
                observations.append(f"Discussion: {action.spectator}")
            elif action.type == ActionType.VOTE:
                # Players cannot see others voting
                return ""
            else:
                # Observed action
                observations.append(action.observer_perspective or action.spectator)
        else:
            # Player didn't see this action
            observations.append(f"You did not see {acting_player.name} during this time.")
    
    return "\n".join(observations) if observations else ""


def append_llm_generation_to_prompt(base_prompt: str, llm_output: str) -> str:
    """Append LLM generation to existing prompt unchanged."""
    return base_prompt + "\n" + llm_output


def append_new_observations_to_prompt(base_prompt: str, observations: str) -> str:
    """Append new environmental observations to prompt."""
    if observations.strip():
        return base_prompt + "\n" + observations
    return base_prompt
