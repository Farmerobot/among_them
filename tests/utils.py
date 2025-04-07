"""Shared utilities for Among Them tests."""
from typing import List, Optional
from unittest.mock import patch, MagicMock

from among_them.models.action import Action, ActionType
from among_them.models.history import History, initialize_history
from among_them.models.location import Location
from among_them.models.phase import GamePhase
from among_them.models.player import Player
from among_them.models.tasks import Task
from among_them.consts import IMPOSTOR_COOLDOWN


def create_base_history(
    all_players: List[Player], 
    acting_player: Player, 
    location: Location = Location.CAFETERIA,
    tasks: Optional[List[Task]] = None,
    phase: GamePhase = GamePhase.TASK,
    actions_until_phase_ends: int = 10
) -> List[History]:
    """
    Create a consistent base history for tests.
    
    Args:
        all_players: List of all players in the game
        acting_player: The player who performed the initial action
        location: Current location (default: CAFETERIA)
        tasks: Optional list of tasks for the acting player
        phase: Game phase (default: TASK)
        actions_until_phase_ends: Number of actions until phase ends (default: 10)
        
    Returns:
        List[History]: A list containing the initial history entry and a subsequent
                      action entry for the acting player
    """
    # Create a task in the current location
    cafeteria_task = Task(name="Fix something", location=location)
    
    # Start with initialized history
    initial_hist = initialize_history(all_players)
    first_entry = initial_hist[0]
    
    # Create a basic action for the acting player (usually WAIT)
    base_action = Action(
        type=ActionType.WAIT,
        player_name=acting_player.name,
        spectator=f"{acting_player.name} waited"
    )
    
    # Update tasks if provided
    tasks_left = first_entry.tasks_left_to_do.copy()
    
    # Manually assign tasks to each player
    for player in all_players:
        tasks_left[player.name] = [cafeteria_task]
    
    # Override with provided tasks if any
    if tasks:
        tasks_left[acting_player.name] = tasks
    
    # Get a list of spectators from alive players
    spectators = [p.name for p in all_players]
    
    # Create a new history entry
    move_history = History(
        player_names_to_play_next=[p.name for p in all_players if p.name != acting_player.name],
        phase=phase,
        actions_until_phase_ends=actions_until_phase_ends,
        location=location,
        impostor_cooldown=IMPOSTOR_COOLDOWN,
        actions_agent_could_take=[],
        spectators_who_saw=spectators,
        llm_cot="",
        llm_response="",
        token_usage={},
        action_taken=base_action,
        tasks_left_to_do=tasks_left,
        votes_before_this_discussion_message={}
    )
    
    initial_hist.append(move_history)
    return initial_hist
