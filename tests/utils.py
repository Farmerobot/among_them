"""Shared utilities for Among Them tests."""
from typing import Dict, List, Optional

from among_them.models.action import Action
from among_them.models.game_config import GameConfig
from among_them.models.history import History
from among_them.models.location import Location
from among_them.models.phase import GamePhase
from among_them.models.player import Player
from among_them.models.tasks import Task


def create_history_entry(
    all_players: List[Player], 
    acting_player: Player, 
    action: Action,
    location: Location = Location.CAFETERIA,
    tasks_left_to_do: Optional[Dict[str, List[Task]]] = None,
    phase: GamePhase = GamePhase.TASK,
    actions_until_phase_ends: int = 10,
    impostor_cooldown: int = 0,
    game_config: GameConfig = GameConfig()
) -> History:
    """
    Create a single history entry for tests.
    
    Args:
        all_players: List of all players in the game
        acting_player: The player who performed the action
        action: The action performed by the acting player
        location: Current location (default: CAFETERIA)
        tasks_left_to_do: Dictionary mapping player names to their tasks
        phase: Game phase (default: TASK)
        actions_until_phase_ends: Number of actions until phase ends (default: 10)
        impostor_cooldown: Cooldown for impostor actions (default: 0)
        game_config: Game configuration (default: default GameConfig)
        
    Returns:
        History: A single history entry with the specified parameters
    """
    # Create a default task in CAFETERIA if tasks not provided
    default_task = Task(name="Fix something", location=Location.CAFETERIA)
    
    # Initialize tasks for each player if not provided
    if tasks_left_to_do is None:
        tasks_left_to_do = {}
        for player in all_players:
            tasks_left_to_do[player.name] = [default_task]
    
    # Get a list of spectators (all players in the location)
    spectators = [p.name for p in all_players]
    
    # Create the next players to play (everyone except the acting player)
    next_players = [p.name for p in all_players if p.name != acting_player.name]
    
    # Create the history entry
    history_entry = History(
        player_names_to_play_next=next_players,
        phase=phase,
        actions_until_phase_ends=actions_until_phase_ends,
        location=location,
        impostor_cooldown=impostor_cooldown,
        actions_agent_could_take=[],
        spectators_who_saw=spectators,
        llm_cot="",
        llm_response="",
        token_usage={},
        action_taken=action,
        tasks_left_to_do=tasks_left_to_do,
        votes_before_this_discussion_message={}
    )
    
    return history_entry
