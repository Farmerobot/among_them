from typing import List

from among_them.models.action_type import ActionType
from among_them.game_config import GameConfig
from among_them.models.history import History
from among_them.models.location import Location
from among_them.models.player import Player
from among_them.models.tasks import Task
from among_them.utils.action_utils import (get_task_phase_actions,
                                           get_vote_actions)

# --- Tests for get_task_phase_actions ---

def test_get_task_phase_actions_base(crewmate_player: Player, cafeteria_location: Location,
                                     crewmate_wait_history: History, initial_history: List[History],
                                     generic_test_players: List[Player], game_config: GameConfig):
    """Test basic actions available to a crewmate."""
    acting_player = crewmate_player
    current_location = cafeteria_location  # For readability in the test

    # Use the crewmate_wait_history fixture
    history = [initial_history, crewmate_wait_history]

    actions = get_task_phase_actions(acting_player, current_location, 0, history, generic_test_players, game_config)
    action_types = {action.type for action in actions}

    assert ActionType.WAIT in action_types
    # Check move actions based on DOORS from CAFETERIA
    assert any(a.type == ActionType.MOVE and a.target_location == Location.WEAPONS for a in actions)
    assert any(a.type == ActionType.MOVE and a.target_location == Location.ADMIN for a in actions)
    assert any(a.type == ActionType.MOVE and a.target_location == Location.MEDBAY for a in actions)


def test_get_task_phase_actions_report(crewmate_player: Player, cafeteria_location: Location,
                                       impostor_kill_history: History,
                                       all_generic_test_players: List[Player], dead_player: Player, initial_history: History,
                                       game_config: GameConfig):
    """Test that REPORT action is available when a dead body is present."""
    # Build history with the fixtures and custom actions
    history = [
        initial_history,
        impostor_kill_history,  # Impostor kills DeadPlayer
    ]

    # Call the function with the constructed history
    actions = get_task_phase_actions(crewmate_player, cafeteria_location, 0, history, all_generic_test_players, game_config)
    report_actions = [a for a in actions if a.type == ActionType.REPORT]
    assert len(report_actions) == 1, "REPORT action should be available"
    assert report_actions[0].target_player_name == dead_player.name  # Check it reports the correct player


def test_get_task_phase_actions_no_report_if_body_elsewhere(crewmate_player: Player, weapons_location: Location, crewmate_move_history: History,
                                                           impostor_kill_history: History, all_generic_test_players: List[Player], 
                                                           initial_history: History, game_config: GameConfig):
    """Test REPORT is not available if the body is in a different location."""
    # Build history using fixtures
    history = [
        initial_history,
        crewmate_move_history,  # Crewmate moves to medbay
        impostor_kill_history   # Impostor kills in cafeteria
    ]

    actions = get_task_phase_actions(crewmate_player, weapons_location, 0, history, all_generic_test_players, game_config)
    report_actions = [a for a in actions if a.type == ActionType.REPORT]
    assert len(report_actions) == 0, "REPORT action should NOT be available"


def test_get_task_phase_actions_task(crewmate_player: Player, cafeteria_location: Location,
                                     initial_history: History, impostor_move_history: History,
                                     task_in_cafeteria: Task,
                                     all_generic_test_players: List[Player], game_config: GameConfig):
    """Test DO_TASK action is available when a task is at the current location."""
    history = [initial_history, impostor_move_history]

    actions = get_task_phase_actions(crewmate_player, cafeteria_location, 0, history, all_generic_test_players, game_config)
    task_actions = [a for a in actions if a.type == ActionType.TASK]

    assert len(task_actions) == 1
    assert task_actions[0].target_task.name == task_in_cafeteria.name


def test_get_task_phase_actions_no_task_if_elsewhere(crewmate_player: Player, medbay_location: Location,
                                                     initial_history: History, crewmate_move_history: History,
                                                     impostor_move_history: History,
                                                     all_generic_test_players: List[Player], game_config: GameConfig):
    """Test DO_TASK is not available if the task is elsewhere."""
    # Build history using fixtures
    history = [
        initial_history,  # Initial state in cafeteria
        crewmate_move_history,  # Moved to medbay
        impostor_move_history,  # Impostor moved to weapons
    ]

    actions = get_task_phase_actions(crewmate_player, medbay_location, 0, history, all_generic_test_players, game_config)
    task_actions = [a for a in actions if a.type == ActionType.TASK]
    assert len(task_actions) == 0


def test_get_task_phase_actions_impostor_kill_available(impostor_player: Player, crewmate_player: Player,
                                                      cafeteria_location: Location, initial_history: History,
                                                      impostor_wait_history: History, crewmate_wait_history: History,
                                                      impostor_kill_history: History,
                                                      generic_test_players: List[Player], game_config: GameConfig):
    """Test KILL action is available for impostor when cooldown is 0 and target is present."""
    # Build history using fixtures
    history = [
        initial_history,  # Initial state with crewmate in cafeteria
        impostor_kill_history, # Impostor has just killed DeadPlayer
        crewmate_wait_history, # Crewmate is not reporting
        impostor_wait_history, # Impostor has cooldown
        crewmate_wait_history # Crewmate is not reporting
    ]

    actions = get_task_phase_actions(impostor_player, cafeteria_location, 0, history, generic_test_players, game_config)
    kill_actions = [a for a in actions if a.type == ActionType.KILL]
    
    assert len(kill_actions) >= 1, "KILL action should be available"
    assert kill_actions[0].target_player_name == crewmate_player.name


def test_get_task_phase_actions_impostor_kill_cooldown(impostor_player: Player, crewmate_player: Player,
                                                      cafeteria_location: Location, initial_history: History,
                                                      impostor_kill_history: History, crewmate_wait_history: History,
                                                      generic_test_players: List[Player], game_config: GameConfig):
    """Test KILL action is NOT available for impostor when cooldown > 0."""
    acting_player = impostor_player
    current_location = cafeteria_location

    # Build history using fixtures - impostor_kill_history has cooldown > 0
    history = [
        initial_history,  # Initial state
        impostor_kill_history,  # Impostor has just killed, so cooldown > 0
        crewmate_wait_history  # Crewmate is not reporting
    ]

    actions = get_task_phase_actions(acting_player, current_location, 1, history, generic_test_players, game_config)
    kill_actions = [a for a in actions if a.type == ActionType.KILL]
    
    assert len(kill_actions) == 0, "KILL action should NOT be available due to cooldown"


def test_get_task_phase_actions_impostor_kill_target_elsewhere(impostor_player: Player, crewmate_player: Player,
                                                               weapons_location: Location, impostor_move_history: History,
                                                               crewmate_move_history: History, initial_history: History,
                                                               generic_test_players: List[Player], game_config: GameConfig):
    """Test KILL action is NOT available for impostor if target is elsewhere."""
    # Build history using fixtures
    history = [
        initial_history,   # Initial state
        impostor_move_history,   # Impostor moves to weapons
        crewmate_move_history    # Crewmate moves to medbay (different location)
    ]

    actions = get_task_phase_actions(impostor_player, weapons_location, 0, history, generic_test_players, game_config)
    kill_actions = [a for a in actions if a.type == ActionType.KILL]
    
    assert len(kill_actions) == 0, "KILL action should NOT be available when target is elsewhere"


def test_get_task_phase_actions_impostor_pretend(impostor_player: Player, cafeteria_location: Location, 
                                                initial_history: History, generic_test_players: List[Player],
                                                game_config: GameConfig):
    """Test PRETEND action is available for impostor."""
    # Use the initial_history fixture
    history = [initial_history]

    actions = get_task_phase_actions(impostor_player, cafeteria_location, 0, history, generic_test_players, game_config)
    pretend_actions = [a for a in actions if a.type == ActionType.PRETEND]
    
    assert len(pretend_actions) > 0, "PRETEND action should be available for impostor"


def test_get_task_phase_actions_crewmate_no_impostor_actions(crewmate_player: Player, cafeteria_location: Location, 
                                                             initial_history: History, generic_test_players: List[Player],
                                                             game_config: GameConfig):
    """Test that crewmates cannot KILL or PRETEND_TASK."""

    # Use the initial_history fixture
    history = [initial_history]

    actions = get_task_phase_actions(crewmate_player, cafeteria_location, 0, history, generic_test_players, game_config)
    kill_actions = [a for a in actions if a.type == ActionType.KILL]
    pretend_actions = [a for a in actions if a.type == ActionType.PRETEND]
    
    assert len(kill_actions) == 0, "KILL action should NOT be available for crewmate"
    assert len(pretend_actions) == 0, "PRETEND action should NOT be available for crewmate"


# --- Tests for get_vote_actions ---

def test_get_vote_actions(crewmate_player: Player, generic_test_players: List[Player], dead_player: Player):
    """Test available VOTE actions, including voting for alive players and nobody."""
    # Get the vote actions
    vote_actions = get_vote_actions(generic_test_players, crewmate_player)
    
    # Check that there's a vote action for each alive player except self
    alive_player_no_self_names = [p.name for p in generic_test_players if p.name != crewmate_player.name and p.name != dead_player.name]
    vote_target_names = [a.target_player_name for a in vote_actions]
    
    # Check that we can vote for each alive player
    for player_name in alive_player_no_self_names:
        assert player_name in vote_target_names, f"Should be able to vote for {player_name}"
    assert "nobody" in vote_target_names, "Should be able to vote for nobody"
    assert dead_player.name not in vote_target_names, f"Should not be able to vote for dead player {dead_player.name}"
    assert crewmate_player.name not in vote_target_names, "Should not be able to vote for self"

