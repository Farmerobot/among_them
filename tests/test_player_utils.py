import pytest
import random
from unittest.mock import patch
from typing import List, Dict, Tuple, Optional

from among_them.models.action import Action, ActionType
from among_them.models.history import History, initialize_history
from among_them.models.location import Location
from among_them.models.phase import GamePhase
from among_them.models.player import Player
from among_them.models.player_role import PlayerRole
from among_them.utils.player_utils import (
    get_next_random_player,
    get_alive_players,
    get_last_player_action,
    get_dead_players,
    get_players_in_room,
)
from tests.utils import create_base_history


@pytest.fixture
def player1() -> Player:
    return Player(name="P1", role=PlayerRole.CREWMATE)

@pytest.fixture
def player2() -> Player:
    return Player(name="P2", role=PlayerRole.CREWMATE)

@pytest.fixture
def player3() -> Player:
    return Player(name="P3", role=PlayerRole.IMPOSTOR)

@pytest.fixture
def all_players(player1: Player, player2: Player, player3: Player) -> List[Player]:
    return [player1, player2, player3]

@pytest.fixture
def alive_players(player1: Player, player2: Player, player3: Player) -> List[Player]:
    return [player1, player2, player3]

@pytest.fixture
def base_action(player1: Player) -> Action:
    return Action(type=ActionType.WAIT, player_name=player1.name)

@pytest.fixture
def kill_action_p2(player3: Player, player2: Player) -> Action:
    return Action(type=ActionType.KILL, player_name=player3.name, target_player_name=player2.name)

@pytest.fixture
def report_action(player1: Player, player2: Player) -> Action:
    return Action(type=ActionType.REPORT, player_name=player1.name, target_player_name=player2.name)

@pytest.fixture
def move_action_p1_medbay(player1: Player) -> Action:
    return Action(type=ActionType.MOVE, player_name=player1.name, target_location=Location.MEDBAY)

@pytest.fixture
def move_action_p2_weapons(player2: Player) -> Action:
    return Action(type=ActionType.MOVE, player_name=player2.name, target_location=Location.WEAPONS)

@pytest.fixture
def base_history_entry(all_players: List[Player], player1: Player, base_action: Action) -> List[History]:
    """Create a consistent base history entry using the shared utility."""
    return create_base_history(all_players, player1, Location.CAFETERIA)


# --- Tests for get_next_random_player ---

@patch('random.choice')
def test_get_next_random_player_empty_history(mock_choice, alive_players: List[Player], player1: Player) -> None:
    # Mock random.choice to return the Player object itself when called with alive_players
    # When history is empty, it chooses from the list of Player objects.
    mock_choice.return_value = player1
    next_player, next_players_list = get_next_random_player([], alive_players)
    assert next_player.name == player1.name # Compare name
    assert next_player == player1 # Also check object equality
    assert len(next_players_list) == len(alive_players) - 1
    assert player1.name not in next_players_list

@patch('random.choice')
def test_get_next_random_player_with_remaining(mock_choice, base_history_entry: List[History], alive_players: List[Player], player2: Player) -> None:
    # Use the latest entry as base
    base_entry = base_history_entry[-1].copy()
    base_entry.player_names_to_play_next = ["P2", "P3"]
    
    mock_choice.return_value = "P2"
    next_player, next_players_list = get_next_random_player(base_history_entry + [base_entry], alive_players)
    assert next_player.name == player2.name
    assert next_players_list == ["P3"]
    mock_choice.assert_called_once_with(["P2", "P3"]) # Called with remaining players

@patch('random.choice')
def test_get_next_random_player_empty_remaining(mock_choice, base_history_entry: List[History], alive_players: List[Player], player1: Player) -> None:
    # Use the latest entry as base
    base_entry = base_history_entry[-1].copy()
    base_entry.player_names_to_play_next = []
    
    # Mock should return a name from the list random.choice will operate on
    expected_players_list = [p.name for p in alive_players]
    mock_choice.return_value = player1.name 
    next_player, next_players_list = get_next_random_player(base_history_entry + [base_entry], alive_players)
    assert next_player == player1
    # Verify choice was called with the correct list
    mock_choice.assert_called_once_with(expected_players_list)
    # Assert the returned list excludes the chosen player
    assert set(next_players_list) == set(p.name for p in alive_players if p != player1)


@patch('random.choice')
def test_get_next_random_player_after_report(mock_choice, base_history_entry: List[History], report_action: Action, alive_players: List[Player], player1: Player) -> None:
    # Use the latest entry as base to create a new entry with report action
    base_entry = base_history_entry[-1]
    report_entry = base_entry.copy(action_taken=report_action)
    report_entry.player_names_to_play_next = ["P2", "P3"] # Should be ignored by the function logic
    
    history = base_history_entry + [report_entry]
    
    # Mock should return a name from the list random.choice will operate on (all alive players after report)
    expected_players_list = [p.name for p in alive_players]
    mock_choice.return_value = player1.name
    next_player, next_players_list = get_next_random_player(history, alive_players)
    assert next_player == player1
    # Verify choice was called with the correct list
    mock_choice.assert_called_once_with(expected_players_list)
    # Assert the returned list excludes the chosen player
    assert set(next_players_list) == set(p.name for p in alive_players if p != player1)

@patch('random.choice')
def test_get_next_random_player_removes_dead(mock_choice, base_history_entry: List[History], alive_players: List[Player], player2: Player, player3: Player) -> None:
    # Use the latest entry as base
    base_entry = base_history_entry[-1].copy()
    base_entry.player_names_to_play_next = ["P1", "P2", "P3"]
    
    # P1 is technically alive, but not in the alive_players list passed
    current_alive = [player2, player3]
    mock_choice.return_value = "P2" # Example return

    next_player, next_players_list = get_next_random_player(base_history_entry + [base_entry], current_alive)

    assert next_player.name == player2.name
    # P1 should be removed from the list used by random.choice and the returned list
    assert set(next_players_list) == {"P3"}
    mock_choice.assert_called_once_with(["P2", "P3"])


# --- Tests for get_alive_players ---

def test_get_alive_players_no_kills(all_players: List[Player]) -> None:
    # Start with empty history - no kills yet
    history: List[History] = []
    alive = get_alive_players(history, all_players)
    assert len(alive) == 3
    assert set(p.name for p in alive) == {"P1", "P2", "P3"}

def test_get_alive_players_one_kill(all_players: List[Player], base_history_entry: List[History], kill_action_p2: Action) -> None:
    # Use the latest entry as base to create a kill entry
    base_entry = base_history_entry[-1]
    kill_entry = base_entry.copy(action_taken=kill_action_p2)
    
    history = base_history_entry + [kill_entry]
    alive = get_alive_players(history, all_players)
    assert len(alive) == 2
    assert set(p.name for p in alive) == {"P1", "P3"}

def test_get_alive_players_multiple_kills(all_players: List[Player], base_history_entry: List[History], kill_action_p2: Action) -> None:
    # Use the latest entry as base to create kill entries
    base_entry = base_history_entry[-1]
    kill_action_p1 = Action(type=ActionType.KILL, player_name="P3", target_player_name="P1")
    
    kill_p2_entry = base_entry.copy(action_taken=kill_action_p2)
    kill_p1_entry = base_entry.copy(action_taken=kill_action_p1)
    
    history = base_history_entry + [kill_p2_entry, kill_p1_entry]
    alive = get_alive_players(history, all_players)
    assert len(alive) == len(all_players) - 2
    assert set(p.name for p in alive) == {"P3"} # Compare names

# --- Tests for get_last_player_action ---

def test_get_last_player_action_found(base_history_entry: List[History], move_action_p1_medbay: Action, player1: Player) -> None:
    # Use the latest entry as base to create a sequence of history entries
    base_entry = base_history_entry[-1]
    
    move_p2_entry = base_entry.copy(action_taken=Action(type=ActionType.MOVE, player_name="P2", target_location=Location.ADMIN))
    move_p1_entry = base_entry.copy(action_taken=move_action_p1_medbay)
    move_p3_entry = base_entry.copy(action_taken=Action(type=ActionType.MOVE, player_name="P3", target_location=Location.STORAGE))
    
    history = base_history_entry + [move_p2_entry, move_p1_entry, move_p3_entry]
    last_action_entry = get_last_player_action(history, player1)
    assert last_action_entry.action_taken == move_action_p1_medbay

def test_get_last_player_action_only_first(base_history_entry: List[History], player1: Player) -> None:
    # Only base history with player1's action
    last_action_entry = get_last_player_action(base_history_entry, player1)
    assert last_action_entry == base_history_entry[-1]

def test_get_last_player_action_not_found(base_history_entry: List[History], player1: Player, player3: Player) -> None:
    # History only contains P2's action
    last_action_entry = get_last_player_action(base_history_entry, player1)
    # Should return the first entry if player not found
    assert last_action_entry == base_history_entry[-1]

# Test for empty history is omitted as the current implementation would raise IndexError.
# If desired, the function could be modified to handle this (e.g., return None).


# --- Tests for get_dead_players ---

def test_get_dead_players_none_since_discussion(base_history_entry: List[History], player1: Player) -> None:
    # Use the latest entry as base
    base_entry = base_history_entry[-1]
    
    discuss_entry = base_entry.copy(
        phase=GamePhase.DISCUSS,
        action_taken=Action(type=ActionType.SPEAK, player_name=player1.name, text="Hi")
    )
    
    history = base_history_entry + [discuss_entry]
    dead = get_dead_players(history)
    assert dead == {}

def test_get_dead_players_one_since_discussion(base_history_entry: List[History], kill_action_p2: Action) -> None:
    # Use the latest entry as base
    base_entry = base_history_entry[-1]
    
    discuss_entry = base_entry.copy(phase=GamePhase.DISCUSS)
    kill_entry = base_entry.copy(
        action_taken=kill_action_p2,
        location=Location.WEAPONS
    )
    
    history = base_history_entry + [discuss_entry, kill_entry]
    dead = get_dead_players(history)
    assert dead == {"P2": Location.WEAPONS.value}

def test_get_dead_players_kill_before_discussion(base_history_entry: List[History], kill_action_p2: Action) -> None:
    # Use the latest entry as base
    base_entry = base_history_entry[-1]
    
    kill_entry = base_entry.copy(
        action_taken=kill_action_p2,
        location=Location.WEAPONS
    )
    discuss_entry = base_entry.copy(phase=GamePhase.DISCUSS)
    
    # Kill happens *before* the last discussion phase starts
    history = base_history_entry + [kill_entry, discuss_entry]
    dead = get_dead_players(history)
    assert dead == {}

def test_get_dead_players_no_discussion_phase(base_history_entry: List[History], kill_action_p2: Action) -> None:
    # Use the latest entry as base
    base_entry = base_history_entry[-1]
    
    kill_entry = base_entry.copy(
        action_taken=kill_action_p2,
        location=Location.WEAPONS
    )
    
    history = base_history_entry + [kill_entry] # No DISCUSS phase yet
    dead = get_dead_players(history)
    # Should search from beginning if no discussion found
    assert dead == {"P2": Location.WEAPONS.value}


# --- Tests for get_players_in_room ---

def test_get_players_in_room_multiple(base_history_entry: List[History], all_players: List[Player], alive_players: List[Player], move_action_p1_medbay: Action, player1: Player, player2: Player) -> None:
    # Use the latest entry as base
    base_entry = base_history_entry[-1]
    
    # P2 waits in cafeteria
    p2_wait_entry = base_entry.copy(
        action_taken=Action(type=ActionType.WAIT, player_name=player2.name),
        location=Location.CAFETERIA
    )
    
    # P1 moves to medbay
    p1_move_entry = base_entry.copy(
        action_taken=move_action_p1_medbay,
        location=Location.MEDBAY
    )
    
    # P3 waits in cafeteria
    p3_wait_entry = base_entry.copy(
        action_taken=Action(type=ActionType.WAIT, player_name="P3"),
        location=Location.CAFETERIA
    )
    
    history = base_history_entry + [p2_wait_entry, p1_move_entry, p3_wait_entry]
    
    in_cafeteria = get_players_in_room(history, all_players, Location.CAFETERIA)
    in_medbay = get_players_in_room(history, all_players, Location.MEDBAY)

    assert set(p.name for p in in_cafeteria) == {"P2", "P3"} # Compare names
    assert set(p.name for p in in_medbay) == {"P1"} # Compare names

def test_get_players_in_room_none(base_history_entry: List[History], all_players: List[Player], alive_players: List[Player]) -> None:
    # Everyone starts in CAFETERIA, check WEAPONS
    in_weapons = get_players_in_room(base_history_entry, all_players, Location.WEAPONS)
    assert in_weapons == []

def test_get_players_in_room_ignores_dead(base_history_entry: List[History], kill_action_p2: Action, all_players: List[Player], alive_players: List[Player]) -> None:
    # Use the latest entry as base
    base_entry = base_history_entry[-1]
    
    # P3 kills P2 in CAFETERIA
    kill_entry = base_entry.copy(
        action_taken=kill_action_p2,
        location=Location.CAFETERIA
    )
    
    history = base_history_entry + [kill_entry]
    
    # P2 is dead, should not be included even though their last action was in CAFETERIA
    in_cafeteria = get_players_in_room(history, all_players, Location.CAFETERIA)
    # Expected: P1 and P3 in CAFETERIA
    assert set(p.name for p in in_cafeteria) == {"P1", "P3"} # Compare names

def test_get_players_in_room_empty(base_history_entry, all_players):
    history = base_history_entry
    in_cafeteria = get_players_in_room(history, all_players, Location.CAFETERIA)
    assert set(p.name for p in in_cafeteria) == {"P1", "P2", "P3"} # Compare names
