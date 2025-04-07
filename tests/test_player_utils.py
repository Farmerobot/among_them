import pytest
import random
from unittest.mock import patch
from typing import List, Dict

from among_them.models.action import Action, ActionType
from among_them.models.history import History, initialize_history # Changed import
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

@pytest.fixture
def player1():
    return Player(name="P1", role=PlayerRole.CREWMATE)

@pytest.fixture
def player2():
    return Player(name="P2", role=PlayerRole.CREWMATE)

@pytest.fixture
def player3():
    return Player(name="P3", role=PlayerRole.IMPOSTOR)

@pytest.fixture
def all_players(player1, player2, player3):
    return [player1, player2, player3]

@pytest.fixture
def alive_players(player1, player2, player3):
    # Assume all players start alive for most tests
    player1.alive = True
    player2.alive = True
    player3.alive = True
    return [player1, player2, player3]

@pytest.fixture
def base_action(player1):
    return Action(type=ActionType.WAIT, player_name=player1.name)

@pytest.fixture
def kill_action_p2(player3, player2):
    return Action(type=ActionType.KILL, player_name=player3.name, target_player_name=player2.name)

@pytest.fixture
def report_action(player1, player2):
    return Action(type=ActionType.REPORT, player_name=player1.name, target_player_name=player2.name)

@pytest.fixture
def move_action_p1_medbay(player1):
    return Action(type=ActionType.MOVE, player_name=player1.name, target_location=Location.MEDBAY)

@pytest.fixture
def move_action_p2_weapons(player2):
    return Action(type=ActionType.MOVE, player_name=player2.name, target_location=Location.WEAPONS)

@pytest.fixture
@patch('among_them.models.history.get_crewmate_tasks', return_value=[]) # Mock task assignment
@patch('among_them.models.history.get_impostor_tasks', return_value=[]) # Mock task assignment
def base_history_entry(mock_impostor_tasks, mock_crew_tasks, all_players, base_action):
    # Use initialize_history to get a valid starting point
    initial_hist = initialize_history(all_players)
    # Modify the first entry or create a new one based on it if needed for specific tests
    # For simplicity, we'll modify the action and next players for this fixture
    entry = initial_hist[0]
    entry.action_taken = base_action
    entry.player_names_to_play_next=[p.name for p in all_players if p.name != base_action.player_name]
    entry.phase = GamePhase.TASK # Set a relevant phase
    entry.actions_until_phase_ends = 10 # Set a relevant count
    entry.location = Location.CAFETERIA # Action location
    return entry


# --- Tests for get_next_random_player ---

@patch('random.choice')
def test_get_next_random_player_empty_history(mock_choice, alive_players, player1):
    # Mock random.choice to return the Player object itself when called with alive_players
    # When history is empty, it chooses from the list of Player objects.
    mock_choice.return_value = player1
    next_player, next_players_list = get_next_random_player([], alive_players)
    assert next_player.name == player1.name # Compare name
    assert next_player == player1 # Also check object equality
    assert len(next_players_list) == len(alive_players) - 1
    assert player1.name not in next_players_list

@patch('random.choice')
def test_get_next_random_player_with_remaining(mock_choice, base_history_entry, alive_players, player2):
    base_history_entry.player_names_to_play_next = ["P2", "P3"]
    mock_choice.return_value = "P2"
    next_player, next_players_list = get_next_random_player([base_history_entry], alive_players)
    assert next_player.name == player2.name
    assert next_players_list == ["P3"]
    mock_choice.assert_called_once_with(["P2", "P3"]) # Called with remaining players

@patch('random.choice')
def test_get_next_random_player_empty_remaining(mock_choice, base_history_entry, alive_players, player1):
    base_history_entry.player_names_to_play_next = []
    # Mock should return a name from the list random.choice will operate on
    expected_players_list = [p.name for p in alive_players]
    mock_choice.return_value = player1.name 
    next_player, next_players_list = get_next_random_player([base_history_entry], alive_players)
    assert next_player == player1
    # Verify choice was called with the correct list
    mock_choice.assert_called_once_with(expected_players_list)
    # Assert the returned list excludes the chosen player
    assert set(next_players_list) == set(p.name for p in alive_players if p != player1)


@patch('random.choice')
def test_get_next_random_player_after_report(mock_choice, base_history_entry, report_action, alive_players, player1):
    history = [base_history_entry, base_history_entry.copy(action_taken=report_action)]
    history[-1].player_names_to_play_next = ["P2", "P3"] # Should be ignored by the function logic
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
def test_get_next_random_player_removes_dead(mock_choice, base_history_entry, alive_players, player2, player3):
    # P1 is technically alive, but not in the alive_players list passed
    current_alive = [player2, player3]
    base_history_entry.player_names_to_play_next = ["P1", "P2", "P3"]
    mock_choice.return_value = "P2" # Example return

    next_player, next_players_list = get_next_random_player([base_history_entry], current_alive)

    assert next_player.name == player2.name
    # P1 should be removed from the list used by random.choice and the returned list
    assert set(next_players_list) == {"P3"}
    mock_choice.assert_called_once_with(["P2", "P3"])


# --- Tests for get_alive_players ---

def test_get_alive_players_no_kills(all_players):
    history = []
    alive = get_alive_players(history, all_players)
    assert len(alive) == 3
    assert set(p.name for p in alive) == {"P1", "P2", "P3"}

def test_get_alive_players_one_kill(all_players, base_history_entry, kill_action_p2):
    history = [base_history_entry, base_history_entry.copy(action_taken=kill_action_p2)]
    alive = get_alive_players(history, all_players)
    assert len(alive) == 2
    assert set(p.name for p in alive) == {"P1", "P3"}

def test_get_alive_players_multiple_kills(all_players, base_history_entry, kill_action_p2):
    kill_action_p1 = Action(type=ActionType.KILL, player_name="P3", target_player_name="P1")
    history = [
        base_history_entry,
        base_history_entry.copy(action_taken=kill_action_p2),
        base_history_entry.copy(action_taken=kill_action_p1)
    ]
    alive = get_alive_players(history, all_players)
    assert len(alive) == len(all_players) - 2
    assert set(p.name for p in alive) == {"P3"} # Compare names

# --- Tests for get_last_player_action ---

def test_get_last_player_action_found(base_history_entry, move_action_p1_medbay, player1):
    history = [
        base_history_entry, # P1 wait
        base_history_entry.copy(action_taken=Action(type=ActionType.MOVE, player_name="P2", target_location=Location.ADMIN)),
        base_history_entry.copy(action_taken=move_action_p1_medbay), # P1 move
        base_history_entry.copy(action_taken=Action(type=ActionType.MOVE, player_name="P3", target_location=Location.STORAGE)),
    ]
    last_action_entry = get_last_player_action(history, player1)
    assert last_action_entry.action_taken == move_action_p1_medbay

def test_get_last_player_action_only_first(base_history_entry, player1):
    history = [base_history_entry] # Only P1's action
    last_action_entry = get_last_player_action(history, player1)
    assert last_action_entry == base_history_entry

def test_get_last_player_action_not_found(base_history_entry, player2, player3):
    # History only contains P1's action
    history = [base_history_entry]
    # Player 2 never acted
    last_action_entry = get_last_player_action(history, player2)
    # Should return the first entry if player not found
    assert last_action_entry == base_history_entry

# Test for empty history is omitted as the current implementation would raise IndexError.
# If desired, the function could be modified to handle this (e.g., return None).


# --- Tests for get_dead_players ---

def test_get_dead_players_none_since_discussion(base_history_entry, player1):
    discuss_entry = base_history_entry.copy(
        phase=GamePhase.DISCUSS,
        action_taken=Action(type=ActionType.SPEAK, player_name=player1.name, text="Hi") # Changed type to SPEAK
    )
    history = [base_history_entry, discuss_entry]
    dead = get_dead_players(history, []) # players list not used by this func
    assert dead == {}

def test_get_dead_players_one_since_discussion(base_history_entry, kill_action_p2):
    discuss_entry = base_history_entry.copy(phase=GamePhase.DISCUSS)
    kill_entry = base_history_entry.copy(
        action_taken=kill_action_p2,
        location=Location.WEAPONS # Location where kill happened
    )
    history = [base_history_entry, discuss_entry, kill_entry]
    dead = get_dead_players(history, [])
    assert dead == {"P2": Location.WEAPONS.value}

def test_get_dead_players_kill_before_discussion(base_history_entry, kill_action_p2):
    discuss_entry = base_history_entry.copy(phase=GamePhase.DISCUSS)
    kill_entry = base_history_entry.copy(
        action_taken=kill_action_p2,
        location=Location.WEAPONS
    )
    # Kill happens *before* the last discussion phase starts
    history = [base_history_entry, kill_entry, discuss_entry]
    dead = get_dead_players(history, [])
    assert dead == {}

def test_get_dead_players_no_discussion_phase(base_history_entry, kill_action_p2):
    kill_entry = base_history_entry.copy(
        action_taken=kill_action_p2,
        location=Location.WEAPONS
    )
    history = [base_history_entry, kill_entry] # No DISCUSS phase yet
    dead = get_dead_players(history, [])
    # Should search from beginning if no discussion found
    assert dead == {"P2": Location.WEAPONS.value}


# --- Tests for get_players_in_room ---

def test_get_players_in_room_multiple(base_history_entry, all_players, alive_players, move_action_p1_medbay, player1, player2):
    # P1 moves to Medbay, P2 stays in Cafeteria, P3 starts Cafeteria (base)
    history = [
        base_history_entry, # P1 acts, all start Cafeteria
        base_history_entry.copy( # P2 acts, stays Cafeteria
            action_taken=Action(type=ActionType.WAIT, player_name=player2.name),
            location=Location.CAFETERIA
            ),
        base_history_entry.copy( # P1 acts, moves to Medbay
            action_taken=move_action_p1_medbay,
            location=Location.MEDBAY
            ),
         base_history_entry.copy( # P3 acts, stays Cafeteria
            action_taken=Action(type=ActionType.WAIT, player_name="P3"),
            location=Location.CAFETERIA
            ),
    ]
    in_cafeteria = get_players_in_room(history, all_players, Location.CAFETERIA)
    in_medbay = get_players_in_room(history, all_players, Location.MEDBAY)

    assert set(p.name for p in in_cafeteria) == {"P2", "P3"} # Compare names
    assert set(p.name for p in in_medbay) == {"P1"} # Compare names

def test_get_players_in_room_none(base_history_entry, all_players, alive_players):
    history = [base_history_entry] # All start in Cafeteria
    in_weapons = get_players_in_room(history, all_players, Location.WEAPONS)
    assert in_weapons == []

def test_get_players_in_room_ignores_dead(base_history_entry, kill_action_p2, all_players, alive_players):
    # P3 kills P2 in Cafeteria
    kill_entry = base_history_entry.copy(
        action_taken=kill_action_p2,
        location=Location.CAFETERIA
    )
    history = [
        base_history_entry, # All start Cafeteria
        kill_entry
    ]
    # P2 is dead, should not be included even though last action was in Cafeteria
    in_cafeteria = get_players_in_room(history, all_players, Location.CAFETERIA)
    # Expected: P1 (last action base_history_entry in Cafeteria) and P3 (last action kill_entry in Cafeteria)
    assert set(p.name for p in in_cafeteria) == {"P1", "P3"} # Compare names

def test_get_players_in_room_empty(base_history_entry, all_players):
    history = [base_history_entry]
    in_cafeteria = get_players_in_room(history, all_players, Location.CAFETERIA)
    assert set(p.name for p in in_cafeteria) == {"P1", "P2", "P3"} # Compare names
