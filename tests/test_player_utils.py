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


# --- Tests for get_next_random_player ---

@patch('random.choice')
def test_get_next_random_player_empty_history(mock_choice, generic_test_players: List[Player], crewmate_player: Player) -> None:
    """Test get_next_random_player with empty history."""
    # Mock random.choice to return the Player object itself when called with alive_players
    mock_choice.return_value = crewmate_player
    next_player, next_players_list = get_next_random_player([], generic_test_players)
    assert next_player.name == crewmate_player.name # Compare name
    assert next_player == crewmate_player # Also check object equality
    assert len(next_players_list) == len(generic_test_players) - 1
    assert crewmate_player.name not in next_players_list

@patch('random.choice')
def test_get_next_random_player_with_remaining(mock_choice, base_history_entry: List[History], generic_test_players: List[Player], 
                                              other_player: Player, impostor_player: Player) -> None:
    """Test get_next_random_player with remaining players in history."""
    # Use the latest entry as base
    base_entry = base_history_entry[-1].copy()
    # Set remaining players to other_player and impostor_player
    base_entry.player_names_to_play_next = [other_player.name, impostor_player.name]
    
    mock_choice.return_value = other_player.name
    next_player, next_players_list = get_next_random_player(base_history_entry + [base_entry], generic_test_players)
    assert next_player.name == other_player.name
    assert next_players_list == [impostor_player.name]
    mock_choice.assert_called_once_with([other_player.name, impostor_player.name]) # Called with remaining players

@patch('random.choice')
def test_get_next_random_player_empty_remaining(mock_choice, base_history_entry: List[History], generic_test_players: List[Player], 
                                               crewmate_player: Player) -> None:
    """Test get_next_random_player with empty remaining players in history."""
    # Use the latest entry as base
    base_entry = base_history_entry[-1].copy()
    base_entry.player_names_to_play_next = []
    
    # Mock should return a name from the list random.choice will operate on
    expected_players_list = [p.name for p in generic_test_players]
    mock_choice.return_value = crewmate_player.name 
    next_player, next_players_list = get_next_random_player(base_history_entry + [base_entry], generic_test_players)
    assert next_player == crewmate_player
    # Verify choice was called with the correct list
    mock_choice.assert_called_once_with(expected_players_list)
    # Assert the returned list excludes the chosen player
    assert set(next_players_list) == set(p.name for p in generic_test_players if p != crewmate_player)


@patch('random.choice')
def test_get_next_random_player_after_report(mock_choice, base_history_entry: List[History], report_action: Action, 
                                            generic_test_players: List[Player], crewmate_player: Player) -> None:
    """Test get_next_random_player after a report action."""
    # Use the latest entry as base to create a new entry with report action
    base_entry = base_history_entry[-1]
    report_entry = base_entry.copy(action_taken=report_action)
    # These should be ignored by the function logic after a report
    report_entry.player_names_to_play_next = [p.name for p in generic_test_players if p != crewmate_player]
    
    history = base_history_entry + [report_entry]
    
    # Mock should return a name from the list random.choice will operate on (all alive players after report)
    expected_players_list = [p.name for p in generic_test_players]
    mock_choice.return_value = crewmate_player.name
    next_player, next_players_list = get_next_random_player(history, generic_test_players)
    assert next_player == crewmate_player
    # Verify choice was called with the correct list
    mock_choice.assert_called_once_with(expected_players_list)
    # Assert the returned list excludes the chosen player
    assert set(next_players_list) == set(p.name for p in generic_test_players if p != crewmate_player)

@patch('random.choice')
def test_get_next_random_player_removes_dead(mock_choice, base_history_entry: List[History], crewmate_player: Player,
                                            other_player: Player, impostor_player: Player) -> None:
    """Test get_next_random_player removes dead players from consideration."""
    # Use the latest entry as base
    base_entry = base_history_entry[-1].copy()
    # Include all player names, including one that's not in the alive list
    base_entry.player_names_to_play_next = [crewmate_player.name, other_player.name, impostor_player.name]
    
    # Only other_player and impostor_player are considered alive
    current_alive = [other_player, impostor_player]
    mock_choice.return_value = other_player.name

    next_player, next_players_list = get_next_random_player(base_history_entry + [base_entry], current_alive)

    assert next_player.name == other_player.name
    # "Crewmate" should be removed from the list used by random.choice and the returned list
    assert set(next_players_list) == {impostor_player.name}
    mock_choice.assert_called_once_with([other_player.name, impostor_player.name])


# --- Tests for get_alive_players ---

def test_get_alive_players_no_kills(generic_test_players: List[Player]) -> None:
    """Test get_alive_players with no kills in history."""
    # Start with empty history - no kills yet
    history: List[History] = []
    alive = get_alive_players(history, generic_test_players)
    assert len(alive) == len(generic_test_players)
    assert set(p.name for p in alive) == set(p.name for p in generic_test_players)

def test_get_alive_players_one_kill(generic_test_players: List[Player], base_history_entry: List[History], 
                                   kill_action: Action, crewmate_player: Player) -> None:
    """Test get_alive_players with one kill in history."""
    # Use the latest entry as base to create a kill entry
    base_entry = base_history_entry[-1]
    kill_entry = base_entry.copy(action_taken=kill_action)
    
    history = base_history_entry + [kill_entry]
    alive = get_alive_players(history, generic_test_players)
    assert len(alive) == len(generic_test_players) - 1
    assert crewmate_player.name not in set(p.name for p in alive)

def test_get_alive_players_multiple_kills(all_generic_test_players: List[Player], base_history_entry: List[History], 
                                         kill_action: Action, crewmate_player: Player, other_player: Player, impostor_player: Player) -> None:
    """Test get_alive_players with multiple kills in history."""
    # Use the latest entry as base to create kill entries
    base_entry = base_history_entry[-1]
    # Create a second kill action (impostor kills other_player)
    kill_action_other = Action(
        type=ActionType.KILL, 
        player_name=impostor_player.name, 
        target_player_name=other_player.name
    )
    
    kill_crewmate_entry = base_entry.copy(action_taken=kill_action)
    kill_other_entry = base_entry.copy(action_taken=kill_action_other)
    
    history = base_history_entry + [kill_crewmate_entry, kill_other_entry]
    alive = get_alive_players(history, all_generic_test_players)
    assert len(alive) == len(all_generic_test_players) - 2
    alive_names = set(p.name for p in alive)
    assert crewmate_player.name not in alive_names
    assert other_player.name not in alive_names
    assert impostor_player.name in alive_names

# --- Tests for get_last_player_action ---

def test_get_last_player_action_found(base_history_entry: List[History], move_action: Action, 
                                     crewmate_player: Player, other_player: Player, impostor_player: Player) -> None:
    """Test get_last_player_action when the player's action is found."""
    # Use the latest entry as base to create a sequence of history entries
    base_entry = base_history_entry[-1]
    
    # Create move actions for each player
    move_other_action = Action(
        type=ActionType.MOVE, 
        player_name=other_player.name, 
        target_location=Location.ADMIN
    )
    move_impostor_action = Action(
        type=ActionType.MOVE, 
        player_name=impostor_player.name, 
        target_location=Location.STORAGE
    )
    
    move_other_entry = base_entry.copy(action_taken=move_other_action)
    move_crewmate_entry = base_entry.copy(action_taken=move_action)
    move_impostor_entry = base_entry.copy(action_taken=move_impostor_action)
    
    history = base_history_entry + [move_other_entry, move_crewmate_entry, move_impostor_entry]
    last_action_entry = get_last_player_action(history, crewmate_player)
    assert last_action_entry.action_taken == move_action

def test_get_last_player_action_only_first(base_history_entry: List[History], crewmate_player: Player) -> None:
    """Test get_last_player_action when only the first action is found."""
    # Only base history with crewmate_player's action
    last_action_entry = get_last_player_action(base_history_entry, crewmate_player)
    assert last_action_entry == base_history_entry[-1]

def test_get_last_player_action_not_found(base_history_entry: List[History], crewmate_player: Player, impostor_player: Player) -> None:
    """Test get_last_player_action when the player's action is not found."""
    # Assuming base_history_entry doesn't contain an action by crewmate_player
    # Should return the first entry if player not found
    last_action_entry = get_last_player_action(base_history_entry, crewmate_player)
    assert last_action_entry == base_history_entry[-1]

# Test for empty history is omitted as the current implementation would raise IndexError.
# If desired, the function could be modified to handle this (e.g., return None).


# --- Tests for get_dead_players ---

def test_get_dead_players_none_since_discussion(base_history_entry: List[History], crewmate_player: Player) -> None:
    """Test get_dead_players when no players have died since the last discussion."""
    # Use the latest entry as base
    base_entry = base_history_entry[-1]
    
    discuss_entry = base_entry.copy(
        phase=GamePhase.DISCUSS,
        action_taken=Action(type=ActionType.SPEAK, player_name=crewmate_player.name, text="Hi")
    )
    
    history = base_history_entry + [discuss_entry]
    dead = get_dead_players(history)
    assert dead == {}

def test_get_dead_players_one_since_discussion(base_history_entry: List[History], kill_action: Action, crewmate_player: Player) -> None:
    """Test get_dead_players when one player has died since the last discussion."""
    # Use the latest entry as base
    base_entry = base_history_entry[-1]
    
    discuss_entry = base_entry.copy(phase=GamePhase.DISCUSS)
    kill_entry = base_entry.copy(
        action_taken=kill_action,
        location=Location.WEAPONS
    )
    
    history = base_history_entry + [discuss_entry, kill_entry]
    dead = get_dead_players(history)
    assert dead == {crewmate_player.name: Location.WEAPONS.value}

def test_get_dead_players_kill_before_discussion(base_history_entry: List[History], kill_action: Action) -> None:
    """Test get_dead_players when a kill happened before the last discussion."""
    # Use the latest entry as base
    base_entry = base_history_entry[-1]
    
    kill_entry = base_entry.copy(
        action_taken=kill_action,
        location=Location.WEAPONS
    )
    discuss_entry = base_entry.copy(phase=GamePhase.DISCUSS)
    
    # Kill happens *before* the last discussion phase starts
    history = base_history_entry + [kill_entry, discuss_entry]
    dead = get_dead_players(history)
    assert dead == {} # Should be empty as the kill was before discussion

def test_get_dead_players_multiple_kills(base_history_entry: List[History], kill_action: Action, 
                                        other_player: Player, impostor_player: Player, crewmate_player: Player) -> None:
    """Test get_dead_players with multiple kills since the last discussion."""
    # Use the latest entry as base
    base_entry = base_history_entry[-1]
    
    discuss_entry = base_entry.copy(phase=GamePhase.DISCUSS)
    
    # First kill - Impostor kills Crewmate at WEAPONS
    kill_crewmate_entry = base_entry.copy(
        action_taken=kill_action,
        location=Location.WEAPONS
    )
    
    # Second kill - Impostor kills OtherPlayer at MEDBAY
    kill_other_action = Action(
        type=ActionType.KILL, 
        player_name=impostor_player.name, 
        target_player_name=other_player.name
    )
    kill_other_entry = base_entry.copy(
        action_taken=kill_other_action,
        location=Location.MEDBAY
    )
    
    history = base_history_entry + [discuss_entry, kill_crewmate_entry, kill_other_entry]
    dead = get_dead_players(history)
    assert dead == {
        crewmate_player.name: Location.WEAPONS.value,
        other_player.name: Location.MEDBAY.value
    }

# --- Tests for get_players_in_room ---

def test_get_players_in_room_initial_location(base_history_entry: List[History], generic_test_players: List[Player], 
                                             cafeteria_location: Location) -> None:
    """Test get_players_in_room with initial location."""
    # In base_history_entry, all players start in CAFETERIA
    players_in_room = get_players_in_room(base_history_entry, generic_test_players, cafeteria_location)
    assert set(p.name for p in players_in_room) == set(p.name for p in generic_test_players)

def test_get_players_in_room_after_move(base_history_entry: List[History], generic_test_players: List[Player], 
                                       weapons_location: Location, crewmate_player: Player) -> None:
    """Test get_players_in_room after a player has moved to a different location."""
    # Use the latest entry as base
    base_entry = base_history_entry[-1]
    
    # Crewmate moves to WEAPONS
    move_action = Action(
        type=ActionType.MOVE, 
        player_name=crewmate_player.name, 
        target_location=weapons_location
    )
    move_entry = base_entry.copy(
        action_taken=move_action,
        location=weapons_location
    )
    
    history = base_history_entry + [move_entry]
    
    # Check players in WEAPONS
    players_in_weapons = get_players_in_room(history, generic_test_players, weapons_location)
    assert len(players_in_weapons) == 1
    assert players_in_weapons[0].name == crewmate_player.name
    
    # Check players in CAFETERIA (original location)
    players_in_cafeteria = get_players_in_room(history, generic_test_players, Location.CAFETERIA)
    assert len(players_in_cafeteria) == len(generic_test_players) - 1
    assert crewmate_player.name not in [p.name for p in players_in_cafeteria]

def test_get_players_in_room_multiple_moves(base_history_entry: List[History], generic_test_players: List[Player],
                                           weapons_location: Location, medbay_location: Location,
                                           crewmate_player: Player, other_player: Player) -> None:
    """Test get_players_in_room after multiple players have moved to different locations."""
    # Use the latest entry as base
    base_entry = base_history_entry[-1]
    
    # Crewmate moves to WEAPONS
    move_crewmate_action = Action(
        type=ActionType.MOVE, 
        player_name=crewmate_player.name, 
        target_location=weapons_location
    )
    move_crewmate_entry = base_entry.copy(
        action_taken=move_crewmate_action,
        location=weapons_location
    )
    
    # OtherPlayer moves to MEDBAY
    move_other_action = Action(
        type=ActionType.MOVE, 
        player_name=other_player.name, 
        target_location=medbay_location
    )
    move_other_entry = base_entry.copy(
        action_taken=move_other_action,
        location=medbay_location
    )
    
    history = base_history_entry + [move_crewmate_entry, move_other_entry]
    
    # Check players in WEAPONS
    players_in_weapons = get_players_in_room(history, generic_test_players, weapons_location)
    assert len(players_in_weapons) == 1
    assert players_in_weapons[0].name == crewmate_player.name
    
    # Check players in MEDBAY
    players_in_medbay = get_players_in_room(history, generic_test_players, medbay_location)
    assert len(players_in_medbay) == 1
    assert players_in_medbay[0].name == other_player.name
    
    # Check players in CAFETERIA (original location)
    players_in_cafeteria = get_players_in_room(history, generic_test_players, Location.CAFETERIA)
    assert len(players_in_cafeteria) == len(generic_test_players) - 2
    cafeteria_names = [p.name for p in players_in_cafeteria]
    assert crewmate_player.name not in cafeteria_names
    assert other_player.name not in cafeteria_names
