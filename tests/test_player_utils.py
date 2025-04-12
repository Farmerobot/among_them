from typing import List
from unittest.mock import patch

from among_them.models.action import Action
from among_them.models.history import History
from among_them.models.location import Location
from among_them.models.player import Player
from among_them.utils.player_utils import (get_alive_players, get_dead_players,
                                           get_last_player_action,
                                           get_next_random_player,
                                           get_players_in_room)

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
def test_get_next_random_player_with_remaining(mock_choice, crewmate_wait_history: History, initial_history: History,
                                              generic_test_players: List[Player], other_player: Player, 
                                              impostor_player: Player) -> None:
    """Test get_next_random_player with remaining players in history."""
    # Create a modified history entry with specific players to play next
    modified_entry = crewmate_wait_history.copy(player_names_to_play_next=[other_player.name, impostor_player.name])
    
    mock_choice.return_value = other_player.name
    next_player, next_players_list = get_next_random_player([initial_history, modified_entry], generic_test_players)
    assert next_player.name == other_player.name
    assert next_players_list == [impostor_player.name]
    mock_choice.assert_called_once_with([other_player.name, impostor_player.name]) # Called with remaining players

@patch('random.choice')
def test_get_next_random_player_empty_remaining(mock_choice, crewmate_wait_history: History, initial_history: History,
                                               generic_test_players: List[Player], crewmate_player: Player) -> None:
    """Test get_next_random_player with empty remaining players in history."""
    # Create a modified history entry with empty players to play next
    modified_entry = crewmate_wait_history.copy(player_names_to_play_next=[])
    
    # Mock should return a name from the list random.choice will operate on
    expected_players_list = [p.name for p in generic_test_players]
    mock_choice.return_value = crewmate_player.name 
    next_player, next_players_list = get_next_random_player([initial_history, modified_entry], generic_test_players)
    assert next_player == crewmate_player
    # Verify choice was called with the correct list
    mock_choice.assert_called_once_with(expected_players_list)
    # Assert the returned list excludes the chosen player
    assert set(next_players_list) == set(p.name for p in generic_test_players if p != crewmate_player)


@patch('random.choice')
def test_get_next_random_player_after_report(mock_choice, initial_history: History, crewmate_report_history: History, impostor_kill_history: History,
                                            generic_test_players: List[Player], crewmate_player: Player) -> None:
    """Test get_next_random_player after a report action."""
    # Create a history with a report action
    history = [initial_history, impostor_kill_history, crewmate_report_history]
    
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
def test_get_next_random_player_removes_dead(mock_choice, initial_history: History, crewmate_wait_history: History, 
                                            crewmate_player: Player, other_player: Player, impostor_player: Player) -> None:
    """Test get_next_random_player removes dead players from consideration."""
    # Create a modified history entry with specific players to play next
    modified_entry = crewmate_wait_history.copy(player_names_to_play_next=[crewmate_player.name, other_player.name, impostor_player.name])
    
    # Only other_player and impostor_player are considered alive
    current_alive = [other_player, impostor_player]
    mock_choice.return_value = other_player.name

    next_player, next_players_list = get_next_random_player([initial_history, modified_entry], current_alive)

    assert next_player.name == other_player.name
    # "Crewmate" should be removed from the list used by random.choice and the returned list
    assert set(next_players_list) == {impostor_player.name}
    mock_choice.assert_called_once_with([other_player.name, impostor_player.name])


# --- Tests for get_alive_players ---

def test_get_alive_players_no_kills(generic_test_players: List[Player], initial_history: History) -> None:
    """Test get_alive_players with no kills in history."""
    # Start with empty history - no kills yet
    history: List[History] = [initial_history]
    alive = get_alive_players(history, generic_test_players)
    assert len(alive) == len(generic_test_players)
    assert set(p.name for p in alive) == set(p.name for p in generic_test_players)

def test_get_alive_players_one_kill(all_generic_test_players: List[Player], initial_history: History, crewmate_wait_history: History, 
                                   impostor_kill_history: History, dead_player: Player) -> None:
    """Test get_alive_players with one kill in history."""
    # Create a history with a kill
    history = [initial_history, crewmate_wait_history, impostor_kill_history]
    alive = get_alive_players(history, all_generic_test_players)
    
    # Check that the target of kill_action is not in alive players
    target_name = dead_player.name # or just impostor_kill_history.action_taken.target_player_name
    assert len(alive) == len(all_generic_test_players) - 1
    assert target_name not in set(p.name for p in alive)

def test_get_alive_players_multiple_kills(all_generic_test_players: List[Player], crewmate_wait_history: History, 
                                         impostor_kill_history: History, other_player: Player, initial_history: History,
                                         impostor_player: Player, kill_action: Action, dead_player: Player) -> None:
    """Test get_alive_players with multiple kills in history."""
    # Create a second kill history entry by copying the first one and modifying the action
    second_kill_history = impostor_kill_history.copy(action_taken=kill_action.copy(target_player_name=other_player.name))
    
    history = [initial_history, impostor_kill_history, crewmate_wait_history, second_kill_history]
    alive = get_alive_players(history, all_generic_test_players)
    
    # Check that both killed players are not in alive players
    assert len(alive) == len(all_generic_test_players) - 2
    alive_names = set(p.name for p in alive)
    assert dead_player.name not in alive_names
    assert other_player.name not in alive_names
    assert impostor_player.name in alive_names

# --- Tests for get_last_player_action ---

def test_get_last_player_action_found(crewmate_move_history: History, initial_history: History,
                                     impostor_move_history: History, crewmate_player: Player) -> None:
    """Test get_last_player_action when the player's action is found."""
    
    history = [initial_history, crewmate_move_history, impostor_move_history]
    last_action_entry = get_last_player_action(history, crewmate_player)
    assert last_action_entry.action_taken == crewmate_move_history.action_taken

def test_get_last_player_action_only_first(initial_history: History, crewmate_player: Player) -> None:
    """Test get_last_player_action when only the first action is found."""
    # Only base history with crewmate_player's action
    history = [initial_history]
    last_action_entry = get_last_player_action(history, crewmate_player)
    assert last_action_entry == initial_history

def test_get_last_player_action_not_found(initial_history: History, crewmate_player: Player) -> None:
    """Test get_last_player_action when the player's action is not found."""
    # Create a history with only impostor's action
    history = [initial_history]
    last_action_entry = get_last_player_action(history, crewmate_player)
    assert last_action_entry == initial_history

# --- Tests for get_dead_players ---

def test_get_dead_players_none_since_discussion(initial_history: History, crewmate_report_history: History, discuss_phase_history: History,
                                              crewmate_player: Player, impostor_kill_history: History, vote_phase_history: History) -> None:
    """Test get_dead_players when no players have died since the last discussion."""
    # Create a history with a discussion phase but no kills
    history = [initial_history, impostor_kill_history, crewmate_report_history, discuss_phase_history, vote_phase_history]
    
    # No players should be reported as dead since the last discussion
    dead_players = get_dead_players(history)
    assert len(dead_players) == 0

def test_get_dead_players_one_since_discussion(initial_history: History, crewmate_report_history: History, discuss_phase_history: History, cafeteria_location: Location,
                                              crewmate_player: Player, impostor_kill_history: History, kill_action: Action, other_player: Player, vote_phase_history: History) -> None:
    """Test get_dead_players when one player has died since the last discussion."""
    # Create history with a discussion phase followed by a kill
    other_kill_history = impostor_kill_history.copy(action_taken=kill_action.copy(target_player_name=other_player.name))
    
    history = [initial_history, impostor_kill_history, crewmate_report_history, discuss_phase_history, vote_phase_history, other_kill_history]
    
    # The player killed after the discussion should be reported
    dead_players = get_dead_players(history)
    assert len(dead_players) == 1
    assert other_player.name in dead_players
    assert dead_players[other_player.name]== cafeteria_location.value

def test_get_dead_players_kill_before_discussion(impostor_kill_history: History, discuss_phase_history: History,
                                               crewmate_wait_history: History) -> None:
    """Test get_dead_players when a kill happened before the last discussion."""
    # Create history with a kill followed by a discussion phase
    history = [crewmate_wait_history, impostor_kill_history, discuss_phase_history]
    
    # No players should be reported as dead since the last discussion
    dead_players = get_dead_players(history)
    assert len(dead_players) == 0

def test_get_dead_players_multiple_kills(discuss_phase_history: History, impostor_kill_history: History,
                                        other_player: Player, impostor_player: Player, crewmate_player: Player,
                                        initial_history: History, kill_action: Action, dead_player: Player) -> None:
    """Test get_dead_players with multiple kills since the last discussion."""
    # Create a second kill history entry by copying the first one and modifying the action
    second_kill_history = impostor_kill_history.copy(action_taken=kill_action.copy(target_player_name=other_player.name))
    
    history = [initial_history, discuss_phase_history, impostor_kill_history, second_kill_history]
    
    # Both players killed after the discussion should be reported
    dead_players = get_dead_players(history)
    assert len(dead_players) == 2
    dead_names = [p for p, _ in dead_players.items()]
    assert dead_player.name in dead_names
    assert other_player.name in dead_names


# --- Tests for get_players_in_room ---

def test_get_players_in_room_initial_location(initial_history: History, all_generic_test_players: List[Player], 
                                             cafeteria_location: Location) -> None:
    """Test get_players_in_room with initial location."""
    players_in_room = get_players_in_room([initial_history], all_generic_test_players, cafeteria_location)
    assert len(players_in_room) == len(all_generic_test_players)
    assert set(p.name for p in players_in_room) == set(p.name for p in all_generic_test_players)

def test_get_players_in_room_multiple_moves(initial_history: History, all_generic_test_players: List[Player], 
                                       crewmate_move_history: History, cafeteria_location: Location,
                                       weapons_location: Location, impostor_player: Player, medbay_location: Location,
                                       crewmate_player: Player, impostor_move_history: History, other_player: Player, dead_player: Player) -> None:
    """Test get_players_in_room after a player has moved to a different location."""
    # Create history with move action
    history = [initial_history, crewmate_move_history, impostor_move_history] # dead player not dead
    
    # Check players in weapons_location
    players_in_weapons = get_players_in_room(history, all_generic_test_players, weapons_location)
    assert len(players_in_weapons) == 1 # only impostor
    assert players_in_weapons[0].name == impostor_player.name

    players_in_medbay = get_players_in_room(history, all_generic_test_players, medbay_location)
    assert len(players_in_medbay) == 1 # only crewmate
    assert players_in_medbay[0].name == crewmate_player.name
    
    # Check players in cafeteria_location
    players_in_cafeteria = get_players_in_room(history, all_generic_test_players, cafeteria_location)
    assert len(players_in_cafeteria) == len(all_generic_test_players) - 2
    assert crewmate_player.name not in [p.name for p in players_in_cafeteria]
    assert impostor_player.name not in [p.name for p in players_in_cafeteria]
    assert other_player.name in [p.name for p in players_in_cafeteria]
    assert dead_player.name in [p.name for p in players_in_cafeteria]
