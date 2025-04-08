from typing import List

import pytest

from among_them.models.action import Action, ActionType
from among_them.models.game_config import GameConfig
from among_them.models.history import History
from among_them.models.phase import GamePhase
from among_them.models.player import Player
from among_them.utils.phase_utils import (count_votes,
                                          determine_ejection_result,
                                          get_last_discussion_action_idx,
                                          get_phase_and_when_it_ends,
                                          handle_phase_change)

# --- Tests for get_last_discussion_action_idx ---

def test_get_last_discussion_action_idx_found(task_phase_history: History, initial_history: History, 
                                             discuss_phase_history: History, vote_phase_history: History) -> None:
    """Test get_last_discussion_action_idx when discussion phase is found."""
    # Create a history list with different phases using fixtures
    history = [
        initial_history,
        task_phase_history,
        discuss_phase_history,
        vote_phase_history,
        discuss_phase_history, 
        task_phase_history,
    ]
    assert get_last_discussion_action_idx(history) == 4

def test_get_last_discussion_action_idx_not_found(task_phase_history: History, vote_phase_history: History) -> None:
    """Test get_last_discussion_action_idx when no discussion phase is found."""
    # Create a history list without DISCUSS phase using fixtures
    history = [
        task_phase_history,
        vote_phase_history,
    ]
    assert get_last_discussion_action_idx(history) == 0

def test_get_last_discussion_action_idx_empty() -> None:
    """Test get_last_discussion_action_idx with empty history."""
    assert get_last_discussion_action_idx([]) == 0


# --- Tests for get_phase_and_when_it_ends ---

def test_get_phase_and_when_it_ends_first_turn(generic_test_players: List[Player], game_config: GameConfig, 
                                              task_phase_history: History, initial_history: History) -> None:
    """Test get_phase_and_when_it_ends on the first turn."""
    expected_actions = (game_config.num_task_phase_actions_per_player * len(generic_test_players))
    
    # Use task_phase_history as a base and modify it
    task_start_history = task_phase_history.copy(actions_until_phase_ends=expected_actions)
    
    history = [initial_history, task_start_history]

    phase, actions_left = get_phase_and_when_it_ends(history, generic_test_players, game_config)
    assert phase == GamePhase.TASK
    # The function returns actions_until_phase_ends - 1
    assert actions_left == expected_actions - 1

def test_get_phase_and_when_it_ends_report(initial_history: History, crewmate_report_history: History, 
                                         generic_test_players: List[Player], dead_player: Player, 
                                         game_config: GameConfig) -> None:
    """Test get_phase_and_when_it_ends after a report action."""
    # Create a history list with a report action using fixtures
    history = [
        initial_history,
        crewmate_report_history  # Report action changes phase to DISCUSS
    ]
    # Calculate players alive for discussion count
    alive_players = [p for p in generic_test_players if p.name != dead_player.name]
    expected_actions = (game_config.num_discuss_phase_actions_per_player * len(alive_players)) - 1 
    phase, actions_left = get_phase_and_when_it_ends(history, alive_players, game_config) # Pass only alive players
    assert phase == GamePhase.DISCUSS
    assert actions_left == expected_actions

def test_get_phase_and_when_it_ends_continue_phase(initial_history: History, task_phase_history: History, 
                                                 generic_test_players: List[Player], game_config: GameConfig) -> None:
    """Test get_phase_and_when_it_ends when continuing in the same phase."""
    # Create a history list with a move action using fixtures
    history = [
        initial_history,
        task_phase_history  # Task phase with actions_until_phase_ends=10
    ]
    phase, actions_left = get_phase_and_when_it_ends(history, generic_test_players, game_config)
    assert phase == GamePhase.TASK
    assert actions_left == 9  # 10 - 1

def test_get_phase_and_when_it_ends_phase_change_task(initial_history: History, task_phase_history: History, 
                                                    generic_test_players: List[Player], game_config: GameConfig) -> None:
    """Test get_phase_and_when_it_ends when task phase is ending."""
    # Create a custom history entry with actions_until_phase_ends=0
    task_ending_history = task_phase_history.copy(actions_until_phase_ends=0)
    
    history = [initial_history, task_phase_history, task_ending_history]
    phase, actions_left = get_phase_and_when_it_ends(history, generic_test_players, game_config)
    assert phase == GamePhase.MAIN_MENU
    assert actions_left == 0

def test_get_phase_and_when_it_ends_phase_change_discuss(initial_history: History, discuss_phase_history: History, 
                                                       generic_test_players: List[Player], game_config: GameConfig) -> None:
    """Test get_phase_and_when_it_ends when discuss phase is ending."""
    # Create a custom history entry with actions_until_phase_ends=0
    discuss_ending_history = discuss_phase_history.copy(actions_until_phase_ends=0)
    
    history = [initial_history, discuss_phase_history, discuss_ending_history]
    phase, actions_left = get_phase_and_when_it_ends(history, generic_test_players, game_config)
    assert phase == GamePhase.VOTE
    assert actions_left == len(generic_test_players) - 1


# --- Tests for handle_phase_change ---

def test_handle_phase_change_task_to_main_menu(generic_test_players: List[Player], initial_history: History, game_config: GameConfig) -> None:
    """Test handle_phase_change from TASK to MAIN_MENU."""
    history = [initial_history] 
    phase, actions_left = handle_phase_change(history, generic_test_players, GamePhase.TASK, game_config)
    assert phase == GamePhase.MAIN_MENU
    assert actions_left == 0

def test_handle_phase_change_discuss_to_vote(initial_history: History, generic_test_players: List[Player], game_config: GameConfig) -> None:
    """Test handle_phase_change from DISCUSS to VOTE."""
    history = [initial_history] 
    phase, actions_left = handle_phase_change(history, generic_test_players, GamePhase.DISCUSS, game_config)
    assert phase == GamePhase.VOTE
    assert actions_left == len(generic_test_players) - 1

def test_handle_phase_change_vote_to_task(vote_phase_history: History, crewmate_vote_history: History, 
                                         impostor_vote_history: History, generic_test_players: List[Player], 
                                         initial_history: History, impostor_kill_history: History, crewmate_report_history: History,
                                         crewmate_speak_history: History, game_config: GameConfig,
                                         vote_impostor_action: Action, other_vote_history: History) -> None:
    """Test handle_phase_change from VOTE to TASK."""
    # Create a custom vote action for other_player voting crewmate by copying an existing vote action
    other_vote_history = other_vote_history.copy(actions_until_phase_ends=len(generic_test_players) - 2)
    
    # Create a history list with all players voting
    history = [
        initial_history,
        impostor_kill_history,
        crewmate_report_history,
        crewmate_speak_history,
        crewmate_vote_history,  # Crewmate votes for impostor
        impostor_vote_history,  # Impostor votes for nobody
        other_vote_history,     # Other votes for crewmate
    ]
    
    initial_history_len = len(history)
    # Calculate expected actions for TASK phase
    expected_actions = game_config.num_task_phase_actions_per_player * len(generic_test_players) - 1
    
    phase, actions_left = handle_phase_change(history, generic_test_players, GamePhase.VOTE, game_config)
    assert phase == GamePhase.TASK
    assert actions_left == expected_actions

    assert len(history) == initial_history_len + 1
    vote_entry = history[-1] # This is the entry *added* by the function
    # The added entry reflects the result (nobody voted out), setting phase to MAIN_MENU as the system speaks
    assert vote_entry.phase == GamePhase.MAIN_MENU
    assert vote_entry.actions_until_phase_ends == 0

def test_handle_phase_change_main_menu_to_main_menu(generic_test_players: List[Player], initial_history: History, game_config: GameConfig) -> None:
    """Test handle_phase_change from MAIN_MENU to MAIN_MENU."""
    history = [initial_history] 
    phase, actions_left = handle_phase_change(history, generic_test_players, GamePhase.MAIN_MENU, game_config)
    assert phase == GamePhase.MAIN_MENU
    assert actions_left == 0


# --- Tests for count_votes ---

def test_count_votes(crewmate_vote_history: History, impostor_vote_history: History, crewmate_report_history: History,
                   crewmate_speak_history: History, crewmate_player: Player, other_player: Player, other_vote_history: History,
                   initial_history: History, impostor_kill_history: History, impostor_player: Player) -> None:
    """Test count_votes with multiple votes."""
    # Create a history list with a discussion phase followed by votes
    history = [
        initial_history,
        impostor_kill_history,
        crewmate_report_history,  # Starts DISCUSS phase
        crewmate_speak_history,   # Discussion
        crewmate_vote_history,    # Crewmate votes for impostor
        impostor_vote_history,    # Impostor votes for crewmate
        other_vote_history,       # Other votes for impostor
    ]
    
    vote_counts, votes = count_votes(history)

    expected_votes = {
        impostor_player.name: "nobody",
        other_player.name: impostor_player.name,
        crewmate_player.name: impostor_player.name,
    }
    expected_vote_counts = {
        impostor_player.name: 2,
        "nobody": 1,
    }
    assert votes == expected_votes
    assert vote_counts == expected_vote_counts

def test_count_votes_no_votes(initial_history: History, crewmate_report_history: History,
                            crewmate_speak_history: History) -> None:
    """Test count_votes with no votes cast."""
    # Create a history list with a discussion phase but no votes
    history = [
        initial_history,
        crewmate_report_history,  # Starts DISCUSS phase
        crewmate_speak_history,   # Discussion
    ]
    
    vote_counts, votes = count_votes(history)
    assert votes == {}
    assert vote_counts == {}


# --- Tests for determine_ejection_result ---

def test_determine_ejection_result_clear_winner(crewmate_player: Player, impostor_player: Player) -> None:
    votes = {impostor_player.name: 2, crewmate_player.name: 1}
    ejected, action_type = determine_ejection_result(votes)
    assert ejected == impostor_player.name
    assert action_type == ActionType.KILL

def test_determine_ejection_result_tie(crewmate_player: Player, impostor_player: Player) -> None:
    votes = {impostor_player.name: 1, crewmate_player.name: 1}
    ejected, action_type = determine_ejection_result(votes)
    assert ejected == "nobody"
    assert action_type == ActionType.WAIT

def test_determine_ejection_result_nobody(impostor_player: Player) -> None:
    votes = {"nobody": 2, impostor_player.name: 1}
    ejected, action_type = determine_ejection_result(votes)
    assert ejected == "nobody"
    assert action_type == ActionType.WAIT

def test_determine_ejection_result_nobody_tie(impostor_player: Player) -> None:
    votes = {"nobody": 1, impostor_player.name: 1}
    ejected, action_type = determine_ejection_result(votes)
    assert ejected == "nobody"
    assert action_type == ActionType.WAIT

def test_determine_ejection_result_no_votes():
    vote_counts = {}
    with pytest.raises(ValueError, match="No votes casted"):
        determine_ejection_result(vote_counts)
