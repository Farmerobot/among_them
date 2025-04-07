import pytest
from typing import List, Dict, Tuple
from unittest.mock import patch

from among_them.consts import NUM_ACTIONS_WITHOUT_REPORT, NUM_CHATS
from among_them.models.action import Action, ActionType
from among_them.models.history import History, initialize_history, create_vote_history_entry
from among_them.models.location import Location
from among_them.models.phase import GamePhase
from among_them.models.player import Player
from among_them.models.player_role import PlayerRole 
from among_them.utils.phase_utils import (
    get_last_discussion_action_idx,
    get_phase_and_when_it_ends,
    handle_phase_change,
    count_votes,
    determine_ejection_result,
)


# --- Tests for get_last_discussion_action_idx ---

def test_get_last_discussion_action_idx_found(base_history_entry: List[History]) -> None:
    # Use the latest history entry as the base for modifications
    base_entry = base_history_entry[-1]
    
    history = [
        base_entry.copy(phase=GamePhase.TASK),
        base_entry.copy(phase=GamePhase.DISCUSS),
        base_entry.copy(phase=GamePhase.VOTE),
        base_entry.copy(phase=GamePhase.DISCUSS), 
        base_entry.copy(phase=GamePhase.TASK),
    ]
    assert get_last_discussion_action_idx(history) == 3

def test_get_last_discussion_action_idx_not_found(base_history_entry: List[History]) -> None:
    # Use the latest history entry as the base for modifications
    base_entry = base_history_entry[-1]
    
    history = [
        base_entry.copy(phase=GamePhase.TASK),
        base_entry.copy(phase=GamePhase.VOTE),
    ]
    assert get_last_discussion_action_idx(history) == 0

def test_get_last_discussion_action_idx_empty() -> None:
    assert get_last_discussion_action_idx([]) == 0


# --- Tests for get_phase_and_when_it_ends ---

def test_get_phase_and_when_it_ends_first_turn(generic_test_players: List[Player]) -> None:
    history = initialize_history(generic_test_players) 
    expected_actions = (NUM_ACTIONS_WITHOUT_REPORT * len(generic_test_players))
    task_start_history = history[0].copy(phase=GamePhase.TASK, actions_until_phase_ends=expected_actions)
    history.append(task_start_history)

    phase, actions_left = get_phase_and_when_it_ends(history, generic_test_players)
    assert phase == GamePhase.TASK
    # The function returns actions_until_phase_ends - 1
    assert actions_left == expected_actions - 1

def test_get_phase_and_when_it_ends_report(base_history_entry: List[History], report_action: Action, generic_test_players: List[Player], dead_player: Player) -> None:
    # Use the latest entry as base
    base_entry = base_history_entry[-1]

    history = [
        base_entry,
        base_entry.copy(action_taken=report_action) # report_action uses dead_player
    ]
    # Calculate players alive for discussion count
    alive_players = [p for p in generic_test_players if p.name != dead_player.name]
    expected_actions = (NUM_CHATS * len(alive_players)) - 1 # Report action counts as first "chat" implicitly
    phase, actions_left = get_phase_and_when_it_ends(history, alive_players) # Pass only alive players
    assert phase == GamePhase.DISCUSS
    assert actions_left == expected_actions

def test_get_phase_and_when_it_ends_continue_phase(base_history_entry: List[History], move_action: Action, generic_test_players: List[Player]) -> None:
    # Use the latest entry as base
    base_entry = base_history_entry[-1]

    history = [
        base_entry,
        # Use move_action (crewmate moves to medbay) instead of old task_action
        base_entry.copy(action_taken=move_action, phase=GamePhase.TASK, actions_until_phase_ends=5)
    ]
    phase, actions_left = get_phase_and_when_it_ends(history, generic_test_players)
    assert phase == GamePhase.TASK
    assert actions_left == 4

def test_get_phase_and_when_it_ends_phase_change_task(base_history_entry: List[History], move_action: Action, generic_test_players: List[Player]) -> None:
    # Use the latest entry as base
    base_entry = base_history_entry[-1]

    history = [
        base_entry,
        # Use move_action (crewmate moves to medbay) instead of old task_action
        base_entry.copy(action_taken=move_action, phase=GamePhase.TASK, actions_until_phase_ends=0)
    ]
    phase, actions_left = get_phase_and_when_it_ends(history, generic_test_players)
    assert phase == GamePhase.MAIN_MENU
    assert actions_left == 0

def test_get_phase_and_when_it_ends_phase_change_discuss(base_history_entry: List[History], discuss_action: Action, generic_test_players: List[Player]) -> None:
    # Use the latest entry as base
    base_entry = base_history_entry[-1]

    history = [
        base_entry,
        base_entry.copy(action_taken=discuss_action, phase=GamePhase.DISCUSS, actions_until_phase_ends=0)
    ]
    phase, actions_left = get_phase_and_when_it_ends(history, generic_test_players)
    assert phase == GamePhase.VOTE
    assert actions_left == len(generic_test_players) - 1


# --- Tests for handle_phase_change ---

def test_handle_phase_change_task_to_main_menu(generic_test_players: List[Player]) -> None:
    history: List[History] = [] 
    phase, actions_left = handle_phase_change(history, generic_test_players, GamePhase.TASK)
    assert phase == GamePhase.MAIN_MENU
    assert actions_left == 0

def test_handle_phase_change_discuss_to_vote(generic_test_players: List[Player]) -> None:
    history: List[History] = [] 
    phase, actions_left = handle_phase_change(history, generic_test_players, GamePhase.DISCUSS)
    assert phase == GamePhase.VOTE
    assert actions_left == len(generic_test_players) - 1

def test_handle_phase_change_vote_to_task(base_history_entry: List[History], vote_action: Action, vote_nobody_action: Action,
                                         generic_test_players: List[Player], crewmate_player: Player, other_player: Player, impostor_player: Player) -> None:
    # Use the latest entry as base
    base_entry = base_history_entry[-1]

    # vote_action fixture already has crewmate voting impostor
    # Create vote for other_player voting crewmate
    vote_action_other = vote_action.copy(player_name=other_player.name, target_player_name=crewmate_player.name)
    # vote_nobody_action fixture already has impostor voting nobody

    history = [
        # Simulate vote actions being added
        base_entry.copy(phase=GamePhase.VOTE, action_taken=vote_action), # Crewmate votes Impostor
        base_entry.copy(phase=GamePhase.VOTE, action_taken=vote_action_other), # OtherPlayer votes Crewmate
        base_entry.copy(phase=GamePhase.VOTE, action_taken=vote_nobody_action), # Impostor votes Nobody
    ]
    initial_history_len = len(history)
    phase, actions_left = handle_phase_change(history, generic_test_players, GamePhase.VOTE)

    assert phase == GamePhase.TASK # Check returned next phase
    expected_actions = (NUM_ACTIONS_WITHOUT_REPORT * len(generic_test_players)) - 1
    assert actions_left == expected_actions # Check returned actions left

    assert len(history) == initial_history_len + 1
    vote_entry = history[-1] # This is the entry *added* by the function
    # The added entry reflects the result (nobody voted out), setting phase to MAIN_MENU
    assert vote_entry.phase == GamePhase.MAIN_MENU

def test_handle_phase_change_main_menu_to_main_menu(generic_test_players: List[Player]) -> None:
    history: List[History] = []
    phase, actions_left = handle_phase_change(history, generic_test_players, GamePhase.MAIN_MENU)
    assert phase == GamePhase.MAIN_MENU
    assert actions_left == 0


# --- Tests for count_votes ---

def test_count_votes(base_history_entry: List[History], vote_action: Action, vote_nobody_action: Action,
                   crewmate_player: Player, other_player: Player, impostor_player: Player) -> None:
    # Use the latest entry as base
    base_entry = base_history_entry[-1]

    # vote_action fixture has crewmate voting impostor
    # Create vote for other_player voting impostor
    vote_action_other = vote_action.copy(player_name=other_player.name, target_player_name=impostor_player.name)
    # vote_nobody_action fixture has impostor voting nobody

    history = [
        base_entry.copy(phase=GamePhase.TASK, action_taken=Action(type=ActionType.REPORT, player_name=crewmate_player.name)), # Report starts discussion
        base_entry.copy(phase=GamePhase.DISCUSS, action_taken=Action(type=ActionType.SPEAK, player_name=crewmate_player.name)),
        base_entry.copy(phase=GamePhase.VOTE, action_taken=vote_action), # Crewmate votes Impostor
        base_entry.copy(phase=GamePhase.VOTE, action_taken=vote_action_other), # OtherPlayer votes Impostor
        base_entry.copy(phase=GamePhase.VOTE, action_taken=vote_nobody_action), # Impostor votes Nobody
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

def test_count_votes_no_votes(base_history_entry: List[History], crewmate_player: Player) -> None:
    base_entry = base_history_entry[-1]
    history = [
        base_entry.copy(phase=GamePhase.TASK),
        base_entry.copy(phase=GamePhase.DISCUSS, action_taken=Action(type=ActionType.REPORT, player_name=crewmate_player.name)),
        base_entry.copy(phase=GamePhase.DISCUSS, action_taken=Action(type=ActionType.SPEAK, player_name=crewmate_player.name)),
        # No VOTE phase entries
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

