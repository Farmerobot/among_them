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
def alive_players(player1: Player, player2: Player, player3: Player) -> List[Player]:
    return [player1, player2, player3]

@pytest.fixture
def base_history_entry(player1: Player, alive_players: List[Player]) -> List[History]:
    """Create a consistent base history entry using the shared utility."""
    return create_base_history(alive_players, player1, Location.CAFETERIA)

@pytest.fixture
def task_action(player1: Player) -> Action:
    return Action(type=ActionType.MOVE, player_name=player1.name, target_location=Location.WEAPONS)

@pytest.fixture
def report_action(player1: Player, player2: Player) -> Action:
    return Action(type=ActionType.REPORT, player_name=player1.name, target_player_name=player2.name)

@pytest.fixture
def discuss_action(player1: Player) -> Action:
    return Action(type=ActionType.SPEAK, player_name=player1.name, text="Hello")

@pytest.fixture
def vote_action(player1: Player, player2: Player) -> Action:
    return Action(type=ActionType.VOTE, player_name=player1.name, target_player_name=player2.name)

@pytest.fixture
def vote_nobody_action(player3: Player) -> Action:
    return Action(type=ActionType.VOTE, player_name=player3.name, target_player_name="nobody")


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

def test_get_phase_and_when_it_ends_first_turn(alive_players: List[Player]) -> None:
    history = initialize_history(alive_players) 
    expected_actions = (NUM_ACTIONS_WITHOUT_REPORT * len(alive_players))
    task_start_history = history[0].copy(phase=GamePhase.TASK, actions_until_phase_ends=expected_actions)
    history.append(task_start_history)

    phase, actions_left = get_phase_and_when_it_ends(history, alive_players)
    assert phase == GamePhase.TASK
    # The function returns actions_until_phase_ends - 1
    assert actions_left == expected_actions - 1

def test_get_phase_and_when_it_ends_report(base_history_entry: List[History], report_action: Action, alive_players: List[Player]) -> None:
    # Use the latest entry as base
    base_entry = base_history_entry[-1]
    
    history = [
        base_entry,
        base_entry.copy(action_taken=report_action)
    ]
    expected_actions = (NUM_CHATS * len(alive_players)) - 1
    phase, actions_left = get_phase_and_when_it_ends(history, alive_players)
    assert phase == GamePhase.DISCUSS
    assert actions_left == expected_actions

def test_get_phase_and_when_it_ends_continue_phase(base_history_entry: List[History], task_action: Action, alive_players: List[Player]) -> None:
    # Use the latest entry as base
    base_entry = base_history_entry[-1]
    
    history = [
        base_entry,
        base_entry.copy(action_taken=task_action, phase=GamePhase.TASK, actions_until_phase_ends=5)
    ]
    phase, actions_left = get_phase_and_when_it_ends(history, alive_players)
    assert phase == GamePhase.TASK
    assert actions_left == 4

def test_get_phase_and_when_it_ends_phase_change_task(base_history_entry: List[History], task_action: Action, alive_players: List[Player]) -> None:
    # Use the latest entry as base
    base_entry = base_history_entry[-1]
    
    history = [
        base_entry,
        base_entry.copy(action_taken=task_action, phase=GamePhase.TASK, actions_until_phase_ends=0)
    ]
    phase, actions_left = get_phase_and_when_it_ends(history, alive_players)
    assert phase == GamePhase.MAIN_MENU 
    assert actions_left == 0

def test_get_phase_and_when_it_ends_phase_change_discuss(base_history_entry: List[History], discuss_action: Action, alive_players: List[Player]) -> None:
    # Use the latest entry as base
    base_entry = base_history_entry[-1]
    
    history = [
        base_entry,
        base_entry.copy(action_taken=discuss_action, phase=GamePhase.DISCUSS, actions_until_phase_ends=0)
    ]
    phase, actions_left = get_phase_and_when_it_ends(history, alive_players)
    assert phase == GamePhase.VOTE
    assert actions_left == len(alive_players) - 1


# --- Tests for handle_phase_change ---

def test_handle_phase_change_task_to_main_menu(alive_players: List[Player]) -> None:
    history: List[History] = [] 
    phase, actions_left = handle_phase_change(history, alive_players, GamePhase.TASK)
    assert phase == GamePhase.MAIN_MENU
    assert actions_left == 0

def test_handle_phase_change_discuss_to_vote(alive_players: List[Player]) -> None:
    history: List[History] = [] 
    phase, actions_left = handle_phase_change(history, alive_players, GamePhase.DISCUSS)
    assert phase == GamePhase.VOTE
    assert actions_left == len(alive_players) - 1

def test_handle_phase_change_vote_to_task(base_history_entry: List[History], vote_action: Action, vote_nobody_action: Action, 
                                         alive_players: List[Player], player1: Player, player2: Player, player3: Player) -> None:
    # Use the latest entry as base
    base_entry = base_history_entry[-1]
    
    history = [
        base_entry.copy(phase=GamePhase.VOTE, action_taken=vote_action), 
        base_entry.copy(phase=GamePhase.VOTE, action_taken=vote_action.copy(player_name=player2.name, target_player_name=player1.name)), 
        base_entry.copy(phase=GamePhase.VOTE, action_taken=vote_nobody_action), 
    ]
    initial_history_len = len(history)
    phase, actions_left = handle_phase_change(history, alive_players, GamePhase.VOTE)

    assert phase == GamePhase.TASK # Check returned next phase
    expected_actions = (NUM_ACTIONS_WITHOUT_REPORT * len(alive_players)) - 1
    assert actions_left == expected_actions # Check returned actions left

    assert len(history) == initial_history_len + 1
    vote_entry = history[-1] # This is the entry *added* by the function
    # The added entry reflects the result (nobody voted out), setting phase to MAIN_MENU
    assert vote_entry.phase == GamePhase.MAIN_MENU

def test_handle_phase_change_main_menu_to_main_menu(alive_players: List[Player]) -> None:
    history: List[History] = []
    phase, actions_left = handle_phase_change(history, alive_players, GamePhase.MAIN_MENU)
    assert phase == GamePhase.MAIN_MENU
    assert actions_left == 0


# --- Tests for count_votes ---

def test_count_votes(base_history_entry: List[History], vote_action: Action, vote_nobody_action: Action, 
                   player1: Player, player2: Player, player3: Player) -> None:
    # Use the latest entry as base
    base_entry = base_history_entry[-1]
    
    history = [
        base_entry.copy(phase=GamePhase.DISCUSS), 
        base_entry.copy(phase=GamePhase.VOTE, action_taken=vote_action), 
        base_entry.copy(phase=GamePhase.VOTE, 
                                action_taken=vote_action.copy(player_name=player2.name, target_player_name=player1.name)), 
        base_entry.copy(phase=GamePhase.VOTE, action_taken=vote_nobody_action)
    ]

    # Pass only the relevant segment to the function, as it stops at the first non-VOTE from the end
    vote_counts, votes = count_votes(history)

    expected_counts = {player1.name: 1, player2.name: 1, "nobody": 1}
    expected_votes = {
        player1.name: player2.name,
        player2.name: player1.name,
        player3.name: "nobody"
    }

    assert vote_counts == expected_counts
    assert votes == expected_votes

def test_count_votes_no_votes(base_history_entry: List[History]) -> None:
    # Use the latest entry as base
    base_entry = base_history_entry[-1]
    
    history = [base_entry.copy(phase=GamePhase.DISCUSS)]
    vote_counts, votes = count_votes(history)
    assert vote_counts == {}
    assert votes == {}


# --- Tests for determine_ejection_result ---

def test_determine_ejection_result_clear_winner():
    vote_counts = {"P1": 2, "P2": 1, "nobody": 0}
    ejected, action_type = determine_ejection_result(vote_counts)
    assert ejected == "P1"
    assert action_type == ActionType.KILL

def test_determine_ejection_result_tie_players():
    vote_counts = {"P1": 2, "P2": 2, "nobody": 1}
    ejected, action_type = determine_ejection_result(vote_counts)
    assert ejected == "nobody"
    assert action_type == ActionType.WAIT

def test_determine_ejection_result_tie_with_nobody():
    vote_counts = {"P1": 2, "nobody": 2, "P2": 1}
    ejected, action_type = determine_ejection_result(vote_counts)
    assert ejected == "nobody"
    assert action_type == ActionType.WAIT

def test_determine_ejection_result_all_nobody():
    vote_counts = {"nobody": 3}
    ejected, action_type = determine_ejection_result(vote_counts)
    assert ejected == "nobody"
    assert action_type == ActionType.WAIT

def test_determine_ejection_result_no_votes():
    vote_counts = {}
    with pytest.raises(ValueError, match="No votes casted"):
        determine_ejection_result(vote_counts)

def test_determine_ejection_result_tie():
    vote_counts = {"Player1": 1, "Player2": 1}
    ejected_player, action_type = determine_ejection_result(vote_counts)
    assert ejected_player == "nobody"
    assert action_type == ActionType.WAIT
