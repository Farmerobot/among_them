import pytest
from typing import Dict, List

from among_them.models.action import Action, ActionType
from among_them.models.phase import GamePhase
from among_them.models.history import History, initialize_history
from among_them.models.location import Location
from among_them.models.player import Player
from among_them.models.player_role import PlayerRole
from among_them.models.tasks import Task
from among_them.utils.action_utils import get_task_phase_actions, get_vote_actions
from among_them.consts import IMPOSTOR_COOLDOWN


@pytest.fixture
def crewmate_player() -> Player:
    return Player(name="Crewmate1", role=PlayerRole.CREWMATE)

@pytest.fixture
def impostor_player() -> Player:
    return Player(name="Impostor1", role=PlayerRole.IMPOSTOR)

@pytest.fixture
def other_player() -> Player:
    return Player(name="OtherPlayer", role=PlayerRole.CREWMATE)

@pytest.fixture
def dead_player() -> Player:
    return Player(name="DeadPlayer", role=PlayerRole.CREWMATE)

@pytest.fixture
def current_location() -> Location:
    return Location.CAFETERIA

@pytest.fixture
def other_location() -> Location:
    return Location.WEAPONS

@pytest.fixture
def task_at_location() -> Task:
    return Task(name="Fix Wires", location=Location.CAFETERIA)

@pytest.fixture
def task_elsewhere() -> Task:
    return Task(name="Upload Data", location=Location.NAVIGATION)

@pytest.fixture
def alive_players(crewmate_player: Player, impostor_player: Player, other_player: Player) -> List[Player]:
    """Fixture for a list of currently alive players for action tests."""
    return [crewmate_player, impostor_player, other_player]

@pytest.fixture
def all_players(alive_players: List[Player], dead_player: Player) -> List[Player]:
    """Fixture for all players including the dead player."""
    return alive_players + [dead_player]

@pytest.fixture
def initial_history(all_players: List[Player]) -> List[History]:
    """Create the initial history entry using initialize_history."""
    return initialize_history(all_players)

@pytest.fixture
def base_history_entry(initial_history: List[History], current_location: Location, 
                       crewmate_player: Player, task_at_location: Task, 
                       task_elsewhere: Task, alive_players: List[Player],
                       dead_player: Player) -> List[History]:
    """
    Base history for testing, starting with the initialized history
    and adding a move action to the current location.
    """
    # Start with the initialized history
    history = initial_history.copy()
    
    # Update tasks for the crewmate player
    first_entry = history[0]
    tasks_left = first_entry.tasks_left_to_do.copy()
    tasks_left[crewmate_player.name] = [task_at_location, task_elsewhere]
    
    # Add a move action to the current location
    move_action = Action(
        type=ActionType.MOVE,
        player_name=crewmate_player.name,
        target_location=current_location,
        spectator=f"{crewmate_player.name} moved to {current_location.value}"
    )
    
    # Create a new history entry after the move
    # Use explicit list of player names rather than trying to iterate all_players fixture
    spectators = [p.name for p in alive_players]
    
    move_history = History(
        player_names_to_play_next=[p for p in first_entry.player_names_to_play_next if p != crewmate_player.name],
        phase=GamePhase.TASK,
        actions_until_phase_ends=10,
        location=current_location,
        impostor_cooldown=IMPOSTOR_COOLDOWN,
        actions_agent_could_take=[],
        spectators_who_saw=spectators,
        llm_cot="",
        llm_response="",
        token_usage={},
        action_taken=move_action,
        tasks_left_to_do=tasks_left,
        votes_before_this_discussion_message={}
    )
    
    history.append(move_history)
    return history


# --- Tests for get_task_phase_actions ---

def test_get_task_phase_actions_base(crewmate_player: Player, current_location: Location, 
                                    base_history_entry: List[History], alive_players: List[Player]):
    """Test basic actions available to a crewmate."""
    acting_player = crewmate_player
    
    actions = get_task_phase_actions(acting_player, current_location, 0, base_history_entry, alive_players)
    action_types = {action.type for action in actions}

    assert ActionType.WAIT in action_types
    # Check move actions based on DOORS from CAFETERIA
    assert any(a.type == ActionType.MOVE and a.target_location == Location.WEAPONS for a in actions)
    assert any(a.type == ActionType.MOVE and a.target_location == Location.ADMIN for a in actions)
    assert any(a.type == ActionType.MOVE and a.target_location == Location.MEDBAY for a in actions)


def test_get_task_phase_actions_report(crewmate_player: Player, current_location: Location, 
                                      base_history_entry: List[History], alive_players: List[Player], 
                                      dead_player: Player, impostor_player: Player):
    """Test that REPORT action is available when a dead body is present."""
    acting_player = crewmate_player
    players_list_with_dead = alive_players + [dead_player]
    
    # Start with the base history
    history = base_history_entry.copy()
    
    # Add kill action - impostor kills dead_player at current_location
    kill_action = Action(
        type=ActionType.KILL,
        player_name=impostor_player.name,
        target_player_name=dead_player.name,
        spectator=f"{impostor_player.name} killed {dead_player.name}"
    )
    
    next_players = [p for p in history[-1].player_names_to_play_next if p != impostor_player.name]
    
    kill_history = history[-1].copy(
        location=current_location,
        action_taken=kill_action,
        player_names_to_play_next=next_players
    )
    history.append(kill_history)
    
    # Add a subsequent action by the reporting player to place them in the room
    wait_action = Action(
        type=ActionType.WAIT,
        player_name=acting_player.name,
        spectator=f"{acting_player.name} waited"
    )
    
    next_players = [p for p in kill_history.player_names_to_play_next if p != acting_player.name]
    
    wait_history = kill_history.copy(
        action_taken=wait_action,
        player_names_to_play_next=next_players
    )
    history.append(wait_history)

    # Call the function with the constructed history
    actions = get_task_phase_actions(acting_player, current_location, 0, history, players_list_with_dead)
    report_actions = [a for a in actions if a.type == ActionType.REPORT]
    assert len(report_actions) == 1, "REPORT action should be available"
    assert report_actions[0].target_player_name == dead_player.name


def test_get_task_phase_actions_no_report_if_body_elsewhere(crewmate_player: Player, current_location: Location, 
                                                          base_history_entry: List[History], alive_players: List[Player], 
                                                          dead_player: Player, other_location: Location):
    """Test REPORT is not available if the body is in a different location."""
    acting_player = crewmate_player
    players_list_with_dead = alive_players + [dead_player]

    # Start with the base history
    history = base_history_entry.copy()
    
    # Dead player moves to other_location
    dead_move_action = Action(
        type=ActionType.MOVE,
        player_name=dead_player.name,
        target_location=other_location,
        spectator=f"{dead_player.name} moved to {other_location.value}"
    )
    
    next_players = [p for p in history[-1].player_names_to_play_next if p != dead_player.name]
    
    dead_move_history = history[-1].copy(
        location=other_location,
        action_taken=dead_move_action,
        player_names_to_play_next=next_players
    )
    history.append(dead_move_history)
    
    # Acting player moves to current_location
    player_move_action = Action(
        type=ActionType.MOVE,
        player_name=acting_player.name,
        target_location=current_location,
        spectator=f"{acting_player.name} moved to {current_location.value}"
    )
    
    next_players = [p for p in dead_move_history.player_names_to_play_next if p != acting_player.name]
    
    player_move_history = dead_move_history.copy(
        location=current_location,
        action_taken=player_move_action,
        player_names_to_play_next=next_players
    )
    history.append(player_move_history)

    # Call the function with the constructed history
    actions = get_task_phase_actions(acting_player, current_location, 0, history, players_list_with_dead)
    report_actions = [a for a in actions if a.type == ActionType.REPORT]
    assert len(report_actions) == 0, "REPORT action should NOT be available"


def test_get_task_phase_actions_task(crewmate_player: Player, current_location: Location, 
                                    base_history_entry: List[History], alive_players: List[Player], 
                                    task_at_location: Task):
    """Test DO_TASK action is available when a task is at the current location."""
    acting_player = crewmate_player
    
    actions = get_task_phase_actions(acting_player, current_location, 0, base_history_entry, alive_players)
    task_actions = [a for a in actions if a.type == ActionType.TASK]

    assert len(task_actions) == 1
    assert task_actions[0].target_task.name == task_at_location.name


def test_get_task_phase_actions_no_task_if_elsewhere(crewmate_player: Player, other_location: Location, 
                                                   base_history_entry: List[History], alive_players: List[Player]):
    """Test DO_TASK is not available if the task is elsewhere."""
    acting_player = crewmate_player
    
    # Start with the base history
    history = base_history_entry.copy()
    
    # Add action to move player to other_location
    move_action = Action(
        type=ActionType.MOVE,
        player_name=acting_player.name,
        target_location=other_location,
        spectator=f"{acting_player.name} moved to {other_location.value}"
    )
    
    next_players = [p for p in history[-1].player_names_to_play_next if p != acting_player.name]
    
    move_history = history[-1].copy(
        location=other_location,
        action_taken=move_action,
        player_names_to_play_next=next_players
    )
    history.append(move_history)
    
    actions = get_task_phase_actions(acting_player, other_location, 0, history, alive_players)
    task_actions = [a for a in actions if a.type == ActionType.TASK]
    assert len(task_actions) == 0


def test_get_task_phase_actions_impostor_kill_available(impostor_player: Player, crewmate_player: Player, 
                                                      current_location: Location, base_history_entry: List[History], 
                                                      alive_players: List[Player]):
    """Test KILL action is available for impostor when cooldown is 0 and target is present."""
    acting_player = impostor_player
    target_player = crewmate_player

    # Start with the base history
    history = base_history_entry.copy()
    
    # Target moves to location
    target_move_action = Action(
        type=ActionType.MOVE,
        player_name=target_player.name,
        target_location=current_location,
        spectator=f"{target_player.name} moved to {current_location.value}"
    )
    
    next_players = [p for p in history[-1].player_names_to_play_next if p != target_player.name]
    
    target_move_history = history[-1].copy(
        location=current_location,
        action_taken=target_move_action,
        player_names_to_play_next=next_players
    )
    history.append(target_move_history)
    
    # Impostor moves to the same location
    impostor_move_action = Action(
        type=ActionType.MOVE,
        player_name=acting_player.name,
        target_location=current_location,
        spectator=f"{acting_player.name} moved to {current_location.value}"
    )
    
    next_players = [p for p in target_move_history.player_names_to_play_next if p != acting_player.name]
    
    impostor_move_history = target_move_history.copy(
        location=current_location,
        impostor_cooldown=0,  # Set cooldown to 0
        action_taken=impostor_move_action,
        player_names_to_play_next=next_players
    )
    history.append(impostor_move_history)

    # Check available actions
    actions = get_task_phase_actions(acting_player, current_location, 0, history, alive_players)
    kill_actions = [a for a in actions if a.type == ActionType.KILL]
    
    assert len(kill_actions) >= 1, "KILL action should be available"
    assert any(a.target_player_name == target_player.name for a in kill_actions), "Should be able to kill the target"


def test_get_task_phase_actions_impostor_kill_cooldown(impostor_player: Player, crewmate_player: Player, 
                                                     current_location: Location, base_history_entry: List[History], 
                                                     alive_players: List[Player]):
    """Test KILL action is not available when cooldown is > 0."""
    acting_player = impostor_player
    target_player = crewmate_player

    # Start with the base history
    history = base_history_entry.copy()
    
    # Target moves to location
    target_move_action = Action(
        type=ActionType.MOVE,
        player_name=target_player.name,
        target_location=current_location,
        spectator=f"{target_player.name} moved to {current_location.value}"
    )
    
    next_players = [p for p in history[-1].player_names_to_play_next if p != target_player.name]
    
    target_move_history = history[-1].copy(
        location=current_location,
        action_taken=target_move_action,
        player_names_to_play_next=next_players
    )
    history.append(target_move_history)
    
    # Impostor moves to the same location with cooldown > 0
    impostor_move_action = Action(
        type=ActionType.MOVE,
        player_name=acting_player.name,
        target_location=current_location,
        spectator=f"{acting_player.name} moved to {current_location.value}"
    )
    
    next_players = [p for p in target_move_history.player_names_to_play_next if p != acting_player.name]
    
    impostor_move_history = target_move_history.copy(
        location=current_location,
        impostor_cooldown=5,  # Set cooldown > 0
        action_taken=impostor_move_action,
        player_names_to_play_next=next_players
    )
    history.append(impostor_move_history)

    # Check available actions
    actions = get_task_phase_actions(acting_player, current_location, 5, history, alive_players)
    kill_actions = [a for a in actions if a.type == ActionType.KILL]
    
    assert len(kill_actions) == 0, "KILL action should not be available when cooldown > 0"


def test_get_task_phase_actions_impostor_kill_no_target(impostor_player: Player, current_location: Location, 
                                                      base_history_entry: List[History], alive_players: List[Player], 
                                                      other_location: Location):
    """Test KILL action is not available when no target is present."""
    acting_player = impostor_player

    # Start with the base history
    history = base_history_entry.copy()
    
    # Other players move to other_location
    for player in [p for p in alive_players if p != impostor_player]:
        move_action = Action(
            type=ActionType.MOVE,
            player_name=player.name,
            target_location=other_location,
            spectator=f"{player.name} moved to {other_location.value}"
        )
        
        next_players = [p for p in history[-1].player_names_to_play_next if p != player.name]
        
        move_history = history[-1].copy(
            location=other_location,
            action_taken=move_action,
            player_names_to_play_next=next_players
        )
        history.append(move_history)
    
    # Impostor moves to current_location
    impostor_move_action = Action(
        type=ActionType.MOVE,
        player_name=acting_player.name,
        target_location=current_location,
        spectator=f"{acting_player.name} moved to {current_location.value}"
    )
    
    next_players = [p for p in history[-1].player_names_to_play_next if p != acting_player.name]
    
    impostor_move_history = history[-1].copy(
        location=current_location,
        impostor_cooldown=0,  # Cooldown is 0
        action_taken=impostor_move_action,
        player_names_to_play_next=next_players
    )
    history.append(impostor_move_history)

    # Check available actions
    actions = get_task_phase_actions(acting_player, current_location, 0, history, alive_players)
    kill_actions = [a for a in actions if a.type == ActionType.KILL]
    
    assert len(kill_actions) == 0, "KILL action should not be available when no target is present"


def test_get_task_phase_actions_impostor_pretend(impostor_player: Player, current_location: Location, 
                                                base_history_entry: List[History], alive_players: List[Player]):
    """Test PRETEND_TASK action is available for impostor."""
    acting_player = impostor_player
    
    actions = get_task_phase_actions(acting_player, current_location, 0, base_history_entry, alive_players)
    pretend_actions = [a for a in actions if a.type == ActionType.PRETEND]
    
    assert len(pretend_actions) >= 1, "PRETEND_TASK action should be available for impostor"


def test_get_task_phase_actions_crewmate_no_impostor_actions(crewmate_player: Player, current_location: Location, 
                                                           base_history_entry: List[History], alive_players: List[Player]):
    """Test that crewmates cannot KILL or PRETEND_TASK."""
    acting_player = crewmate_player
    
    actions = get_task_phase_actions(acting_player, current_location, 0, base_history_entry, alive_players)
    impostor_action_types = [ActionType.KILL, ActionType.PRETEND]
    
    for action in actions:
        assert action.type not in impostor_action_types, f"Crewmate should not have {action.type} action available"


# --- Tests for get_vote_actions ---

def test_get_vote_actions(crewmate_player: Player, impostor_player: Player, other_player: Player):
    alive_players = [crewmate_player, impostor_player, other_player]
    actions = get_vote_actions(alive_players, crewmate_player)
    vote_targets = {a.target_player_name for a in actions if a.type == ActionType.VOTE}

    assert len(actions) == 3 # nobody + 2 other players
    assert "nobody" in vote_targets
    assert impostor_player.name in vote_targets
    assert other_player.name in vote_targets
    assert crewmate_player.name not in vote_targets # Can't vote for self

def test_get_vote_actions_only_self(crewmate_player: Player):
    alive_players = [crewmate_player]
    actions = get_vote_actions(alive_players, crewmate_player)
    vote_targets = {a.target_player_name for a in actions if a.type == ActionType.VOTE}

    assert len(actions) == 1
    assert "nobody" in vote_targets
