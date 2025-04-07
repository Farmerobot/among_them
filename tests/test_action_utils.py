import pytest
from typing import List

from among_them.models.action import Action, ActionType
from among_them.models.action_type import ActionType
from among_them.models.phase import GamePhase
from among_them.models.history import History
from among_them.models.location import Location
from among_them.models.player import Player
from among_them.models.player_role import PlayerRole
from among_them.models.tasks import Task
from among_them.utils.action_utils import get_task_phase_actions, get_vote_actions


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
def base_history_entry(crewmate_player: Player, impostor_player: Player, other_player: Player, dead_player: Player, current_location: Location, task_at_location: Task, task_elsewhere: Task) -> History:
    all_players = [crewmate_player, impostor_player, other_player] # Exclude dead player for next turn
    # Construct tasks_left_to_do based on fixture tasks
    tasks_left = {
        crewmate_player.name: [task_at_location, task_elsewhere],
        impostor_player.name: [],
        other_player.name: [],
        dead_player.name: [] # Dead player has no tasks
    }

    # Define a sample previous action, e.g., crewmate moving to the current location
    previous_action = Action(
        type=ActionType.MOVE,
        player_name=crewmate_player.name,
        target_location=current_location,
        spectator="Moved"
    )

    return History(
        phase=GamePhase.TASK, # Set phase to TASK
        player_names_to_play_next=[p.name for p in all_players if p != crewmate_player], # Next players exclude the one who just acted
        actions_until_phase_ends=10, # Example value
        location=current_location, # Current location of the acting player (crewmate_player)
        impostor_cooldown=0, # Example value, assumes kill is available
        actions_agent_could_take=[], # This will be populated by the function under test
        # Determine spectators based on the fixture setup for this specific state
        spectators_who_saw=[p.name for p in [crewmate_player, impostor_player] if p in all_players], # Only crewmate and impostor are in current_location
        llm_cot="", # Not relevant for these tests
        llm_response="", # Not relevant for these tests
        token_usage={}, # Not relevant for these tests
        action_taken=previous_action,
        tasks_left_to_do=tasks_left,
        votes_before_this_discussion_message={} # Not relevant for TASK phase
    )


# --- Tests for get_task_phase_actions ---

def test_get_task_phase_actions_base(crewmate_player: Player, current_location: Location, base_history_entry: History, alive_players: List[Player]):
    """Test basic actions available to a crewmate."""
    acting_player = crewmate_player
    # Pass the list of players, not the map
    actions = get_task_phase_actions(acting_player, current_location, 0, [base_history_entry], alive_players)
    action_types = {action.type for action in actions}

    assert ActionType.WAIT in action_types
    # Check move actions based on DOORS from CAFETERIA
    assert any(a.type == ActionType.MOVE and a.target_location == Location.WEAPONS for a in actions)
    assert any(a.type == ActionType.MOVE and a.target_location == Location.ADMIN for a in actions)
    # assert any(a.type == ActionType.MOVE and a.target_location == Location.STORAGE for a in actions) # Incorrect path from Cafeteria
    assert any(a.type == ActionType.MOVE and a.target_location == Location.MEDBAY for a in actions)


def test_get_task_phase_actions_report(crewmate_player: Player, current_location: Location, base_history_entry: History, alive_players: List[Player], dead_player: Player, impostor_player: Player):
    """Test that REPORT action is available when a dead body is present."""
    acting_player = crewmate_player
    # Need impostor for the KILL action
    players_list_with_dead = alive_players + [dead_player]

    # History: Impostor KILLS dead_player at current_location
    history = [
        base_history_entry.copy(
            location=current_location,
            action_taken=Action(
                type=ActionType.KILL,
                player_name=impostor_player.name, # Impostor performs the kill
                target_player_name=dead_player.name, # Target is the dead player
                spectator=f"{impostor_player.name} killed {dead_player.name}"
            )
        ),
        # Add a subsequent action by the reporting player to place them in the room
        base_history_entry.copy(
            location=current_location,
            action_taken=Action(
                type=ActionType.WAIT, # Or MOVE, just needs to be player's latest action
                player_name=acting_player.name,
                spectator=f"{acting_player.name} waited"
            )
        )
    ]

    # Call the function with the constructed history
    # Note: The players list passed here might not strictly matter for finding the body
    # as get_dead_players uses the KILL history, but pass it for consistency.
    actions = get_task_phase_actions(acting_player, current_location, 0, history, players_list_with_dead)
    report_actions = [a for a in actions if a.type == ActionType.REPORT]
    assert len(report_actions) == 1, "REPORT action should be available"
    assert report_actions[0].target_player_name == dead_player.name


def test_get_task_phase_actions_no_report_if_body_elsewhere(crewmate_player: Player, current_location: Location, base_history_entry: History, alive_players: List[Player], dead_player: Player, other_location: Location):
    """Test REPORT is not available if the body is in a different location."""
    acting_player = crewmate_player
    players_list_with_dead = alive_players + [dead_player]

    # History: Dead player moves to other_location, acting player moves to current_location
    history = [
        base_history_entry.copy(
            location=other_location,
            action_taken=Action(type=ActionType.MOVE, player_name=dead_player.name, target_location=other_location, spectator=f"{dead_player.name} moved to {other_location.value}")
        ),
         base_history_entry.copy(
            location=current_location,
            action_taken=Action(type=ActionType.MOVE, player_name=acting_player.name, target_location=current_location, spectator=f"{acting_player.name} moved to {current_location.value}")
        )
    ]

    # Call the function with the constructed history
    actions = get_task_phase_actions(acting_player, current_location, 0, history, players_list_with_dead)
    report_actions = [a for a in actions if a.type == ActionType.REPORT]
    assert len(report_actions) == 0, "REPORT action should NOT be available"


def test_get_task_phase_actions_task(crewmate_player: Player, current_location: Location, base_history_entry: History, alive_players: List[Player], task_at_location: Task):
    """Test DO_TASK action is available when a task is at the current location."""
    acting_player = crewmate_player
    # Pass the list of players, not the map
    actions = get_task_phase_actions(acting_player, current_location, 0, [base_history_entry], alive_players)
    task_actions = [a for a in actions if a.type == ActionType.TASK]

    assert len(task_actions) == 1
    assert task_actions[0].target_task.name == task_at_location.name

def test_get_task_phase_actions_no_task_if_elsewhere(crewmate_player: Player, other_location: Location, base_history_entry: History, alive_players: List[Player]):
    """Test DO_TASK is not available if the task is elsewhere."""
    acting_player = crewmate_player
    # Pass the list of players, not the map
    actions = get_task_phase_actions(acting_player, other_location, 0, [base_history_entry], alive_players)
    task_actions = [a for a in actions if a.type == ActionType.TASK]
    assert len(task_actions) == 0

def test_get_task_phase_actions_impostor_kill_available(impostor_player: Player, crewmate_player: Player, current_location: Location, base_history_entry: History, alive_players: List[Player]):
    """Test KILL action is available for impostor when cooldown is 0 and target is present."""
    acting_player = impostor_player
    target_player = crewmate_player # Target is the crewmate

    # History: Target moves to location, then Impostor moves to the same location
    history = [
        base_history_entry.copy(
            location=current_location,
            action_taken=Action(type=ActionType.MOVE, player_name=target_player.name, target_location=current_location, spectator=f"{target_player.name} moved to {current_location.value}")
        ),
        base_history_entry.copy(
            location=current_location, # Impostor is here
            impostor_cooldown=0, # Set cooldown in the *latest* history entry for the actor
            action_taken=Action(type=ActionType.MOVE, player_name=acting_player.name, target_location=current_location, spectator=f"{acting_player.name} moved to {current_location.value}")
        )
    ]

    # Call the function with 0 cooldown and constructed history
    actions = get_task_phase_actions(acting_player, current_location, 0, history, alive_players)
    kill_actions = [a for a in actions if a.type == ActionType.KILL]
    assert len(kill_actions) > 0, "KILL action should be available"
    assert any(a.target_player_name == target_player.name for a in kill_actions), f"KILL action targeting {target_player.name} should exist"


def test_get_task_phase_actions_impostor_kill_cooldown(impostor_player: Player, crewmate_player: Player, current_location: Location, base_history_entry: History, alive_players: List[Player]):
    """Test KILL action is not available when cooldown is > 0."""
    acting_player = impostor_player
    target_player = crewmate_player
    cooldown = 10

    # Ensure crewmate is alive
    target_player.alive = True

    # History: Target moves to location, then Impostor moves to the same location
    history = [
        base_history_entry.copy(
            location=current_location,
            action_taken=Action(type=ActionType.MOVE, player_name=target_player.name, target_location=current_location, spectator=f"{target_player.name} moved to {current_location.value}")
        ),
        base_history_entry.copy(
            location=current_location, # Impostor is here
            impostor_cooldown=cooldown, # Set cooldown in the *latest* history entry for the actor
            action_taken=Action(type=ActionType.MOVE, player_name=acting_player.name, target_location=current_location, spectator=f"{acting_player.name} moved to {current_location.value}")
        )
    ]

    # Call the function with the specific cooldown and constructed history
    actions = get_task_phase_actions(acting_player, current_location, cooldown, history, alive_players)
    kill_actions = [a for a in actions if a.type == ActionType.KILL]
    assert len(kill_actions) == 0, "KILL action should NOT be available due to cooldown"


def test_get_task_phase_actions_impostor_kill_no_target(impostor_player: Player, current_location: Location, base_history_entry: History, alive_players: List[Player], other_location: Location):
    """Test KILL action is not available when no target is present."""
    acting_player = impostor_player

    # Find a crewmate to explicitly move elsewhere
    crewmate_to_move = next((p for p in alive_players if p.role == PlayerRole.CREWMATE and p != acting_player), None)
    assert crewmate_to_move is not None, "Test setup needs at least one crewmate"

    # History: Crewmate moves to other_location, Impostor moves to current_location
    history = [
         base_history_entry.copy(
            location=other_location, # Crewmate is elsewhere
            action_taken=Action(type=ActionType.MOVE, player_name=crewmate_to_move.name, target_location=other_location, spectator=f"{crewmate_to_move.name} moved to {other_location.value}")
        ),
        base_history_entry.copy(
            location=current_location, # Impostor is here
            impostor_cooldown=0,
            action_taken=Action(type=ActionType.MOVE, player_name=acting_player.name, target_location=current_location, spectator=f"{acting_player.name} moved to {current_location.value}")
        )
    ]

    # Call the function with 0 cooldown and constructed history
    actions = get_task_phase_actions(acting_player, current_location, 0, history, alive_players)
    kill_actions = [a for a in actions if a.type == ActionType.KILL]
    assert len(kill_actions) == 0, "KILL action should NOT be available as no target is present"

def test_get_task_phase_actions_impostor_pretend(impostor_player: Player, current_location: Location, base_history_entry: History, alive_players: List[Player]):
    """Test PRETEND_TASK action is available for impostor."""
    acting_player = impostor_player
    # Pass the list of players, not the map
    actions = get_task_phase_actions(acting_player, current_location, 0, [base_history_entry], alive_players)
    pretend_actions = [a for a in actions if a.type == ActionType.PRETEND]
    # Check the *specific* pretend task returned for Cafeteria based on get_impostor_pretend_tasks_at_location implementation
    assert len(pretend_actions) == 1, "Should generate exactly one PRETEND action"
    assert pretend_actions[0].target_task.name == "Empty the cafeteria trash", "Incorrect PRETEND task generated for Cafeteria"

def test_get_task_phase_actions_crewmate_no_impostor_actions(crewmate_player: Player, current_location: Location, base_history_entry: History, alive_players: List[Player]):
    """Test that crewmates cannot KILL or PRETEND_TASK."""
    acting_player = crewmate_player
    # Pass the list of players, not the map
    actions = get_task_phase_actions(acting_player, current_location, 0, [base_history_entry], alive_players)
    action_types = {action.type for action in actions}
    assert ActionType.KILL not in action_types
    assert ActionType.PRETEND not in action_types


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
