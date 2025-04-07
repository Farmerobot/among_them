"""Common fixtures for all tests."""
import pytest
from typing import List

from among_them.models.phase import GamePhase
from among_them.models.player import Player
from among_them.models.location import Location
from among_them.models.player_role import PlayerRole
from among_them.models.action import Action
from among_them.models.action_type import ActionType
from among_them.models.history import History, initialize_history
from among_them.models.tasks import Task
from among_them.consts import IMPOSTOR_COOLDOWN
from tests.utils import create_base_history


# --- Generic Player Fixtures ---

@pytest.fixture
def crewmate_player() -> Player:
    """Create a generic crewmate player."""
    return Player(name="Crewmate", role=PlayerRole.CREWMATE)


@pytest.fixture
def impostor_player() -> Player:
    """Create a generic impostor player."""
    return Player(name="Impostor", role=PlayerRole.IMPOSTOR)


@pytest.fixture
def other_player() -> Player:
    """Create another generic crewmate player."""
    return Player(name="OtherPlayer", role=PlayerRole.CREWMATE)


@pytest.fixture
def dead_player() -> Player:
    """Create a generic dead player."""
    player = Player(name="DeadPlayer", role=PlayerRole.CREWMATE)
    return player


@pytest.fixture
def generic_test_players(crewmate_player: Player, impostor_player: Player, other_player: Player) -> List[Player]:
    """Fixture for a list of generic test players (crewmate, impostor, other)."""
    return [crewmate_player, impostor_player, other_player]


@pytest.fixture
def all_generic_test_players(generic_test_players: List[Player], dead_player: Player) -> List[Player]:
    """Fixture for all generic test players including the dead one."""
    return generic_test_players + [dead_player]


# --- Location & Task Fixtures ---

@pytest.fixture
def cafeteria_location() -> Location:
    """The cafeteria location."""
    return Location.CAFETERIA


@pytest.fixture
def weapons_location() -> Location:
    """The weapons location."""
    return Location.WEAPONS


@pytest.fixture
def medbay_location() -> Location:
    """The medbay location."""
    return Location.MEDBAY


@pytest.fixture
def task_in_cafeteria() -> Task:
    """Create a task in the cafeteria."""
    return Task(name="Fix Wires", location=Location.CAFETERIA)


@pytest.fixture
def task_in_navigation() -> Task:
    """Create a task in navigation."""
    return Task(name="Upload Data", location=Location.NAVIGATION)


# --- Basic Action Fixtures ---

@pytest.fixture
def base_action(crewmate_player: Player) -> Action:
    """Create a basic WAIT action for crewmate player."""
    return Action(type=ActionType.WAIT, player_name=crewmate_player.name)


@pytest.fixture
def kill_action(impostor_player: Player, crewmate_player: Player) -> Action:
    """Create an action for impostor player killing crewmate player."""
    return Action(type=ActionType.KILL, player_name=impostor_player.name, target_player_name=crewmate_player.name)


@pytest.fixture
def report_action(crewmate_player: Player, dead_player: Player) -> Action:
    """Create an action for crewmate player reporting dead player."""
    return Action(type=ActionType.REPORT, player_name=crewmate_player.name, target_player_name=dead_player.name)


@pytest.fixture
def move_action(crewmate_player: Player, medbay_location: Location) -> Action:
    """Create an action for crewmate player moving to MEDBAY."""
    return Action(type=ActionType.MOVE, player_name=crewmate_player.name, target_location=medbay_location)


@pytest.fixture
def move_action_p2_weapons(impostor_player: Player) -> Action:
    """Create an action for impostor player moving to WEAPONS."""
    return Action(type=ActionType.MOVE, player_name=impostor_player.name, target_location=Location.WEAPONS)


@pytest.fixture
def discuss_action(crewmate_player: Player) -> Action:
    """Create a SPEAK action for crewmate player."""
    return Action(type=ActionType.SPEAK, player_name=crewmate_player.name, text="Hello")


@pytest.fixture
def vote_action(crewmate_player: Player, impostor_player: Player) -> Action:
    """Create a VOTE action for crewmate player voting impostor player."""
    return Action(type=ActionType.VOTE, player_name=crewmate_player.name, target_player_name=impostor_player.name)


@pytest.fixture
def vote_nobody_action(impostor_player: Player) -> Action:
    """Create a VOTE action for impostor player voting nobody."""
    return Action(type=ActionType.VOTE, player_name=impostor_player.name, target_player_name="nobody")


# --- Basic History Fixture ---

@pytest.fixture
def base_history_entry(all_generic_test_players: List[Player], crewmate_player: Player) -> List[History]:
    """Create a consistent base history entry using generic test players."""
    return create_base_history(all_generic_test_players, crewmate_player, Location.CAFETERIA)


# --- History Fixtures for Generic Test Scenarios ---

@pytest.fixture
def generic_initial_history(all_generic_test_players: List[Player]) -> List[History]:
    """Create the initial history entry using generic test players."""
    return initialize_history(all_generic_test_players)


@pytest.fixture
def generic_test_history_entry(generic_initial_history: List[History], cafeteria_location: Location, 
                         crewmate_player: Player, task_in_cafeteria: Task, 
                         task_in_navigation: Task, generic_test_players: List[Player]) -> List[History]:
    """
    Base history for generic testing, starting with the initialized generic history
    and adding a move action to the cafeteria.
    """
    # Start with the initialized history
    history = generic_initial_history.copy()
    
    # Update tasks for the crewmate player
    first_entry = history[0]
    tasks_left = first_entry.tasks_left_to_do.copy()
    tasks_left[crewmate_player.name] = [task_in_cafeteria, task_in_navigation]
    
    # Add a move action to the location
    move_action = Action(
        type=ActionType.MOVE,
        player_name=crewmate_player.name,
        target_location=cafeteria_location,
        spectator=f"{crewmate_player.name} moved to {cafeteria_location.value}"
    )
    
    # Create a new history entry after the move
    spectators = [p.name for p in generic_test_players]
    
    move_history = History(
        player_names_to_play_next=[p for p in first_entry.player_names_to_play_next if p != crewmate_player.name],
        phase=GamePhase.TASK,
        actions_until_phase_ends=10,
        location=cafeteria_location,
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
