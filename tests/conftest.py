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
    return Task(name="Fix something", location=Location.CAFETERIA)


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

