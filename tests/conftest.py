"""Common fixtures for all tests."""
from typing import Callable, Dict, List
from unittest.mock import patch

import pytest

from among_them.models.action import Action
from among_them.models.action_type import ActionType
from among_them.models.game_config import GameConfig
from among_them.models.history import History, initialize_history
from among_them.models.location import Location
from among_them.models.phase import GamePhase
from among_them.models.player import Player
from among_them.models.player_role import PlayerRole
from among_them.models.tasks import Task
from tests.utils import create_history_entry

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
def wait_action(crewmate_player: Player) -> Action:
    """Create a basic WAIT action for crewmate player."""
    return Action(type=ActionType.WAIT, player_name=crewmate_player.name)


@pytest.fixture
def kill_action(impostor_player: Player, dead_player: Player) -> Action:
    """Create an action for impostor player killing dead player."""
    return Action(type=ActionType.KILL, player_name=impostor_player.name, target_player_name=dead_player.name)


@pytest.fixture
def report_action(crewmate_player: Player, dead_player: Player) -> Action:
    """Create an action for crewmate player reporting dead player."""
    return Action(type=ActionType.REPORT, player_name=crewmate_player.name, target_player_name=dead_player.name)


@pytest.fixture
def move_to_medbay_action(crewmate_player: Player, medbay_location: Location) -> Action:
    """Create an action for crewmate player moving to MEDBAY."""
    return Action(type=ActionType.MOVE, player_name=crewmate_player.name, target_location=medbay_location)


@pytest.fixture
def move_to_weapons_action(impostor_player: Player, weapons_location: Location) -> Action:
    """Create an action for impostor player moving to WEAPONS."""
    return Action(type=ActionType.MOVE, player_name=impostor_player.name, target_location=weapons_location)


@pytest.fixture
def speak_action(crewmate_player: Player) -> Action:
    """Create a SPEAK action for crewmate player."""
    message = "Hello everyone, I think we should investigate"
    return Action(
        type=ActionType.SPEAK, 
        player_name=crewmate_player.name, 
        text="speak",
        result=f"[{crewmate_player.name}]: {message}",
        spectator=f"[{crewmate_player.name}]: {message}"
    )


@pytest.fixture
def vote_impostor_action(crewmate_player: Player, impostor_player: Player) -> Action:
    """Create a VOTE action for crewmate player voting impostor player."""
    return Action(type=ActionType.VOTE, player_name=crewmate_player.name, target_player_name=impostor_player.name)


@pytest.fixture
def vote_nobody_action(impostor_player: Player) -> Action:
    """Create a VOTE action for impostor player voting nobody."""
    return Action(type=ActionType.VOTE, player_name=impostor_player.name, target_player_name="nobody")


# --- Game Config Fixture ---

@pytest.fixture
def game_config() -> GameConfig:
    """Create a default game configuration for tests."""
    return GameConfig(
        num_tasks=4,
        num_players=5,
        num_impostors=1,
        map_size=2,
        num_task_phase_actions_per_player=7,
        num_discuss_phase_actions_per_player=2,
        impostor_cooldown=1
    )


# --- Specific History Entry Fixtures ---

@pytest.fixture
def crewmate_wait_history(all_generic_test_players: List[Player], crewmate_player: Player, 
                         wait_action: Action, cafeteria_location: Location, 
                         game_config: GameConfig) -> History:
    """Create a history entry for crewmate waiting in cafeteria."""
    return create_history_entry(
        all_players=all_generic_test_players, 
        acting_player=crewmate_player, 
        action=wait_action,
        location=cafeteria_location,
        phase=GamePhase.TASK,
        game_config=game_config
    )

@pytest.fixture
def impostor_wait_history(all_generic_test_players: List[Player], impostor_player: Player, 
                         wait_action: Action, cafeteria_location: Location, 
                         game_config: GameConfig) -> History:
    """Create a history entry for impostor waiting in cafeteria."""
    return create_history_entry(
        all_players=all_generic_test_players, 
        acting_player=impostor_player, 
        action=wait_action.copy(player_name=impostor_player.name),
        location=cafeteria_location,
        phase=GamePhase.TASK,
        game_config=game_config
    )


@pytest.fixture
def impostor_kill_history(all_generic_test_players: List[Player], impostor_player: Player, 
                         kill_action: Action, cafeteria_location: Location, 
                         game_config: GameConfig) -> History:
    """Create a history entry for impostor killing a player in cafeteria."""
    return create_history_entry(
        all_players=all_generic_test_players, 
        acting_player=impostor_player, 
        action=kill_action,
        location=cafeteria_location,
        phase=GamePhase.TASK,
        impostor_cooldown=1,  # Set cooldown after kill
        game_config=game_config
    )


@pytest.fixture
def crewmate_report_history(all_generic_test_players: List[Player], crewmate_player: Player, 
                           report_action: Action, cafeteria_location: Location, 
                           game_config: GameConfig) -> History:
    """Create a history entry for crewmate reporting a dead body in cafeteria."""
    return create_history_entry(
        all_players=all_generic_test_players, 
        acting_player=crewmate_player, 
        action=report_action,
        location=cafeteria_location,
        phase=GamePhase.TASK,
        game_config=game_config
    )


@pytest.fixture
def crewmate_move_history(all_generic_test_players: List[Player], crewmate_player: Player, 
                         move_to_medbay_action: Action, medbay_location: Location, 
                         game_config: GameConfig) -> History:
    """Create a history entry for crewmate moving to medbay."""
    return create_history_entry(
        all_players=all_generic_test_players, 
        acting_player=crewmate_player, 
        action=move_to_medbay_action,
        location=medbay_location,  # Location after move
        phase=GamePhase.TASK,
        game_config=game_config
    )


@pytest.fixture
def impostor_move_history(all_generic_test_players: List[Player], impostor_player: Player, 
                         move_to_weapons_action: Action, weapons_location: Location, 
                         game_config: GameConfig) -> History:
    """Create a history entry for impostor moving to weapons."""
    return create_history_entry(
        all_players=all_generic_test_players, 
        acting_player=impostor_player, 
        action=move_to_weapons_action,
        location=weapons_location,  # Location after move
        phase=GamePhase.TASK,
        game_config=game_config
    )


@pytest.fixture
def crewmate_speak_history(all_generic_test_players: List[Player], crewmate_player: Player, 
                          speak_action: Action, cafeteria_location: Location, 
                          game_config: GameConfig) -> History:
    """Create a history entry for crewmate speaking during discussion."""
    return create_history_entry(
        all_players=all_generic_test_players, 
        acting_player=crewmate_player, 
        action=speak_action,
        location=cafeteria_location,
        phase=GamePhase.DISCUSS,
        game_config=game_config
    )


@pytest.fixture
def crewmate_vote_history(all_generic_test_players: List[Player], crewmate_player: Player, 
                         vote_impostor_action: Action, cafeteria_location: Location, 
                         game_config: GameConfig) -> History:
    """Create a history entry for crewmate voting for impostor."""
    return create_history_entry(
        all_players=all_generic_test_players, 
        acting_player=crewmate_player, 
        action=vote_impostor_action,
        location=cafeteria_location,
        phase=GamePhase.VOTE,
        game_config=game_config
    )


@pytest.fixture
def other_vote_history(all_generic_test_players: List[Player], other_player: Player, 
                      vote_impostor_action: Action, cafeteria_location: Location, 
                      game_config: GameConfig) -> History:
    """Create a history entry for other player voting for impostor."""
    return create_history_entry(
        all_players=all_generic_test_players, 
        acting_player=other_player, 
        action=vote_impostor_action.copy(player_name=other_player.name),
        location=cafeteria_location,
        phase=GamePhase.VOTE,
        game_config=game_config
    )


@pytest.fixture
def impostor_vote_history(all_generic_test_players: List[Player], impostor_player: Player, 
                         vote_nobody_action: Action, cafeteria_location: Location, 
                         game_config: GameConfig) -> History:
    """Create a history entry for impostor voting for nobody."""
    return create_history_entry(
        all_players=all_generic_test_players, 
        acting_player=impostor_player, 
        action=vote_nobody_action,
        location=cafeteria_location,
        phase=GamePhase.VOTE,
        game_config=game_config
    )


# --- Phase-specific History Entry Fixtures ---

@pytest.fixture
def task_phase_history(all_generic_test_players: List[Player], crewmate_player: Player,
                      wait_action: Action, cafeteria_location: Location, 
                      game_config: GameConfig) -> History:
    """Create a history entry in TASK phase."""
    return create_history_entry(
        all_players=all_generic_test_players, 
        acting_player=crewmate_player, 
        action=wait_action,
        location=cafeteria_location,
        phase=GamePhase.TASK,
        actions_until_phase_ends=10,
        game_config=game_config
    )


@pytest.fixture
def discuss_phase_history(all_generic_test_players: List[Player], crewmate_player: Player,
                         speak_action: Action, cafeteria_location: Location, 
                         game_config: GameConfig) -> History:
    """Create a history entry in DISCUSS phase."""
    return create_history_entry(
        all_players=all_generic_test_players, 
        acting_player=crewmate_player, 
        action=speak_action,
        location=cafeteria_location,
        phase=GamePhase.DISCUSS,
        actions_until_phase_ends=10,
        game_config=game_config
    )


@pytest.fixture
def vote_phase_history(all_generic_test_players: List[Player], crewmate_player: Player,
                      vote_impostor_action: Action, cafeteria_location: Location, 
                      game_config: GameConfig) -> History:
    """Create a history entry in VOTE phase."""
    return create_history_entry(
        all_players=all_generic_test_players, 
        acting_player=crewmate_player, 
        action=vote_impostor_action,
        location=cafeteria_location,
        phase=GamePhase.VOTE,
        actions_until_phase_ends=len(all_generic_test_players),
        game_config=game_config
    )


# --- Initialized History Fixtures ---

@pytest.fixture
@patch('among_them.models.tasks.get_crewmate_tasks')
def initial_history(get_crewmate_tasks, all_generic_test_players: List[Player], game_config: GameConfig) -> History:
    """Create the initial history entry using generic test players."""
    get_crewmate_tasks.return_value = [Task(name="Do something", location=Location.CAFETERIA)]
    return initialize_history(all_generic_test_players, game_config)[0]
