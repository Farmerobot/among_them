from typing import List

from among_them.models.action import Action, ActionType
from among_them.models.history import History
from among_them.models.location import DOORS, Location
from among_them.models.player import Player
from among_them.models.player_role import PlayerRole
from among_them.models.tasks import get_impostor_pretend_tasks_at_location
from among_them.utils.player_utils import get_dead_players, get_players_in_room


def get_task_phase_actions(
    player: Player,
    location: Location,
    cooldown: int,
    history: List[History],
    players: List[Player],
) -> list[Action]:
    """Creates available actions based on the circumstances.

    Returns:
        A list of actions
    """
    actions = []

    # actions for WAIT
    actions.append(Action(type=ActionType.WAIT, player_name=player.name))

    # action for REPORT
    dead_players_in_room_names = [name for name, dead_location in get_dead_players(history, players).items() if dead_location == location]
    for dead_name in dead_players_in_room_names:
        actions.append(
            Action(
                type=ActionType.REPORT,
                player_name=player.name,
                target_player_name=dead_name,
            )
        )

    # actions for MOVE
    for room in DOORS[location]:
        actions.append(
            Action(type=ActionType.MOVE, player_name=player.name, target_location=room)
        )

    # actions for tasks TASK
    for task in history[-1].tasks_left_to_do[player.name]:
        if task.location == location and not task.completed:
            actions.append(
                Action(type=ActionType.TASK, player_name=player.name, target_task=task)
            )

    # actions for impostors KILL
    if player.role == PlayerRole.IMPOSTOR and cooldown == 0:
        targets = get_players_in_room(history, players, location)
        for target in targets:
            if target.name != player.name:
                actions.append(
                    Action(
                        type=ActionType.KILL,
                        player_name=player.name,
                        target_player_name=target.name,
                    )
                )

    # actions for impostros PRETEND
    if player.role == PlayerRole.IMPOSTOR:
        for task in get_impostor_pretend_tasks_at_location(location):
            actions.append(
                Action(
                    type=ActionType.PRETEND, player_name=player.name, target_task=task
                )
            )

    return actions


def get_vote_actions(
    alive_players: List[Player], player: Player
) -> list[Action]:
    """Creates voting options.

    Returns:
        A list of game actions
    """
    actions = []
    actions.append(
        Action(
            type=ActionType.VOTE, player_name=player.name, target_player_name="nobody"
        )
    )
    for other_player in alive_players:
        if other_player != player:
            actions.append(
                Action(
                    type=ActionType.VOTE,
                    player_name=player.name,
                    target_player_name=other_player.name,
                )
            )
    return actions