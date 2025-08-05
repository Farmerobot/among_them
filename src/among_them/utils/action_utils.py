from typing import List

from among_them.models.action import Action, ActionType
from among_them.game_config import GameConfig
from among_them.models.history import History
from among_them.models.location import get_map
from among_them.models.player import Player
from among_them.models.player_role import PlayerRole
from among_them.models.tasks import get_impostor_pretend_tasks_at_location
from among_them.utils.player_utils import get_dead_players, get_players_in_room, get_last_player_action


def get_task_phase_actions(
    player: Player,
    history: List[History],
    players: List[Player],
    game_config: GameConfig,
) -> list[Action]:
    """Creates available actions based on the circumstances.

    Returns:
        A list of actions
    """
    last_player_action = get_last_player_action(history, player)
    location = last_player_action.location
    cooldown = last_player_action.impostor_cooldown

    actions = []

    # actions for WAIT
    # actions.append(Action(type=ActionType.WAIT, player_name=player.name))

    # action for REPORT
    dead_players_in_room_names = [name for name, dead_location in get_dead_players(history).items() if dead_location == location.value]
    for dead_name in dead_players_in_room_names:
        actions.append(
            Action(
                type=ActionType.REPORT,
                player_name=player.name,
                target_player_name=dead_name,
            )
        )

    # actions for MOVE
    for room in get_map(game_config.map_size)[0][location]:
        actions.append(
            Action(type=ActionType.MOVE, player_name=player.name, target_location=room)
        )

    # actions for tasks TASK
    for task in history[-1].tasks_left_to_do[player.name]:
        if player.role == PlayerRole.CREWMATE and task.location == location and not task.completed:
            actions.append(
                Action(type=ActionType.TASK, player_name=player.name, target_task=task)
            )

    # actions for impostors KILL
    if player.role == PlayerRole.IMPOSTOR and cooldown == 0:
        targets = get_players_in_room(history, players, player)
        for target in targets:
            if target.name != player.name and target.role != PlayerRole.IMPOSTOR: # cannot kill impostors
                actions.append(
                    Action(
                        type=ActionType.KILL,
                        player_name=player.name,
                        target_player_name=target.name,
                    )
                )

    # actions for impostros PRETEND
    if player.role == PlayerRole.IMPOSTOR:
        for task in get_impostor_pretend_tasks_at_location(location, game_config):
            actions.append(
                Action(
                    type=ActionType.PRETEND, player_name=player.name, target_task=task
                )
            )

    return actions


def get_vote_actions(
    history: List[History], players: List[Player], player: Player
) -> list[Action]:
    """Creates voting options.

    Args:
        history: List of history items
        players: List of all players
        player: The player who is voting
    Returns:
        A list of game actions
    """
    other_alive_players = [p for p in players if p.name in history[-1].alive_player_names and p.name != player.name]
    
    actions = []
    actions.append(
        Action(
            type=ActionType.VOTE, player_name=player.name, target_player_name="nobody"
        )
    )
    for other_player in other_alive_players:
        actions.append(
            Action(
                type=ActionType.VOTE,
                player_name=player.name,
                target_player_name=other_player.name,
            )
        )
    return actions