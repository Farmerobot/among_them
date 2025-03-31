from typing import List, Dict
from among_them.models.history import History
from among_them.models.location import Location
from among_them.utils.phase_utils import get_last_discussion_action_idx
from among_them.models.player import Player
from among_them.models.action_type import ActionType
import random


def get_next_random_player(
    history: List[History], alive_players: List[Player]
) -> tuple[Player, List[Player]]:
    """Given a history of actions it selects random player from last history item with players_to_play_next.
    If this variable is empty, it selects random player from all alive players.

    Args:
        history: List of not empty history items
        players: List of players
    Returns:
        The next player to act
        The list of players who will play in next round.
    """
    if not history:
        return random.choice(alive_players), alive_players
    players_to_play_next = history[-1].player_names_to_play_next

    # Remove any player who is not alive from players_to_play_next
    players_to_play_next = [p for p in players_to_play_next if p in [player.name for player in alive_players]]
    if not players_to_play_next or history[-1].action_type == ActionType.REPORT:
        players_to_play_next = [p.name for p in alive_players]

    next_player_name = random.choice(players_to_play_next)
    next_player = [player for player in alive_players if player.name == next_player_name][0]
    return next_player, [p for p in players_to_play_next if p != next_player_name]


def get_alive_players(history: List[History], players: List[Player]) -> List[Player]:
    kill_history = [history_item for history_item in history if history_item.action_type == ActionType.KILL]
    dead_players = [history_item.killed_or_reported_player_name for history_item in kill_history]
    return [player for player in players if player.name not in dead_players]


def get_last_player_action(history: List[History], player: Player) -> History:
    for i in range(len(history) - 1, -1, -1):
        if history[i].acted_by_player == player.name:
            return history[i]
    print(f"Player {player.name} has no actions in history")
    return history[0]


def get_dead_players(history: List[History], players: List[Player]) -> Dict[str, str]:
    """Returns the dictionary of dead players and their location that can be reported (without ghosts)"""
    search_from = get_last_discussion_action_idx(history)
    kill_history = [history_item for history_item in history[search_from:] if history_item.action_type == ActionType.KILL]
    return {history_item.killed_or_reported_player_name: history_item.location for history_item in kill_history}


def get_players_in_room(
    history: List[History], players: List[Player], location: Location
) -> List[Player]:
    alive = get_alive_players(history, players)
    in_room = []
    for player in alive:
        last_player_action = get_last_player_action(history, player)
        last_player_location = last_player_action.location
        if last_player_location == location:
            in_room.append(player)
    return in_room