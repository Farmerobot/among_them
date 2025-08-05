import random
from typing import Dict, List, Optional

from among_them.models.action_type import ActionType
from among_them.models.history import History
from among_them.models.location import Location
from among_them.models.player import Player
from among_them.utils.phase_utils import get_last_voting_action_idx


def get_next_random_player(
    history: List[History], players: List[Player]
) -> tuple[Player, List[str]]:
    """Given a history of actions it selects random player from last history item with players_to_play_next.
    If this variable is empty, it selects random player from all alive players.

    Args:
        history: List of not empty history items
        players: List of players
    Returns:
        The next player to act
        The list of players who will play in next round.
    """
    alive_players = [p for p in players if p.name in history[-1].alive_player_names]
    players_to_play_next = history[-1].player_names_to_play_next

    # Remove any player who is not alive from players_to_play_next
    players_to_play_next = [p for p in players_to_play_next if p in history[-1].alive_player_names]
    if not players_to_play_next or history[-1].action_taken.type == ActionType.REPORT:
        players_to_play_next = [p.name for p in alive_players]

    next_player_name = random.choice(players_to_play_next)
    next_player = [player for player in alive_players if player.name == next_player_name][0]
    return next_player, [p for p in players_to_play_next if p != next_player_name]


def get_last_player_action(history: List[History], player: Player) -> History:
    """Returns the last action taken by the player. Used to get the player's last location and impostor cooldown."""
    for i in range(len(history) - 1, -1, -1):
        if history[i].action_taken.player_name == player.name:
            return history[i]
        elif getattr(history[i].action_taken, "target_message", None) and history[i].action_taken.target_message.startswith("Everyone is in the cafeteria and start from there"):
            return history[i]
    return history[0]


def get_dead_players(history: List[History]) -> Dict[str, str]:
    """Returns the dictionary of dead players and their location that can be reported (without ghosts)"""
    search_from = get_last_voting_action_idx(history)
    kill_history = [history_item for history_item in history[search_from:] if history_item.action_taken.type == ActionType.KILL]
    return {history_item.action_taken.target_player_name: history_item.location.value for history_item in kill_history} # type: ignore


def get_players_in_room(
    history: List[History], players: List[Player], player: Player, location: Optional[Location] = None
) -> List[Player]:
    """Returns the list of alive players in a specific location.

    Args:
        history: List of history items
        players: List of all players
        player: The player to check
        location: The location to check - if None, uses the last player action location
    Returns:
        List of alive players in the player location
    """
    other_alive = [p for p in players if p.name in history[-1].alive_player_names and p.name != player.name]
    if location is None:
        location = get_last_player_action(history, player).location
    in_room = []
    for other_player in other_alive:
        last_player_action = get_last_player_action(history, other_player)
        last_player_location = last_player_action.location
        if last_player_location == location:
            in_room.append(other_player)
    return in_room