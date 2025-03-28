from typing import List
from among_them.players.player import Player
from among_them.models.player_role import PlayerRole
import random


def check_and_get_players(
    players: List[Player], impostor_count: int = 1
) -> List[Player]:
    """Checks if the players have already been assigned roles, checks balance and assigns roles (crewmate or impostor).

    Args:
        players: List of players with or without impostors
        impostor_count: Expected number of impostors

    Returns:
        List of players with roles assigned

    Raises:
        ValueError: Inconsistent configuration
    """
    if len(players) < 3:
        raise ValueError("Minimum number of players is 3.")

    if impostor_count >= len(players) or impostor_count <= 0:
        raise ValueError("Invalid number of impostors")

    # Count existing impostors
    existing_impostors = sum(
        1 for player in players if player.role == PlayerRole.IMPOSTOR
    )

    # Assign impostors randomly, only if needed
    impostors_to_assign = impostor_count - existing_impostors
    while impostors_to_assign > 0:
        available_players = [p for p in players if p.role != PlayerRole.IMPOSTOR]
        if not available_players:
            break  # No more players to assign as impostors
        chosen_player = random.choice(available_players)
        chosen_player.role = PlayerRole.IMPOSTOR
        impostors_to_assign -= 1

    # Check for imbalanced team sizes AFTER role assignment
    crewmates_count = len(players) - impostor_count
    if impostor_count >= crewmates_count:
        raise ValueError(
            "Number of impostors cannot be greater than "
            "or equal to the number of crewmates."
        )
    random.shuffle(players)
    print("Players:", {p.name: p.role.value for p in players})
    return players
