from typing import List, Optional

from among_them.models.history import History
from among_them.models.player import Player
from among_them.models.end_game import EndGameReason
from among_them.models.phase import GamePhase
from among_them.models.player_role import PlayerRole
from among_them.models.action_type import ActionType

def get_end_game_reason(history: List[History], players: List[Player]) -> Optional[EndGameReason]:
    last_history_item = history[-1]
    # For system messages, check the target_message
    if last_history_item.action_taken.player_name == "System" and last_history_item.action_taken.target_message.startswith("The game ended"):
        # Extract reason from target_message
        return EndGameReason(last_history_item.action_taken.target_message.split("(")[1].split(")")[0])
    if last_history_item.actions_until_phase_ends == 0 and last_history_item.phase == GamePhase.TASKS and last_history_item.action_taken.type != ActionType.REPORT:
        return EndGameReason.NO_ACTIONS_LEFT

    alive_players = [p for p in players if p.name in last_history_item.alive_player_names]
    crewmates = [player for player in alive_players if player.role == PlayerRole.CREWMATE]
    impostors = [player for player in alive_players if player.role == PlayerRole.IMPOSTOR]
    if len(impostors) == 0:
        return EndGameReason.NO_IMPOSTORS_LEFT
    if len(impostors) >= len(crewmates):
        return EndGameReason.TOO_SMALL_NUMBER_OF_CREWMATES_LEFT

    if [len(last_history_item.tasks_left_to_do[p.name]) for p in crewmates] == [0] * len(crewmates):
        return EndGameReason.ALL_TASKS_DONE
    return None
