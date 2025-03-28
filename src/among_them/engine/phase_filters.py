from typing import List
from among_them.models.history import History
from among_them.models.game_phase import GamePhase
from among_them.models.action_type import ActionType
from among_them.consts import NUM_ACTIONS_WITHOUT_REPORT, NUM_CHATS

def get_phase_and_actions_until_phase_ends(history: List[History], players_number: int) -> tuple[GamePhase, int]:
    """
    Returns the next phase and the number of actions until the phase ends.
    
    Args:
        history: List of History objects
    
    Returns:
        Next phase
        Number of actions until the phase ends
    """
    if len(history) == 1:
        return GamePhase.TASK, (NUM_ACTIONS_WITHOUT_REPORT * players_number) - 1
    previous_phase = history[-1].phase
    if history[-1].action_type == ActionType.REPORT:
        return GamePhase.DISCUSS, (NUM_CHATS * players_number) - 1
    if history[-1].actions_until_phase_ends == 0:
        return handle_phase_change(previous_phase, players_number)
    return previous_phase, history[-1].actions_until_phase_ends - 1


def handle_phase_change(previous_phase: GamePhase, players_number: int) -> tuple[GamePhase, int]:
    if previous_phase == GamePhase.TASK:
        return GamePhase.MAIN_MENU, 0
    elif previous_phase == GamePhase.DISCUSS:
        return GamePhase.VOTE, players_number - 1
    elif previous_phase == GamePhase.VOTE:
        return GamePhase.TASK, (NUM_ACTIONS_WITHOUT_REPORT * players_number) - 1
    elif previous_phase == GamePhase.MAIN_MENU:
        return GamePhase.MAIN_MENU, 0

def get_last_discussion_action_idx(history: List[History]) -> int:
    for i in range(len(history) - 1, -1, -1):
        if history[i].phase == GamePhase.DISCUSS:
            return i
    return 0