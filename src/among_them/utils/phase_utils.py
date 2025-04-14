from typing import List

from among_them.models.action_type import ActionType
from among_them.game_config import GameConfig
from among_them.models.history import History
from among_them.models.phase import GamePhase
from among_them.models.player import Player
from among_them.utils.end_utils import get_end_game_reason


def get_last_voting_action_idx(history: List[History]) -> int:
    for i in range(len(history) - 1, -1, -1):
        if history[i].phase == GamePhase.VOTE_RESULTS:
            return i+1
    return 0


def get_phase_and_when_it_ends(history: List[History], game_config: GameConfig, players: List[Player]) -> tuple[GamePhase, int]:
    """
    Handles sudden phase changes - first phase, report - and automatic phase change using actions_until_phase_ends.
    Returns the next phase and the number of actions until the phase ends.
    """
    if get_end_game_reason(history, players) is not None:
        return GamePhase.GAME_END, 0
    previous_phase = history[-1].phase
    
    if history[-1].action_taken.type == ActionType.REPORT: # if report start discussion
        return GamePhase.DISCUSS, (game_config.num_discuss_phase_actions_per_player * len(history[-1].alive_player_names)) - 1
    
    if history[-1].actions_until_phase_ends == 0:
        return handle_phase_change(history, previous_phase, game_config)
    
    return previous_phase, history[-1].actions_until_phase_ends - 1


def handle_phase_change(history: List[History], previous_phase: GamePhase, game_config: GameConfig) -> tuple[GamePhase, int]:
    """
    This is for automatic phase change when actions_until_phase_ends is 0.
    Returns the next phase and the number of actions until the phase ends.
    """
    if previous_phase == GamePhase.TASK:
        return GamePhase.GAME_END, 0 # signal end of game
    elif previous_phase == GamePhase.DISCUSS:
        return GamePhase.VOTING, len(history[-1].alive_player_names) - 1
    elif previous_phase == GamePhase.VOTING:
        return GamePhase.VOTE_RESULTS, 0
    elif previous_phase == GamePhase.VOTE_RESULTS or previous_phase == GamePhase.GAME_START:
        return GamePhase.TASK, (game_config.num_task_phase_actions_per_player * len(history[-1].alive_player_names)) - 1
    elif previous_phase == GamePhase.GAME_END:
        return GamePhase.GAME_END, 0


def count_votes(history: List[History]) -> tuple[dict[str, int], dict[str, str]]:
    """Count votes from the history. Returns vote counts and votes dictionary."""
    vote_counts: dict[str, int] = {}
    votes: dict[str, str] = {}
    for item in range(len(history) - 1, -1, -1):
        if history[item].action_taken.type == ActionType.VOTE:
            voted_for: str = history[item].action_taken.target_player_name # type: ignore
            vote_counts[voted_for] = vote_counts.get(voted_for, 0) + 1
            votes[history[item].action_taken.player_name] = voted_for
        else:
            break 
    return vote_counts, votes


def determine_ejection_result(vote_counts: dict[str, int]) -> tuple[str, ActionType]:
    """Determine the ejected player and action type based on vote counts."""
    if not vote_counts: # Added check for empty votes
        raise ValueError("No votes casted")

    most_voted = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)

    is_tie = len(most_voted) >= 2 and most_voted[0][1] == most_voted[1][1]
    is_tie = is_tie or (len(most_voted) >= 3 and most_voted[0][1] == most_voted[1][1] == most_voted[2][1])

    ejected_player = "nobody" if is_tie else most_voted[0][0]
    action_type = ActionType.WAIT if ejected_player == "nobody" else ActionType.KILL
    return ejected_player, action_type