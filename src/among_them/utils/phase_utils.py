from typing import List

from among_them.models.action_type import ActionType
from among_them.models.history import History
from among_them.models.phase import GamePhase

def get_last_voting_action_idx(history: List[History]) -> int:
    """Return index after the vote-result system message (now the final VOTING entry)."""
    for i in range(len(history) - 1, -1, -1):
        # Check if this is a system message indicating someone was voted out
        if (history[i].phase == GamePhase.VOTING and 
            history[i].action_taken.player_name == "System" and
            history[i].action_taken.target_message.endswith("was voted out.")):
            return i + 1
    return 0


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
    action_type = ActionType.SPEAK if ejected_player == "nobody" else ActionType.KILL
    return ejected_player, action_type