from typing import List

from langchain_core.messages.ai import UsageMetadata
from among_them.models.history import History
from among_them.models.game_phase import GamePhase
from among_them.models.action_type import ActionType
from among_them.models.location import Location
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
        return handle_phase_change(previous_phase, players_number, history)
    return previous_phase, history[-1].actions_until_phase_ends - 1


def handle_phase_change(previous_phase: GamePhase, players_number: int, history: List[History]) -> tuple[GamePhase, int]:
    if previous_phase == GamePhase.TASK:
        return GamePhase.MAIN_MENU, 0
    elif previous_phase == GamePhase.DISCUSS:
        return GamePhase.VOTE, players_number - 1
    elif previous_phase == GamePhase.VOTE:
        # Count votes and check for tie
        vote_counts = {}
        for item in range(len(history) - 1, -1, -1):
            if history[item].action_type == ActionType.VOTE:
                voted_for = history[item].action_result_spectator_sees.split(" voted for ")[-1]
                vote_counts[voted_for] = vote_counts.get(voted_for, 0) + 1
            else:
                break
        
        # Handle voting result
        most_voted = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
        is_tie = len(most_voted) >= 2 and most_voted[0][1] == most_voted[1][1]
        is_tie = is_tie or (len(most_voted) >= 3 and most_voted[0][1] == most_voted[2][1])
        action_type = ActionType.WAIT if not most_voted or is_tie else ActionType.KILL
        ejected_player = None if not most_voted or is_tie else most_voted[0][0]
        
        # Result message
        if not vote_counts:
            action_result = "No votes were cast. No one was voted out."
        elif is_tie:
            action_result = "There was a tie in the vote. No one was voted out."
        else:
            action_result = f"{ejected_player} was voted out."
        
        history.append(History(
            player_names_to_play_next = history[-1].player_names_to_play_next,
            acted_by_player = "System",
            phase = GamePhase.MAIN_MENU,
            actions_until_phase_ends = 0,
            location = Location.CAFETERIA,
            impostor_cooldown = 0,
            actions_agent_could_take = [],
            spectators_who_saw = history[-1].spectators_who_saw,
            action_type = action_type,
            llm_cot = "",
            llm_response = "",
            token_usage = UsageMetadata(),
            killed_or_reported_player_name = ejected_player,
            action_result_agent_sees = "",
            action_result_spectator_sees = action_result,
            tasks_left_to_do = history[-1].tasks_left_to_do,
        ))
        return GamePhase.TASK, (NUM_ACTIONS_WITHOUT_REPORT * players_number) - 1
    elif previous_phase == GamePhase.MAIN_MENU:
        return GamePhase.MAIN_MENU, 0

def get_last_discussion_action_idx(history: List[History]) -> int:
    for i in range(len(history) - 1, -1, -1):
        if history[i].phase == GamePhase.DISCUSS:
            return i
    return 0