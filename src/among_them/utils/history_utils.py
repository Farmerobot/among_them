
from typing import List, Optional
from among_them.models.player import Player
from among_them.game_config import GameConfig
from among_them.models.history import History
from among_them.models.location import Location
from among_them.models.action import Action
from among_them.models.end_game import EndGameReason
from among_them.models.player_role import PlayerRole
from among_them.models.action_type import ActionType
from among_them.models.phase import GamePhase
from among_them.models.tasks import get_impostor_tasks, get_crewmate_tasks
from among_them.utils.phase_utils import get_phase_and_when_it_ends
from among_them.utils.player_utils import get_players_in_room, get_last_player_action

def initialize_history(players: List[Player], game_config: GameConfig) -> List[History]:
    tasks = {}
    for player in players:
        tasks[player.name] = get_impostor_tasks() if player.role == PlayerRole.IMPOSTOR else get_crewmate_tasks(game_config)

    first_entry = History(
        player_names_to_play_next = [p.name for p in players],
        phase = GamePhase.GAME_START,
        actions_until_phase_ends = 0,
        location = Location.CAFETERIA,
        impostor_cooldown = game_config.impostor_cooldown,
        actions_agent_could_take = [],
        spectators_who_saw = [player.name for player in players],
        llm_cot = "",
        llm_response = "",
        token_usage = {},
        action_taken = Action(ActionType.MOVE, "System", spectator="The game started"),
        tasks_left_to_do = tasks,
        votes_before_this_discussion_message = {},
        alive_player_names = [p.name for p in players],
    )
    return [first_entry]

def end_game_history(history: List[History], players: List[Player], game_config: GameConfig, reason: EndGameReason) -> History:
    return History(
        player_names_to_play_next = [],
        phase = GamePhase.GAME_END,
        actions_until_phase_ends = 0,
        location = Location.CAFETERIA,
        impostor_cooldown = game_config.impostor_cooldown,
        actions_agent_could_take = [],
        spectators_who_saw = [p.name for p in players],
        llm_cot = "",
        llm_response = "",
        token_usage = {},
        action_taken = Action(ActionType.MOVE, "System", spectator=f"The game ended ({reason})"),
        tasks_left_to_do = history[-1].tasks_left_to_do,
        votes_before_this_discussion_message = {},
        alive_player_names = [p.name for p in players],
    )
    
def create_vote_history_entry(
    history: List[History],
    alive_players: List[Player],
    ejected_player: str,
    action_result: str,
    action_type: ActionType,
    game_config: GameConfig
) -> History:
    """
    Create a new history entry for a vote action. 
    
    Args:
        votes: dictionary of player name to voted for. This is for votes_before_this_discussion_message variable so that it includes votes after discussion.
    
    Returns:
        A new history entry for the vote action.
    """
    return History(
        player_names_to_play_next = [], # handled automatically
        phase = GamePhase.VOTE_RESULTS,
        actions_until_phase_ends = 0,
        location = Location.CAFETERIA,
        impostor_cooldown = game_config.impostor_cooldown,
        actions_agent_could_take = [],
        spectators_who_saw = [p.name for p in alive_players],
        llm_cot = "",
        llm_response = "",
        token_usage = {},
        action_taken = Action(type=action_type, player_name="System", target_player_name=ejected_player, spectator=action_result),
        tasks_left_to_do = history[-1].tasks_left_to_do,
        votes_before_this_discussion_message = {},
        alive_player_names = [p.name for p in alive_players if p.name != ejected_player],
    )

def get_action_history_str(history: List[History], players: List[Player], player: Player, game_config: GameConfig, phase: Optional[GamePhase] = None) -> str:
    """Returns all actions seen by agent and actions that agent saw/spectated in history in order."""
    if phase is None:
        try:
            phase, _ = get_phase_and_when_it_ends(history, game_config, players)
        except:
            phase = GamePhase.TASK
    players_in_room = get_players_in_room(history, players, player)
    alive_players = [p for p in players if p.name in history[-1].alive_player_names]
    location = get_last_player_action(history, player).location

    # Agent description
    history_str = "<player_info>\n"
    history_str += f"<player_name>{player.name}</player_name>\n<current_role>{player.role.value}</current_role>\n<current_location>{location.value}</current_location>\n"
    other_impostors = [p for p in alive_players if p.role == PlayerRole.IMPOSTOR and p.name != player.name]
    if player.role == PlayerRole.IMPOSTOR:
        other_crewmates = [p for p in alive_players if p.role == PlayerRole.CREWMATE and p.name != player.name]
        if other_impostors:
            history_str += "<other_impostors>\n"
            for imp in other_impostors:
                history_str += f"<impostor>{imp.name}</impostor>\n"
            history_str += "</other_impostors>\n"
        else:
            history_str += "<other_impostors>none</other_impostors>\n"
        
        if other_crewmates:
            history_str += "<crewmates>\n"
            for crew in other_crewmates:
                history_str += f"<crewmate>{crew.name}</crewmate>\n"
            history_str += "</crewmates>\n"
    else:
        other_players = [p.name for p in alive_players if p.name != player.name]
        history_str += f"<impostor_count>{len(other_impostors)}</impostor_count>\n"
        history_str += "<potential_impostors>\n"
        for other_player in other_players:
            history_str += f"<player>{other_player}</player>\n"
        history_str += "</potential_impostors>\n"
    
    # Tasks left
    tasks_left = []
    for h in history:
        if player.name in h.tasks_left_to_do:
            tasks_left = h.tasks_left_to_do[player.name]
            break
    
    if tasks_left:
        history_str += "<tasks_left>\n"
        for task in tasks_left:
            history_str += f"<task><name>{task.name}</name><location>{task.location.value if task.location else 'N/A'}</location></task>\n"
        history_str += "</tasks_left>\n"
    history_str += "</player_info>\n\n"

    # Game history
    history_str += "<game_history>\n"
    for h in history:
        if h.action_taken:
            if h.action_taken.player_name == player.name:
                if hasattr(h, 'llm_cot') and h.llm_cot:
                    cot_without_think_tags = h.llm_cot.replace("<think>", "<thought>").replace("</think>", "</thought>")
                    history_str += f"<player_thought>{cot_without_think_tags}</player_thought>\n"
                if h.action_taken.type == ActionType.SPEAK:
                    history_str += f"<action type=\"player_message\">{h.action_taken.spectator}</action>\n"
                else:
                    history_str += f"<action type=\"player_action\">{h.action_taken.result}</action>\n"
            elif player.name in h.spectators_who_saw:
                if h.action_taken.type == ActionType.SPEAK:
                    history_str += f"<action type=\"discussion\">{h.action_taken.spectator}</action>\n"
                elif h.action_taken.type != ActionType.VOTE: # players cannot see others voting
                    history_str += f"<action type=\"observed\">{h.action_taken.spectator}</action>\n"
    history_str += "</game_history>\n\n"

    # Current game state
    history_str += "<current_state>\n"
    other_players = [p.name for p in players_in_room if p.name != player.name]
    if phase == GamePhase.TASK:
        if len(other_players) == 0:
            history_str += "<companions>none</companions>\n"
            history_str += f"<location>{location.value}</location>\n"
        else:
            history_str += "<companions>\n"
            for other_player in other_players:
                history_str += f"<player>{other_player}</player>\n"
            history_str += "</companions>\n"
            history_str += f"<location>{location.value}</location>\n"
        history_str += "<phase>Task</phase>\n<note>You cannot speak during this phase.</note>\n"
    elif phase == GamePhase.VOTING:
        history_str += "<phase>Voting</phase>\n"
    else:
        history_str += "<phase>Discussion</phase>\n<note>You can speak now. Respond to the crewmates.</note>\n"
    history_str += "</current_state>\n"
    return history_str
