
from typing import List, Optional
from among_them.models.player import Player
from among_them.game_config import GameConfig
from among_them.models.history import History
from among_them.models.location import Location
from among_them.models.action import Action
from among_them.models.player_role import PlayerRole
from among_them.models.action_type import ActionType
from among_them.models.phase import GamePhase
from among_them.models.tasks import get_impostor_tasks, get_crewmate_tasks
from among_them.utils.player_utils import get_players_in_room, get_last_player_action

def initialize_history(players: List[Player], game_config: GameConfig) -> List[History]:
    """Create first history entry – a system message that the game started (TASK phase)."""
    tasks = {p.name: (get_impostor_tasks() if p.role == PlayerRole.IMPOSTOR else get_crewmate_tasks(game_config)) for p in players}

    first_entry = History(
        player_names_to_play_next=[p.name for p in players],
        phase=GamePhase.TASKS,
        actions_until_phase_ends=(game_config.num_task_phase_actions_per_player * len(players)),
        location=Location.CAFETERIA,
        impostor_cooldown=game_config.impostor_cooldown,
        actions_agent_could_take=[],
        spectators_who_saw=[p.name for p in players],
        llm_cot="",
        llm_response="",
        token_usage={},
        action_taken=Action(ActionType.SPEAK, "System", target_message="The game started"),
        tasks_left_to_do=tasks,
        votes_before_this_discussion_message={},
        alive_player_names=[p.name for p in players],
    )
    return [first_entry]


def create_system_message(
    history: List[History], 
    phase: GamePhase, 
    text: str, 
    game_config: GameConfig, 
    alive_player_names: List[str],
    ejected_player: str = "nobody",
    no_more_actions: bool = False,
):
    """Append a generic system History entry with given phase and spectator text."""
    actions_until_phase_ends = 0
    if phase == GamePhase.TASKS:
        actions_until_phase_ends = (game_config.num_task_phase_actions_per_player * len(alive_player_names))
    elif phase == GamePhase.DISCUSS:
        actions_until_phase_ends = (game_config.num_discuss_phase_actions_per_player * len(alive_player_names))
    elif phase == GamePhase.VOTING:
        actions_until_phase_ends = len(alive_player_names)
    return History(
        player_names_to_play_next=alive_player_names.copy(),
        phase=phase,
        actions_until_phase_ends=0 if no_more_actions else actions_until_phase_ends,
        location=Location.CAFETERIA,
        impostor_cooldown=game_config.impostor_cooldown,
        actions_agent_could_take=[],
        spectators_who_saw=alive_player_names.copy(),
        llm_cot="",
        llm_response="",
        token_usage={},
        action_taken = Action(
            type=ActionType.SPEAK if ejected_player == "nobody" else ActionType.KILL,
            player_name="System", 
            target_player_name=ejected_player if ejected_player != "nobody" else None, 
            target_message=text
        ),
        tasks_left_to_do={k: v.copy() for k, v in history[-1].tasks_left_to_do.items()},
        votes_before_this_discussion_message={},
        alive_player_names=alive_player_names.copy(),
    )

def get_action_history_str(history: List[History], players: List[Player], player: Player, game_config: GameConfig, phase: Optional[GamePhase] = None) -> str:
    """
    DEPRECATED: see prompt_utils.py
    Returns all actions seen by agent and actions that agent saw/spectated in history in order.
    """
    phase = history[-1].phase
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
                    history_str += f"<action type=\"player_message\">{h.action_taken.agent_perspective}</action>\n"
                else:
                    history_str += f"<action type=\"player_action\">{h.action_taken.agent_perspective}</action>\n"
            elif player.name in h.spectators_who_saw:
                if h.action_taken.type == ActionType.SPEAK:
                    history_str += f"<action type=\"discussion\">{h.action_taken.observer_perspective}</action>\n"
                elif h.action_taken.type != ActionType.VOTE: # players cannot see others voting
                    history_str += f"<action type=\"observed\">{h.action_taken.observer_perspective}</action>\n"
    history_str += "</game_history>\n\n"

    # Current game state
    history_str += "<current_state>\n"
    other_players = [p.name for p in players_in_room if p.name != player.name]
    if phase == GamePhase.TASKS:
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
