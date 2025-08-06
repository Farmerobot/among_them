
from typing import List
from among_them.models.player import Player
from among_them.game_config import GameConfig
from among_them.models.history import History
from among_them.models.location import Location
from among_them.models.action import Action
from among_them.models.player_role import PlayerRole
from among_them.models.action_type import ActionType
from among_them.models.phase import GamePhase
from among_them.models.tasks import get_impostor_tasks, get_crewmate_tasks

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
