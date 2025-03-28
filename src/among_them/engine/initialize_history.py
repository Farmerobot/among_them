from pkgutil import get_importer
from typing import List
import random
from among_them.models.tasks import get_crewmate_tasks, get_impostor_tasks
from among_them.players.player import Player
from among_them.models.history import History
from among_them.models.game_phase import GamePhase
from among_them.models.location import Location
from among_them.models.action_type import ActionType
from langchain_core.messages.ai import UsageMetadata
from among_them.models.player_role import PlayerRole

def initialize_history(players: List[Player]) -> List[History]:
    tasks = {}
    for player in players:
        tasks[player.name] = get_impostor_tasks() if player.role == PlayerRole.IMPOSTOR else get_crewmate_tasks()

    first_entry = History(
        player_names_to_play_next = [p.name for p in players],
        acted_by_player = "System",
        phase = GamePhase.MAIN_MENU,
        actions_until_phase_ends = 0,
        location = Location.CAFETERIA,
        impostor_cooldown = 0,
        actions_agent_could_take = [],
        spectators_who_saw = [player.name for player in players],
        action_type = ActionType.WAIT,
        llm_cot = "",
        llm_response = "",
        token_usage = UsageMetadata(),
        killed_or_reported_player_name = "",
        action_result_agent_sees = "",
        action_result_spectator_sees = f"The game started",
        tasks_left_to_do = tasks,
    )
    return [first_entry]
        