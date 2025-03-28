from typing import List, Dict

from langchain_core.messages.ai import UsageMetadata

from among_them.models.game_phase import GamePhase
from among_them.models.location import Location
from among_them.models.action_type import ActionType
from among_them.models.tasks import Task


class History:
    """Item in the sequence of actions that have occurred in the game."""

    def __init__(
        self,
        player_names_to_play_next: List[str],
        acted_by_player: str,
        phase: GamePhase,
        actions_until_phase_ends: int,
        location: Location, # This is after the action
        impostor_cooldown: int,
        actions_agent_could_take: List[str],
        spectators_who_saw: List[str],
        action_type: ActionType,
        llm_cot: str,
        llm_response: str,
        token_usage: UsageMetadata,
        killed_or_reported_player_name: str,
        action_result_agent_sees: str,
        action_result_spectator_sees: str,
        tasks_left_to_do: Dict[str, List[Task]],
    ):
        self.player_names_to_play_next: List[str] = player_names_to_play_next
        self.acted_by_player: str = acted_by_player

        self.phase: GamePhase = phase
        self.actions_until_phase_ends: int = actions_until_phase_ends
        self.location: Location = location
        self.impostor_cooldown: int = impostor_cooldown
        self.killed_or_reported_player_name: str = killed_or_reported_player_name

        self.action_type: ActionType = action_type
        self.actions_agent_could_take: List[str] = actions_agent_could_take
        self.llm_cot: str = llm_cot
        self.llm_response: str = llm_response
        self.token_usage: dict = token_usage

        self.action_result_agent_sees: str = action_result_agent_sees
        self.action_result_spectator_sees: str = action_result_spectator_sees
        self.spectators_who_saw: List[str] = spectators_who_saw

        self.tasks_left_to_do: Dict[str, List[Task]] = tasks_left_to_do

    def __repr__(self) -> str:
        # Format the token usage for better readability
        token_usage_str = ""
        
        # Format task information with indentation
        tasks_str = ""
        for player, tasks in self.tasks_left_to_do.items():
            if player == self.acted_by_player:
                task_list = "\n    - ".join(str(task) for task in tasks)
                tasks_str += f"\n    - {player}:\n    - {task_list}"
        if not tasks_str:
            tasks_str = " None"
            
        # Format the lists for better readability
        players_next = ", ".join(self.player_names_to_play_next)
        spectators = ", ".join(self.spectators_who_saw)
        actions_available = "\n    - ".join(self.actions_agent_could_take)
            
        # Build the full representation with clear sections
        return f"""{self.phase}({self.actions_until_phase_ends}) [{self.action_result_agent_sees}{"- "+self.killed_or_reported_player_name if self.action_type == ActionType.KILL else ""}({self.impostor_cooldown}), {self.location}] next: {players_next}||{spectators} saw it"""