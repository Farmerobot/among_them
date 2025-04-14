from typing import Dict, List

from among_them.models.action import Action
from among_them.models.action_type import ActionType
from among_them.models.location import Location
from among_them.models.phase import GamePhase
from among_them.models.tasks import Task


class History:
    """Item in the sequence of actions that have occurred in the game."""

    def __init__(
        self,
        player_names_to_play_next: List[str],
        phase: GamePhase,
        actions_until_phase_ends: int,
        location: Location, # This is after the action
        impostor_cooldown: int,
        actions_agent_could_take: List[str],
        spectators_who_saw: List[str],
        llm_cot: str,
        llm_response: str,
        token_usage: Dict[str, int],
        action_taken: Action,
        tasks_left_to_do: Dict[str, List[Task]],
        votes_before_this_discussion_message: Dict[str, dict],
        alive_player_names: List[str],
    ):
        self.player_names_to_play_next: List[str] = player_names_to_play_next
        self.phase: GamePhase = phase
        self.actions_until_phase_ends: int = actions_until_phase_ends
        self.location: Location = location
        self.impostor_cooldown: int = impostor_cooldown

        self.actions_agent_could_take: List[str] = actions_agent_could_take
        self.llm_cot: str = llm_cot
        self.llm_response: str = llm_response
        self.token_usage: dict = token_usage
        self.action_taken: Action = action_taken

        self.spectators_who_saw: List[str] = spectators_who_saw

        self.tasks_left_to_do: Dict[str, List[Task]] = tasks_left_to_do
        self.votes_before_this_discussion_message: Dict[str, dict] = votes_before_this_discussion_message
        self.alive_player_names: List[str] = alive_player_names

    def __repr__(self) -> str:
        # Format the token usage for better readability
        token_usage_str = "\n".join([f"{key}: {value}" for key, value in self.token_usage.items()])
        
        # Format task information with indentation
        tasks_str = ""
        for player, tasks in self.tasks_left_to_do.items():
            if player == self.action_taken.player_name:
                task_list = "\n    - ".join(str(task) for task in tasks)
                tasks_str += f"\n    - {player}:\n    - {task_list}"
        if not tasks_str:
            tasks_str = " None"
            
        # Format the lists for better readability
        players_next = ", ".join(self.player_names_to_play_next)
        spectators = ", ".join(self.spectators_who_saw)
            
        # Build the full representation with clear sections
        return f"""{self.phase}({self.actions_until_phase_ends}) [\033[33m{self.action_taken.spectator}\033[0m{"- "+self.action_taken.target_player_name if self.action_taken.type == ActionType.KILL and self.action_taken.target_player_name else ""}({self.impostor_cooldown}), {self.location}] next: {players_next}||{spectators} saw it"""

    def copy(self, **kwargs): # Added copy method
        data = self.__dict__.copy()
        data.update(kwargs)
        return History(**data)
