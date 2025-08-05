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
        """
        Return a condensed, colour-coded one-liner of the history item in the format:
        [A:2,B:3] [PHASE n] actor <icon> target @LOCATION next: A,B seen: C,D
        Raw ANSI escape sequences are used so the string stays portable and
        dependency-free (non-ANSI terminals will just display the codes).
        """
        RESET = "\033[0m"
        COLORS = {
            "blue": "\033[34m",
            "cyan": "\033[36m",
            "magenta": "\033[35m",
            "green": "\033[32m",
            "yellow": "\033[33m",
            "red": "\033[31m",
            "gray": "\033[90m",
            "dim": "\033[2m",
            "underline": "\033[4m",
        }

        # Alive players with task counts
        alive_players_str = ""
        if self.alive_player_names and self.tasks_left_to_do is not None:
            player_task_info = []
            for player_name in self.alive_player_names:
                if player_name in self.tasks_left_to_do:
                    task_count = len(self.tasks_left_to_do[player_name])
                    first_letter = player_name[0] if player_name else "?"
                    player_task_info.append(f"{first_letter}:{task_count}")
                else:
                    # If player not in tasks_left_to_do, show 0 tasks
                    first_letter = player_name[0] if player_name else "?"
                    player_task_info.append(f"{first_letter}:0")
            alive_players_str = f'[{",".join(player_task_info)}] '

        # Phase segment (override to green if game ended)
        # For system messages, check the target_message
        if (self.action_taken.player_name == "System" and
            self.action_taken.target_message.startswith("The game ended")):
            phase_color = COLORS["green"]
        else:
            phase_color_map = {
                GamePhase.TASKS: COLORS["blue"],
                GamePhase.DISCUSS: COLORS["cyan"],
                GamePhase.VOTING: COLORS["magenta"],
            }
            phase_color = phase_color_map.get(self.phase, '')
        phase_str = f"{phase_color}[{self.phase.name} {self.actions_until_phase_ends}]{RESET}"

        # Actor and action icon
        # Call set_stories if needed to ensure perspectives are populated
        self.action_taken.set_stories()
        
        actor_text = self.action_taken.global_perspective
        actor = f'{COLORS["yellow"]}{actor_text}{RESET}'
        icon_map = {
            ActionType.WAIT: "·",
            ActionType.MOVE: "→",
            ActionType.TASK: "✓",
            ActionType.KILL: "×",
            ActionType.REPORT: "⚑",
            ActionType.SPEAK: "💬",
            ActionType.VOTE: "✔",
            ActionType.PRETEND: "🎭",
        }
        icon = icon_map.get(self.action_taken.type, "")
        if self.action_taken.type == ActionType.KILL and self.action_taken.target_player_name:
            target_part = f' {icon} {COLORS["red"]}{self.action_taken.target_player_name}{RESET}'
        elif self.action_taken.type == ActionType.REPORT and self.action_taken.target_player_name:
            target_part = f' {icon} {COLORS["green"]}{self.action_taken.target_player_name}{RESET}'
        else:
            target_part = f" {icon}" if icon else ""

        # Location
        location_str = f' @{COLORS["gray"]}{self.location.name}{RESET}'

        # Impostor cooldown (only show if non-zero)
        cooldown_part = f" cd={self.impostor_cooldown}" if self.impostor_cooldown else ""

        # Next players and spectators who saw the action
        players_next = ", ".join(self.player_names_to_play_next)
        spectators = ", ".join(self.spectators_who_saw)
        next_part = f' next: {COLORS["dim"]}{players_next}{RESET}' if players_next else ""
        seen_part = f' seen by: {COLORS["underline"]}{spectators}{RESET}' if spectators else ""

        return f"{alive_players_str}{phase_str}{target_part}{location_str} {actor}{cooldown_part}{next_part}{seen_part}"

    def copy(self, **kwargs): # Added copy method
        data = self.__dict__.copy()
        data.update(kwargs)
        return History(**data)
