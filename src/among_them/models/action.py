from typing import Optional

from among_them.models.location import Location
from among_them.models.tasks import Task
from among_them.models.action_type import ActionType
from typing import List 
import re


class Action:
    def __init__(
        self,
        type: ActionType,
        player_name: str,
        target_player_name: Optional[str] = None,
        target_location: Optional[Location] = None,
        target_task: Optional[Task] = None,
        text: str = "",
        result: str = "",
        spectator: str = "",
    ):
        self.type = type
        self.player_name = player_name
        self.target_player_name = target_player_name
        self.target_location = target_location
        self.target_task = target_task
        self.text = text
        self.result = result
        self.spectator = spectator
        self.set_stories()

    def set_stories(self):
        if self.type == ActionType.MOVE:
            self.text = f"move to location {self.target_location.value}"
            self.result = (
                f"You [{self.player_name}] moved to {self.target_location.value}"
            )
            self.spectator = f"{self.player_name} moved to {self.target_location.value}"
        elif self.type == ActionType.WAIT:
            self.text = f"wait"
            self.result = f"You [{self.player_name}] are waiting"
            self.spectator = f"{self.player_name} waited"
        elif self.type == ActionType.TASK:
            self.text = f"complete task: {self.target_task.name}"
            self.result = f"You [{self.player_name}] {self.target_task}"
            self.spectator = f"{self.player_name} doing task {self.target_task.name}"
        elif self.type == ActionType.REPORT:
            self.text = f"report dead body of {str(self.target_player_name)}"
            self.result = (
                f"You [{self.player_name}] reported {str(self.target_player_name)}"
            )
            self.spectator = (
                f"{self.player_name} reported dead body of {self.target_player_name}"
            )
        elif self.type == ActionType.KILL:
            self.text = f"eliminate {str(self.target_player_name)}"
            self.result = (
                f"You [{self.player_name}] eliminated {str(self.target_player_name)}"
            )
            self.spectator = f"{self.player_name} eliminated {self.target_player_name}"
        elif self.type == ActionType.VOTE:
            self.text = f"vote for {str(self.target_player_name)}"
            self.result = (
                f"You [{self.player_name}] voted for {str(self.target_player_name)}"
            )
            self.spectator = f"{self.player_name} voted for {self.target_player_name}"
        elif self.type == ActionType.PRETEND:
            self.text = f"pretend doing task: {self.target_task.name}"
            self.result = f"You [{self.player_name}] pretended {self.target_task}"
            self.spectator = f"{self.player_name} doing task {self.target_task.name}"
        elif self.type == ActionType.SPEAK:
            pass
        else:
            raise ValueError(f"Unknown action type: {self.type}")



def normalize_and_check_action_valid(
    available_actions: List[str], chosen_action: str, player_name: str = ""
) -> tuple[int, str]:
    normalized_chosen_action = normalize_action(chosen_action)
    normalized_available_actions = [
        normalize_action(action) for action in available_actions if action != "wait"
    ]

    for action in normalized_available_actions:
        if action in normalized_chosen_action:
            return normalized_available_actions.index(action)+1, action
    if "wait" in normalized_chosen_action:
        return 0, "wait"

    warning_str = (
        f"{player_name} LLM did not conform to output format. "
        f"Expected one of {normalized_available_actions}, but got '{chosen_action}'"
    )
    print(warning_str)
    raise ValueError(warning_str)


def normalize_action(action: str) -> str:
    return re.sub(r"^\d+[\s:.)-]*", "", action).strip().strip(".").strip("- ").lower()
