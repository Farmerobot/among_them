from typing import Optional

from among_them.models.action_type import ActionType
from among_them.models.location import Location
from among_them.models.tasks import Task


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
        if text == "" and result == "" and spectator == "":
            self.set_stories()

    def set_stories(self):
        if self.type == ActionType.MOVE:
            self.text = f"move to location {self.target_location.value}"
            self.result = f"You [{self.player_name}] moved to location {self.target_location.value}"
            self.spectator = f"{self.player_name} moved to location {self.target_location.value}"
        # elif self.type == ActionType.WAIT:
        #     self.text = "wait"
        #     self.result = f"You [{self.player_name}] are waiting"
        #     self.spectator = f"{self.player_name} waited"
        elif self.type == ActionType.TASK:
            self.text = f"complete task: {self.target_task.name}"
            self.result = f"You [{self.player_name}] completed task: {self.target_task.name}"
            self.spectator = f"{self.player_name} completed task: {self.target_task.name}"
        elif self.type == ActionType.REPORT:
            self.text = f"report dead body of {str(self.target_player_name)} and start a discussion"
            self.result = f"You [{self.player_name}] reported dead body of {str(self.target_player_name)} to other players and started discussion"
            self.spectator = f"{self.player_name} reported dead body of {self.target_player_name} to everyone and started discussion"
        elif self.type == ActionType.KILL:
            self.text = f"kill {str(self.target_player_name)}"
            self.result = f"You [{self.player_name}] killed {str(self.target_player_name)}"
            self.spectator = f"{self.player_name} killed {self.target_player_name}"
        elif self.type == ActionType.VOTE:
            self.text = f"vote for {str(self.target_player_name)}"
            self.result = f"You [{self.player_name}] voted for {str(self.target_player_name)}"
            self.spectator = f"{self.player_name} voted for {self.target_player_name}"
        elif self.type == ActionType.PRETEND:
            self.text = f"pretend doing task: {self.target_task.name}"
            self.result = f"You [{self.player_name}] pretended {self.target_task.name}"
            self.spectator = f"{self.player_name} doing task {self.target_task.name}"
        elif self.type == ActionType.SPEAK:
            self.text = "speak"
        else:
            raise ValueError(f"Unknown action type: {self.type}")

    def copy(self, **kwargs):
        data = self.__dict__.copy()
        data.update(kwargs)
        return Action(**data)
