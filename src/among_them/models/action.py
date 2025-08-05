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
        target_message: Optional[str] = None,
    ):
        self.type = type
        self.player_name = player_name
        self.target_player_name = target_player_name
        self.target_location = target_location
        self.target_task = target_task
        self.target_message = target_message
            

    def set_stories(self):
        if self.type == ActionType.MOVE:
            self.command_perspective = f"move to {self.target_location.value}"
            self.agent_perspective = f"You moved to {self.target_location.value}."
            self.observer_perspective = f"You saw {self.player_name} move to {self.target_location.value}."
            self.global_perspective = f"{self.player_name} moved to {self.target_location.value}."
        elif self.type == ActionType.WAIT:
            self.command_perspective = "wait"
            self.agent_perspective = "You waited."
            self.observer_perspective = f"You noticed {self.player_name} was waiting."
            self.global_perspective = f"{self.player_name} waited."
        elif self.type == ActionType.TASK:
            self.command_perspective = f"{self.target_task.name}"
            self.agent_perspective = f"You completed the {self.target_task.name} task."
            self.observer_perspective = f"You saw {self.player_name} complete the {self.target_task.name} task."
            self.global_perspective = f"{self.player_name} completed the {self.target_task.name} task."
        elif self.type == ActionType.REPORT:
            self.command_perspective = f"report the dead body of {str(self.target_player_name)}"
            self.agent_perspective = f"You reported the dead body of {str(self.target_player_name)} and started a discussion."
            self.observer_perspective = f"You saw {self.player_name} report the dead body of {str(self.target_player_name)} to everyone. Discussion started."
            self.global_perspective = f"A dead body was reported by {self.player_name}. Discussion started."
        elif self.type == ActionType.KILL:
            self.command_perspective = f"kill {str(self.target_player_name)}"
            self.agent_perspective = f"You killed {str(self.target_player_name)}."
            self.observer_perspective = f"You witnessed {self.player_name} kill {str(self.target_player_name)}!"
            self.global_perspective = f"{str(self.target_player_name)} was killed."
        elif self.type == ActionType.VOTE:
            self.command_perspective = f"vote for {str(self.target_player_name)}"
            self.agent_perspective = f"You voted for {str(self.target_player_name)}."
            self.observer_perspective = f"You heard {self.player_name} vote for {str(self.target_player_name)}."
            self.global_perspective = f"{self.player_name} voted for {str(self.target_player_name)}."
        elif self.type == ActionType.PRETEND:
            self.command_perspective = f"pretend to do task: {self.target_task.name}"
            self.agent_perspective = f"You pretended to do the {self.target_task.name} task."
            self.observer_perspective = f"You saw {self.player_name} complete the {self.target_task.name} task."
            self.global_perspective = f"{self.player_name} pretended to do the {self.target_task.name} task."
        elif self.type == ActionType.SPEAK:
            self.command_perspective = f"{self.target_message}"
            self.agent_perspective = f"You said: {self.target_message}"
            self.observer_perspective = f"{self.player_name} said: {self.target_message}"
            self.global_perspective = f"{self.player_name} said: {self.target_message}"
        else:
            raise ValueError(f"Unknown action type: {self.type}")
        return self

    def copy(self, **kwargs):
        data = self.__dict__.copy()
        data.update(kwargs)
        return Action(**data)

    def __str__(self):
        # obj_dict = {k: v for k, v in self.__dict__.items() if v is not None and v != ""}
        # perspective_fields = ["agent_perspective", "observer_perspective", "global_perspective", "command_perspective", "text", "result", "spectator"]
        # for field in perspective_fields:
        #     obj_dict.pop(field, None)
        # return str(obj_dict)
        return self.global_perspective

    def __repr__(self):
        # obj_dict = {k: v for k, v in self.__dict__.items() if v is not None and v != ""}
        # perspective_fields = ["agent_perspective", "observer_perspective", "global_perspective", "command_perspective", "text", "result", "spectator"]
        # for field in perspective_fields:
        #     obj_dict.pop(field, None)
        # return str(obj_dict)
        return self.global_perspective

    def __eq__(self, other):
        if not isinstance(other, Action):
            return False
        
        obj_dict = {k: v for k, v in self.__dict__.items() if v is not None and v != ""}
        other_dict = {k: v for k, v in other.__dict__.items() if v is not None and v != ""}
        # Remove perspective fields for Action objects
        perspective_fields = ["agent_perspective", "observer_perspective", "global_perspective", "command_perspective", "text", "result", "spectator"]
        for field in perspective_fields:
            obj_dict.pop(field, None)
            other_dict.pop(field, None)
        return obj_dict == other_dict
