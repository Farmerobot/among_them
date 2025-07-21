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
        agent_perspective: str = "",
        observer_perspective: str = "",
        global_perspective: str = "",
        # Keep old fields for backward compatibility during transition
        result: str = "",
        spectator: str = "",
    ):
        self.type = type
        self.player_name = player_name
        self.target_player_name = target_player_name
        self.target_location = target_location
        self.target_task = target_task
        self.text = text
        self.agent_perspective = agent_perspective
        self.observer_perspective = observer_perspective
        self.global_perspective = global_perspective
        # Backward compatibility
        self.result = result
        self.spectator = spectator
        
        if (text == "" and agent_perspective == "" and observer_perspective == "" 
            and global_perspective == "" and result == "" and spectator == ""):
            self.set_stories()

    def set_stories(self):
        if self.type == ActionType.MOVE:
            self.text = f"move to {self.target_location.value}"
            self.agent_perspective = f"You are in {self.target_location.value}."
            self.observer_perspective = f"You saw {self.player_name} move to {self.target_location.value}."
            self.global_perspective = f"{self.player_name} moved to {self.target_location.value}."
            # Backward compatibility
            self.result = f"You [{self.player_name}] moved to {self.target_location.value}"
            self.spectator = f"{self.player_name} moved to {self.target_location.value}"
        elif self.type == ActionType.WAIT:
            self.text = "wait"
            self.agent_perspective = "You waited."
            self.observer_perspective = f"You noticed {self.player_name} was waiting."
            self.global_perspective = f"{self.player_name} waited."
            # Backward compatibility
            self.result = f"You [{self.player_name}] are waiting"
            self.spectator = f"{self.player_name} waited"
        elif self.type == ActionType.TASK:
            self.text = f"{self.target_task.name}"
            self.agent_perspective = f"You completed the {self.target_task.name} task."
            self.observer_perspective = f"You saw {self.player_name} complete the {self.target_task.name} task."
            self.global_perspective = f"{self.player_name} completed the {self.target_task.name} task."
            # Backward compatibility
            self.result = f"You [{self.player_name}] completed task: {self.target_task.name}"
            self.spectator = f"{self.player_name} completed task: {self.target_task.name}"
        elif self.type == ActionType.REPORT:
            self.text = f"report the dead body of {str(self.target_player_name)} and start a discussion"
            self.agent_perspective = f"You reported the dead body of {str(self.target_player_name)} and started a discussion."
            self.observer_perspective = f"You saw {self.player_name} report the dead body of {str(self.target_player_name)}."
            self.global_perspective = f"A dead body was reported by {self.player_name}. Discussion started."
            # Backward compatibility
            self.result = f"You [{self.player_name}] reported dead body of {str(self.target_player_name)} to other players and started discussion"
            self.spectator = f"{self.player_name} reported dead body of {self.target_player_name} to everyone and started discussion"
        elif self.type == ActionType.KILL:
            self.text = f"kill {str(self.target_player_name)}"
            self.agent_perspective = f"You killed {str(self.target_player_name)}."
            self.observer_perspective = f"You witnessed {self.player_name} kill {str(self.target_player_name)}!"
            self.global_perspective = f"{str(self.target_player_name)} was killed."
            # Backward compatibility
            self.result = f"You [{self.player_name}] killed {str(self.target_player_name)}"
            self.spectator = f"{self.player_name} killed {self.target_player_name}"
        elif self.type == ActionType.VOTE:
            self.text = f"vote for {str(self.target_player_name)}"
            self.agent_perspective = f"You voted for {str(self.target_player_name)}."
            self.observer_perspective = f"You heard {self.player_name} vote for {str(self.target_player_name)}."
            self.global_perspective = f"{self.player_name} voted for {str(self.target_player_name)}."
            # Backward compatibility
            self.result = f"You [{self.player_name}] voted for {str(self.target_player_name)}"
            self.spectator = f"{self.player_name} voted for {self.target_player_name}"
        elif self.type == ActionType.PRETEND:
            self.text = f"pretend to do task: {self.target_task.name}"
            self.agent_perspective = f"You pretended to do the {self.target_task.name} task."
            self.observer_perspective = f"You saw {self.player_name} working on the {self.target_task.name} task."
            self.global_perspective = f"{self.player_name} appeared to be doing the {self.target_task.name} task."
            # Backward compatibility
            self.result = f"You [{self.player_name}] pretended {self.target_task.name}"
            self.spectator = f"{self.player_name} doing task {self.target_task.name}"
        elif self.type == ActionType.SPEAK:
            self.text = "speak"
            # Speech actions are handled differently as they contain the actual message
        else:
            raise ValueError(f"Unknown action type: {self.type}")

    def copy(self, **kwargs):
        data = self.__dict__.copy()
        data.update(kwargs)
        return Action(**data)

    def __str__(self):
        return self.spectator

    def __repr__(self):
        return self.spectator
