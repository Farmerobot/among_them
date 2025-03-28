from enum import Enum


class GamePhase(str, Enum):
    MAIN_MENU = "Main Menu"
    TASK = "Task"
    DISCUSS = "Discuss"
    VOTE = "Vote"
        
    def __repr__(self):
        return self.value