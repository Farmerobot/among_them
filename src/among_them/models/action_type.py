from enum import Enum


class ActionType(Enum):
    VOTE = "Vote"
    SPEAK = "Speak"
    # WAIT = "Wait"
    MOVE = "Move"
    TASK = "Task"
    KILL = "Kill"
    REPORT = "Report"
    PRETEND = "Pretend"
