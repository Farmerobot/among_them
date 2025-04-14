from enum import Enum


class GamePhase(Enum):
    GAME_START = "Game Start"
    TASK = "Task"
    DISCUSS = "Discuss"
    VOTE = "Vote"
    VOTE_RESULTS = "Vote Results"
    GAME_END = "Game End"

    def __repr__(self):
        return self.value
