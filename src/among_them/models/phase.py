from enum import Enum


class GamePhase(Enum):
    GAME_START = "Game Start"
    TASK = "Task"
    DISCUSS = "Discuss"
    VOTING = "Voting"
    VOTE_RESULTS = "Vote Results"
    GAME_END = "Game End"
