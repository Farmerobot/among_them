from enum import Enum


class PlayerRole(str, Enum):
    CREWMATE = "Crewmate"
    IMPOSTOR = "Impostor"

    def __repr__(self):
        return self.value