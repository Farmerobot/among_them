from enum import Enum


class PlayerRole(Enum):
    CREWMATE = "Crewmate"
    IMPOSTOR = "Impostor"

    def __repr__(self):
        return self.value
