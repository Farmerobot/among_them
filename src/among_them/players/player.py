from typing import List
from among_them.models.player_role import PlayerRole
from among_them.models.action import Action
from langchain_core.messages.ai import UsageMetadata


class Player:
    def __init__(self, name: str, role: PlayerRole = PlayerRole.CREWMATE):
        self.name = name
        self.role = role

    def prompt_action(self, actions: List[Action], history_str: str) -> tuple[int, str, str, UsageMetadata]:
        pass
