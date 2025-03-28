from argparse import Action
from typing import List

from among_them.players.player import Player
from among_them.models.player_role import PlayerRole
from among_them.agents.unified_agent import UnifiedAgent
from among_them.models.action import Action
from langchain_core.messages.ai import UsageMetadata


class AIPlayer(Player):
    agent: UnifiedAgent

    def __init__(self, name: str, agent: UnifiedAgent, role: PlayerRole = PlayerRole.CREWMATE):
        super().__init__(name, role)
        self.agent = agent

    def prompt_action(self, actions: List[Action], history_str: str) -> tuple[int, str, str, UsageMetadata]:
        chosen_action_idx, response, cot, token_usage = self.agent.act(
            prompt=history_str,
            actions=actions,
        )
        return chosen_action_idx, response, cot, token_usage
