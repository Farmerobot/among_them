from typing import Dict, List, Tuple
from among_them.agents.unified_agent import UnifiedAgent
from among_them.models.action_type import ActionType
from among_them.models.player_role import PlayerRole
from among_them.models.action import Action


class Player:
    def __init__(self, name: str, role: PlayerRole = PlayerRole.CREWMATE):
        self.name = name
        self.role = role

    def prompt_action(
        self, actions: List[Action], history_str: str
    ) -> tuple[int, str, str, Dict[str, int]]:
        pass


class AIPlayer(Player):
    agent: UnifiedAgent

    def __init__(
        self, name: str, agent: UnifiedAgent, role: PlayerRole = PlayerRole.CREWMATE
    ):
        super().__init__(name, role)
        self.agent = agent

    def prompt_action(
        self, actions: List[Action], history_str: str
    ) -> tuple[int, str, str, Dict[str, int]]:
        chosen_action_idx, response, cot, token_usage = self.agent.act(
            prompt=history_str,
            actions=actions,
        )
        return chosen_action_idx, response, cot, token_usage


class HumanPlayer(Player):
    def prompt_action(
        self, actions: List[Action], history_str: str
    ) -> Tuple[int, str, str, Dict[str, int]]:
        print(history_str)
        if actions[0].type == ActionType.SPEAK:
            return 0, input("Your message to others:"), "", {}
        else:
            action_prompt = "\n".join(
                [f"{i}: {action.text}" for i, action in enumerate(actions)]
            )
            prompt = "========================================\n"
            prompt += f"Your turn {self.name}: Choose an action\n{action_prompt}\n\n"
            prompt += "========================================\n"
            print(prompt)
            while True:
                try:
                    chosen_action = int(input("Choose action (enter the number): "))
                    if 0 <= chosen_action < len(actions):
                        return (
                            chosen_action,
                            actions[chosen_action],
                            "",
                            {},
                        )
                    else:
                        print(f"Please enter a number between 0 and {len(actions) - 1}")
                except ValueError:
                    print("Invalid input. Please enter a number.")
