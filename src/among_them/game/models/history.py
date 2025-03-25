from enum import Enum
from typing import Dict, List

from pydantic import BaseModel, Field

from among_them.game.models.engine import GameLocation, GamePhase
from among_them.game.models.tasks import Task
from among_them.game.models.usage_metadata import UsageMetadata


class PlayerState(str, Enum):
    ALIVE = "Alive"
    DEAD = "Dead"
    DEAD_REPORTED = "Dead Reported"


class RoundData(BaseModel):
    location: GameLocation = GameLocation.LOC_CAFETERIA
    stage: GamePhase = GamePhase.MAIN_MENU
    life: PlayerState = PlayerState.ALIVE
    tasks: List[Task] = Field(default_factory=list)
    llm_responses: List[str] = Field(default_factory=list)
    token_usage: UsageMetadata = Field(default_factory=UsageMetadata)
    prompts: List[str] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)
    response: str = ""
    action_result: str = ""
    seen_actions: List[str] = Field(default_factory=list)
    player_in_room: str = ""
    observations: List[str] = Field(default_factory=list)
    chat_messages: List[str] = Field(default_factory=list)
    votes: Dict[str, str] = Field(default_factory=dict)

    def to_dict(self):
        return {
            "location": self.location.value,
            "stage": self.stage.value,
            "life": self.life.value,
            "token_usage": self.token_usage.to_dict(),
            "tasks": [str(task) for task in self.tasks],
            "llm_responses": self.llm_responses,
            "prompts": self.prompts,
            "actions": self.actions,
            "response": self.response,
            "action_result": self.action_result,
            "seen_actions": self.seen_actions,
            "player_in_room": self.player_in_room,
            "observations": self.observations,
            "chat_messages": self.chat_messages,
        }


class PlayerHistory(BaseModel):
    rounds: List[RoundData] = Field(default_factory=list)

    def add_round(self, round_data: RoundData):
        self.rounds.append(round_data)

    def get_history(self) -> List[RoundData]:
        return self.rounds

    def get_history_str(self) -> str:
        history_str = ""
        last_action_idx = 0
        for i, round in enumerate(self.rounds):
            if round.stage == GamePhase.ACTION_PHASE:
                last_action_idx = i
                seen_actions = "\n".join(round.seen_actions)
                observations = "\n".join(round.observations)
                history_str += f"Round {i + 1}\n"
                history_str += f"Location: {round.location.value}\n"
                history_str += f"Seen Actions:\n{seen_actions}\n"
                if len(round.llm_responses) > 1:
                    history_str += f"Your previous plan:\n{round.llm_responses[0]}\n"
                    history_str += f"Your action: {round.llm_responses[1]}\n"
                elif len(round.llm_responses) == 1:
                    history_str += f"Your action:\n{round.llm_responses[0]}\n"
                else:
                    history_str += f"Your action:\n{round.response}\n"
                history_str += f"Observations:\n{observations}\n"
                history_str += f"{round.player_in_room}\n"
            elif (
                i == len(self.rounds) - 1
                or self.rounds[i + 1].stage == GamePhase.ACTION_PHASE
            ):
                observations = "\n".join(round.observations)
                history_str += (
                    f"Round {i + 1}\n"
                    if last_action_idx == i - 1
                    else f"Rounds {last_action_idx + 2}-{i + 1}\n"
                )
                nl = "\n"
                history_str += (
                    ""
                    if last_action_idx == i - 1
                    else f"Chat Messages:\n{nl.join(round.chat_messages)}\n"
                )
                history_str += f"Location: {round.location}\n"
                history_str += f"Observations:\n{observations}\n"
        return history_str

    def to_dict(self):
        return {"rounds": [round_data.to_dict() for round_data in self.rounds]}
