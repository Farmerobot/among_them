from typing import Any, List

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import Field

from among_them.config import OPENROUTER_API_KEY
from among_them.game.agents.base_agent import Agent
from among_them.llm_prompts import (
    ADVENTURE_ACTION_SYSTEM_PROMPT,
    ADVENTURE_ACTION_USER_PROMPT,
    ADVENTURE_PLAN_SYSTEM_PROMPT,
    ADVENTURE_PLAN_USER_PROMPT,
)
from among_them.game.utils import check_action_valid


class AdventureAgent(Agent):
    llm: ChatOpenAI = None
    response_llm: ChatOpenAI = None
    llm_model_name: str
    history: str = Field(default="")
    current_tasks: List[Any] = Field(default_factory=list)
    available_actions: List[str] = Field(default_factory=list)
    current_location: str = Field(default="")
    in_room: str = Field(default="")

    def __init__(self, **data):
        super().__init__(**data)
        self.init_llm()

    def init_llm(self):
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "Missing OpenRouter API key. "
                "Please set OPENROUTER_API_KEY in your environment."
            )
        self.llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            model=self.llm_model_name,
            temperature=1,
        )
        self.response_llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            model=self.llm_model_name,
            temperature=0,
        )

    def act(
        self,
        observations: str,
        tasks: List[str],
        actions: List[str],
        current_location: str,
        in_room: str,
    ) -> Any:
        self.history = observations
        self.current_tasks = tasks
        self.available_actions = actions
        self.current_location = current_location
        self.in_room = in_room

        plan_prompt, plan = self.create_plan()
        action_prompt, action_idx, action = self.choose_action(plan)
        self.responses.append(plan)
        self.responses.append(action)
        return [plan_prompt, action_prompt], action_idx

    def create_plan(self) -> str:
        plan_prompt = ADVENTURE_PLAN_USER_PROMPT.format(
            player_name=self.player_name,
            player_role=self.role,
            history=self.history,
            tasks=[str(task) for task in self.current_tasks],
            actions="- " + "\n- ".join(self.available_actions),
            in_room=self.in_room,
            current_location=self.current_location,
        )

        messages = [
            SystemMessage(content=ADVENTURE_PLAN_SYSTEM_PROMPT),
            HumanMessage(content=plan_prompt),
        ]

        plan = self.llm.invoke(messages)
        # print("\nCreated plan:", plan.content)
        self.add_token_usage(plan.usage_metadata)
        return plan_prompt, plan.content.strip()

    def choose_action(self, plan: str) -> int:
        action_prompt = ADVENTURE_ACTION_USER_PROMPT.format(
            player_name=self.player_name,
            player_role=self.role,
            plan=plan,
            actions="- " + "\n- ".join(self.available_actions),
        )

        messages = [
            SystemMessage(content=ADVENTURE_ACTION_SYSTEM_PROMPT),
            HumanMessage(content=action_prompt),
        ]

        chosen_action = self.response_llm.invoke(messages)
        # print("\nChosen action:", chosen_action.content)
        self.add_token_usage(chosen_action.usage_metadata)
        chosen_action = chosen_action.content.strip()
        return action_prompt, *check_action_valid(
            self.available_actions, chosen_action, self.player_name
        )
