import re
from typing import Dict, List, Optional, Tuple

import tiktoken
from ollama import chat

from among_them.config import OLLAMA_LLM_MODEL_NAME
from among_them.llm_prompts import UNIVERSAL_SYSTEM_PROMPT
from among_them.models.action import Action, ActionType
from among_them.models.player_role import PlayerRole


class Player:
    """
    A unified player class that can be either human-controlled or AI-controlled.
    
    The player's behavior is determined by the manual_human_control flag:
    - If True: The player is controlled by human input
    - If False: The player is controlled by an LLM
    """

    def __init__(
        self, 
        name: str, 
        role: PlayerRole = PlayerRole.CREWMATE,
        manual_human_control: bool = False,
        llm_model_name: str = OLLAMA_LLM_MODEL_NAME
    ):
        self.name = name
        self.role = role
        self.manual_human_control = manual_human_control
        self.llm_model_name = llm_model_name
    
    def prompt_action(
        self, actions: List[Action], history_str: str
    ) -> Tuple[int, str, str, Dict[str, int]]:
        """
        Prompts the player for an action, either via human input or LLM.
        
        Args:
            actions: List of available actions
            history_str: String representation of game history
            
        Returns:
            Tuple containing:
            - Index of chosen action
            - Response text
            - Chain of thought (if AI player)
            - Token usage details (if AI player)
        """
        if self.manual_human_control:
            return self._handle_human_action(actions, history_str)
        else:
            return self._handle_ai_action(actions, history_str)
    
    def _handle_human_action(
        self, actions: List[Action], history_str: str
    ) -> Tuple[int, str, str, Dict[str, int]]:
        """Handle action selection for human-controlled player"""
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
                            actions[chosen_action].text,
                            "",
                            {},
                        )
                    else:
                        print(f"Please enter a number between 0 and {len(actions) - 1}")
                except ValueError:
                    print("Invalid input. Please enter a number.")
    
    def _handle_ai_action(
        self, actions: List[Action], history_str: str
    ) -> Tuple[int, str, str, Dict[str, int]]:
        """Handle action selection for AI-controlled player"""
        system_prompt = UNIVERSAL_SYSTEM_PROMPT
        prompt = history_str

        # Add available actions to prompt if needed
        if actions and actions[0].type != ActionType.SPEAK:
            actions_text = "<actions>\n" + "\n".join(f"<action>{action.text}</action>" for action in actions) + "\n</actions>"
            prompt += f"\n\nAvailable actions you can take at the moment:\n{actions_text}\nChosen action:"
        elif actions and actions[0].type == ActionType.SPEAK:
            prompt += "\n\nRespond in the following format: [Your name]: message"

        # Print prompts for debugging
        print("\033[91m" + system_prompt + "\033[0m")  # Light red for system prompt
        print("\033[92m" + prompt + "\033[0m")  # Light green for user prompt

        # Invoke LLM
        stream = chat(
            model=self.llm_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )

        # Process the response
        cot = None
        response_text = ""
        for chunk in stream:
            print(
                "\033[94m" + chunk["message"]["content"] + "\033[0m", end="", flush=True
            )
            response_text += chunk["message"]["content"]
        print("")

        cot_match = re.search(r"<think>.*?</think>", response_text, re.DOTALL)
        if cot_match:
            cot = cot_match.group(0)
            response_text = re.sub(
                r"<think>.*?</think>", "", response_text, flags=re.DOTALL
            )
        else:
            raise ValueError("No chain of thought found in response")

        # Clean up the response
        response_text = response_text.strip()

        # Determine the chosen action index
        action_idx = None
        if actions and not actions[0].type == ActionType.SPEAK:
            try:
                action_idx, _ = self._normalize_and_check_action_valid(
                    [action.text for action in actions], response_text
                )
            except ValueError:
                stream = chat(
                    model=self.llm_model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                        {
                            "role": "assistant",
                            "content": f"<think>{cot}</think>\n{response_text}",
                        },
                        {
                            "role": "assistant",
                            "content": f"<think>But wait, i need to choose one of the available actions without explanations. My actions are:\n{actions_text}\nSo the correct one would be ",
                        },
                    ],
                    stream=True,
                )
                print(
                    f"\033[91m<think>But wait, i need to choose one of the available actions without explanations. My actions are:\n{actions_text}\nSo the correct one would be \033[0m"
                )

                # Process the response
                response_text = ""
                for chunk in stream:
                    print(
                        "\033[94m" + chunk["message"]["content"] + "\033[0m",
                        end="",
                        flush=True,
                    )
                    response_text += chunk["message"]["content"]
                print("")

                # Clean up the response
                response_text = response_text.strip()
                action_idx, _ = self._normalize_and_check_action_valid(
                    [action.text for action in actions], response_text
                )
        elif actions[0].type == ActionType.SPEAK:
            # Extract text after "[something]: "
            match = re.search(r'\[(.*?)\]:\s*(.*)', response_text)
            if match:
                response_text = match.group(2)
            action_idx = 0

        # Calculate token usage with tiktoken
        encoding = tiktoken.encoding_for_model("gpt-4o")
        input_tokens = len(encoding.encode(system_prompt + prompt))
        output_tokens = len(encoding.encode(response_text + (cot or "")))

        return (
            action_idx, # type: ignore
            response_text,
            cot,
            {"input_tokens": input_tokens, "output_tokens": output_tokens},
        )

    def _normalize_and_check_action_valid(
        self, available_actions: List[str], chosen_action: str
    ) -> tuple[int, str]:
        """Check if the chosen action is valid and return its index"""
        chosen_action = chosen_action.strip().lower()
        available_actions = [a.lower() for a in available_actions]
        for action in range(len(available_actions) - 1, -1, -1): 
            if re.search(rf"\b{re.escape(available_actions[action])}\b", chosen_action, re.IGNORECASE):
                return action, available_actions[action]

        warning_str = (
            f"LLM did not conform to output format. "
            f"Expected one of {available_actions}, but got '{chosen_action}'"
        )
        print(warning_str)
        raise ValueError(warning_str)
