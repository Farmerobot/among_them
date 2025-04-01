import re
from typing import List, Optional, Tuple
import tiktoken
from among_them.models.action import Action, ActionType
from among_them.llm_prompts import UNIVERSAL_SYSTEM_PROMPT
from ollama import chat


class UnifiedAgent:
    """A unified agent that can handle different types of actions.
    Note that initialization of ChatOpenAI might take a long time so it is recommended to initialize it once on the application startup.
    """

    def __init__(self, llm_model_name: str = "deepseek-r1:1.5b"):
        self.llm = None
        self.llm_model_name = llm_model_name

    def act(
        self, prompt: str, actions: List[Action]
    ) -> Tuple[Optional[int], str, Optional[str], int, int]:
        """
        Process the action based on type and return the appropriate response.

        Args:
            action_type: Type of action
            prompt: The prompt to send to LLM
            actions: Available actions (for action and vote types)

        Returns:
            Tuple containing:
            - Index of chosen action
            - Response text
            - Chain of thought (if present)
            - Token usage details
        """
        system_prompt = UNIVERSAL_SYSTEM_PROMPT

        # Add available actions to prompt if needed
        if actions and actions[0].type != ActionType.SPEAK:
            actions_text = "\n".join(f"- {action.text}" for action in actions)
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
                action_idx, _ = self.normalize_and_check_action_valid(
                    [action.text for action in actions], response_text
                )
            except ValueError as e:
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
                action_idx, _ = self.normalize_and_check_action_valid(
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
            action_idx,
            response_text,
            cot,
            {"input_tokens": input_tokens, "output_tokens": output_tokens},
        )


    def normalize_and_check_action_valid(
        self, available_actions: List[str], chosen_action: str
    ) -> tuple[int, str]:
        chosen_action = chosen_action.strip().lower()
        available_actions = [a.lower() for a in available_actions]
        for action in range(len(available_actions) - 1, -1, -1): # wait is last
            if re.search(rf"\b{re.escape(available_actions[action])}\b", chosen_action, re.IGNORECASE):
                return action, available_actions[action]

        warning_str = (
            f"LLM did not conform to output format. "
            f"Expected one of {available_actions}, but got '{chosen_action}'"
        )
        print(warning_str)
        raise ValueError(warning_str)