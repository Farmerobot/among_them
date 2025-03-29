import re
from typing import List, Optional, Tuple
import tiktoken
from among_them.models.action import Action, ActionType
from among_them.models.action import normalize_and_check_action_valid
from among_them.llm_prompts import (
    ADVENTURE_ACTION_SYSTEM_PROMPT,
    DISCUSSION_RESPONSE_SYSTEM_PROMPT,
    UNIVERSAL_SYSTEM_PROMPT,
    VOTING_SYSTEM_PROMPT,
)
from ollama import chat


class UnifiedAgent:
    """A unified agent that can handle different types of actions.
    Note that initialization of ChatOpenAI might take a long time so it is recommended to initialize it once on the application startup.
    """

    def __init__(self, llm_model_name: str = "deepseek-r1:1.5b"):
        self.llm = None
        self.llm_model_name = llm_model_name
        # self.init_llm()

    def get_system_prompt(self, action_type: ActionType) -> str:
        """Get the appropriate system prompt based on action type."""
        if action_type == ActionType.SPEAK:
            return DISCUSSION_RESPONSE_SYSTEM_PROMPT
        elif action_type == ActionType.VOTE:
            return VOTING_SYSTEM_PROMPT
        else:
            return ADVENTURE_ACTION_SYSTEM_PROMPT

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
        if actions:
            actions_text = "\n".join(f"- {action.text}" for action in actions)
            prompt += f"\n\nAvailable actions you can take at the moment:\n{actions_text}\n"

        # Print prompts for debugging
        # print("\033[91m" + system_prompt + "\033[0m")  # Light red for system prompt
        print("\033[92m" + prompt + "\033[0m")  # Light green for user prompt

        # Invoke LLM
        stream = chat(
            model=self.llm_model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ],
            stream=True
        )

        # Process the response
        cot = None
        response_text = ""
        for chunk in stream:
            print("\033[94m" + chunk['message']['content'] + "\033[0m", end='', flush=True)
            response_text += chunk['message']['content']

        cot_match = re.search(r'<think>.*?</think>', response_text, re.DOTALL)
        if cot_match:
            cot = cot_match.group(0)
            response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        else:
            raise ValueError("No chain of thought found in response")

        # Clean up the response
        response_text = response_text.strip()

        # Determine the chosen action index
        action_idx = None
        if actions and not actions[0].type == ActionType.SPEAK:
            action_idx, _ = normalize_and_check_action_valid([action.text for action in actions], response_text)
        elif actions[0].type == ActionType.SPEAK:
            action_idx = 0
    
        # Calculate token usage with tiktoken
        encoding = tiktoken.encoding_for_model("gpt-4o")
        input_tokens = len(encoding.encode(system_prompt + prompt))
        output_tokens = len(encoding.encode(response_text + (cot or "")))
        
        return action_idx, response_text, cot, {"input_tokens": input_tokens, "output_tokens": output_tokens}

    def init_llm(self):
        """Initialize the language model client."""
        if not OPENROUTER_API_KEY and not RUN_LOCALLY:
            raise ValueError(
                "Missing OpenRouter API key. "
                "Please set OPENROUTER_API_KEY in your environment."
            )

        # Support for local Ollama models
        if RUN_LOCALLY:
            self.llm = ChatOpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
                model=self.llm_model_name or "deepseek-r1:1.5b",
                temperature=0,
            )
        else:
            # Default to OpenRouter
            self.llm = chat(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
                model=self.llm_model_name,
                temperature=0,
            )
