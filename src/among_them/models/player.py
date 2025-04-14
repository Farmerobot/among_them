import re
from typing import Dict, List, Tuple, Optional

import tiktoken
from ollama import chat

from among_them.config import OLLAMA_LLM_MODEL_NAME
from among_them.llm_prompts import RULES, UNIVERSAL_SYSTEM_PROMPT
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
    
    def _handle_manual_action(self, actions: List[Action]) -> Tuple[int, str, str, Dict[str, int]]:
        """Handles manual action selection when AI generation is interrupted."""
        print("\nAI action generation interrupted. Please choose an action manually:")
        for i, action in enumerate(actions):
            print(f"{i + 1}. {action.text}")

        if actions[0].type == ActionType.SPEAK:
            try:
                return 0, input("Your message to others:"), "", {}
            except EOFError: # Handle Ctrl+D or similar EOF signals gracefully
                print("\nInput stream closed. Defaulting to \"Who did it?\"")
                return 0, "Who did it?", "", {}
        while True:
            try:
                choice = input(f"Enter the number of your choice (1-{len(actions)}): ")
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(actions):
                    selected_action = actions[choice_idx]
                    print(f"You chose: {selected_action.text}")
                    return (
                        choice_idx,
                        selected_action.text,
                        "",
                        {},
                    )
                else:
                    print("Invalid choice. Please enter a number within the range.")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except EOFError: # Handle Ctrl+D or similar EOF signals gracefully
                print("\nInput stream closed. Defaulting to first action.")
                return (
                    0,
                    actions[0].text,
                    "",
                    {},
                )

    def _invoke_llm(
        self, system_prompt: str, prompt: str, previous_messages: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Invoke the LLM with the given prompts and handle exceptions.
        
        Args:
            system_prompt: The system prompt to use
            prompt: The user prompt to send to the LLM
            previous_messages: Optional list of previous messages to include in the conversation
        
        Returns:
            Tuple of (response_text, chain_of_thought)
        
        Raises:
            ValueError: If no chain of thought is found in the response
        """
        # Initialize messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        
        # Add previous messages if provided
        if previous_messages:
            messages.extend(previous_messages)
        
        # Invoke LLM
        response_text = ""
        cot = ""
        try:
            stream = chat(
                model=self.llm_model_name,
                messages=messages,
                stream=True,
            )

            # Process the response
            for chunk in stream:
                print(
                    "\033[94m" + chunk["message"]["content"] + "\033[0m", end="", flush=True
                )
                response_text += chunk["message"]["content"]
                # Check for any XML-like tags in the response which would indicate hallucination
                if re.search(r"<(?!/?(?:action|message|think))[^>]+>", response_text, re.DOTALL):
                    raise Exception("LLM did hallucinate")
            print("")

        except KeyboardInterrupt:
            print("\n\033[93mKeyboardInterrupt detected! Switching to manual action selection.\033[0m")
            raise KeyboardInterrupt("User interrupted LLM generation")

        # Extract chain of thought
        cot_match = re.search(r"<think>.*?</think>", response_text, re.DOTALL)
        if cot_match:
            cot = cot_match.group(0)
            response_text = re.sub(
                r"<think>.*?</think>", "", response_text, flags=re.DOTALL
            )
        elif previous_messages is None:
            raise ValueError("No chain of thought found in response")

        # Clean up the response
        response_text = response_text.strip()
        
        return response_text, cot

    def _handle_ai_action(
        self, actions: List[Action], history_str: str
    ) -> Tuple[int, str, str, Dict[str, int]]:
        """Handle action selection for AI-controlled player"""
        system_prompt = UNIVERSAL_SYSTEM_PROMPT
        prompt = history_str
        prompt += RULES

        # Add available actions to prompt if needed
        if actions and actions[0].type != ActionType.SPEAK:
            actions_text = "<available_actions>\n" + "\n".join(f"<action>{action.text}</action>" for action in actions) + "\n</available_actions>"
            prompt += f"\n\n{actions_text}\n"
            prompt += "\n\nChoose one action. Respond in the following xml format: <action>action</action>"
        elif actions and actions[0].type == ActionType.SPEAK:
            prompt += "\n\nIt is discussion phase now. Respond to others in the following xml format: <message>message</message>"

        # Print prompts for debugging
        print("\033[91m" + system_prompt + "\033[0m")  # Light red for system prompt
        print("\033[92m" + prompt + "\033[0m")  # Light green for user prompt

        try:
            # Invoke LLM and get response
            response_text, cot = self._invoke_llm(system_prompt, prompt)
        except KeyboardInterrupt:
            # If interrupted, fall back to manual selection
            selected_action = self._handle_manual_action(actions)
            return selected_action

        # Determine the chosen action index
        action_idx = None
        if actions and not actions[0].type == ActionType.SPEAK:
            try:
                action_idx, _ = self._normalize_and_check_action_valid(
                    [action.text for action in actions], response_text
                )
            except ValueError:
                # Try again with a more explicit prompt
                retry_prompt = f"<think>But wait, i need to choose one of the available actions without explanations. My actions are:\n{actions_text}\nSo the correct one would be "
                print(f"\033[91m{retry_prompt}\033[0m")
                
                try:
                    # Create previous messages for the retry
                    previous_messages = [
                        {
                            "role": "assistant",
                            "content": f"<think>{cot}</think>\n{response_text}",
                        },
                        {
                            "role": "assistant",
                            "content": retry_prompt,
                        },
                    ]
                    
                    # Invoke LLM again with the retry prompt
                    response_text, _ = self._invoke_llm(system_prompt, prompt, previous_messages)
                    
                    # Try to extract the action again
                    action_idx, _ = self._normalize_and_check_action_valid(
                        [action.text for action in actions], response_text
                    )
                except KeyboardInterrupt:
                    # If interrupted during retry, fall back to manual selection
                    selected_action = self._handle_manual_action(actions)
                    return selected_action
        elif actions[0].type == ActionType.SPEAK:
            # Extract text after "[name]: " or "name: " using the player's name
            player_name = actions[0].player_name
            # Try to match both formats with the player's name
            pattern = r'<message>(.*?)</message>'
            match = re.search(pattern, response_text, re.DOTALL)
            
            if match:
                # Extract just the message part
                response_text = match.group(1)
            else:
                # Fallback to generic pattern if player name format not found
                generic_match = re.search(fr'(?:\[{re.escape(player_name)}\]|^{re.escape(player_name)}):\s*(.*)', response_text, re.DOTALL)
                if generic_match:
                    # Groups: 1=name in brackets, 2=name without brackets, 3=message
                    response_text = generic_match.group(3)
                else:
                    raise Exception("LLM did not provide a message with correct format")
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
