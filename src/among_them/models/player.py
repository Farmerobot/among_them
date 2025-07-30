import re
from typing import Dict, List, Optional, Tuple
import tiktoken
from among_them.config import LLMBackend, LLM_BACKEND, OLLAMA_LLM_MODEL_NAME, OPENROUTER_MODEL_NAME, HUGGINGFACE_MODEL_NAME
from among_them.llm_prompts import RULES, UNIVERSAL_SYSTEM_PROMPT
from among_them.models.action import Action, ActionType
from among_them.models.player_role import PlayerRole


class Player:
    """
    A player in the game. This class holds player-specific state and is responsible
    for generating LLM prompts for the player's turn.
    """

    def __init__(
        self, 
        name: str, 
        role: PlayerRole = PlayerRole.CREWMATE,
        manual_human_control: bool = False,
        llm_model_name: str = None
    ):
        self.name = name
        self.role = role
        self.manual_human_control = manual_human_control
        
        # Set model name based on backend if not provided
        if llm_model_name is None:
            if LLM_BACKEND == LLMBackend.OLLAMA:
                self.llm_model_name = OLLAMA_LLM_MODEL_NAME
            elif LLM_BACKEND == LLMBackend.OPENROUTER:
                self.llm_model_name = OPENROUTER_MODEL_NAME
            elif LLM_BACKEND == LLMBackend.HUGGINGFACE:
                self.llm_model_name = HUGGINGFACE_MODEL_NAME
            else:  # MLX
                self.llm_model_name = "mlx-model"  # MLX uses loaded model, not name
        else:
            self.llm_model_name = llm_model_name
            
        print(f"Using LLM backend: {LLM_BACKEND.value} with model: {self.llm_model_name}")
    
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

    def _invoke_llm(self, system_prompt: str, prompt: str) -> Tuple[str, Optional[str]]:
        """
        Invoke the LLM with the given prompts and handle exceptions.

        Args:
            system_prompt: The system prompt to use
            prompt: The user prompt to send to the LLM

        Returns:
            Tuple of (response_text, chain_of_thought)

        Raises:
            ValueError: If no chain of thought is found in the response
        """
        # Use the centralized invoke_llm function from llm_utils
        from among_them.utils.llm_utils import invoke_llm
        return invoke_llm(system_prompt, prompt, self.llm_model_name)

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
            if actions[0].type == ActionType.VOTE:
                prompt += "\n\nChoose one action. Please put your final answer within <action></action> xml tags"
            else:
                prompt += "\n\nChoose one action. Please put your final vote within <action></action> xml tags"
        elif actions and actions[0].type == ActionType.SPEAK:
            prompt += "\n\nIt is discussion phase now. Respond to others. Please put your message between <message></message> xml tags"

        # Print prompts for debugging
        # print("\033[91m" + system_prompt + "\033[0m")  # Light red for system prompt
        print("\033[92m" + prompt + "\033[0m")  # Light green for user prompt

        try:
            # Invoke LLM (Ollama or MLX)
            response_text, cot = self._invoke_llm(system_prompt, prompt)
        except KeyboardInterrupt:
            # If interrupted, fall back to manual selection
            selected_action = self._handle_manual_action(actions)
            return selected_action

        # Determine the chosen action index
        action_idx = None
        if actions and not actions[0].type == ActionType.SPEAK:
            action_idx, _ = self._normalize_and_check_action_valid(
                [action.text for action in actions], response_text
            )
        elif actions[0].type == ActionType.SPEAK:
            # Extract text after "[name]: " or "name: " using the player's name
            player_name = actions[0].player_name
            # Try to match both formats with the player's name
            pattern = r'<message>(.*?)</message>'
            pattern2 = r'<action type="player_message">(.*?)</action>'
            match = re.search(pattern, response_text, re.DOTALL)
            match2 = re.search(pattern2, response_text, re.DOTALL)
            
            if match:
                # Extract just the message part
                response_text = match.group(1)
            elif match2:
                response_text = match2.group(1)
            else:
                # Fallback to generic pattern if player name format not found
                generic_match = re.search(fr'(?:\[{re.escape(player_name)}\]|^{re.escape(player_name)}):\s*(.*)', response_text, re.DOTALL)
                if generic_match:
                    # Groups: 1=message
                    response_text = generic_match.group(1)
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
