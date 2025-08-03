import re
from typing import List, Optional, Tuple

from among_them.config import USE_MLX
from among_them.llm_prompts import UNIVERSAL_SYSTEM_PROMPT
from among_them.models.action import Action, ActionType


def create_llm_prompts(actions: List[Action], history_str: str) -> Tuple[str, str]:
    """Constructs the prompt for the LLM.
    Args:
        actions: List of available actions
        history_str: String representation of game history
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = UNIVERSAL_SYSTEM_PROMPT
    prompt = history_str

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
    
    return system_prompt, prompt




def invoke_llm(system_prompt: str, prompt: str, model_name: str) -> Tuple[str, Optional[str]]:
    """
    Invoke the LLM with the given prompts and handle exceptions.
    
    Args:
        system_prompt: The system prompt to use
        prompt: The user prompt to send to the LLM
        model_name: The name of the LLM model to use
    
    Returns:
        Tuple of (response_text, chain_of_thought)
    
    Raises:
        ValueError: If no chain of thought is found in the response
    """
    # Debug print
    print("\n\033[93mSystem Prompt:\033[0m") #yellow
    print("\033[91m" + system_prompt + "\033[0m") #green
    print("\033[93mUser Prompt:\033[0m") #yellow
    print("\033[92m" + prompt + "\033[0m") #green

    # Build messages list
    messages = [
        {"role": "user", "content": system_prompt + "\n" + prompt},
    ]

    # Generate raw output via MLX or streaming chat
    raw = "<think>" if USE_MLX else ""
    try:
        if USE_MLX:
            from mlx_lm import stream_generate
            from mlx_lm.sample_utils import make_sampler
            from among_them.config import MLX_MODEL, MLX_TOKENIZER
            model, tokenizer = (MLX_MODEL, MLX_TOKENIZER)
            prompt_chat = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            chunks = stream_generate(
                model, tokenizer, prompt=prompt_chat,
                max_tokens=2000, sampler=make_sampler(temp=0.0)
            )
        else:
            from ollama import chat
            # Streaming Ollama chat
            chunks = chat(model=model_name, messages=messages, stream=True)

        for chunk in chunks:
            content = chunk["message"]["content"] if not USE_MLX else chunk.text
            print("\033[94m" + content + "\033[0m", end="", flush=True)
            raw += content
            # Count all non-<think> tags; Hallucination Early Check (HEC) System
            all_tags = re.findall(r"<(?!/?(?:think))[^>]+>", raw)
            if len(all_tags) > 2:
                raise Exception("LLM did hallucinate")
        print("")

    except KeyboardInterrupt:
        print("\n\033[93mKeyboardInterrupt detected! Switching to manual action selection.\033[0m")
        raise KeyboardInterrupt("User interrupted LLM generation")

    # Extract chain of thought and cleanup
    cot = ""
    cot_match = re.search(r"<think>.*?</think>", raw, re.DOTALL)
    cot_match_end = re.search(r".*?</think>", raw, re.DOTALL)
    if cot_match:
        cot = cot_match.group(0)
        response_text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    elif cot_match_end:
        cot = cot_match_end.group(0)
        response_text = re.sub(r".*?</think>", "", raw, flags=re.DOTALL).strip()
    else:
        raise ValueError("No chain of thought found in response")

    return response_text, cot

def normalize_and_check_action_valid(
    available_actions: List[str], chosen_action: str
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

def parse_llm_response_to_action(
    actions: List['Action'], llm_response: str, player_name: str
) -> tuple[int, str]:
    """
    Parses the raw LLM response to determine the chosen action index and response text.
    """
    from among_them.models.action import ActionType

    if not actions:
        raise ValueError("Cannot parse LLM response without a list of available actions.")

    if actions[0].type == ActionType.SPEAK:
        # Try to match both formats with the player's name
        pattern = r'<message>(.*?)</message>'
        pattern2 = r'<action type="player_message">(.*?)</action>'
        match = re.search(pattern, llm_response, re.DOTALL)
        match2 = re.search(pattern2, llm_response, re.DOTALL)
        
        response_text = ""
        if match:
            response_text = match.group(1)
        elif match2:
            response_text = match2.group(1)
        else:
            # Fallback to generic pattern if player name format not found
            generic_match = re.search(fr'(?:\[{re.escape(player_name)}\]|^{re.escape(player_name)}):\s*(.*)', llm_response, re.DOTALL)
            if generic_match:
                response_text = generic_match.group(1)
            else:
                # If no other format matches, assume the whole response is the message
                response_text = llm_response
        
        return 0, response_text
    else:
        action_idx, _ = normalize_and_check_action_valid(
            [action.text for action in actions], llm_response
        )
        return action_idx, llm_response
