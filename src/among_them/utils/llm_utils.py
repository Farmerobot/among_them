import re
from typing import List, Optional, Tuple

from among_them.config import LLMBackend, LLM_BACKEND, OPENROUTER_API_KEY, MODEL_NAME
from among_them.models.action import Action, ActionType


def invoke_llm(
    conversation: List[dict],
    model_name: str,
    allowed_actions: Optional[List[str]] = None,
    single_line_only: bool = False,
    max_output_chars: Optional[int] = None,
) -> Tuple[str, Optional[str]]:
    """
    Invoke the LLM with the given conversation history and handle exceptions.

    Args:
        conversation: List of message dicts [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        model_name: The model name to use
        allowed_actions: Optional list of allowed actions for validation
        single_line_only: Whether to enforce single-line responses
        max_output_chars: Maximum output characters allowed

    Returns:
        Tuple of (response_text, chain_of_thought)

    Raises:
        ValueError: If no chain of thought is found in the response
    """
    messages = conversation

    raw = "<think>" if LLM_BACKEND == LLMBackend.MLX else ""
    raw_reasoning = ""
    raw_content = ""

    max_reasoning_retries = 3 if LLM_BACKEND in [LLMBackend.OPENROUTER, LLMBackend.OLLAMA] else 1
    attempt = 0

    while True:
        # Reset buffers for each attempt
        if attempt > 0:
            raw = "<think>" if LLM_BACKEND == LLMBackend.MLX else ""
            raw_reasoning = ""
            raw_content = ""

        try:
            if LLM_BACKEND == LLMBackend.MLX:
                from mlx_lm import stream_generate
                from mlx_lm.sample_utils import make_sampler
                from among_them.config import MLX_MODEL, MLX_TOKENIZER

                model, tokenizer = MLX_MODEL, MLX_TOKENIZER
                prompt_chat = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                chunks = stream_generate(
                    model, tokenizer, prompt=prompt_chat,
                    max_tokens=2000, sampler=make_sampler(temp=0.0)
                )
            
            elif LLM_BACKEND == LLMBackend.OLLAMA:
                import ollama
                
                print(f"Using Ollama with model: {model_name}")
                
                response = ollama.chat(
                    model=model_name,
                    messages=messages,
                    stream=True,
                )
                chunks = response
            
            elif LLM_BACKEND == LLMBackend.OPENROUTER:
                from openai import OpenAI

                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=OPENROUTER_API_KEY,
                )

                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    stream=True
                )
                chunks = response
            
            else:
                raise ValueError(f"Unsupported LLM backend: {LLM_BACKEND}")

            stop_stream = False
            for chunk in chunks:
                if LLM_BACKEND == LLMBackend.MLX:
                    content = chunk.text
                    raw += content
                    print("\033[94m" + content + "\033[0m", end="", flush=True)
                    if max_output_chars is not None and len(raw) >= max_output_chars:
                        stop_stream = True
                        break
                    
                elif LLM_BACKEND == LLMBackend.OLLAMA:
                    # Handle both native .thinking attribute and <think> tags in content
                    reasoning = getattr(chunk["message"], "thinking", "")
                    content = getattr(chunk["message"], "content", "")
                    
                    if reasoning:
                        # Model provides separate thinking attribute
                        raw_reasoning += reasoning
                        print("\033[90m" + reasoning + "\033[0m", end="", flush=True)
                    
                    if content:
                        # Accumulate in raw for <think> tag extraction
                        raw += content
                        raw_content += content
                        
                        # Color code the output based on whether it's in <think> tags
                        # Count opening and closing tags to determine if we're inside a think block
                        open_tags = raw.count("<think>")
                        close_tags = raw.count("</think>")
                        inside_think = open_tags > close_tags
                        
                        if inside_think:
                            # Inside think block - use cyan (light blue)
                            print("\033[96m" + content + "\033[0m", end="", flush=True)
                        else:
                            # Outside think block - use blue
                            print("\033[94m" + content + "\033[0m", end="", flush=True)
                        
                        if max_output_chars is not None and len(raw) >= max_output_chars:
                            stop_stream = True
                            break
                        if single_line_only and allowed_actions:
                            # Only check content outside <think> tags
                            content_without_think = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL)
                            first_line = next((ln for ln in content_without_think.splitlines() if ln.strip()), "")
                            normalized_first = first_line.lstrip("*- ").strip().lower()
                            if normalized_first in [a.strip().lower() for a in allowed_actions]:
                                stop_stream = True
                                break
                    
                elif LLM_BACKEND == LLMBackend.OPENROUTER:
                    try:
                        content = chunk.choices[0].delta.content
                        if content:
                            raw_content += content
                            print("\033[94m" + content + "\033[0m", end="", flush=True)
                            if max_output_chars is not None and len(raw_content) >= max_output_chars:
                                stop_stream = True
                                break
                            if single_line_only and allowed_actions:
                                first_line = next((ln for ln in raw_content.splitlines() if ln.strip()), "")
                                normalized_first = first_line.lstrip("*- ").strip().lower()
                                if normalized_first in [a.strip().lower() for a in allowed_actions]:
                                    stop_stream = True
                                    break
                        
                        reasoning = chunk.choices[0].delta.reasoning
                        if reasoning:
                            raw_reasoning += reasoning
                            print("\033[90m" + reasoning + "\033[0m", end="", flush=True)
                    except:
                        print("\n\033[91mError parsing OpenRouter response\033[0m")
                
                # Check for hallucination (only for local backends that use raw)
                if LLM_BACKEND in [LLMBackend.MLX, LLMBackend.OLLAMA]:
                    all_tags = re.findall(r"<(?!/?(?:think))[^>]+>", raw)
                    if len(all_tags) > 2:
                        raise Exception("LLM did hallucinate")

            # Ensure a newline after streaming
            print("")
            if stop_stream:
                # Best-effort: stop consuming the stream after capturing first valid line
                pass

        except KeyboardInterrupt:
            print("\n\033[93mKeyboardInterrupt detected! Switching to manual action selection.\033[0m")
            raise KeyboardInterrupt("User interrupted LLM generation")
        finally:
            # Always reset terminal color to avoid leaking color to subsequent output
            print("\033[0m", end="")

        # If this backend expects separate reasoning and we didn't get any, optionally retry
        if LLM_BACKEND == LLMBackend.OPENROUTER and not raw_reasoning:
            attempt += 1
            if attempt < max_reasoning_retries:
                print("\n\033[93mWarning: No reasoning provided by the LLM, retrying...\033[0m")
                continue
        break

    # Extract chain of thought and cleanup
    if LLM_BACKEND in [LLMBackend.MLX, LLMBackend.OLLAMA]:
        # Check if we have reasoning from .thinking attribute first
        if raw_reasoning:
            cot = "<think>\n" + raw_reasoning + "\n</think>"
            response_text = raw_content
        else:
            # Extract <think> tags from raw content
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
                cot = ""
                response_text = raw.strip()
    else:  # OpenRouter
        if not raw_reasoning:
            print("\n\033[93mWarning: LLM did not provide chain of thought. Proceeding without CoT.\033[0m")
            cot = ""
            response_text = raw_content
        else:
            cot = "<think>\n" + raw_reasoning + "\n</think>"
            response_text = raw_content

    return response_text, cot

def normalize_and_check_action_valid(
    available_actions: List[str], chosen_action: str
) -> tuple[int, str]:
    """Check if the chosen action is valid and return its index"""
    # Take only the first non-empty line and strip bullets/markdown artifacts
    line = next((ln for ln in chosen_action.splitlines() if ln.strip()), "")
    line = line.lstrip("*- ").strip()
    chosen_action = line.lower()
    available_actions = [a.strip().lower() for a in available_actions]
    
    for action in range(len(available_actions) - 1, -1, -1): 
        if re.search(rf"\b{re.escape(available_actions[action])}\b", chosen_action, re.IGNORECASE):
            return action, available_actions[action]
        if "vote" in available_actions[action]:
            if re.search(rf"\b{re.escape(' '.join(available_actions[action].split(' for ')))}\b", chosen_action, re.IGNORECASE):
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
            [action.set_stories().command_perspective for action in actions], llm_response
        )
        return action_idx, llm_response
