#!/usr/bin/env python3
"""
Calculate action probabilities for Among Us game AI using DeepSeek 1.5b model
and TWOSOME methodology for fair action comparison.
"""

import os
import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache
from typing import List, Tuple, Dict
import math

from among_them.game_engine import GameEngine
from among_them.game_config import GameConfig
from among_them.models.action import Action


def load_deepseek_model():
    """Load DeepSeek-R1-Distill-Qwen-1.5B model and tokenizer."""
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    load_start_time = time.time()
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Check if MPS is available and use it, otherwise use CPU
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS device")
    else:
        device = "cpu"
        print("MPS not available, using CPU")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map=device
    )
    model.eval()
    load_end_time = time.time()
    load_time = load_end_time - load_start_time
    
    print(f"Model loaded in {load_time:.2f}s")
    
    compile_start_time = time.time()
    model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
    compile_end_time = time.time()
    compile_time = compile_end_time - compile_start_time
    
    print(f"Model compiled in {compile_time:.2f}s")
    
    return model, tokenizer

def stream_print(token_id, tokenizer):
    try:
        text = tokenizer.decode([token_id])
    except Exception:
        text = f"<id:{token_id}>"
    print(text, end="", flush=True)


def generate_reasoning_and_calculate_probabilities(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    actions: List[Action]
) -> Tuple[str, List[Tuple[Action, float, float]], List[Tuple[Action, float, float]]]:
    """
    Generate reasoning until </think> then efficiently calculate action probabilities
    using KV-cache checkpoint to avoid memory issues with large batches.
    """
    if len(actions) == 0:
        return "", [], []
    
    # 1. Prepare base prompt
    messages = [
        # {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    tokenizer_start_time = time.time()
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 2. Generate reasoning until </think> - this creates our KV-cache checkpoint
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    think_close_token = tokenizer.encode("</think>", add_special_tokens=False)[0]
    tokenizer_end_time = time.time()
    tokenizer_time = tokenizer_end_time - tokenizer_start_time

    # Static cache to persist KV states across manual decoding
    static_cache_start_time = time.time()
    past_key_values = StaticCache(
        config=model.config,
        max_batch_size=1,
        # If you plan to reuse the cache, make sure the cache length is large enough for all cases
        max_cache_len=inputs.input_ids.shape[1] + 700 + len(actions) * 7,
        device=next(model.parameters()).device,
        dtype=next(model.parameters()).dtype,
    )
    static_cache_end_time = time.time()
    static_cache_time = static_cache_end_time - static_cache_start_time

    print("\nStreaming reasoning (until </think>)...")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To prevent long warnings :)
    print("<think>") # already in chat template

    first_token_start_time = time.time()
    device = next(model.parameters()).device
    attention_mask = inputs.get("attention_mask", torch.ones_like(inputs.input_ids))
    with torch.no_grad():
        # 1) Fill cache with the prompt
        prompt_len = inputs.input_ids.shape[1]
        cache_position = torch.arange(prompt_len, device=device)
        logits = model(
            inputs.input_ids,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=False,
        )[0]
        first_token_end_time = time.time()
        first_token_time = first_token_end_time - first_token_start_time
    
        # Generate reasoning and get KV cache checkpoint (streamed)
        reasoning_start_time = time.time()

        # 2) Greedy decode until </think> (one-token stop) or step cap
        max_reason_tokens = 50
        generated_reason_ids = []
        cur_pos = torch.tensor([prompt_len], device=device)
        for step in range(max_reason_tokens):
            next_token = torch.argmax(logits[:, -1], dim=-1)  # [1]
            token_id = next_token.item()
            # Do not include </think> in generation; append it later
            if token_id == think_close_token:
                break
            generated_reason_ids.append(token_id)
            # stream token text
            stream_print(token_id, tokenizer)

            # Write token into cache and compute next logits
            out = model(
                next_token[:, None],
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=cur_pos,
            )
            logits = out.logits
            cur_pos += 1

        reasoning_end_time = time.time()
        reasoning_time = reasoning_end_time - reasoning_start_time
    
    appending_start_time = time.time()

    base_sequence = torch.cat([inputs.input_ids[0], torch.tensor(generated_reason_ids, device=device)])
    generated_text = tokenizer.decode(base_sequence, skip_special_tokens=False)
    
    # Append </think> and Action: prefix directly into the cache (continue from cur_pos)
    end_think_token_ids = tokenizer.encode("\n</think>", add_special_tokens=False)
    action_prefix_token_ids = tokenizer.encode("\n\nAction:", add_special_tokens=False)
    base_seq_len = base_sequence.shape[0]
    current_logits = None
    with torch.no_grad():
        for tid in end_think_token_ids + action_prefix_token_ids:
            stream_print(tid, tokenizer)
            tok = torch.tensor([[tid]], dtype=torch.long, device=device)
            out = model(
                tok,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=cur_pos,
            )
            current_logits = out.logits[0, -1, :]
            cur_pos += 1
    # Logits after the Action: prefix
    last_logits = current_logits
    prefix_end_pos = base_seq_len + len(end_think_token_ids) + len(action_prefix_token_ids)
    # For display, you can still decode the extended text if desired
    # extended_sequence = torch.cat([base_sequence, torch.tensor(action_prefix_token_ids, device=model.device)])
    # generated_text = tokenizer.decode(extended_sequence, skip_special_tokens=False)
    
    appending_end_time = time.time()
    appending_time = appending_end_time - appending_start_time
    
    # 3. Add newline token and get checkpoint (cached KV states)
    action_calc_start_time = time.time()
    
    # 4. Calculate probabilities for each action using the checkpoint
    results = []
    
    for action in actions:
        action_text = action.command_perspective
        action_tokens = tokenizer(action_text, add_special_tokens=False, return_tensors="pt")
        action_token_ids = action_tokens.input_ids[0].to(model.device)
        
        num_tokens = len(action_token_ids)
        num_words = len(action_text.split())
        
        if num_tokens == 0:
            results.append((action, 0.0, 0.0, 0.0))
            continue
        
        # Calculate probability using teacher forcing with cached states
        total_log_prob = 0.0
        current_logits = last_logits
        token_probabilities = []  # Store individual token probabilities
        token_texts = []  # Store individual token texts
        
        # Reset cache position to the end of the Action: prefix so we overwrite any prior action tokens
        cache_position_act = torch.tensor([prefix_end_pos], device=next(model.parameters()).device)

        for i, token_id in enumerate(action_token_ids):
            # Get probability of this token
            probs = torch.softmax(current_logits, dim=-1)
            token_prob = probs[token_id].item()
            token_log_prob = math.log(max(token_prob, 1e-10))
            total_log_prob += token_log_prob
            
            # Store token info
            token_probabilities.append(token_prob * 100)  # Convert to percentage
            token_texts.append(tokenizer.decode([token_id]))
            
            # Stream this token
            stream_print(token_id, tokenizer)
            
            # If not the last token, get next logits using cached states
            if i < len(action_token_ids) - 1:
                with torch.no_grad():
                    next_output = model(
                        token_id.unsqueeze(0).unsqueeze(0),
                        past_key_values=past_key_values,
                        use_cache=True,
                        cache_position=cache_position_act,
                    )
                    current_logits = next_output.logits[0, -1, :]
                    cache_position_act += 1
        # End of stream for this action
        print("")
        
        # Calculate BOTH normalizations
        token_normalized_prob = total_log_prob / num_tokens if num_tokens > 0 else total_log_prob
        word_normalized_prob = total_log_prob / num_words if num_words > 0 else total_log_prob
        
        results.append((action, total_log_prob, token_normalized_prob, word_normalized_prob, token_texts, token_probabilities))
    
    # Create separate results for each normalization method
    token_results = [(action, log_prob, token_norm, token_texts, token_probs) for action, log_prob, token_norm, _, token_texts, token_probs in results]
    word_results = [(action, log_prob, word_norm, token_texts, token_probs) for action, log_prob, _, word_norm, token_texts, token_probs in results]
    
    # Sort both by their respective normalized probabilities (descending)
    token_results.sort(key=lambda x: x[2], reverse=True)
    word_results.sort(key=lambda x: x[2], reverse=True)

    action_calc_end_time = time.time()
    action_calc_time = action_calc_end_time - action_calc_start_time
    
    print(f"\n📊 Total time breakdown:")
    print(f"   • Tokenizer: {tokenizer_time:.2f}s")
    print(f"   • Static cache: {static_cache_time:.2f}s")
    print(f"   • First token: {first_token_time:.2f}s")
    print(f"   • Reasoning generation: {reasoning_time:.2f}s ({len(generated_reason_ids) / reasoning_time:.2f} tokens/s)")
    action_prefix_and_action_tokens_per_s = (len(end_think_token_ids) + len(action_prefix_token_ids) + len(action_token_ids)) / (action_calc_time + appending_time)
    print(f"   • Appending Action: prefix: {appending_time:.2f}s ({action_prefix_and_action_tokens_per_s:.2f} tokens/s)")
    print(f"   • Action probabilities: {action_calc_time:.2f}s ({action_prefix_and_action_tokens_per_s:.2f} tokens/s)")
    print(f"   • Total: {tokenizer_time + static_cache_time + first_token_time + reasoning_time + appending_time + action_calc_time:.2f}s")
    
    # Show top token probabilities after Action: prefix
    print(f"\n🎯 TOP TOKEN PROBABILITIES after Action: prefix:")
    print("-" * 60)
    probs_after_think = torch.softmax(last_logits, dim=-1)
    top_k = 10
    top_values, top_indices = torch.topk(probs_after_think, top_k)
    
    for i in range(top_k):
        token_id = top_indices[i].item()
        token_prob = top_values[i].item()
        try:
            token_text = tokenizer.decode([token_id])
            # Handle special characters for display
            if token_text == '\n':
                token_display = '<newline>'
            elif token_text == '\n\n':
                token_display = '<double_newline>'
            elif token_text == ' ':
                token_display = '<space>'
            elif token_text == '\t':
                token_display = '<tab>'
            # elif token_text.strip() == '':
            #     token_display = '<whitespace>'
            else:
                token_display = repr(token_text)
        except:
            token_display = f'<token_id_{token_id}>'
        
        print(f"{i+1:2d}. {token_display:<20} {token_prob:.4%} (log: {math.log(token_prob):.4f})")
    
    return generated_text, token_results, word_results


# All action probability calculation is now done in generate_reasoning_and_calculate_probabilities


def format_game_context(engine: GameEngine) -> str:
    """Format the current game context for display."""
    history = engine.history[-1] if engine.history else None
    if not history:
        return "No game history available"
    
    context = []
    context.append(f"Game Phase: {history.phase}")
    context.append(f"Current Player: {history.action_taken.player_name}")
    context.append(f"Location: {history.location}")
    context.append(f"Alive Players: {', '.join(history.alive_player_names)}")
    context.append(f"Actions Until Phase Ends: {history.actions_until_phase_ends}")
    
    return "\n".join(context)


def main():
    """Main function to calculate action probabilities."""
    start_time = time.time()
    
    # 1. Initialize game engine
    print("=" * 80)
    print("AMONG US ACTION PROBABILITY CALCULATOR")
    print("=" * 80)
    print("\n1. Initializing game engine...")
    
    game_config = GameConfig(
        num_tasks=2,
        num_players=5,
        num_impostors=1,
        map_size=0,
        num_task_phase_actions_per_player=10,
        num_discuss_phase_actions_per_player=2,
        impostor_cooldown=1
    )
    
    engine = GameEngine(game_config)
    
    # 2. Get turn context
    print("\n2. Getting turn context...")
    turn_context = engine.get_turn_context()
    
    if turn_context is None or turn_context[0] is None:
        print("Game is over or no turn context available")
        return
    
    turn_context_history, actions_player_can_take, system_prompt, user_prompt, pre_discussion_vote_prompts = turn_context
    
    print(f"   Found {len(actions_player_can_take)} available actions: {[a.command_perspective for a in actions_player_can_take]}")
    
    # 3. Load DeepSeek model
    print("\n3. Loading DeepSeek-R1-Distill-Qwen-1.5B model...")
    model, tokenizer = load_deepseek_model()
    print(f"   Model loaded successfully")
    
    # 4. Generate reasoning and calculate ALL action probabilities with BOTH normalizations in ONE forward pass
    print("\n4. Generating reasoning and calculating action probabilities with BOTH normalizations in ONE forward pass...")
    
    generated_text, token_norm_results, word_norm_results = generate_reasoning_and_calculate_probabilities(
        model, tokenizer, system_prompt, user_prompt, actions_player_can_take
    )
    
    # 6. Output results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    print("\n📍 CURRENT GAME CONTEXT:")
    print("-" * 40)
    print(format_game_context(engine))
    
    print("\n📊 DETAILED ACTION PROBABILITY BREAKDOWN:")
    print("=" * 120)
    print(f"{'Action':<30} {'Tokens':<25} {'Token Probabilities (%)':<35} {'W/O Norm':<12} {'Token Norm':<12} {'Word Norm':<12}")
    print("=" * 120)
    
    wo_norm_sum = sum(math.exp(log_prob) for _, log_prob, _, _, _ in token_norm_results[:10])
    token_norm_sum = sum(math.exp(prob) for _, _, prob, _, _ in token_norm_results[:10])
    word_norm_sum = sum(math.exp(prob) for _, _, prob, _, _ in word_norm_results[:10])
    
    for action, log_prob, token_norm_prob, token_texts, token_probs in token_norm_results[:10]:
        # Find corresponding word norm result
        word_norm_prob = next((w_norm for w_action, _, w_norm, _, _ in word_norm_results if w_action == action), token_norm_prob)
        
        action_text = action.command_perspective[:27] + "..." if len(action.command_perspective) > 30 else action.command_perspective
        
        # Format tokens as [token1, token2, ...]
        token_list = "[" + ", ".join([repr(t) for t in token_texts]) + "]"
        if len(token_list) > 25:
            token_list = token_list[:22] + "...]"
        
        # Format probabilities as [prob1, prob2, ...]
        prob_list = "[" + ", ".join([f"{p:.3f}" for p in token_probs]) + "]"
        if len(prob_list) > 35:
            prob_list = prob_list[:32] + "...]"
        
        # Calculate action probability without normalization (geometric mean)
        action_prob_wo_norm = math.exp(log_prob) * 100 / wo_norm_sum 
        action_prob_token_norm = math.exp(token_norm_prob) * 100 / token_norm_sum
        action_prob_word_norm = math.exp(word_norm_prob) * 100 / word_norm_sum
        
        print(f"{action_text:<30} {token_list:<25} {prob_list:<35} {action_prob_wo_norm:<11.4f} {action_prob_token_norm:<11.4f} {action_prob_word_norm:<11.4f}")
    
    # Most probable action
    print("\n🎯 MOST PROBABLE ACTION (Token Normalization):")
    print("-" * 40)
    best_action_token, best_log_prob_token, best_norm_prob_token, best_tokens, best_token_probs = token_norm_results[0]
    print(f"Action: {best_action_token.command_perspective}")
    print(f"Tokens: {best_tokens}")
    print(f"Token Probabilities: {[f'{p:.3f}%' for p in best_token_probs]}")
    print(f"Log Probability: {best_log_prob_token:.4f}")
    print(f"Normalized Probability: {best_norm_prob_token:.4f}")
    print(f"Confidence Score: {math.exp(best_norm_prob_token):.4%}")
    
    print("\n🎯 MOST PROBABLE ACTION (Word Normalization):")
    print("-" * 40)
    best_action_word, best_log_prob_word, best_norm_prob_word, _, _ = word_norm_results[0]
    print(f"Action: {best_action_word.command_perspective}")
    print(f"Log Probability: {best_log_prob_word:.4f}")
    print(f"Normalized Probability: {best_norm_prob_word:.4f}")
    print(f"Confidence Score: {math.exp(best_norm_prob_word):.4%}")
    
    # Execution details
    end_time = time.time()
    print("\n⚙️ EXECUTION DETAILS:")
    print("-" * 40)
    print(f"Model: DeepSeek-R1-Distill-Qwen-1.5B")
    print(f"Device: {next(model.parameters()).device}")
    print(f"Total Execution Time: {end_time - start_time:.2f} seconds")
    print(f"TWOSOME Normalization: Token and Word")
    
    print("\n" + "=" * 80)
    print("CALCULATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
