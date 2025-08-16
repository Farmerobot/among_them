#!/usr/bin/env python3
"""
Calculate action probabilities for Among Us game AI using MLX-LM
(DeepSeek-R1-Distill-Qwen-1.5B) and TWOSOME methodology for fair comparison.

This is a functional rewrite of scripts/calculate_action_probabilities.py
using MLX-LM for model loading, tokenization, manual decoding, and
teacher-forced scoring of candidate actions.
"""

import os
import time
import math
from typing import List, Tuple

import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm import stream_generate

from among_them.game_engine import GameEngine
from among_them.game_config import GameConfig
from among_them.models.action import Action


def load_deepseek_model_mlx():
    """Load DeepSeek-R1-Distill-Qwen-1.5B using MLX-LM."""
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    load_start = time.time()
    print(f"Loading model (MLX-LM): {model_name}")

    # Qwen-based tokenizers require trust_remote_code; eos token is inferred
    model, tokenizer = mlx_load(
        model_name,
        tokenizer_config={"trust_remote_code": True},
    )

    mx.eval(model.parameters())
    model.eval()
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s (device: {mx.default_device()})")
    return model, tokenizer


def stream_print(token_id: int, tokenizer) -> None:
    try:
        text = tokenizer.decode([int(token_id)])
    except Exception:
        text = f"<id:{token_id}>"
    print(text, end="", flush=True)


def _forced_sampler(forced_ids: List[int]):
    """Create a stateful sampler that returns the next forced token id each step."""
    idx = {"i": 0}

    def sampler(logprobs):  # logprobs: mx.array [vocab]
        # Ignore logprobs, always pick the next forced token.
        i = idx["i"]
        tid = forced_ids[i] if i < len(forced_ids) else forced_ids[-1]
        idx["i"] = i + 1
        # Important: return shape [1] so generate_step sees y as 1D and y[None] -> [1, 1]
        return mx.array([int(tid)], dtype=mx.uint32)

    return sampler


def generate_reasoning_and_calculate_probabilities(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    actions: List[Action],
) -> Tuple[str, List[Tuple[Action, float, float, List[str], List[float]]], List[Tuple[Action, float, float, List[str], List[float]]]]:
    """
    1) Generate reasoning greedily until </think>.
    2) Append "\n</think>\n\nAction:" and compute action probabilities via teacher forcing.

    Returns:
    - generated_text (string with the streamed reasoning)
    - token-normalized results list
    - word-normalized results list
    """
    if len(actions) == 0:
        return "", [], []

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 1. Prepare base prompt
    messages = [
        # {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    t_tok_start = time.time()
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    think_close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    think_close_token = think_close_ids[0] if len(think_close_ids) > 0 else None
    t_tok = time.time() - t_tok_start

    # 2. Generate reasoning until </think>
    print("\nStreaming reasoning (until </think>)...")
    print("<think>")

    max_reason_tokens = 50
    reason_token_ids: List[int] = []

    t_reason_start = time.time()
    for resp in stream_generate(
        model,
        tokenizer,
        prompt=input_text,
        # temp=0.0,
        # top_p=1.0,
        max_tokens=max_reason_tokens,
    ):
        tok = int(resp.token)
        if think_close_token is not None and tok == think_close_token:
            break
        reason_token_ids.append(tok)
        stream_print(tok, tokenizer)
    t_reason = time.time() - t_reason_start

    # Build strings
    generated_text = tokenizer.decode(reason_token_ids, skip_special_tokens=False)

    # 3. Build prefix with </think> and Action:
    end_think_str = "\n</think>"
    action_prefix_str = "\n\nAction:"
    prefix_str = input_text + generated_text + end_think_str + action_prefix_str

    # Probe top-k after Action: prefix
    print()
    print("\n🎯 TOP TOKEN PROBABILITIES after Action: prefix:")
    print("-" * 60)
    first_step_logprobs = None
    for i, resp in enumerate(
        stream_generate(
            model,
            tokenizer,
            prompt=prefix_str,
            # temp=0.0,
            # top_p=1.0,
            max_tokens=1,
        )
    ):
        if i == 0:
            first_step_logprobs = resp.logprobs
            break

    if first_step_logprobs is not None:
        # top-10 by probability
        import numpy as np

        # Convert MLX array to NumPy safely via .tolist() to avoid buffer dtype issues
        logp_np = np.asarray(first_step_logprobs.tolist(), dtype=np.float32)
        probs_np = np.exp(logp_np)
        top_k = 10
        top_idx = probs_np.argsort()[-top_k:][::-1]
        for rank, tid in enumerate(top_idx, start=1):
            token_prob = float(probs_np[tid])
            try:
                token_text = tokenizer.decode([int(tid)])
                if token_text == "\n":
                    token_display = "<newline>"
                elif token_text == "\n\n":
                    token_display = "<double_newline>"
                elif token_text == " ":
                    token_display = "<space>"
                elif token_text == "\t":
                    token_display = "<tab>"
                else:
                    token_display = repr(token_text)
            except Exception:
                token_display = f"<token_id_{int(tid)}>"
            print(f"{rank:2d}. {token_display:<20} {token_prob:.4%} (log: {math.log(max(token_prob,1e-12)):.4f})")

    # 4. Score each action with teacher forcing
    t_actions_start = time.time()
    results = []

    for action in actions:
        action_text = action.command_perspective
        action_token_ids = tokenizer.encode(action_text, add_special_tokens=False)
        num_tokens = len(action_token_ids)
        num_words = len(action_text.split())

        if num_tokens == 0:
            results.append((action, 0.0, 0.0, 0.0, [], []))
            continue

        # Teacher-force the model to emit action tokens and collect per-token probs
        total_log_prob = 0.0
        token_probs: List[float] = []
        token_texts: List[str] = []

        sampler = _forced_sampler(action_token_ids)
        step_idx = 0
        for resp in stream_generate(
            model,
            tokenizer,
            prompt=prefix_str,
            # temp=0.0,
            # top_p=1.0,
            max_tokens=num_tokens,
            sampler=sampler,
        ):
            # stream_generate yields an extra final response with finish_reason set.
            if getattr(resp, "finish_reason", None) is not None:
                break
            # Guard against extra iterations
            if step_idx >= num_tokens:
                break

            forced_tid = action_token_ids[step_idx]
            # Probability under the returned distribution
            logprobs = resp.logprobs
            # Access log p(token)
            lp = float(logprobs[int(forced_tid)].item())
            total_log_prob += lp
            p = math.exp(lp)
            token_probs.append(p * 100.0)  # percent
            token_texts.append(tokenizer.decode([int(forced_tid)]))

            # Stream the token text as in the original script
            stream_print(forced_tid, tokenizer)

            step_idx += 1

        # Newline after streaming this action's tokens
        print()

        # TWOSOME normalizations
        token_norm = total_log_prob / num_tokens
        word_norm = total_log_prob / num_words if num_words > 0 else total_log_prob

        results.append((action, total_log_prob, token_norm, word_norm, token_texts, token_probs))

    # Separate and sort
    token_results = [(a, lp, tn, tt, tp) for (a, lp, tn, wn, tt, tp) in results]
    word_results = [(a, lp, wn, tt, tp) for (a, lp, tn, wn, tt, tp) in results]
    token_results.sort(key=lambda x: x[2], reverse=True)
    word_results.sort(key=lambda x: x[2], reverse=True)

    t_actions = time.time() - t_actions_start

    # Brief timing summary
    print(f"\n📊 Total time breakdown:")
    print(f"   • Tokenizer: {t_tok:.2f}s")
    print(f"   • Reasoning generation: {t_reason:.2f}s ({(len(reason_token_ids) / max(t_reason,1e-6)):.2f} tokens/s)")
    print(f"   • Action probabilities: {t_actions:.2f}s")

    return generated_text, token_results, word_results


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
    start_time = time.time()

    # 1. Initialize game engine
    print("=" * 80)
    print("AMONG US ACTION PROBABILITY CALCULATOR (MLX-LM)")
    print("=" * 80)
    print("\n1. Initializing game engine...")

    game_config = GameConfig(
        num_tasks=2,
        num_players=5,
        num_impostors=1,
        map_size=0,
        num_task_phase_actions_per_player=10,
        num_discuss_phase_actions_per_player=2,
        impostor_cooldown=1,
    )

    engine = GameEngine(game_config)

    # 2. Get turn context
    print("\n2. Getting turn context...")
    turn_context = engine.get_turn_context()

    if turn_context is None or turn_context[0] is None:
        print("Game is over or no turn context available")
        return

    turn_context_history, actions_player_can_take, system_prompt, user_prompt, pre_discussion_vote_prompts = turn_context

    print(
        f"   Found {len(actions_player_can_take)} available actions: "
        f"{[a.command_perspective for a in actions_player_can_take]}"
    )

    # 3. Load model
    print("\n3. Loading DeepSeek-R1-Distill-Qwen-1.5B model (MLX-LM)...")
    model, tokenizer = load_deepseek_model_mlx()
    print("   Model loaded successfully")

    # 4. Generate reasoning and calculate probabilities
    print("\n4. Generating reasoning and calculating action probabilities...")
    generated_text, token_norm_results, word_norm_results = generate_reasoning_and_calculate_probabilities(
        model, tokenizer, system_prompt, user_prompt, actions_player_can_take
    )

    # 5. Output results
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

        token_list = "[" + ", ".join([repr(t) for t in token_texts]) + "]"
        if len(token_list) > 25:
            token_list = token_list[:22] + "...]"

        prob_list = "[" + ", ".join([f"{p:.3f}" for p in token_probs]) + "]"
        if len(prob_list) > 35:
            prob_list = prob_list[:32] + "...]"

        action_prob_wo_norm = math.exp(log_prob) * 100 / max(wo_norm_sum, 1e-12)
        action_prob_token_norm = math.exp(token_norm_prob) * 100 / max(token_norm_sum, 1e-12)
        action_prob_word_norm = math.exp(word_norm_prob) * 100 / max(word_norm_sum, 1e-12)

        print(f"{action_text:<30} {token_list:<25} {prob_list:<35} {action_prob_wo_norm:<11.4f} {action_prob_token_norm:<11.4f} {action_prob_word_norm:<11.4f}")

    # Most probable actions
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
    print(f"Model: DeepSeek-R1-Distill-Qwen-1.5B (MLX-LM)")
    print(f"Device: {mx.default_device()}")
    print(f"Total Execution Time: {end_time - start_time:.2f} seconds")
    print(f"TWOSOME Normalization: Token and Word")

    print("\n" + "=" * 80)
    print("CALCULATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
