#!/usr/bin/env python3
"""
Run the current environment model on the benchmark examples and score results.

- Loads data/benchmark_examples.json
- For each example, reconstructs the environment prompt (already stored as full_prompt in dataset)
- Invokes the LLM using existing env (through among_them.utils.llm_utils.invoke_llm)
- Parses the LLM output into an action using among_them.utils.llm_utils.parse_llm_response_to_action
- Compares with correct_action; scores 0/1
- Repeats per example N times (configurable)
- Writes detailed and aggregated results to generated/benchmarks/<timestamp>_results.txt

Usage:
  python scripts/run_benchmark.py --repeats 3 --model deepseek-r1:14b

Notes:
- We assume examples contain either `full_prompt` or enough context to rebuild prompts later.
- We run with temperature 0.0 via backend defaults to reduce variance; repeats are optional.
"""

import argparse
import json
import os
import sys
import time
import math
from typing import Any, Dict, List, Tuple, Optional

# Ensure src is importable
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from among_them.utils.llm_utils import invoke_llm  # type: ignore
from among_them.models.action import Action  # type: ignore
from among_them.models.action_type import ActionType  # type: ignore
from among_them.models.player import Player  # type: ignore
from among_them.game_jsonencoder import game_object_hook  # type: ignore
from among_them.utils.prompt_utils import reconstruct_environment_prompt_from_history  # type: ignore
from among_them.utils.action_utils import get_task_phase_actions, get_vote_actions  # type: ignore
from among_them.models.phase import GamePhase  # type: ignore
from among_them.config import (  # type: ignore
    LLMBackend,
    LLM_BACKEND,
    OLLAMA_LLM_MODEL_NAME,
    OPENROUTER_MODEL_NAME,
    HUGGINGFACE_MODEL_NAME,
)  


def load_benchmark(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        data = json.load(f)
        # Handle both formats: benchmark_examples or candidates
        if "candidates" in data and "benchmark_examples" not in data:
            # Convert candidates format to benchmark_examples format
            data["benchmark_examples"] = data["candidates"]
        return data


def actions_from_strings(actions_list: List[str]) -> List[str]:
    # The `parse_llm_response_to_action` expects the list of command-perspective action strings.
    # Our benchmark stores strings already, so we can pass them through.
    return [a.strip() for a in actions_list]


def resolve_backend_model_name() -> str:
    """Return the model name to use for execution based on current backend config (.env).

    This ignores the CLI --model for execution to ensure we use the same config as other scripts.
    The CLI --model is treated as a display label only.
    """
    if LLM_BACKEND == LLMBackend.OLLAMA:
        return OLLAMA_LLM_MODEL_NAME
    if LLM_BACKEND == LLMBackend.OPENROUTER:
        return OPENROUTER_MODEL_NAME
    if LLM_BACKEND == LLMBackend.HUGGINGFACE:
        return HUGGINGFACE_MODEL_NAME
    # MLX or others may not need a model string; return empty string
    return ""


def try_reconstruct_full_prompt(example: Dict[str, Any], dataset_meta: Dict[str, Any]) -> Optional[str]:
    """Reconstruct the full prompt from original game_state using example id and current_player.

    Expects example['id'] like 'game_state_10.json:52' and example['current_player'].
    """
    ex_id = example.get('id', '')
    if ':' not in ex_id:
        return None
    game_state_name, idx_str = ex_id.split(':', 1)
    try:
        hist_idx = int(idx_str)
    except Exception:
        return None

    # Resolve game_state path: prefer data/<game_state_name>
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    candidate_path = os.path.normpath(os.path.join(data_dir, game_state_name))
    possible_paths = [candidate_path]

    # Also search any listed source_files (these are candidate JSONs; we still derive the game_state path)
    for sf in (dataset_meta.get('source_files') or []):
        # Derive sibling data directory from candidate file location
        base_dir = os.path.dirname(os.path.normpath(sf))
        possible_paths.append(os.path.normpath(os.path.join(base_dir, '..', game_state_name)))

    # Deduplicate
    seen = set()
    possible_paths = [p for p in possible_paths if not (p in seen or seen.add(p))]

    history = None
    players = None
    game_config = None
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    loaded = json.load(f, object_hook=game_object_hook)
                if len(loaded) == 3:
                    history, players, game_config = loaded
                elif len(loaded) == 2:
                    history, players = loaded
                else:
                    continue
                break
            except Exception:
                continue

    if history is None or players is None or hist_idx >= len(history):
        return None

    current_player_name = example.get('current_player')
    if not current_player_name:
        return None
    current_player_obj = next((p for p in players if p.name == current_player_name), None)
    if not current_player_obj:
        return None

    history_slice = history[: hist_idx + 1]
    try:
        return reconstruct_environment_prompt_from_history(current_player_obj, history_slice, players, game_config)
    except Exception:
        return None


def call_model_with_retry(full_prompt: str, exec_model_name: str, max_retries: int = 10) -> Tuple[str, Optional[str]]:
    """Call invoke_llm with simple retry/backoff on rate-limit (429) errors."""
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return invoke_llm(system_prompt="", prompt=full_prompt, model_name=exec_model_name, single_line_only=False)
        except Exception as e:
            last_err = e
            msg = str(e)
            # Retry on rate limit signals
            if "429" in msg or "rate limit" in msg.lower() or "temporarily rate-limited" in msg.lower():
                time.sleep(min(1 + attempt, 10))
                continue
            # Non-retriable
            raise
    # Exhausted retries
    raise last_err if last_err else Exception("invoke_llm failed after retries")


def reconstruct_actions_from_source(example: Dict[str, Any]) -> List[str]:
    """If available_actions are missing, rebuild them from the original game state slice."""
    source_file = example.get('source_file')
    history_index = example.get('history_index')
    player_name = example.get('current_player')
    if source_file is None or history_index is None or player_name is None:
        return []

    # Load original state
    try:
        with open(source_file, 'r') as f:
            loaded = json.load(f, object_hook=game_object_hook)
        history, players, game_config = loaded if len(loaded) == 3 else (loaded[0], loaded[1], None)
    except Exception:
        return []

    if history_index < 0 or history_index >= len(history):
        return []

    # Slice up to this entry
    history_slice = history[: history_index + 1]
    hist_entry = history[history_index]
    phase = getattr(hist_entry, 'phase', None)
    if phase is None or game_config is None:
        return []

    player = next((p for p in players if p.name == player_name), None)
    if player is None:
        return []

    # Compute actions
    actions: List[Action] = []
    if phase == GamePhase.TASKS:
        actions = get_task_phase_actions(player, history_slice, players, game_config)
    elif phase == GamePhase.VOTING:
        actions = get_vote_actions(history_slice, players, player)
    else:
        # Discussion-only: no discrete actions to score
        return []

    return [a.set_stories().command_perspective.strip() for a in actions]


def get_action_score(chosen_action: str, correct_actions: List[Dict[str, Any]]) -> float:
    """Get the score for a chosen action based on the correct actions list."""
    chosen_action = chosen_action.strip()
    
    # Find exact match first
    for correct in correct_actions:
        if correct.get('action', '').strip() == chosen_action:
            return correct.get('score', 0.0)
    
    # If no exact match, return 0.0
    return 0.0


def run_example(example: Dict[str, Any], exec_model_name: str, dataset_meta: Dict[str, Any]) -> Tuple[str, int, str]:
    """Run the model once on an example.
    Returns: (raw_output, chosen_idx, chosen_action_str)
    """
    # Build prompts: we store full_prompt in the example; system prompt is included.
    full_prompt = example.get('full_prompt', '')
    if not full_prompt:
        # Reconstruct from original game state if possible
        full_prompt = try_reconstruct_full_prompt(example, dataset_meta) or ''
    if not full_prompt:
        raise ValueError(f"Example {example.get('id')} missing full_prompt")

    # Our invoke_llm expects system_prompt and prompt; we pass system empty and put all into user prompt
    response_text, cot = call_model_with_retry(full_prompt, exec_model_name)

    actions = actions_from_strings(example.get('available_actions', []))
    if not actions:
        # Try to rebuild from source
        actions = reconstruct_actions_from_source(example)
    # If still none, we cannot score this example (likely discussion-only); raise a clear error
    if not actions:
        raise ValueError('Cannot parse LLM response without a list of available actions.')

    # Normalize choice against the provided actions
    from among_them.utils.llm_utils import normalize_and_check_action_valid  # type: ignore
    chosen_idx, _ = normalize_and_check_action_valid(actions, response_text)

    return response_text, chosen_idx, actions[chosen_idx] if 0 <= chosen_idx < len(actions) else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', default=os.path.join(os.path.dirname(__file__), '..', 'data', 'benchmark_examples.json'))
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--model', dest='model_name', default=None)
    parser.add_argument('--outdir', default=os.path.join(os.path.dirname(__file__), '..', 'generated', 'benchmarks'))
    args = parser.parse_args()

    data = load_benchmark(args.benchmark)
    examples: List[Dict[str, Any]] = data.get('benchmark_examples', [])
    if not examples:
        print('No examples found in benchmark file.')
        return

    os.makedirs(args.outdir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(args.outdir, f'{ts}_results.txt')

    # Resolve execution model from backend config; CLI --model is display-only
    exec_model_name = resolve_backend_model_name()
    display_model_name = args.model_name or exec_model_name or "(backend default)"

    # Run
    detailed_lines: List[str] = []
    summary_correct_per_example: List[Tuple[str, float, int]] = []  # (id, total_score, repeats)
    per_example_run_scores: Dict[str, List[float]] = {}

    # Global stats
    non_conform_errors = 0
    non_ideal_outputs = 0
    total_runs = 0
    successful_runs = 0

    for ex in examples:
        ex_id = ex.get('id', 'unknown')
        actions = actions_from_strings(ex.get('available_actions', []))
        correct_actions = ex.get('correct_actions', []) or ex.get('llm_scored_actions', [])
        total_score = 0.0
        per_run_lines: List[str] = []
        per_run_scores: List[float] = []

        for r in range(args.repeats):
            try:
                raw_output, chosen_idx, chosen_action = run_example(ex, exec_model_name, data.get('metadata', {}))
                score = get_action_score(chosen_action, correct_actions)
                total_score += score
                # Non-ideal detection: exact match only is ideal
                is_ideal = (raw_output.strip() == chosen_action)
                if not is_ideal:
                    non_ideal_outputs += 1
                successful_runs += 1
                per_run_scores.append(score)
                per_run_lines.append(f"- run {r+1}: chosen_idx={chosen_idx}, score={score:.2f}, action='{chosen_action}', output={raw_output.strip()}")
            except Exception as e:
                err_str = str(e)
                if "LLM did not conform to output format" in err_str:
                    non_conform_errors += 1
                per_run_lines.append(f"- run {r+1}: error={err_str}")

        avg_score = total_score / max(1, args.repeats)
        summary_correct_per_example.append((ex_id, total_score, args.repeats))
        detailed_lines.append(f"Example {ex_id}: avg_score={avg_score:.2f} (total={total_score:.2f}/{args.repeats})")
        # Show scoring strategy map (correct actions with scores)
        if correct_actions:
            detailed_lines.append("Scoring map:")
            try:
                for ca in correct_actions:
                    a = (ca.get('action') or '').strip()
                    s = float(ca.get('score', 0.0) or 0.0)
                    detailed_lines.append(f"* {a} - {s:.2f}")
            except Exception:
                # Fallback if format unexpected
                pass
        # Record per-example run scores for SD
        per_example_run_scores[ex_id] = per_run_scores
        detailed_lines.extend(per_run_lines)
        detailed_lines.append("")
        total_runs += args.repeats

    # Aggregate
    total_score = sum(c for _, c, _ in summary_correct_per_example)
    total_runs = sum(r for _, _, r in summary_correct_per_example)
    overall = (total_score / max(1, total_runs)) if total_runs else 0.0

    # SD calculation across examples for repeats
    sd_lines: List[str] = []
    if args.repeats > 1:
        # Compute SD of totals per example (sum of scores across repeats)
        totals = [sum(scores) for _, scores in per_example_run_scores.items() if scores]
        if len(totals) >= 2:
            mean_total = sum(totals) / len(totals)
            variance = sum((t - mean_total) ** 2 for t in totals) / (len(totals) - 1)
            sd_total = math.sqrt(variance)
            sd_lines.append(f"Score SD across examples (sum over repeats): {sd_total:.3f}")
        else:
            sd_lines.append("Score SD across examples: N/A (insufficient data)")
    else:
        sd_lines.append("Score SD across examples: N/A (repeats=1)")

    # Write report
    header = [
        f"Model: {display_model_name}",
        f"Repeats: {args.repeats}",
        f"Examples: {len(examples)}",
        f"Successful runs: {successful_runs}",
        f"Overall average score: {overall:.3f} (total={total_score:.2f}/{total_runs})",
        "",
        f"LLM non-conform errors: {non_conform_errors} ({(non_conform_errors / max(1, total_runs))*100:.1f}%)",
        f"Non-ideal outputs: {non_ideal_outputs} of {successful_runs} successful runs ({(non_ideal_outputs / max(1, successful_runs))*100:.1f}%)",
        *sd_lines,
        "",
        "Per-example results:",
        ""
    ]

    with open(out_path, 'w') as f:
        f.write("\n".join(header + detailed_lines))

    print(f"Wrote benchmark results to {out_path}")


if __name__ == '__main__':
    main()
