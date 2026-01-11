#!/usr/bin/env python3

"""
Run the current environment model on the benchmark examples and score results.

- Loads data/benchmarks/benchmark_dataset.json (or specified benchmark file)
- For each example, reconstructs the environment prompt (already stored as full_prompt in dataset)
- Invokes the LLM using existing env (through among_them.utils.llm_utils.invoke_llm)
- Parses the LLM output into an action using among_them.utils.llm_utils.parse_llm_response_to_action
- Compares with correct_action; scores 0/1
- Repeats per example N times (configurable)
- Writes detailed and aggregated results to generated/benchmarks/<timestamp>_results.txt
- Also writes JSON results to generated/benchmarks/<timestamp>_results.json for later analysis

Usage:
  python scripts/run_benchmark.py --repeats 3 --model deepseek-r1:14b --backend ollama

Notes:
- We assume examples contain either `full_prompt` or enough context to rebuild prompts later.
- We run with temperature 0.0 via backend defaults to reduce variance; repeats are optional.
- Run-to-run spread calculation works for any number of repeats (2, 10, etc.)
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple, Optional

# Ensure src is importable
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from among_them.utils.llm_utils import invoke_llm, parse_llm_response_to_action  # type: ignore
from among_them.models.action import Action  # type: ignore
from among_them.models.action_type import ActionType  # type: ignore
from among_them.models.player import Player  # type: ignore
from among_them.game_jsonencoder import game_object_hook  # type: ignore
from among_them.utils.prompt_utils import build_conversation_for_player  # type: ignore
from among_them.utils.action_utils import get_task_phase_actions, get_vote_actions  # type: ignore
from among_them.models.phase import GamePhase  # type: ignore
from among_them.config import (  # type: ignore
    LLMBackend,
    LLM_BACKEND,
    MODEL_NAME,
)

def load_benchmark(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        data = json.load(f)
        # Handle both formats: benchmark_dataset or candidates
        if "candidates" in data and "benchmark_dataset" not in data:
            # Convert candidates format to benchmark_dataset format
            data["benchmark_dataset"] = data["candidates"]
        return data

def actions_from_strings(actions_list: List[str]) -> List[str]:
    # The `parse_llm_response_to_action` expects the list of command-perspective action strings.
    # Our benchmark stores strings already, so we can pass them through.
    return [a.strip() for a in actions_list]

def try_reconstruct_conversation(example: Dict[str, Any], dataset_meta: Dict[str, Any]) -> Optional[List[dict]]:
    """Reconstruct the conversation from original game_state using example id and current_player.

    Expects example['id'] like 'game_state_10.json:52' and example['current_player'].
    Returns a conversation (list of message dicts) or None if reconstruction fails.
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
        conversation, _ = build_conversation_for_player(current_player_obj, history_slice, players, game_config)
        return conversation
    except Exception:
        return None

def call_model_with_retry(conversation: List[dict], exec_model_name: str, max_retries: int = 10) -> Tuple[str, Optional[str]]:
    """Call invoke_llm with simple retry/backoff on rate-limit (429) errors."""
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return invoke_llm(conversation=conversation, model_name=exec_model_name, single_line_only=False)
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

def get_actions_as_objects(example: Dict[str, Any]) -> List[Action]:
    """Get Action objects (not strings) from example, needed for LOCAL_PROBABILITY backend."""
    # Try to get source_file directly, or reconstruct from id
    source_file = example.get('source_file')
    history_index = example.get('history_index')
    player_name = example.get('current_player')

    # If source_file or history_index is not provided, try to reconstruct from id
    if source_file is None or history_index is None:
        ex_id = example.get('id', '')
        if ':' not in ex_id:
            return []
        game_state_name, idx_str = ex_id.split(':', 1)
        try:
            if history_index is None:
                history_index = int(idx_str)
        except Exception:
            return []

        # Use the same logic as try_reconstruct_conversation to find the file
        if source_file is None:
            data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
            candidate_path = os.path.normpath(os.path.join(data_dir, game_state_name))
            if os.path.exists(candidate_path):
                source_file = candidate_path
            else:
                return []

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
        return []

    return actions

def run_example(example: Dict[str, Any], exec_model_name: str, dataset_meta: Dict[str, Any], backend_override: Optional[str] = None) -> Tuple[str, int, str, Optional[str]]:
    """Run the model once on an example using the exact same logic as manual_llm_game.py.

    Returns: (raw_output, chosen_idx, chosen_action_str, chain_of_thought)
    """
    # Load the actual game state from the benchmark example (same as manual_llm_game.py)
    ex_id = example.get('id', '')
    if ':' not in ex_id:
        raise ValueError(f"Invalid example id format: {ex_id}")

    game_state_name, idx_str = ex_id.split(':', 1)
    history_index = int(idx_str)

    # Load the original game state file
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    game_state_path = os.path.join(data_dir, game_state_name)

    if not os.path.exists(game_state_path):
        raise ValueError(f"Game state file not found: {game_state_path}")

    # Create GameEngine with the game state file (same as manual_llm_game.py)
    from among_them.game_engine import GameEngine  # type: ignore

    # Load game state to get game_config
    with open(game_state_path, 'r') as f:
        loaded = json.load(f, object_hook=game_object_hook)

    if len(loaded) == 3:
        _, _, game_config = loaded
    elif len(loaded) == 2:
        game_config = None
    else:
        raise ValueError(f"Invalid game state format in {game_state_path}")

    engine = GameEngine(game_config, file_path=game_state_path)

    # Load the state
    engine.load_state()

    # Truncate history to the exact point BEFORE the benchmark example action
    # If history_index is 49, we want entries 0-48, so get_turn_context() gives us context for action 49
    engine.history = engine.history[:history_index]

    # Get turn context (exact same call as manual_llm_game.py line 52)
    turn_context_history, actions_player_can_take, conversation, pre_discussion_vote_prompts = engine.get_turn_context()

    if not turn_context_history:
        # Game may have already ended at this point - try to extract the action from the original history
        # Load the full history again to get the action at history_index
        with open(game_state_path, 'r') as f:
            full_loaded = json.load(f, object_hook=game_object_hook)
        full_history = full_loaded[0] if len(full_loaded) >= 1 else []

        if history_index < len(full_history):
            # Extract the action from the history entry at history_index
            history_entry = full_history[history_index]
            if hasattr(history_entry, 'action_taken') and history_entry.action_taken:
                action_taken = history_entry.action_taken
                chosen_action_cmd = action_taken.set_stories().command_perspective.strip()

                # Get available actions from the example (as strings)
                actions = actions_from_strings(example.get('available_actions', []))
                if not actions:
                    actions = reconstruct_actions_from_source(example)

                if actions:
                    # Find matching index in string actions list
                    chosen_idx = 0
                    for i, action_str in enumerate(actions):
                        if action_str.strip().lower() == chosen_action_cmd.lower():
                            chosen_idx = i
                            break
                    # Return with empty CoT since game already ended
                    return "", chosen_idx, chosen_action_cmd, None

        raise ValueError("No turn context available - game may be over")

    # Find current player
    current_player_name = turn_context_history.action_taken.player_name
    current_player = next((p for p in engine.players if p.name == current_player_name), None)
    if not current_player:
        raise ValueError(f"Current player {current_player_name} not found")

    # Override the model name to use the exec_model_name (for benchmark consistency)
    original_model_name = current_player.llm_model_name
    current_player.llm_model_name = exec_model_name

    try:
        # Exact same LLM call logic as manual_llm_game.py (lines 139-150)
        allowed_actions_main = [a.set_stories().command_perspective for a in actions_player_can_take if a.type.name != "SPEAK"]
        single_line_only = len(allowed_actions_main) > 0
        allowed_actions_for_call = allowed_actions_main if single_line_only else None

        # Temporarily override backend if provided
        original_backend_env = os.environ.get("LLM_BACKEND")
        try:
            if backend_override:
                os.environ["LLM_BACKEND"] = backend_override.lower()
                # Reload config to pick up new backend
                import importlib
                from among_them import config
                importlib.reload(config)
            
            llm_response, cot = invoke_llm(
                conversation=conversation,
                model_name=exec_model_name,
                allowed_actions=allowed_actions_for_call,
                single_line_only=single_line_only,
                max_output_chars=None if single_line_only else 1500,
                actions=actions_player_can_take,
            )
        finally:
            # Restore original backend
            if original_backend_env is not None:
                os.environ["LLM_BACKEND"] = original_backend_env
            elif "LLM_BACKEND" in os.environ:
                del os.environ["LLM_BACKEND"]
            if backend_override:
                import importlib
                from among_them import config
                importlib.reload(config)

        # Parse response (exact same as manual_llm_game.py lines 154-156)
        action_idx, response_text = parse_llm_response_to_action(
            actions_player_can_take, llm_response, current_player.name
        )

        # Get the chosen action
        chosen_action_obj = actions_player_can_take[action_idx]
        chosen_action_cmd = chosen_action_obj.set_stories().command_perspective.strip()

        # Convert back to string actions list for compatibility with benchmark scoring
        actions = [a.set_stories().command_perspective for a in actions_player_can_take]

        return response_text, action_idx, chosen_action_cmd, cot

    finally:
        # Restore original model name
        current_player.llm_model_name = original_model_name

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', default=os.path.join(os.path.dirname(__file__), '..', 'data', 'benchmarks', 'benchmark_dataset.json'))
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--model', dest='model_name', default=None, help="Model name to use (overrides config)")
    parser.add_argument('--backend', dest='backend', default=None, help="Backend to use (ollama, openrouter, etc.)")
    parser.add_argument('--outdir', default=os.path.join(os.path.dirname(__file__), '..', 'generated', 'benchmarks'))
    args = parser.parse_args()

    data = load_benchmark(args.benchmark)
    examples: List[Dict[str, Any]] = data.get('benchmark_dataset', [])
    if not examples:
        print('No examples found in benchmark file.')
        return

    os.makedirs(args.outdir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(args.outdir, f'{ts}_results.txt')

    # Resolve execution model: use CLI --model if provided, otherwise use config
    exec_model_name = args.model_name or MODEL_NAME
    backend_override = args.backend
    display_model_name = exec_model_name or "(backend default)"

    # Run
    detailed_lines: List[str] = []
    summary_correct_per_example: List[Tuple[str, float, int]] = []  # (id, total_score, repeats)
    per_example_run_scores: Dict[str, List[float]] = {}

    # JSON data structure
    json_data: Dict[str, Any] = {
        "model": display_model_name,
        "exec_model": exec_model_name,
        "repeats": args.repeats,
        "timestamp": ts,
        "benchmark_file": args.benchmark,
        "examples": []
    }

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

        # JSON structure for this example
        example_json: Dict[str, Any] = {
            "id": ex_id,
            "runs": [],
            "avg_score": 0.0,
            "total_score": 0.0,
            "correct_actions": correct_actions
        }

        for r in range(args.repeats):
            run_json: Dict[str, Any] = {
                "run_number": r + 1,
                "score": 0.0,
                "success": False,
                "error": None
            }
            try:
                raw_output, chosen_idx, chosen_action, cot = run_example(ex, exec_model_name, data.get('metadata', {}), backend_override=backend_override)
                score = get_action_score(chosen_action, correct_actions)
                total_score += score
                # Non-ideal detection: exact match only is ideal
                is_ideal = (raw_output.strip() == chosen_action)
                if not is_ideal:
                    non_ideal_outputs += 1
                successful_runs += 1
                per_run_scores.append(score)
                cot_display = f"\n  CoT: {cot}" if cot else ""
                per_run_lines.append(f"- run {r+1}: chosen_idx={chosen_idx}, score={score:.2f}, action='{chosen_action}', output={raw_output.strip()}{cot_display}")

                # JSON run data
                run_json["success"] = True
                run_json["score"] = score
                run_json["chosen_idx"] = chosen_idx
                run_json["chosen_action"] = chosen_action
                run_json["raw_output"] = raw_output.strip()
                run_json["chain_of_thought"] = cot if cot else None
                run_json["is_ideal"] = is_ideal

            except Exception as e:
                err_str = str(e)
                if "LLM did not conform to output format" in err_str:
                    non_conform_errors += 1
                # Count this run as 0 score to include in spread/averages
                per_run_scores.append(0.0)
                per_run_lines.append(f"- run {r+1}: error={err_str}")

                # JSON run data for error
                run_json["success"] = False
                run_json["error"] = err_str
                run_json["score"] = 0.0

            example_json["runs"].append(run_json)

        avg_score = total_score / max(1, args.repeats)
        summary_correct_per_example.append((ex_id, total_score, args.repeats))
        example_json["avg_score"] = avg_score
        example_json["total_score"] = total_score

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
        
        json_data["examples"].append(example_json)

    # Aggregate
    total_score = sum(c for _, c, _ in summary_correct_per_example)
    total_runs = sum(r for _, _, r in summary_correct_per_example)
    overall = (total_score / max(1, total_runs)) if total_runs else 0.0

    # Run-to-run spread: average per-example absolute drift across repeats
    # For 2 repeats: |r1 - r2|; for >2 repeats: mean pairwise absolute difference
    # This works for any number of repeats (2, 10, etc.)
    spread_lines: List[str] = []
    avg_spread: Optional[float] = None
    per_example_spreads: List[float] = []
    
    if args.repeats > 1:
        for scores in per_example_run_scores.values():
            if len(scores) < 2:
                continue
            if len(scores) == 2:
                spread = abs(scores[0] - scores[1])
            else:
                # For >2 repeats: mean pairwise absolute difference
                n = len(scores)
                pair_count = n * (n - 1) / 2
                total_abs_diff = 0.0
                for i in range(n):
                    for j in range(i + 1, n):
                        total_abs_diff += abs(scores[i] - scores[j])
                spread = total_abs_diff / pair_count if pair_count > 0 else 0.0
            per_example_spreads.append(spread)
        
        if per_example_spreads:
            avg_spread = sum(per_example_spreads) / len(per_example_spreads)
            spread_lines.append(f"Run-to-run spread (avg per example): {avg_spread:.3f}")
        else:
            spread_lines.append("Run-to-run spread: N/A (insufficient data)")
    else:
        spread_lines.append("Run-to-run spread: N/A (repeats=1)")

    # Add statistics to JSON
    json_data["statistics"] = {
        "total_examples": len(examples),
        "total_runs": total_runs,
        "successful_runs": successful_runs,
        "overall_average_score": overall,
        "total_score": total_score,
        "non_conform_errors": non_conform_errors,
        "non_conform_error_rate": (non_conform_errors / max(1, total_runs)) * 100,
        "non_ideal_outputs": non_ideal_outputs,
        "non_ideal_output_rate": (non_ideal_outputs / max(1, successful_runs)) * 100,
        "run_to_run_spread": avg_spread,
        "per_example_spreads": per_example_spreads
    }

    # Write text report
    header = [
        f"Model: {display_model_name}",
        f"Repeats: {args.repeats}",
        f"Examples: {len(examples)}",
        f"Successful runs: {successful_runs}",
        f"Overall average score: {overall:.3f} (total={total_score:.2f}/{total_runs})",
        "",
        f"LLM non-conform errors: {non_conform_errors} ({(non_conform_errors / max(1, total_runs))*100:.1f}%)",
        f"Non-ideal outputs: {non_ideal_outputs} of {successful_runs} successful runs ({(non_ideal_outputs / max(1, successful_runs))*100:.1f}%)",
        *spread_lines,
        "",
        "Per-example results:",
        ""
    ]

    with open(out_path, 'w') as f:
        f.write("\n".join(header + detailed_lines))

    # Write JSON report
    json_out_path = os.path.join(args.outdir, f'{ts}_results.json')
    with open(json_out_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"Wrote benchmark results to {out_path}")
    print(f"Wrote benchmark results (JSON) to {json_out_path}")

if __name__ == '__main__':
    main()
