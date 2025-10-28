#!/usr/bin/env python3
"""
Extract benchmark examples using LLM analysis to determine obvious correct/wrong actions.

This script:
1. Takes a specific game state file as input
2. Uses OpenAI to analyze each action's context, prompt, and response
3. Determines if the scenario has obvious correct/wrong answers
4. Generates benchmark examples with proper scoring
5. Stops when reaching the specified maximum number of examples
6. Saves with timestamp to avoid conflicts
"""

import json
import os
import sys
import time
import random
from typing import List, Dict, Any, Optional, Tuple

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from among_them.game_jsonencoder import game_object_hook
from among_them.models.history import History
from among_them.models.action_type import ActionType
from among_them.utils.prompt_utils import build_conversation_for_player
from among_them.utils.player_utils import get_last_player_action, get_players_in_room, get_dead_players


def load_game_state(file_path: str) -> tuple[List[History], List[Any], Any]:
    """Load a game state file and return history, players, config."""
    with open(file_path, 'r') as f:
        data = json.load(f, object_hook=game_object_hook)
        if len(data) == 3:
            return data[0], data[1], data[2]
        elif len(data) == 2:
            return data[0], data[1], None
        else:
            raise ValueError(f"Invalid game state format in {file_path}")


def analyze_action_with_llm(
    history_entry: History,
    history: List[History],
    players: List[Any],
    game_config: Any,
    openai_api_key: str
) -> Optional[Dict[str, Any]]:
    """
    Use OpenAI to analyze whether this action is suitable for benchmarking.
    Returns None if not suitable, or a dict with scoring information if suitable.
    """
    import openai
    
    client = openai.OpenAI(api_key=openai_api_key)
    
    # Find the player who took this action
    player_name = history_entry.action_taken.player_name
    current_player = next((p for p in players if p.name == player_name), None)
    if not current_player:
        return None
    
    # Get available actions
    actions_list = getattr(history_entry, "actions_agent_could_take", None) or []
    if not actions_list:
        return None
    
    # Skip discussion-only actions
    if history_entry.phase.name == "DISCUSS":
        return None
    
    # Build conversation up to this point
    try:
        conversation, _ = build_conversation_for_player(current_player, history, players, game_config)
        # Get last user message as the "full prompt" for analysis
        full_prompt = conversation[-1]["content"] if conversation and conversation[-1]["role"] == "user" else ""
    except Exception as e:
        print(f"Error building conversation for {player_name}: {e}")
        return None
    
    # Get LLM response and chain of thought
    llm_cot = getattr(history_entry, "llm_cot", "")
    llm_response = getattr(history_entry, "llm_response", "")
    
    # Create analysis prompt
    analysis_prompt = f"""
You are analyzing a game action to determine if it's suitable for benchmarking AI models in a social deduction game.

GAME CONTEXT:
- Player: {player_name}
- Role: {current_player.role.name}
- Phase: {history_entry.phase.name}
- Location: {getattr(history_entry.location, 'value', 'Unknown')}
- Available actions: {actions_list}

FULL PROMPT SENT TO LLM:
{full_prompt}

LLM CHAIN OF THOUGHT:
{llm_cot}

LLM RESPONSE:
{llm_response}

ANALYSIS TASK:
Determine if this scenario is suitable for benchmarking by evaluating:
1. Are there clearly correct actions (score ≥ 0.9)?
2. Are there clearly wrong actions (score ≤ 0.2)?
3. Is the score spread large (max - min ≥ 0.6)?
4. Is the context unambiguous, with enough evidence to justify the scoring?
5. Would this help distinguish strong vs. weak models (non-trivial, not all moves of equal value)?

For each available action, provide:
- action: the exact action string
- score: 0.0-1.0 (0.0=clearly wrong, 1.0=clearly correct)
- reasoning: brief explanation

Respond with JSON only:
{{
  "suitable": true/false,
  "reason": "why suitable/not suitable",
  "actions": [
    {{"action": "exact string", "score": 0.0-1.0, "reasoning": "explanation"}},
    ...
  ]
}}

Only mark as suitable if ALL of the following hold:
- At least one action has score ≥ 0.9
- At least one action has score ≤ 0.2
- Score spread (max - min) ≥ 0.6
- Reasoning is specific to the provided context (avoid generic/templated explanations)
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.0,
            max_tokens=2000
        )
        
        result = json.loads(response.choices[0].message.content)
        
        if result.get("suitable", False):
            return result
        else:
            print(f"  Action not suitable: {result.get('reason', 'Unknown reason')}")
            return None
            
    except Exception as e:
        print(f"Error analyzing action with LLM: {e}")
        return None


def analyze_cot_quality_with_llm(
    history_entry: History,
    current_player_name: str,
    full_prompt: str,
    llm_cot: str,
    llm_response: str,
    openai_api_key: str,
) -> Optional[Dict[str, Any]]:
    """Judge the chain-of-thought quality, novelty, evidence grounding, and hallucinations."""
    import openai

    client = openai.OpenAI(api_key=openai_api_key)

    judge_prompt = f"""
You are scoring the original model's chain-of-thought (CoT) and answer quality for a social deduction game turn.

Consider:
- Depth: multi-step planning, future positioning, trade-offs (0-1)
- Evidence: concrete references to rooms/players/rules/cooldown (0-1)
- Rule-use: task/report/kill/vote rules, visibility, phase constraints (0-1)
- Consistency: no contradictions or hallucinated facts (0-1)
- Novelty: non-trivial insight beyond obvious heuristics (0-1)

Context (abbreviated):
Player: {current_player_name}
FULL PROMPT:
{full_prompt}

MODEL COT:
{llm_cot}

MODEL ANSWER:
{llm_response}

Return JSON only:
{{
  "reasoning_score": 0.0-1.0,
  "evidence_score": 0.0-1.0,
  "rule_use_score": 0.0-1.0,
  "consistency_score": 0.0-1.0,
  "novelty_score": 0.0-1.0,
  "evidence_count": integer,
  "hallucination_flags": [],
  "summary": "one-line justification"
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            max_tokens=1200,
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"Error judging CoT quality: {e}")
        return None


def stringify_taken_action(history_entry: History) -> Optional[str]:
    """Map the taken action into the exact command-style string found in available_actions."""
    a = history_entry.action_taken
    if a.type == ActionType.REPORT and a.target_player_name:
        return f" Report the dead body of {a.target_player_name}"
    if a.type == ActionType.TASK and a.target_task:
        return f" {a.target_task.name}"
    if a.type == ActionType.MOVE and a.target_location:
        return f" Move to {a.target_location.value}"
    if a.type == ActionType.KILL and a.target_player_name:
        return f" Kill {a.target_player_name}"
    if a.type == ActionType.VOTE:
        return f" Vote for {a.target_player_name}"
    return None


def baseline_score_action(
    action_str: str,
    history_entry: History,
    history: List[History],
    players: List[Any],
    game_config: Any,
    current_player_name: str,
) -> float:
    """A naive heuristic to score actions; used to ensure the chosen action beats obvious baselines."""
    action = (action_str or "").strip().lower()
    # Get player and context
    current_player = next((p for p in players if p.name == current_player_name), None)
    if not current_player:
        return 0.0
    last_action = get_last_player_action(history, current_player)
    player_location = getattr(last_action, "location", None)
    player_loc_name = getattr(player_location, "value", "") if player_location else ""

    # Witnesses
    others_in_room = get_players_in_room(history, players, current_player)
    num_others = len(others_in_room)

    # Dead body present?
    dead_map = get_dead_players(history)
    body_here = any(loc == player_loc_name for loc in dead_map.values()) if player_loc_name else False

    # Heuristics
    if "report" in action:
        return 0.95 if body_here else 0.2
    if "kill" in action:
        # Bad if witnesses are present; better if alone
        if num_others == 0:
            return 0.85
        if num_others == 1:
            return 0.5
        return 0.15
    if "task" in action or "fix" in action or "empty" in action or "start" in action or "calibrate" in action:
        # Reward doing tasks for crewmates, neutral otherwise (we don't check role here to keep it simple)
        return 0.7
    if "vote" in action:
        # Voting during task phase is usually irrelevant; during voting phase it's necessary
        return 0.4
    if "move to" in action:
        # Slight preference to central rooms
        central = ["cafeteria", "admin"]
        for c in central:
            if c in action:
                return 0.6
        return 0.5
    return 0.3

def extract_scenario_data(
    history_entry: History,
    history: List[History],
    players: List[Any],
    game_config: Any,
    source_file: str,
    history_index: int,
    llm_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """Extract scenario data using LLM analysis results."""
    
    player_name = history_entry.action_taken.player_name
    actions_list = getattr(history_entry, "actions_agent_could_take", None) or []
    llm_cot = getattr(history_entry, "llm_cot", "")
    llm_response = getattr(history_entry, "llm_response", "")
    
    # Build conversation up to this point
    try:
        current_player = next((p for p in players if p.name == player_name), None)
        if not current_player:
            return None
        conversation, _ = build_conversation_for_player(current_player, history, players, game_config)
        # Get last user message as the "full prompt" for analysis
        full_prompt = conversation[-1]["content"] if conversation and conversation[-1]["role"] == "user" else ""
    except Exception as e:
        print(f"Error building conversation for {player_name}: {e}")
        return None
    
    return {
        "id": f"{os.path.basename(source_file)}:{history_index}",
        "description": f"{player_name} {history_entry.action_taken.type.name} scenario (LLM-analyzed)",
        "source_file": source_file,
        "history_index": history_index,
        "phase": getattr(history_entry, "phase", None).name if getattr(history_entry, "phase", None) else None,
        "current_location": getattr(getattr(history_entry, "location", None), "value", None),
        "current_player": player_name,
        "available_actions": actions_list,
        "correct_actions": llm_analysis.get("actions", []),
        "full_prompt": full_prompt,
        "llm_cot": llm_cot,
        "llm_response": llm_response,
        "difficulty": "easy",  # LLM-analyzed scenarios are considered clear
        "category": history_entry.action_taken.type.name.lower(),
        "llm_analysis_reason": llm_analysis.get("reason", "")
    }


def main():
    """Extract benchmark examples using LLM analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract benchmark examples using LLM analysis")
    parser.add_argument("game_state_file", help="Path to game state file (e.g., game_state_20.json)")
    parser.add_argument("--max_examples", type=int, default=10, help="Maximum number of examples to extract")
    parser.add_argument("--openai_api_key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    args = parser.parse_args()
    
    # Get OpenAI API key
    openai_api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OpenAI API key required. Set OPENAI_API_KEY env var or use --openai_api_key")
        return
    
    # Load game state
    if not os.path.exists(args.game_state_file):
        print(f"Error: Game state file not found: {args.game_state_file}")
        return
    
    print(f"Loading game state from {args.game_state_file}...")
    try:
        history, players, game_config = load_game_state(args.game_state_file)
    except Exception as e:
        print(f"Error loading game state: {e}")
        return
    
    print(f"Found {len(history)} history entries")
    
    examples = []
    candidates: List[Dict[str, Any]] = []
    
    # Build randomized order of candidate indices (no repeats within one run)
    candidate_indices = list(range(len(history)))
    random.shuffle(candidate_indices)
    
    # Analyze each history entry in random order
    for idx_pos, i in enumerate(candidate_indices, start=1):
        history_entry = history[i]
        print(f"Analyzing action {idx_pos}/{len(history)} (history index {i})...")
        
        # Skip system actions
        if history_entry.action_taken.player_name == "System":
            continue
        
        # Skip discussion phase and entries with no available actions
        if getattr(history_entry, "phase", None) and history_entry.phase.name == "DISCUSS":
            continue
        if not (getattr(history_entry, "actions_agent_could_take", None) or []):
            continue
        
        # Analyze with LLM
        history_slice = history[:i+1]
        llm_analysis = analyze_action_with_llm(history_entry, history_slice, players, game_config, openai_api_key)

        # Enforce strict post-filters even if LLM declared suitable
        if llm_analysis and isinstance(llm_analysis.get("actions"), list):
            # VALIDATION: filter LLM-proposed actions by actual available actions at this turn
            def _norm(s: str) -> str:
                return (s or "").strip().lstrip("* ").strip().lower()
            allowed = [ _norm(a) for a in (getattr(history_entry, "actions_agent_could_take", None) or []) ]
            filtered_actions = []
            for a in llm_analysis["actions"]:
                if not isinstance(a, dict):
                    continue
                act_text = _norm(a.get("action", ""))
                if act_text in allowed:
                    filtered_actions.append(a)
            llm_analysis["actions"] = filtered_actions

            scores = [a.get("score", 0.0) for a in llm_analysis["actions"] if isinstance(a, dict)]
            if scores:
                has_high = any(s >= 0.9 for s in scores)
                has_low = any(s <= 0.2 for s in scores)
                spread_ok = (max(scores) - min(scores)) >= 0.6
            else:
                has_high = has_low = spread_ok = False

            # Prefer action variety; if all actions are of the same coarse type, require strong spread
            actions_lower = [str(a.get("action", "")).lower() for a in llm_analysis["actions"] if isinstance(a, dict)]
            variety_keywords = ["report", "task", "kill", "vote"]
            has_variety = any(k in " ".join(actions_lower) for k in variety_keywords) or spread_ok

            # Baseline heuristic margin: chosen must beat baseline best by ≥ 0.15
            taken_str = stringify_taken_action(history_entry) or ""
            baseline_scores = [
                baseline_score_action(astr.get("action", ""), history_entry, history_slice, players, game_config, history_entry.action_taken.player_name)
                for astr in llm_analysis["actions"] if isinstance(astr, dict)
            ]
            baseline_best = max(baseline_scores) if baseline_scores else 0.0
            chosen_baseline = baseline_score_action(taken_str, history_entry, history_slice, players, game_config, history_entry.action_taken.player_name)
            margin_ok = (chosen_baseline - baseline_best) >= 0.15

            # CoT quality judge
            current_player_cot = next((p for p in players if p.name == history_entry.action_taken.player_name), players[0])
            conversation, _ = build_conversation_for_player(current_player_cot, history_slice, players, game_config)
            full_prompt = conversation[-1]["content"] if conversation and conversation[-1]["role"] == "user" else ""
            cot_quality = analyze_cot_quality_with_llm(
                history_entry,
                history_entry.action_taken.player_name,
                full_prompt,
                getattr(history_entry, "llm_cot", ""),
                getattr(history_entry, "llm_response", ""),
                openai_api_key,
            ) or {}

            reasoning_score = float(cot_quality.get("reasoning_score", 0.0) or 0.0)
            novelty_score = float(cot_quality.get("novelty_score", 0.0) or 0.0)
            evidence_count = int(cot_quality.get("evidence_count", 0) or 0)
            hallucinations = cot_quality.get("hallucination_flags", []) or []

            cot_ok = (
                reasoning_score >= 0.85 and
                novelty_score >= 0.7 and
                evidence_count >= 2 and
                len(hallucinations) == 0
            )

            # Compute a continuous utility score to rank all candidates (even rejected)
            spread = (max(scores) - min(scores)) if scores else 0.0
            # Normalize components to [0,1]
            spread_component = min(max(spread / 1.0, 0.0), 1.0)  # typical spread <= 1.0
            margin_component = min(max((chosen_baseline - baseline_best) / 0.5, 0.0), 1.0)  # cap margin at 0.5
            variety_component = 1.0 if has_variety else 0.0
            cot_component = min(max((reasoning_score - 0.5) / 0.5, 0.0), 1.0) * 0.7 + min(max((novelty_score - 0.5) / 0.5, 0.0), 1.0) * 0.3

            utility_score = 0.35 * cot_component + 0.25 * spread_component + 0.2 * margin_component + 0.2 * variety_component

            # Record candidate regardless of acceptance
            candidates.append({
                "id": f"{os.path.basename(args.game_state_file)}:{i}",
                "history_index": i,
                "current_player": history_entry.action_taken.player_name,
                "phase": getattr(history_entry, "phase", None).name if getattr(history_entry, "phase", None) else None,
                "available_actions": [a.get("action", "") for a in llm_analysis["actions"] if isinstance(a, dict)],
                "llm_scored_actions": llm_analysis.get("actions", []),
                "has_high": has_high,
                "has_low": has_low,
                "spread": spread,
                "baseline_best": baseline_best,
                "chosen_baseline": chosen_baseline,
                "margin": chosen_baseline - baseline_best,
                "cot_quality": {
                    "reasoning_score": reasoning_score,
                    "novelty_score": novelty_score,
                    "evidence_count": evidence_count,
                    "hallucinations": hallucinations,
                },
                "utility_score": round(utility_score, 4),
                "accepted": False,
            })

            if has_high and has_low and spread_ok and has_variety and margin_ok and cot_ok and len(actions_lower) >= 3:
                scenario = extract_scenario_data(history_entry, history_slice, players, game_config, args.game_state_file, i, llm_analysis)
                if scenario:
                    # Attach CoT judge summary for auditing
                    scenario["cot_quality"] = {
                        "reasoning_score": reasoning_score,
                        "novelty_score": novelty_score,
                        "evidence_count": evidence_count,
                        "summary": cot_quality.get("summary", ""),
                    }
                    examples.append(scenario)
                    # Mark last candidate as accepted
                    candidates[-1]["accepted"] = True
                    print(f"  ✓ Added extraordinary example: {scenario['description']}")
                    if len(examples) >= args.max_examples:
                        print(f"Reached maximum of {args.max_examples} examples")
                        break
            else:
                print("  ✗ Skipped (failed strict extraordinary filters)")
        else:
            print(f"  ✗ Skipped action {i}")
            # If we have no LLM action scoring, still record a minimal candidate with low utility
            candidates.append({
                "id": f"{os.path.basename(args.game_state_file)}:{i}",
                "history_index": i,
                "current_player": history_entry.action_taken.player_name,
                "phase": getattr(history_entry, "phase", None).name if getattr(history_entry, "phase", None) else None,
                "available_actions": getattr(history_entry, "actions_agent_could_take", None) or [],
                "llm_scored_actions": [],
                "has_high": False,
                "has_low": False,
                "spread": 0.0,
                "baseline_best": 0.0,
                "chosen_baseline": 0.0,
                "margin": 0.0,
                "cot_quality": {},
                "utility_score": 0.0,
                "accepted": False,
            })
    
    # Always write a ranked candidate list for manual selection
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(args.game_state_file))[0]
    candidates_out = os.path.join(os.path.dirname(__file__), '..', 'data', 'benchmarks', f'{base_name}_benchmark_candidates_{timestamp}.json')
    candidates_sorted = sorted(candidates, key=lambda c: c.get("utility_score", 0.0), reverse=True)
    with open(candidates_out, 'w') as f:
        json.dump({
            "source_file": args.game_state_file,
            "generated_at": timestamp,
            "candidates": candidates_sorted,
        }, f, indent=2)
    print(f"Wrote ranked candidates to {candidates_out}")

    if not examples:
        print("No suitable examples passed strict filters; use candidates file to pick thresholds or percentiles.")
        return
    
    # Create output filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(args.game_state_file))[0]
    output_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'benchmarks', f'{base_name}_benchmark_{timestamp}.json')
    
    # Create the benchmark dataset
    benchmark_data = {
        "benchmark_examples": examples,
        "metadata": {
            "total_examples": len(examples),
            "source_file": args.game_state_file,
            "extraction_timestamp": timestamp,
            "categories": {
                "report": len([e for e in examples if e["category"] == "report"]),
                "task": len([e for e in examples if e["category"] == "task"]),
                "move": len([e for e in examples if e["category"] == "move"]),
                "kill": len([e for e in examples if e["category"] == "kill"]),
                "vote": len([e for e in examples if e["category"] == "vote"])
            },
            "difficulty_levels": {
                "easy": len([e for e in examples if e["difficulty"] == "easy"]),
                "medium": 0,
                "hard": 0
            },
            "created_date": time.strftime("%Y-%m-%d"),
            "description": f"LLM-analyzed benchmark dataset extracted from {os.path.basename(args.game_state_file)}"
        }
    }
    
    # Write to file
    with open(output_file, 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    
    print(f"\nExtracted {len(examples)} benchmark examples to {output_file}")
    print("Categories:", benchmark_data["metadata"]["categories"])


if __name__ == "__main__":
    main()
