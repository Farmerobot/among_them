#!/usr/bin/env python3
"""
Run discussion phase benchmarks by replaying discussion episodes.

This script:
1. Loads discussion benchmark episodes from the dataset
2. For each episode, loads the original game state up to discussion start
3. Assigns the model under test to the player who was originally ejected
4. Assigns other models to other players (via --other_players_model if specified)
5. Replays the discussion and voting phases using existing game loop
6. Records who was actually ejected and compares to original

Usage:
  python scripts/run_discussion_benchmark.py \
    --benchmark data/benchmarks/discussion_benchmark_dataset.json \
    --other_players_model deepseek-r1:14b \
    --max_episodes 5
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
import time
import tiktoken
from typing import Any, Dict, List, Optional, Tuple

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from among_them.game_jsonencoder import game_object_hook, GameJSONEncoder  # type: ignore
from among_them.game_engine import GameEngine  # type: ignore
from among_them.models.history import History  # type: ignore
from among_them.models.player import Player  # type: ignore
from among_them.models.phase import GamePhase  # type: ignore
from among_them.models.action_type import ActionType  # type: ignore
from among_them.utils.llm_utils import invoke_llm, parse_llm_response_to_action  # type: ignore
from among_them.utils.phase_utils import count_votes, determine_ejection_result  # type: ignore
from among_them.config import MODEL_NAME, STATE_FILE  # type: ignore


def load_benchmark(path: str) -> Dict[str, Any]:
    """Load discussion benchmark dataset."""
    with open(path, 'r') as f:
        return json.load(f)


def slice_history_at_discussion_start(source_file: str, target_idx: int) -> Tuple[List[History], List[Player], Any]:
    """Load game state and slice history up to (just before) discussion start index.
    
    Returns: (history_slice, players, game_config)
    """
    with open(source_file, 'r') as f:
        data = json.load(f, object_hook=game_object_hook)
    
    history, players, game_config = (data[0], data[1], data[2]) if len(data) >= 3 else (data[0], data[1], None)
    
    # Slice history up to target index (inclusive)
    sliced_history = history[: target_idx + 1]
    
    # Make deep copies
    sliced_players = [copy.deepcopy(p) for p in players]
    
    return sliced_history, sliced_players, game_config


def find_target_discuss_indices(
    source_file: str,
    discuss_start_idx: int,
    target_player: str,
    max_count: int = 2,
) -> Tuple[int, List[int]]:
    """Find the discussion segment start anchor and the first max_count message indices
    by target_player within that segment.

    Strategy:
    - Move forward from discuss_start_idx to find the system discussion-start message if needed
    - Define the end of the discussion segment as the first VOTING outcome/TASKS entry
    - Collect the target player's SPEAK message indices strictly within this segment
    """
    with open(source_file, 'r') as f:
        data = json.load(f, object_hook=game_object_hook)
    history: List[History] = data[0]

    if discuss_start_idx < 0 or discuss_start_idx >= len(history):
        return discuss_start_idx, []

    # 1) Find the actual discussion start anchor
    # The dataset start_idx might point to the END of discussion (before voting), so we need to look BACKWARD
    start_anchor = discuss_start_idx
    found_anchor = False
    
    # First, try to find the system "It is discussion phase now" message by looking backward
    for i in range(discuss_start_idx, max(-1, discuss_start_idx - 50), -1):
        if i < 0:
            break
        h = history[i]
        if (
            h.phase == GamePhase.DISCUSS and
            getattr(h.action_taken, 'player_name', '') == 'System' and
            isinstance(getattr(h.action_taken, 'target_message', None), str) and
            h.action_taken.target_message.startswith("It is discussion phase now")
        ):
            start_anchor = i
            found_anchor = True
            break
    
    # If not found backward, look forward from start_idx
    if not found_anchor:
        for i in range(discuss_start_idx, len(history)):
            h = history[i]
            if (
                h.phase == GamePhase.DISCUSS and
                getattr(h.action_taken, 'player_name', '') == 'System' and
                isinstance(getattr(h.action_taken, 'target_message', None), str) and
                h.action_taken.target_message.startswith("It is discussion phase now")
            ):
                start_anchor = i
                found_anchor = True
                break
    
    # If still not found, find the first DISCUSS entry before or at start_idx
    if not found_anchor:
        for i in range(discuss_start_idx, max(-1, discuss_start_idx - 50), -1):
            if i < 0:
                break
            if history[i].phase == GamePhase.DISCUSS:
                start_anchor = i
                found_anchor = True
                break

    # 2) Find the end boundary of this discussion segment
    end_exclusive = len(history)
    for j in range(start_anchor + 1, len(history)):
        hh = history[j]
        # TASKS marks the next phase definitively
        if hh.phase == GamePhase.TASKS:
            end_exclusive = j
            break
        # Voting outcome announced by system
        if (
            getattr(hh.action_taken, 'player_name', '') == 'System' and
            hh.phase == GamePhase.VOTING and
            isinstance(getattr(hh.action_taken, 'target_message', None), str) and
            ("was voted out" in hh.action_taken.target_message or "was ejected" in hh.action_taken.target_message)
        ):
            end_exclusive = j
            break

    # 3) Collect target player's SPEAK messages within (start_anchor, end_exclusive)
    # Stop when we hit TASKS phase (that's the boundary)
    indices: List[int] = []
    for k in range(start_anchor + 1, end_exclusive):
        hk = history[k]
        # Stop if we hit TASKS phase
        if hk.phase == GamePhase.TASKS:
            break
        # Only look at DISCUSS phase entries
        if hk.phase != GamePhase.DISCUSS:
            continue
        # Check if this is the target player
        player_name = getattr(hk.action_taken, 'player_name', None)
        if player_name != target_player:
            continue
        # Check if this is a SPEAK action (discussion message)
        action_type = getattr(hk.action_taken, 'type', None)
        if action_type != ActionType.SPEAK:
            continue
        # Must have a message
        has_msg = bool(getattr(hk.action_taken, 'target_message', None)) or bool(getattr(hk, 'llm_response', None))
        if not has_msg:
            continue
        indices.append(k)
        if len(indices) >= max_count:
            break

    return start_anchor, indices


def slice_history_before_index(
    source_file: str,
    cut_idx: int,
    min_start_idx: int = 0
) -> Tuple[List[History], List[Player], Any]:
    """Return (history_slice, players, game_config) sliced from min_start_idx up to but excluding cut_idx.
    
    This ensures we don't include earlier voting results that would confuse is_voting_complete().
    """
    with open(source_file, 'r') as f:
        data = json.load(f, object_hook=game_object_hook)
    history, players, game_config = (data[0], data[1], data[2]) if len(data) >= 3 else (data[0], data[1], None)
    # Slice from min_start_idx (usually discussion start) to cut_idx (exclusive)
    sliced_history = history[min_start_idx:cut_idx]
    sliced_players = [copy.deepcopy(p) for p in players]
    return sliced_history, sliced_players, game_config


def collect_discussion_messages(history: List[History]) -> List[Dict[str, str]]:
    """Collect all player discussion messages from history."""
    messages = []
    for h in history:
        if (
            h.phase == GamePhase.DISCUSS and
            getattr(h.action_taken, 'player_name', None) and
            h.action_taken.player_name != 'System' and
            getattr(h.action_taken, 'target_message', None)
        ):
            messages.append({
                "speaker": h.action_taken.player_name,
                "message": h.action_taken.target_message,
            })
    return messages


def compute_voting_outcome(history: List[History]) -> Tuple[str, Dict[str, str]]:
    """Compute voting outcome robustly by finding all votes in the voting phase.

    Finds the last vote, then slices history up to and including it.
    count_votes will scan backward from there and stop at the first non-vote.
    """
    # Find the last vote index by scanning backward
    last_vote_idx = -1
    for i in range(len(history) - 1, -1, -1):
        h = history[i]
        if getattr(h.action_taken, 'type', None) == ActionType.VOTE:
            last_vote_idx = i
            break
    
    # If no votes found, return nobody
    if last_vote_idx == -1:
        return "nobody", {}
    
    # Slice history up to and including the last vote
    # count_votes scans backward from the end and stops at first non-vote
    hist_for_count = history[: last_vote_idx + 1]
    vote_counts, votes = count_votes(hist_for_count)
    
    ejected = "nobody"
    if vote_counts:
        ejected, _ = determine_ejection_result(vote_counts)
    
    return ejected, votes


def call_llm_with_retry(
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    *,
    allowed_actions: Optional[List[str]] = None,
    single_line_only: bool = False,
    max_output_chars: Optional[int] = None,
    max_retries: int = 10,
) -> Tuple[str, Optional[str]]:
    """Invoke LLM with simple retry/backoff for OpenRouter/429 rate limits."""
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return invoke_llm(
                system_prompt=system_prompt,
                prompt=user_prompt,
                model_name=model_name,
                allowed_actions=allowed_actions,
                single_line_only=single_line_only,
                max_output_chars=max_output_chars,
            )
        except Exception as e:
            last_err = e
            msg = str(e)
            # Retry on common rate limit messages
            if ("429" in msg) or ("rate limit" in msg.lower()) or ("temporarily rate-limited" in msg.lower()):
                # Linear backoff capped at 10s
                time.sleep(min(1 + attempt, 10))
                continue
            raise
    raise last_err if last_err else Exception("invoke_llm failed after retries")


def is_voting_complete(engine: GameEngine) -> bool:
    """Check if voting phase has completed by looking for TASKS phase or 'was voted out' message."""
    if not engine.history:
        return False
    
    # Check last few entries
    for entry in reversed(engine.history[-5:]):
        # Check for TASKS phase (voting done, tasks resumed)
        if entry.phase == GamePhase.TASKS:
            return True
        # Check for "was voted out" or "was ejected" system message
        if (hasattr(entry, 'action_taken') and 
            hasattr(entry.action_taken, 'player_name') and
            entry.action_taken.player_name == "System" and
            hasattr(entry.action_taken, 'target_message') and
            entry.action_taken.target_message and
            ("was voted out" in entry.action_taken.target_message or 
             "was ejected" in entry.action_taken.target_message or
             "Voting concluded" in entry.action_taken.target_message)):
            return True
    
    return False


def run_discussion_to_voting(
    engine: GameEngine,
    target_player: str,
    model_for_target: str,
    model_for_others: Optional[str],
    log_file: Optional[str] = None,
    disable_prevotes: bool = False,
    force_first_player: bool = False,
    initial_history_len: int = 0,
    respect_initial_discuss_counter: bool = True,
) -> Tuple[str, List[Dict[str, str]], Dict[str, str]]:
    """Run game using existing game loop pattern until voting completes.
    
    Uses the same pattern as manual_llm_game.py but with model overrides.
    Logs full LLM responses (COT + response) to log_file.
    
    Returns: (ejected_player, messages, votes)
    """
    log_fp = open(log_file, 'a') if log_file else None
    voting_completed_once = False  # Track if we've already completed one voting phase
    discuss_counter_reset_done = False  # Ensure we only reset DISCUSS counter once at start
    pre_votes_disabled_logged = False  # Log the disable notice only once
    
    try:
        first_turn_done = False
        while True:
            # Check if voting already completed (before getting turn context)
            # This prevents continuing if a new discussion starts after voting
            if is_voting_complete(engine):
                if not voting_completed_once:
                    voting_completed_once = True
                    if log_fp:
                        log_fp.write("[INFO] First voting phase completed. Stopping now.\n")
                        log_fp.flush()
                        os.fsync(log_fp.fileno())
                    print("  Stopping: Voting completed")
                    break
                else:
                    # Already completed voting once, stop immediately
                    if log_fp:
                        log_fp.write("[INFO] Voting already completed once - stopping to prevent new discussion phases.\n")
                        log_fp.flush()
                        os.fsync(log_fp.fileno())
                    print("  Stopping: Voting already completed once")
                    break
            
            # Get turn context (same as manual_llm_game.py)
            if force_first_player and not first_turn_done:
                turn_context_history, actions_player_can_take, system_prompt, user_prompt, pre_discussion_vote_prompts = engine.get_turn_context(player_name=target_player)
            else:
                turn_context_history, actions_player_can_take, system_prompt, user_prompt, pre_discussion_vote_prompts = engine.get_turn_context()

            # Safety: only at the very first turn, if stored counter on the last history item is <= 0,
            # reset once to allow a discussion; do NOT reset on later turns when it reaches 0 naturally
            if turn_context_history and turn_context_history.phase == GamePhase.DISCUSS:
                try:
                    history_counter = getattr(engine.history[-1], 'actions_until_phase_ends', 1)
                    if history_counter <= 0 and not discuss_counter_reset_done and not first_turn_done:
                        alive_count = len(getattr(turn_context_history, 'alive_player_names', []) or [])
                        per_player = getattr(engine.game_config, 'num_discuss_phase_actions_per_player', 2)
                        # get_turn_context pre-decrements by 1, so set to (N*alive - 1)
                        reset_value = max(0, (max(1, per_player) * max(1, alive_count)) - 1)
                        if log_fp:
                            log_fp.write(f"[WARN] actions_until_phase_ends <= 0 at DISCUSS start; resetting to {reset_value} to start discussion.\n")
                            log_fp.flush()
                            os.fsync(log_fp.fileno())
                        turn_context_history.actions_until_phase_ends = reset_value
                        discuss_counter_reset_done = True
                except Exception:
                    pass

            # Clamp negative discuss counter to 0 to allow immediate transition to voting
            if turn_context_history and turn_context_history.phase == GamePhase.DISCUSS and getattr(turn_context_history, 'actions_until_phase_ends', 0) < 0:
                turn_context_history.actions_until_phase_ends = 0
            
            if not turn_context_history:
                if log_fp:
                    log_fp.write("[INFO] No more turn context - game ended\n")
                    log_fp.flush()
                    os.fsync(log_fp.fileno())
                break
            
            # Check if we're in TASKS phase (voting done) - stop immediately
            if turn_context_history.phase == GamePhase.TASKS:
                if log_fp:
                    log_fp.write("[INFO] TASKS phase detected - voting completed. Stopping now.\n")
                    log_fp.flush()
                    os.fsync(log_fp.fileno())
                print("  Stopping: Entered TASKS phase")
                break
            
            # Check if voting completed after getting context (but before processing turn)
            if is_voting_complete(engine):
                if not voting_completed_once:
                    voting_completed_once = True
                    if log_fp:
                        log_fp.write("[INFO] Voting completed - detected in history. Stopping now.\n")
                        log_fp.flush()
                        os.fsync(log_fp.fileno())
                    print("  Stopping: Voting completed")
                    break
                else:
                    # Already completed once, stop
                    if log_fp:
                        log_fp.write("[INFO] Voting already completed once - stopping.\n")
                        log_fp.flush()
                        os.fsync(log_fp.fileno())
                    print("  Stopping: Voting already completed once")
                    break
            
            # Handle pre-discussion votes if needed
            pre_discussion_votes = None
            if pre_discussion_vote_prompts and not disable_prevotes:
                if log_fp:
                    log_fp.write("[INFO] Collecting pre-discussion votes\n")
                    log_fp.flush()
                    os.fsync(log_fp.fileno())
                pre_discussion_votes = {}
                for vote_prompt in pre_discussion_vote_prompts:
                    player = vote_prompt["player"]
                    system_prompt_pd = vote_prompt["system_prompt"]
                    user_prompt_pd = vote_prompt["user_prompt"]
                    actions_pd = vote_prompt["actions"]
                    
                    # Override model for this player
                    original_model = player.llm_model_name
                    model_to_use = model_for_target if player.name == target_player else (model_for_others or model_for_target)
                    player.llm_model_name = model_to_use
                    
                    try:
                        allowed_actions_pd = [a.set_stories().command_perspective for a in actions_pd]
                        llm_response, cot = call_llm_with_retry(
                            system_prompt_pd,
                            user_prompt_pd,
                            model_to_use,
                            allowed_actions=allowed_actions_pd,
                            single_line_only=True,
                        )
                        action_idx, _ = parse_llm_response_to_action(actions_pd, llm_response, player.name)
                        action_taken = actions_pd[action_idx]
                        pre_discussion_votes[player.name] = {
                            "voted_player": action_taken.target_player_name,
                            "chain_of_thought": cot,
                        }
                        # Log full response (COT + response)
                        if log_fp:
                            log_fp.write(f"[PRE-VOTE] {player.name}:\n")
                            if cot:
                                log_fp.write(f"COT: {cot}\n")
                            log_fp.write(f"Response: {llm_response}\n")
                            log_fp.write(f"Voted for: {action_taken.target_player_name}\n\n")
                            log_fp.flush()
                            os.fsync(log_fp.fileno())
                    except Exception as e:
                        error_msg = f"[ERROR] Pre-vote error for {player.name}: {e}"
                        print(f"  {error_msg}")
                        if log_fp:
                            log_fp.write(f"{error_msg}\n\n")
                            log_fp.flush()
                            os.fsync(log_fp.fileno())
                    finally:
                        # Restore original model
                        player.llm_model_name = original_model
            elif pre_discussion_vote_prompts and disable_prevotes:
                # Explicitly ignore any pre-discussion vote prompts and log once
                if log_fp and not pre_votes_disabled_logged:
                    log_fp.write("[INFO] Pre-discussion votes disabled for this run\n")
                    log_fp.flush()
                    os.fsync(log_fp.fileno())
                    pre_votes_disabled_logged = True
                pre_discussion_vote_prompts = []
            
            # Get current player
            current_player_name = turn_context_history.action_taken.player_name
            
            # Verify player is actually alive (safety check)
            if current_player_name not in engine.history[-1].alive_player_names:
                if log_fp:
                    log_fp.write(f"[WARNING] Player {current_player_name} is dead but was selected for turn. Skipping.\n")
                    log_fp.flush()
                    os.fsync(log_fp.fileno())
                print(f"  Warning: Player {current_player_name} is dead, skipping turn")
                # Continue to next iteration to get a valid player
                continue
            
            current_player = next((p for p in engine.players if p.name == current_player_name), None)
            if current_player is None:
                print(f"  Warning: Player {current_player_name} not found")
                break
            
            # Override model for this player
            original_model = current_player.llm_model_name
            model_to_use = model_for_target if current_player_name == target_player else (model_for_others or model_for_target)
            current_player.llm_model_name = model_to_use
            
            try:
                print(f"  Turn: {current_player_name} ({turn_context_history.phase.name})")
                
                # Get LLM response (same pattern as manual_llm_game.py)
                allowed_actions_main = [a.set_stories().command_perspective for a in actions_player_can_take if a.type != ActionType.SPEAK]
                single_line_only = len(allowed_actions_main) > 0
                allowed_actions_for_call = allowed_actions_main if single_line_only else None
                
                llm_response, cot = call_llm_with_retry(
                    system_prompt,
                    user_prompt,
                    model_to_use,
                    allowed_actions=allowed_actions_for_call,
                    single_line_only=single_line_only,
                    max_output_chars=None if single_line_only else 1500,
                )
                
                # Parse response
                action_idx, response_text = parse_llm_response_to_action(
                    actions_player_can_take, llm_response, current_player_name
                )
                action_taken = actions_player_can_take[action_idx]
                
                # For SPEAK actions, take only the first line to avoid hallucinated content
                if action_taken.type == ActionType.SPEAK:
                    first_line = response_text.split('\n')[0].strip()
                    if first_line:
                        response_text = first_line
                
                # Log full LLM response (COT + response) to file
                if log_fp:
                    log_fp.write(f"{current_player_name} ({turn_context_history.phase.name}):\n")
                    if cot:
                        log_fp.write(f"COT: {cot}\n")
                    log_fp.write(f"Response: {llm_response}\n")
                    log_fp.write(f"Parsed action: {response_text}\n\n")
                    log_fp.flush()
                    os.fsync(log_fp.fileno())
                
            except Exception as e:
                error_msg = f"[ERROR] {current_player_name} - LLM error: {e}"
                print(f"  {error_msg}")
                if log_fp:
                    log_fp.write(f"{error_msg}\n\n")
                    log_fp.flush()
                    os.fsync(log_fp.fileno())
                # Fallback: use first action
                action_taken = actions_player_can_take[0] if actions_player_can_take else None
                response_text = ""
                cot = ""
                if not action_taken:
                    break
            finally:
                # Restore original model
                current_player.llm_model_name = original_model
            
            # Calculate token usage (same as manual_llm_game.py)
            encoding = tiktoken.encoding_for_model("gpt-4o")
            input_tokens = len(encoding.encode(system_prompt + user_prompt))
            output_tokens = len(encoding.encode(response_text + (cot or "")))
            token_usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
            
            # Step the environment (same as manual_llm_game.py)
            end_game, end_reason = engine.step(
                turn_context_history, 
                action_taken, 
                response_text, 
                cot, 
                token_usage, 
                pre_discussion_votes
            )
            first_turn_done = True
            
            # Check if voting completed after step
            if is_voting_complete(engine):
                if not voting_completed_once:
                    voting_completed_once = True
                    if log_fp:
                        log_fp.write("[INFO] Voting completed after step. Stopping now.\n")
                        log_fp.flush()
                        os.fsync(log_fp.fileno())
                    print("  Stopping: Voting completed")
                    break
                else:
                    # Already completed voting once, stop immediately
                    if log_fp:
                        log_fp.write("[INFO] Voting already completed once after step - stopping.\n")
                        log_fp.flush()
                        os.fsync(log_fp.fileno())
                    print("  Stopping: Voting already completed once")
                    break
            
            if end_game:
                if log_fp:
                    log_fp.write(f"[INFO] Game ended: {end_reason}\n")
                    log_fp.flush()
                    os.fsync(log_fp.fileno())
                break
    
    finally:
        if log_fp:
            log_fp.close()
    
    # Extract voting result robustly from final history
    ejected_player, votes = compute_voting_outcome(engine.history)
    
    # Collect discussion messages that happened during this simulation only
    history_after_start = engine.history[initial_history_len:]
    messages = collect_discussion_messages(history_after_start)
    
    return ejected_player, messages, votes


def run_discussion_episode(
    episode: Dict[str, Any],
    model_for_target: str,
    model_for_others: Optional[str],
    temp_dir: str,
    log_file: Optional[str] = None,
    disable_prevotes: bool = False,
    start_variant: Optional[str] = None,  # None | 'first_msg' | 'second_msg'
) -> Dict[str, Any]:
    """Run a single discussion episode and return results."""
    print(f"Running episode: {episode['id']}")
    
    # Write episode header to log file
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Episode: {episode['id']}\n")
            f.write(f"Target Player: {episode['ejected_player']} ({episode['ejected_player_role']})\n")
            f.write(f"{'='*80}\n\n")
            f.flush()
            os.fsync(f.fileno())
    
    # Create temp copy of game state
    source_file = episode['source_file']
    start_idx = episode['start_history_index']
    target_player = episode['ejected_player']
    original_role = episode['ejected_player_role']
    
    # Determine slice based on variant
    if start_variant in ("first_msg", "second_msg"):
        anchor_idx, msg_indices = find_target_discuss_indices(source_file, start_idx, target_player, max_count=2)
        
        # Debug: print what we found
        print(f"  Debug: start_idx={start_idx}, anchor_idx={anchor_idx}, msg_indices={msg_indices}")
        
        # Additional debug: check what's in the history if no messages found
        if not msg_indices:
            print(f"  WARNING: No message indices found for {target_player} from start_idx={start_idx}")
            # Try to debug what's in the history
            with open(source_file, 'r') as f:
                data = json.load(f, object_hook=game_object_hook)
                history = data[0]
                print(f"  History length: {len(history)}")
                print(f"  Scanning from {anchor_idx} to find {target_player} messages:")
                for i in range(anchor_idx, min(anchor_idx + 50, len(history))):
                    h = history[i]
                    player = getattr(h.action_taken, 'player_name', '?')
                    action_type = getattr(h.action_taken, 'type', None)
                    phase = h.phase.name if hasattr(h.phase, 'name') else str(h.phase)
                    if i < anchor_idx + 20:  # Print first 20 for debugging
                        print(f"    [{i}] phase={phase} player={player} action_type={action_type}")
        
        # Determine which message index to use
        pick_idx = msg_indices[0] if start_variant == "first_msg" and msg_indices else None
        if start_variant == "second_msg" and len(msg_indices) >= 2:
            pick_idx = msg_indices[1]
        if pick_idx is None:
            raise ValueError(f"Could not find {start_variant} index for {target_player} in {source_file} from {start_idx}")
        
        print(f"  Debug: pick_idx={pick_idx}")
        
        # Ensure pick_idx > anchor_idx, otherwise slicing will be empty
        if pick_idx <= anchor_idx:
            raise ValueError(
                f"Invalid indices: pick_idx ({pick_idx}) must be > anchor_idx ({anchor_idx}). "
                f"Message indices found: {msg_indices}. "
                f"Anchor/start mismatch likely means the dataset start index points after the actual discussion start."
            )
        
        # slice from discussion start up to just before the target player's message
        # This avoids including earlier voting results that would confuse is_voting_complete()
        history_slice, players, game_config = slice_history_before_index(source_file, pick_idx, min_start_idx=anchor_idx)
        
        # Ensure we have at least the discussion start message
        if len(history_slice) == 0:
            raise ValueError(f"Empty history slice: anchor_idx={anchor_idx}, pick_idx={pick_idx}")
        
        print(f"  Debug: sliced history has {len(history_slice)} entries")
    else:
        # Slice to actual discussion start anchor for full-discussion runs
        anchor_idx, _ = find_target_discuss_indices(source_file, start_idx, target_player, max_count=0)
        history_slice, players, game_config = slice_history_at_discussion_start(source_file, anchor_idx)
    
    # Write sliced state to game_state.json (default STATE_FILE - engine will auto-save here)
    state_file_path = STATE_FILE  # Use default game_state.json
    print(f"  Writing to game_state.json: {os.path.abspath(state_file_path)}")
    
    # Write sliced state to game_state.json
    with open(state_file_path, 'w') as f:
        json.dump((history_slice, players, game_config), f, cls=GameJSONEncoder, indent=2)
        f.flush()
        os.fsync(f.fileno())
    
    # Write file path to log
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f"[INFO] Game state file: {os.path.abspath(state_file_path)}\n")
            f.flush()
            os.fsync(f.fileno())
    
    # Initialize engine from game_state.json (same as manual_llm_game.py)
    # Engine will auto-save to this file during step()
    if game_config is None:
        from among_them.game_config import GameConfig
        game_config = GameConfig()
    
    engine = GameEngine(game_config=game_config, file_path=state_file_path)
    engine.load_state()  # Loads history, players, and game_config from the file

    # Calibrate remaining discussion turns so total does not exceed 2 rounds (per_player) for alive players,
    # accounting for messages already spoken in this discussion segment and our start point.
    try:
        if engine.history and engine.history[-1].phase == GamePhase.DISCUSS:
            # Find the anchor (system message that starts the discussion) within the sliced state
            anchor_idx = 0
            for i in range(len(engine.history) - 1, -1, -1):
                h = engine.history[i]
                if (
                    h.phase == GamePhase.DISCUSS and
                    getattr(h.action_taken, 'player_name', '') == 'System' and
                    isinstance(getattr(h.action_taken, 'target_message', None), str) and
                    h.action_taken.target_message.startswith("It is discussion phase now")
                ):
                    anchor_idx = i
                    break
            # Count messages already spoken in this discussion segment (after anchor)
            messages_so_far = 0
            for j in range(anchor_idx + 1, len(engine.history)):
                hj = engine.history[j]
                if hj.phase != GamePhase.DISCUSS:
                    break
                if getattr(hj.action_taken, 'player_name', '') != 'System' and getattr(hj.action_taken, 'type', None) == ActionType.SPEAK:
                    messages_so_far += 1

            alive_player_count = len(getattr(engine.history[-1], 'alive_player_names', []) or [])
            per_player = getattr(engine.game_config, 'num_discuss_phase_actions_per_player', 2)
            total_allowed = max(1, per_player) * max(1, alive_player_count)

            remaining = max(0, total_allowed - messages_so_far)
            # get_turn_context typically pre-decrements by 1, so set to remaining-1 (but not negative)
            desired_counter = max(0, remaining - 1)

            # Apply only if current counter is missing or larger than desired (to cap the discussion length)
            current_counter = getattr(engine.history[-1], 'actions_until_phase_ends', None)
            if current_counter is None or current_counter > desired_counter:
                engine.history[-1].actions_until_phase_ends = desired_counter
                if log_file:
                    with open(log_file, 'a') as f:
                        f.write(
                            f"[INFO] Calibrated discussion counter: messages_so_far={messages_so_far}, "
                            f"total_allowed={total_allowed}, remaining={remaining}, set actions_until_phase_ends={desired_counter}\n"
                        )
                        f.flush()
                        os.fsync(f.fileno())
    except Exception:
        pass

    print(f"  ✓ Loaded game state: {len(engine.history)} history entries, {len(engine.players)} players")
    
    # Run simulation using existing game loop pattern
    ejected_player, messages, votes = run_discussion_to_voting(
        engine,
        target_player,
        model_for_target,
        model_for_others,
        log_file=log_file,
        disable_prevotes=disable_prevotes,
        force_first_player=(start_variant in ("first_msg", "second_msg")),
        initial_history_len=len(history_slice),
        respect_initial_discuss_counter=True
    )
    
    # After voting completes, copy game_state.json to temp file for inspection
    temp_state_file = os.path.join(temp_dir, f"temp_game_{episode['id'].replace(':', '_')}.json")
    import shutil
    shutil.copy2(state_file_path, temp_state_file)
    print(f"  ✓ Copied final game state to: {os.path.abspath(temp_state_file)}")
    
    # Write temp file path to log
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f"[INFO] Final game state copied to: {os.path.abspath(temp_state_file)}\n\n")
            f.flush()
            os.fsync(f.fileno())
    
    # Determine outcome
    survived = (ejected_player != target_player)
    outcome = "survived" if survived else "ejected"
    
    result = {
        "episode_id": episode['id'],
        "start_variant": start_variant or "discussion_start",
        "target_player": target_player,
        "target_role": original_role,
        "outcome": outcome,
        "ejected_player": ejected_player,
        "survived": survived,
        "messages": messages,
        "votes": votes,
    }
    
    print(f"  Result: {target_player} {outcome}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Run discussion phase benchmarks")
    parser.add_argument("--benchmark", default=os.path.join(os.path.dirname(__file__), '..', 'data', 'benchmarks', 'discussion_benchmark_dataset.json'))
    parser.add_argument("--other_players_model", type=str, default=None, help="Model for non-target players (default: same as target)")
    parser.add_argument("--max_episodes", type=int, default=None, help="Maximum episodes to run")
    parser.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), '..', 'generated', 'benchmarks'))
    parser.add_argument("--disable_prevotes", action="store_true", help="Disable pre-discussion votes; only vote at end of discussion")
    args = parser.parse_args()
    
    # Load benchmark
    data = load_benchmark(args.benchmark)
    episodes: List[Dict[str, Any]] = data.get('benchmark_dataset', [])
    if not episodes:
        print('No episodes found in benchmark file.')
        return
    
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
    
    # Determine models
    model_for_target = MODEL_NAME  # Use env config
    model_for_others = args.other_players_model
    
    print(f"Model for target player: {model_for_target}")
    print(f"Model for other players: {model_for_others or model_for_target}")
    print(f"Episodes to run: {len(episodes)}")
    print()
    
    # Generate report
    ts = time.strftime('%Y%m%d_%H%M%S')
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f'discussion_{ts}_results.txt')
    
    # Create log file for full LLM responses (COT + response)
    log_file = os.path.join(args.outdir, f'discussion_{ts}_messages.txt')
    
    # Clear/create log file
    with open(log_file, 'w') as f:
        f.write(f"Discussion Benchmark - Full LLM Responses\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model (target): {model_for_target}\n")
        f.write(f"Model (others): {model_for_others or model_for_target}\n")
        f.write(f"Episodes: {len(episodes)}\n\n")
        f.flush()
        os.fsync(f.fileno())
    
    print(f"Logging full LLM responses (COT + response) to: {log_file}\n")
    
    # Create temp directory (NOT auto-deleted - kept for testing purposes)
    # To enable auto-cleanup later, uncomment the cleanup code at the end and use:
    # with tempfile.TemporaryDirectory() as temp_dir:
    temp_dir = tempfile.mkdtemp(prefix="discussion_benchmark_")
    temp_dir_abs = os.path.abspath(temp_dir)
    print(f"Temp directory (will be kept): {temp_dir_abs}\n")
    
    # Write temp directory to log file
    with open(log_file, 'a') as f:
        f.write(f"Temp directory: {temp_dir_abs}\n")
        f.write(f"(All temp game state files will be saved here)\n\n")
        f.flush()
        os.fsync(f.fileno())
    
    results = []
    
    for i, episode in enumerate(episodes, 1):
        print(f"[{i}/{len(episodes)}] ", end="")
        try:
            # Variant 1: start at target player's first message
            res_first = run_discussion_episode(
                episode, 
                model_for_target, 
                model_for_others, 
                temp_dir, 
                log_file, 
                disable_prevotes=args.disable_prevotes,
                start_variant="first_msg"
            )
            results.append(res_first)
        except Exception as e:
            print(f"  Error (first_msg): {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "episode_id": episode.get('id', 'unknown'),
                "start_variant": "first_msg",
                "error": str(e),
            })

        try:
            # Variant 2: start at target player's second message
            res_second = run_discussion_episode(
                episode, 
                model_for_target, 
                model_for_others, 
                temp_dir, 
                log_file, 
                disable_prevotes=args.disable_prevotes,
                start_variant="second_msg"
            )
            results.append(res_second)
        except Exception as e:
            print(f"  Error (second_msg): {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "episode_id": episode.get('id', 'unknown'),
                "start_variant": "second_msg",
                "error": str(e),
            })
    
    # Uncomment below to enable auto-cleanup of temp directory:
    # import shutil
    # if os.path.exists(temp_dir):
    #     shutil.rmtree(temp_dir)
    #     print(f"Cleaned up temp directory: {temp_dir}")
    
    survived_count = sum(1 for r in results if r.get('survived'))
    total_count = len([r for r in results if 'survived' in r])
    survival_rate = survived_count / max(1, total_count)
    
    header = [
        f"Discussion Benchmark Results",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model (target): {model_for_target}",
        f"Model (others): {model_for_others or model_for_target}",
        f"Episodes: {len(episodes)}",
        f"",
        f"Survival Rate: {survival_rate:.1%} ({survived_count}/{total_count})",
        f"",
        "=" * 80,
        "",
    ]
    
    detailed_lines = []
    for r in results:
        detailed_lines.append(f"Episode: {r.get('episode_id', 'unknown')}")
        detailed_lines.append(f"  Start Variant: {r.get('start_variant', 'unknown')}")
        detailed_lines.append(f"  Target: {r.get('target_player', '?')} ({r.get('target_role', '?')})")
        detailed_lines.append(f"  Outcome: {r.get('outcome', 'error')}")
        if 'ejected_player' in r:
            detailed_lines.append(f"  Ejected: {r['ejected_player']}")
        if 'votes' in r and r['votes']:
            detailed_lines.append(f"  Votes:")
            for voter, voted_for in r['votes'].items():
                detailed_lines.append(f"    {voter} -> {voted_for}")
            # Also show vote counts if available
            vote_counts = {}
            for voted_for in r['votes'].values():
                vote_counts[voted_for] = vote_counts.get(voted_for, 0) + 1
            if vote_counts:
                detailed_lines.append(f"  Vote Counts:")
                for player, count in sorted(vote_counts.items(), key=lambda x: x[1], reverse=True):
                    detailed_lines.append(f"    {player}: {count}")
        elif 'votes' in r:
            detailed_lines.append(f"  Votes: None (no votes cast)")
        if 'messages' in r:
            detailed_lines.append(f"  Messages: {len(r['messages'])}")
            for msg in r['messages']:
                detailed_lines.append(f"    {msg['speaker']}: {msg['message'][:100]}")
        if 'error' in r:
            detailed_lines.append(f"  Error: {r['error']}")
        detailed_lines.append("")
    
    with open(out_path, 'w') as f:
        f.write("\n".join(header + detailed_lines))
    
    print(f"\n✓ Wrote results to {out_path}")
    print(f"✓ Wrote full LLM responses (COT + response) to {log_file}")


if __name__ == "__main__":
    main()
