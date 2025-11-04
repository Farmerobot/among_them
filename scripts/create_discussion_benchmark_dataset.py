#!/usr/bin/env python3
"""
Create a discussion benchmark dataset by sampling random discussion episodes
that end in an ejection across existing game_state JSON files.

This script uses OpenAI to summarize the likely reason for ejection,
reading the API key from the OPENAI_API_KEY environment variable,
or via the --openai_api_key flag (same pattern as other scripts).

How to run:
  - Ensure OPENAI_API_KEY is set in your environment (e.g., via your .env) or pass --openai_api_key.
  - Examples:
      OPENAI_API_KEY=sk-... \
      python scripts/create_discussion_benchmark_dataset.py --max_examples 5 --seed 42
    or
      python scripts/create_discussion_benchmark_dataset.py --max_examples 5 --openai_api_key sk-... 

Output format (written to data/benchmarks/discussion_benchmark_<ts>.json):
{
  "benchmark_dataset": [
    {
      "id": "game_state_10.json:142",  # start history idx of discussion episode
      "source_file": ".../data/game_state_10.json",
      "start_history_index": 142,
      "ejected_player": "Alice",
      "ejected_player_role": "CREWMATE|IMPOSTOR",
      "players_alive": ["..."],
      "discussion_message_count": 7,
      "original_messages": [
        {"speaker": "Alice", "message": "..."},
        {"speaker": "Bob",   "message": "..."}
      ],
      "reason_summary": "one-line OpenAI analysis"
    }
  ],
  "metadata": {
    "generated_at": "YYYYMMDD_HHMMSS",
    "max_examples": 5,
    "source_game_files": ["..."],
    "notes": "Sampled randomly from discussion episodes that end with ejection"
  }
}
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from among_them.game_jsonencoder import game_object_hook  # type: ignore
from among_them.models.history import History  # type: ignore
from among_them.models.phase import GamePhase  # type: ignore
from among_them.config import OPENAI_API_KEY  # type: ignore
from among_them.utils.phase_utils import count_votes  # type: ignore


def load_state(file_path: str) -> Tuple[List[History], List[Any], Any]:
    with open(file_path, 'r') as f:
        data = json.load(f, object_hook=game_object_hook)
    if not isinstance(data, list) or len(data) < 2:
        raise ValueError(f"Invalid game state format: {file_path}")
    if len(data) == 3:
        return data[0], data[1], data[2]
    return data[0], data[1], None


def find_discussion_episodes(history: List[History]) -> List[Tuple[int, int, str]]:
    """Return list of (start_idx, vote_result_idx, ejected_player).

    We detect a discussion episode when a DISCUSS phase begins and later a VOTING
    system message states "<name> was voted out.". The vote_result_idx points to
    that system message entry.
    """
    episodes: List[Tuple[int, int, str]] = []
    i = 0
    while i < len(history):
        if history[i].phase == GamePhase.DISCUSS:
            # Discussion region starts here. Find following voting result message
            start_idx = i
            vote_result_idx = -1
            ejected_player = ""
            j = i + 1
            while j < len(history):
                if (history[j].phase == GamePhase.VOTING and
                    history[j].action_taken.player_name == "System" and
                    history[j].action_taken.target_message.endswith("was voted out.")):
                    vote_result_idx = j
                    # Parse ejected player from message prefix
                    msg = history[j].action_taken.target_message
                    # Message format: "<name> was voted out."
                    ejected_player = msg[: msg.index(" was voted out.")]
                    break
                # If a new DISCUSS starts before a vote result, stop this episode
                if j > start_idx and history[j].phase == GamePhase.DISCUSS:
                    break
                j += 1
            if vote_result_idx != -1 and ejected_player and ejected_player != "nobody":
                episodes.append((start_idx, vote_result_idx, ejected_player))
                i = vote_result_idx + 1
                continue
        i += 1
    return episodes


def role_of_player(players: List[Any], player_name: str) -> Optional[str]:
    p = next((p for p in players if getattr(p, 'name', None) == player_name), None)
    if not p:
        return None
    role = getattr(getattr(p, 'role', None), 'name', None)
    return role


def collect_original_messages(history: List[History], start_idx: int, end_idx_inclusive: int) -> List[Dict[str, str]]:
    """Collect all player discussion messages from the start of the discussion phase
    up to and including the index before voting result.

    We anchor at the most recent DISCUSS system message that announces the phase start,
    then gather all DISCUSS entries (excluding System) until vote result.
    """
    # Find most recent "It is discussion phase now" system message at/above start_idx
    anchor = start_idx
    for k in range(start_idx, -1, -1):
        h = history[k]
        if (
            h.phase == GamePhase.DISCUSS and
            getattr(h.action_taken, 'player_name', '') == 'System' and
            isinstance(getattr(h.action_taken, 'target_message', None), str) and
            h.action_taken.target_message.startswith("It is discussion phase now")
        ):
            anchor = k
            break

    messages: List[Dict[str, str]] = []
    for k in range(anchor + 1, min(end_idx_inclusive + 1, len(history))):
        h = history[k]
        if (
            h.phase == GamePhase.DISCUSS and
            getattr(h.action_taken, 'player_name', None) and
            getattr(h.action_taken, 'player_name') != 'System' and
            getattr(h.action_taken, 'target_message', None)
        ):
            messages.append({
                "speaker": h.action_taken.player_name,
                "message": h.action_taken.target_message,
            })
    return messages


def analyze_reason_with_openai(messages: List[Dict[str, str]], ejected_player: str, openai_api_key: Optional[str]) -> Optional[str]:
    if not openai_api_key:
        return None
    try:
        import openai  # type: ignore
        client = openai.OpenAI(api_key=openai_api_key)
        convo = "\n".join([f"{m['speaker']}: {m['message']}" for m in messages])
        prompt = f"""
You are analyzing a discussion in a social deduction game. The following dialogue occurred during the discussion phase (each line is Speaker: message). Briefly explain in one sentence the most likely reason why the player was voted out.

Ejected player: {ejected_player}

Discussion transcript:
{convo}

Return JSON only: {{"reason": "one sentence"}}
"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        return data.get("reason")
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Create discussion benchmark dataset by sampling episodes ending in ejection")
    parser.add_argument("--max_examples", type=int, default=5, help="Maximum number of discussion episodes to include")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--openai_api_key", type=str, default=None, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    data_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    pattern = os.path.join(data_dir, 'game_state_*.json')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No game_state files found via pattern: {pattern}")
        return

    # Randomize file order
    random.shuffle(files)

    examples: List[Dict[str, Any]] = []
    chosen_ids: set[str] = set()
    scanned_files: List[str] = []

    # OpenAI key: CLI flag or config default (config loads .env automatically)
    openai_api_key = args.openai_api_key or OPENAI_API_KEY
    if not openai_api_key:
        print("Error: OpenAI API key required. Set OPENAI_API_KEY env var or use --openai_api_key")
        return

    for fp in files:
        if len(examples) >= args.max_examples:
            break
        try:
            history, players, _ = load_state(fp)
        except Exception as e:
            print(f"Skipping {os.path.basename(fp)}: {e}")
            continue
        scanned_files.append(fp)

        episodes = find_discussion_episodes(history)
        if not episodes:
            continue

        # Shuffle episodes within file to avoid always picking first/last
        random.shuffle(episodes)

        for (start_idx, vote_idx, ejected_player) in episodes:
            if len(examples) >= args.max_examples:
                break

            # Build id and ensure uniqueness
            ex_id = f"{os.path.basename(fp)}:{start_idx}"
            if ex_id in chosen_ids:
                continue

            role = role_of_player(players, ejected_player) or "UNKNOWN"

            # Collect original discussion messages (from DISCUSS start to vote result)
            msgs = collect_original_messages(history, start_idx, vote_idx)

            # Collect voting breakdown by scanning backwards from just before the vote result
            # count_votes scans backwards from the end, so include everything up to but not including the "was voted out" message
            slice_for_votes = history[: vote_idx]
            vote_counts, votes = count_votes(slice_for_votes)

            reason = analyze_reason_with_openai(msgs, ejected_player, openai_api_key)

            example = {
                "id": ex_id,
                "source_file": fp,
                "start_history_index": start_idx,
                "ejected_player": ejected_player,
                "ejected_player_role": role,
                "players_alive": history[start_idx].alive_player_names if hasattr(history[start_idx], 'alive_player_names') else [],
                "discussion_message_count": len(msgs),
                "original_messages": msgs,
                "voting": {
                    "votes": votes,              # {voter: votee}
                    "counts": vote_counts,       # {votee: count}
                },
            }
            if reason:
                example["reason_summary"] = reason

            examples.append(example)
            chosen_ids.add(ex_id)
            print(f"✓ Added discussion episode {len(examples)}/{args.max_examples}: {ex_id} (ejected: {ejected_player}, role: {role})")

    ts = time.strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'benchmarks')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'discussion_benchmark_{ts}.json')

    out = {
        "benchmark_dataset": examples,
        "metadata": {
            "generated_at": ts,
            "max_examples": args.max_examples,
            "source_game_files": scanned_files,
            "notes": "Sampled randomly from discussion episodes that end with ejection",
        },
    }

    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\n✓ Wrote {len(examples)} discussion episodes to {out_path}")


if __name__ == "__main__":
    main()


