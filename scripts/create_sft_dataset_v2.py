#!/usr/bin/env python3
"""
Simple script that:
1. Loads each JSON file in the data folder
2. Extracts discussion messages
3. Gets the prompt using get_action_history_str
4. Outputs a CSV file with the required fields
"""

import os
import json
import csv
from pathlib import Path
import traceback
import sys
import random

from among_them.game_engine import GameEngine
from among_them.models.phase import GamePhase
from among_them.utils.history_utils import get_action_history_str
from among_them.models.action_type import ActionType
from among_them.llm_prompts import UNIVERSAL_SYSTEM_PROMPT
from among_them.utils.phase_utils import count_votes

def process_game_file(file_path: str) -> list:
    """Process a single game file and extract discussion messages"""
    
    # Use GameEngine to load the state
    engine = GameEngine()
    engine.file_path = file_path
    engine.load_state()
    
    results = []
    
    # Process each history item that is a discussion message
    for i, event in enumerate(engine.history):
        # if event.phase != GamePhase.DISCUSS or event.action_taken.type != ActionType.SPEAK:
        #     continue
        
        player_name = event.action_taken.player_name
        player = next((p for p in engine.players if p.name == player_name), None)
        if player is None:
            if player_name == "System":
                continue
            else:
                raise ValueError(f"Player {player_name} not found in game")
        
        history_until_now = engine.history[:i+1]
        history_str = get_action_history_str(
            history_until_now, 
            engine.players, 
            player, 
            engine.game_config
        )
        
        # copied from player.py
        if event.action_taken.type == ActionType.SPEAK:
            history_str += "\n\nIt is discussion phase now. Respond to others in the following xml format: <message>message</message>"
        else:
            actions_text = "<available_actions>\n" + "\n".join(f"<action>{action}</action>" for action in event.actions_agent_could_take) + "\n</available_actions>"
            history_str += f"\n\n{actions_text}\n"
            history_str += "\n\nChoose one action. Respond in the following xml format: <action>action</action>"

        # Get votes before (from the current message)
        votes_before = {}
        if event.votes_before_this_discussion_message:
            votes_before = {p.name: event.votes_before_this_discussion_message.get(p.name, {}).get("voted_player", None) for p in engine.players if p.name in event.votes_before_this_discussion_message}
        
        # Get votes after (from the next message)
        votes_after = {}
        j = 0
        for next_event in engine.history[i+1:]:
            if next_event.votes_before_this_discussion_message and event.phase == GamePhase.DISCUSS:
                votes_after = {p.name: next_event.votes_before_this_discussion_message.get(p.name, {}).get("voted_player", None) for p in engine.players if p.name in next_event.votes_before_this_discussion_message}
                break
            elif next_event.phase == GamePhase.VOTE_RESULTS and event.phase == GamePhase.DISCUSS:
                _, votes_after = count_votes(engine.history[:i+1+j])
                break
            j += 1
        
        # Combine chain of thought and response if both exist
        response = f"<message>{event.llm_response}</message>" if event.action_taken.type == ActionType.SPEAK else f"<action>{event.action_taken.text}</action>"
        model_output = f"<think>{event.llm_cot}</think>\n{response}"
        
        results.append({
            "json_file_name": os.path.basename(file_path),
            "player_name": player_name,
            "player_role": player.role.value, # type: ignore
            "votes_before": json.dumps(votes_before),
            "votes_after": json.dumps(votes_after),
            "prompt": history_str,
            "model_cot_and_cleaned_output": model_output
        })
    
    return results


def write_jsonl(data, output_file):
    """Write data to a JSONL file in the format required for training."""
    with open(output_file, 'w') as f:
        for item in data:
            conversation = [
                {"role": "system", "content": UNIVERSAL_SYSTEM_PROMPT},
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["model_cot_and_cleaned_output"]}
            ]
            f.write(json.dumps({"messages": conversation}) + "\n")


def main():
    data_dir = Path("data")
    output_file = Path("data/sft_dataset2.csv")
    
    # Create sft_data directory for jsonl files
    sft_data_dir = Path("data/sft")
    sft_data_dir.mkdir(exist_ok=True, parents=True)
    
    train_file = sft_data_dir / "train.jsonl"
    valid_file = sft_data_dir / "valid.jsonl"
    test_file = sft_data_dir / "test.jsonl"
    
    all_results = []
    
    # Process all JSON files in the data directory
    for file_path in data_dir.glob("*.json"):
        if file_path.name in ["game_state.json", "14b to be continued.json", "test_game.json"] or file_path.name.endswith("1.5b.json"):
            continue
        try:
            results = process_game_file(str(file_path))
            print(f"Processed {len(results)} actions from {file_path}")
            all_results.extend(results)
        except Exception as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)
            traceback.print_exc()
    
    # Write results to CSV
    if all_results:
        # Replace newlines in all string fields to avoid CSV formatting issues
        for result in all_results:
            for key, value in result.items():
                if isinstance(value, str):
                    # Replace newlines with a special token
                    result[key] = value.replace('\n', '\\n')
        
        with open(output_file, 'w', newline='') as f:
            fieldnames = ["json_file_name", "player_name", "player_role", "votes_before", "votes_after", "prompt", "model_cot_and_cleaned_output"]
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(all_results)
        
        # Split data into train/valid/test sets
        random.shuffle(all_results)
        data_size = len(all_results)
        train_size = int(data_size * 0.8)
        valid_size = int(data_size * 0.1)
        
        train_data = all_results[:train_size]
        valid_data = all_results[train_size:train_size+valid_size]
        test_data = all_results[train_size+valid_size:]
        
        # Write JSONL files
        write_jsonl(train_data, train_file)
        write_jsonl(valid_data, valid_file)
        write_jsonl(test_data, test_file)
        
        print(f"Successfully wrote {len(all_results)} rows to {output_file}")
        print(f"  {len(train_data)} examples to {train_file}")
        print(f"  {len(valid_data)} examples to {valid_file}")
        print(f"  {len(test_data)} examples to {test_file}")
    else:
        print("No data was processed. Check the input directory and file format.")


if __name__ == "__main__":
    main()
