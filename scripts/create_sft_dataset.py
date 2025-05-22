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
import statistics
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from among_them.game_engine import GameEngine
from among_them.models.phase import GamePhase
from among_them.utils.history_utils import get_action_history_str
from among_them.models.action_type import ActionType
from among_them.llm_prompts import UNIVERSAL_SYSTEM_PROMPT
from among_them.utils.phase_utils import count_votes

UNIVERSAL_SYSTEM_PROMPT = "You are an AI assistant observing a game. Based on the game state and player actions, predict the AI's next thought process and action. Provide your response in a clear, step-by-step manner if explaining reasoning, followed by the specific action in the required format."
CHARS_PER_TOKEN = 4 # Fallback if not using actual tokenizer

def format_num(n):
    if isinstance(n, float):
        return f"{n:,.2f}" # For mean, std dev
    try:
        return f"{int(n):,}" # For integers
    except (ValueError, TypeError):
        return str(n) # Fallback for non-numeric

def process_game_file(file_path: str, tokenizer_for_counting=None) -> list:
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
        model_output = f"{event.llm_cot}\n{response}" # without first think tag https://huggingface.co/deepseek-ai/DeepSeek-R1/commit/8a58a132790c9935686eb97f042afa8013451c9f
        
        item_data = {
            "json_file_name": os.path.basename(file_path),
            "player_name": player_name,
            "player_role": player.role.value, # type: ignore
            "votes_before": json.dumps(votes_before),
            "votes_after": json.dumps(votes_after),
            "prompt": history_str,
            "model_cot_and_cleaned_output": model_output,
        }
        
        if tokenizer_for_counting:
            instruction_text_for_count = UNIVERSAL_SYSTEM_PROMPT + "\n" + history_str
            item_data['instruction_token_count_actual'] = len(tokenizer_for_counting.encode(instruction_text_for_count))
            item_data['output_token_count_actual'] = len(tokenizer_for_counting.encode(model_output))
        
        results.append(item_data)
    
    return results


def write_jsonl(data, output_file):
    """Write data to a JSONL file in the format required for training."""
    with open(output_file, 'w') as f:
        for item in data:
            conversation = [
                {"role": "user", "content": UNIVERSAL_SYSTEM_PROMPT + "\n" + item["prompt"]},
                {"role": "assistant", "content": item["model_cot_and_cleaned_output"]}
            ]
            f.write(json.dumps({"messages": conversation}) + "\n")


def write_alpaca_json(data, output_file, use_tokenizer_for_stats: bool):
    """Writes data to Alpaca JSON and calculates token statistics using pre-tokenized counts if available."""
    output_data = []
    total_instruction_tokens = 0
    total_output_tokens = 0
    max_instruction_tokens = 0
    max_output_tokens = 0

    for item in data:
        instruction_text = UNIVERSAL_SYSTEM_PROMPT + "\n" + item["prompt"]
        output_text = item["model_cot_and_cleaned_output"]
        
        instruction_tokens_count = 0
        output_tokens_count = 0

        if use_tokenizer_for_stats:
            instruction_tokens_count = item.get('instruction_token_count_actual')
            output_tokens_count = item.get('output_token_count_actual')
            # Fallback if actual counts are somehow not present (should not happen with new logic)
            if instruction_tokens_count is None:
                print(f"Warning: Missing actual instruction token count for an item from {item.get('json_file_name', 'Unknown Game')}. Approximating.")
                instruction_tokens_count = len(instruction_text) // CHARS_PER_TOKEN
            if output_tokens_count is None:
                print(f"Warning: Missing actual output token count for an item from {item.get('json_file_name', 'Unknown Game')}. Approximating.")
                output_tokens_count = len(output_text) // CHARS_PER_TOKEN
        else:
            instruction_tokens_count = len(instruction_text) // CHARS_PER_TOKEN
            output_tokens_count = len(output_text) // CHARS_PER_TOKEN

        total_instruction_tokens += instruction_tokens_count
        total_output_tokens += output_tokens_count
        max_instruction_tokens = max(max_instruction_tokens, instruction_tokens_count)
        max_output_tokens = max(max_output_tokens, output_tokens_count)

        output_data.append({
            "instruction": instruction_text,
            "output": output_text
        })
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    return {
        "max_instruction_tokens": max_instruction_tokens,
        "max_output_tokens": max_output_tokens,
        "total_instruction_tokens": total_instruction_tokens,
        "total_output_tokens": total_output_tokens,
        "examples_count": len(data)
    }


def calculate_and_plot_token_metrics(all_results, output_path: Path, use_tokenizer_for_calc: bool, tokenizer_model_name_for_report: str):
    """Calculate token metrics using pre-tokenized counts if available, plot distributions, and print summary stats. Returns a dictionary of stats."""
    stats_data = {}
    input_list = []
    output_list = []

    # Tokenizer is not loaded here anymore; we use pre-calculated counts or approximation

    for r_item in all_results:
        input_tokens = 0
        output_tokens = 0

        if use_tokenizer_for_calc:
            input_tokens = r_item.get('instruction_token_count_actual')
            output_tokens = r_item.get('output_token_count_actual')
            # Fallback if actual counts are somehow not present
            if input_tokens is None:
                print(f"Warning: Missing actual instruction token count for distribution summary for an item from {r_item.get('json_file_name', 'Unknown Game')}. Approximating.")
                instruction_text_for_approx = UNIVERSAL_SYSTEM_PROMPT + "\n" + r_item["prompt"]
                input_tokens = len(instruction_text_for_approx) // CHARS_PER_TOKEN
            if output_tokens is None:
                print(f"Warning: Missing actual output token count for distribution summary for an item from {r_item.get('json_file_name', 'Unknown Game')}. Approximating.")
                output_text_for_approx = r_item["model_cot_and_cleaned_output"]
                output_tokens = len(output_text_for_approx) // CHARS_PER_TOKEN
        else:
            instruction_text_for_approx = UNIVERSAL_SYSTEM_PROMPT + "\n" + r_item["prompt"]
            output_text_for_approx = r_item["model_cot_and_cleaned_output"]
            input_tokens = len(instruction_text_for_approx) // CHARS_PER_TOKEN
            output_tokens = len(output_text_for_approx) // CHARS_PER_TOKEN
        
        input_list.append(input_tokens)
        output_list.append(output_tokens)

    # Calculate total tokens (input + output)
    total_list = [i + o for i, o in zip(input_list, output_list)]
    
    if not input_list or not output_list or not total_list:
        print("No token data available to calculate metrics or plot.")
        return stats_data
    
    print(f"\nToken Count Distribution Summary:")
    
    for name, arr in [("Input", input_list), ("Output", output_list), ("Total", total_list)]:
        count = len(arr)
        min_val = min(arr)
        median_val = statistics.median(arr)
        mean_val = statistics.mean(arr)
        max_val = max(arr)
        std_val = statistics.stdev(arr)
        stats_data[name.lower()] = {
            "count": count,
            "min": min_val,
            "median": median_val,
            "mean": mean_val,
            "max": max_val,
            "std": std_val
        }
        print(f"  {name} tokens: count={format_num(count)}, min={format_num(min_val)}, median={format_num(median_val)}, mean={format_num(mean_val)}, max={format_num(max_val)}, std={format_num(std_val)}")
    
    fig, axs = plt.subplots(1, 3, figsize=(18,5))
    axs[0].hist(input_list, bins=50, color="C0", alpha=0.7)
    axs[0].set_title("Input tokens distribution")
    axs[0].set_xlabel("Tokens")
    axs[0].set_ylabel("Count")
    axs[1].hist(output_list, bins=50, color="C1", alpha=0.7)
    axs[1].set_title("Output tokens distribution")
    axs[1].set_xlabel("Tokens")
    axs[2].hist(total_list, bins=50, color="C2", alpha=0.7)
    axs[2].set_title("Total tokens distribution")
    axs[2].set_xlabel("Tokens")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"\nSaved token usage plot to {output_path}")
    return stats_data


def main():
    # --- Configuration ---
    data_dir = Path("data")
    output_csv_file = data_dir / "sft_dataset.csv"
    sft_data_dir = data_dir / "sft"
    alpaca_data_dir = data_dir / "alpaca"
    generated_dir = Path("generated") # Centralized generated folder
    use_actual_tokenizer = True  # Set to False to use character approximation globally
    tokenizer_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" # For actual token counting
    
    # Create output directories if they don't exist
    sft_data_dir.mkdir(exist_ok=True, parents=True)
    alpaca_data_dir.mkdir(exist_ok=True, parents=True)
    generated_dir.mkdir(exist_ok=True, parents=True)
    # --- End Configuration ---
    
    all_results = []
    tokenizer_instance_for_counting = None

    if use_actual_tokenizer:
        try:
            tokenizer_instance_for_counting = AutoTokenizer.from_pretrained(tokenizer_model_name)
            print(f"Using tokenizer: {tokenizer_model_name} for token counts.")
        except Exception as e:
            print(f"Critical: Could not load tokenizer {tokenizer_model_name}. Error: {e}. Token counts will be approximated.")
            use_actual_tokenizer = False # Fallback to approximation globally if tokenizer fails
    else:
        print("Token counts will be approximated (4 chars = 1 token).")

    # Process all JSON files in the data directory
    for file_path in data_dir.glob("*.json"):
        if file_path.name in ["game_state.json", "to_be_continued_14b.json", "test_game.json"] or file_path.name.endswith("1.5b.json"):
            continue
        try:
            results = process_game_file(str(file_path), tokenizer_instance_for_counting)
            print(f"Processed {len(results)} actions from {file_path}")
            all_results.extend(results)
        except Exception as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)
            traceback.print_exc()
    
    if not all_results:
        print("No data was processed. Check the input directory and file format.")
        return

    # Write results to CSV
    with open(output_csv_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["json_file_name", "player_name", "player_role", "votes_before", "votes_after", "prompt", "model_cot_and_cleaned_output", "input_tokens", "output_tokens", "instruction_token_count_actual", "output_token_count_actual"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL, extrasaction='ignore') # Ignore extra fields not in fieldnames for robustness
        for result in all_results:
            row_to_write = {}
            for field_name in fieldnames:
                value = result.get(field_name) # Use .get() to avoid KeyError if a field is missing
                if isinstance(value, str):
                    row_to_write[field_name] = value.replace('\n', '\\n')
                else:
                    row_to_write[field_name] = value
            writer.writerow(row_to_write)
        
    # Split data into train/valid/test sets
    random.seed(42)
    random.shuffle(all_results)
    data_size = len(all_results)
    train_size = int(data_size * 0.8)
    valid_size = int(data_size * 0.1)
    
    train_data = all_results[:train_size]
    valid_data = all_results[train_size:train_size+valid_size]
    test_data = all_results[train_size+valid_size:]
    
    # For Alpaca format, combine validation and test into a single eval set
    eval_data = all_results[train_size:]
    
    # Write JSONL files
    train_file = sft_data_dir / "train.jsonl"
    valid_file = sft_data_dir / "valid.jsonl"
    test_file = sft_data_dir / "test.jsonl"
    write_jsonl(train_data, train_file)
    write_jsonl(valid_data, valid_file)
    write_jsonl(test_data, test_file)
        
    # Write Alpaca JSON files
    alpaca_train_file = alpaca_data_dir / "among_them_train.json"
    alpaca_eval_file = alpaca_data_dir / "among_them_eval.json"
    train_stats = write_alpaca_json(train_data, alpaca_train_file, use_actual_tokenizer)
    eval_stats = write_alpaca_json(eval_data, alpaca_eval_file, use_actual_tokenizer)
        
    # Create a single dataset_info.json file with both datasets
    dataset_info = {
        "among_them_train": {
            "file_name": "among_them_train.json",
            "columns": {
                "prompt": "instruction",
                "response": "output"
            },
            "formatting": "alpaca"
        },
        "among_them_eval": {
            "file_name": "among_them_eval.json",
            "columns": {
                "prompt": "instruction",
                "response": "output"
            },
            "formatting": "alpaca"
        }
    }
        
    info_file = os.path.join(alpaca_data_dir, "dataset_info.json")
    with open(info_file, 'w') as f:
        json.dump(dataset_info, f, indent=2)
        
    # Calculate overall stats
    max_instruction_tokens = max(train_stats["max_instruction_tokens"], eval_stats["max_instruction_tokens"])
    max_output_tokens = max(train_stats["max_output_tokens"], eval_stats["max_output_tokens"])
        
    total_instruction_tokens = train_stats["total_instruction_tokens"] + eval_stats["total_instruction_tokens"]
    total_output_tokens = train_stats["total_output_tokens"] + eval_stats["total_output_tokens"]
        
    print(f"Successfully wrote {format_num(len(all_results))} rows to {output_csv_file}")
    print(f"  {format_num(len(train_data))} examples to {train_file}")
    print(f"  {format_num(len(valid_data))} examples to {valid_file}")
    print(f"  {format_num(len(test_data))} examples to {test_file}")
    print(f"\nAlpaca format datasets:")
    print(f"  {format_num(train_stats['examples_count'])} examples to {alpaca_train_file}")
    print(f"  {format_num(eval_stats['examples_count'])} examples to {alpaca_eval_file}")
        
    token_counting_method_info_oneline = f"Tokenizer: {tokenizer_model_name}" if use_actual_tokenizer else "Token Count Method: Estimated (4 chars = 1 token)"
    print(f"\n--- Alpaca Dataset Token Statistics ({token_counting_method_info_oneline}) ---")

    print(f"\nMaximum Token Lengths Per Example:")
    print(f"  Longest instruction: {format_num(max_instruction_tokens)} tokens")
    print(f"  Longest output: {format_num(max_output_tokens)} tokens")

    print(f"\nTotal Token Counts:")
    print(f"  Total instruction tokens: {format_num(total_instruction_tokens)} tokens")
    print(f"  Total output tokens: {format_num(total_output_tokens)} tokens")
    print(f"  Overall total (all instructions + all outputs): {format_num(total_instruction_tokens + total_output_tokens)} tokens")

    # Calculate and plot token metrics using the selected method
    # The tokenizer_model_name is passed for reporting purposes in the MD file, actual tokenization uses stored counts or approximation.
    token_distribution_stats = calculate_and_plot_token_metrics(
        all_results, 
        generated_dir / "token_usage_distribution.png", # Plot saved in generated
        use_actual_tokenizer, # This flag determines if we use stored actual counts or approximate
        tokenizer_model_name # For report generation
    )
    # Now print the max combined for console, using the calculated distribution stats
    if token_distribution_stats and 'total' in token_distribution_stats and 'max' in token_distribution_stats['total']:
        print(f"  Longest combined (instruction + output): {format_num(token_distribution_stats['total']['max'])} tokens (from distribution summary)")

    # Write stats to markdown file
    stats_md_path = generated_dir / "alpaca_dataset_stats.md"
        
    with open(stats_md_path, 'w', encoding='utf-8') as f:
        f.write("# Alpaca Dataset Statistics\n\n")
        if use_actual_tokenizer:
            f.write(f"Tokenizer used for token counts: `{tokenizer_model_name}`\n\n")
        else:
            f.write("Token counts are estimated (4 characters = 1 token).\n\n")
            
        total_examples_count = train_stats['examples_count'] + eval_stats['examples_count']
        f.write(f"- **Training examples:** {format_num(train_stats['examples_count'])}\n")
        f.write(f"- **Evaluation examples:** {format_num(eval_stats['examples_count'])}\n")
        f.write(f"- **Total examples:** {format_num(total_examples_count)}\n")
            
        f.write("\n## Maximum Token Lengths Per Example\n")
        f.write(f"- **Longest instruction:** {format_num(max_instruction_tokens)} tokens\n")
        f.write(f"- **Longest output:** {format_num(max_output_tokens)} tokens\n")
        if token_distribution_stats and 'total' in token_distribution_stats and 'max' in token_distribution_stats['total']:
             f.write(f"- **Longest combined (instruction + output):** {format_num(token_distribution_stats['total']['max'])} tokens\n")
            
        f.write("\n## Total Token Counts\n")
        f.write(f"- **Total instruction tokens:** {format_num(total_instruction_tokens)}\n")
        f.write(f"- **Total output tokens:** {format_num(total_output_tokens)}\n")
        f.write(f"- **Overall total (all instructions + all outputs):** {format_num(total_instruction_tokens + total_output_tokens)} tokens\n")

        if token_distribution_stats:
            f.write("\n## Token Count Distribution Summary\n")
            f.write("| Statistic | Input Tokens | Output Tokens | Total Tokens |\n")
            f.write("| :-------- | :----------- | :------------ | :----------- |\n")
            stat_names = ["count", "min", "median", "mean", "max", "std"]
            display_names = {"count": "Count", "min": "Min", "median": "Median", "mean": "Mean", "max": "Max", "std": "Std Dev"}

            for stat_key in stat_names:
                f.write(f"| **{display_names[stat_key]}** | {format_num(token_distribution_stats['input'][stat_key])} | {format_num(token_distribution_stats['output'][stat_key])} | {format_num(token_distribution_stats['total'][stat_key])} |\n")

    print(f"\nAlpaca dataset stats also written to {stats_md_path}")


if __name__ == "__main__":
    main()
