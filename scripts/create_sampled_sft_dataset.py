#!/usr/bin/env python3
"""
Creates a sampled SFT dataset by filtering high-quality traces (score >= TRACE_QUALITY_THRESHOLD).
Requires trace_analyzer.py to be run first to generate generated/trace_analysis.txt.

Uses the trace evaluation on a scale from 0 (halucination) to n (outstanding performance).
It prints the sampled dataset to the new file and generates its summary.
TRACE_QUALITY_THRESHOLD defines what is the lower bound on evaluation for entering the sampled dataset.
"""

from pathlib import Path
import numpy as np
import json
from create_sft_dataset import write_alpaca_json, format_num
import random
import os

TRACE_QUALITY_THRESHOLD = 3

def main():
    # --- Configuration ---
    data_dir = Path("data")
    alpaca_source_dir = data_dir / "alpaca"  # Read from the main alpaca dataset
    generated_dir = Path("generated") # Centralized generated folder
    trace_analysis_file = generated_dir / "trace_analysis.txt"
    alpaca_data_dir = data_dir / "alpaca_sampled"
    
    # Ensure required directories exist
    data_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    alpaca_data_dir.mkdir(parents=True, exist_ok=True)

    # Load the full dataset (conversations format)
    alpaca_train_file = alpaca_source_dir / "among_them_train.json"
    alpaca_eval_file = alpaca_source_dir / "among_them_eval.json"
    
    with open(alpaca_train_file, 'r') as f:
        train_data = json.load(f)
    with open(alpaca_eval_file, 'r') as f:
        eval_data = json.load(f)
    
    # Combine all conversations
    all_conversations = train_data + eval_data

    # Load trace analysis scores
    quality_idx, scores = [], []
    with open(trace_analysis_file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip() == "":
                continue
            try:
                eval = json.loads(line)
                score = eval[0]
                scores.append(score)
                if score >= TRACE_QUALITY_THRESHOLD:
                    quality_idx.append(i)
            except json.JSONDecodeError:
                print(f"Failed to decode line {i}: {line}")

    quality_idx, scores = np.array(quality_idx), np.array(scores)

    # Filter conversations based on quality
    filtered_conversations = [all_conversations[i] for i in quality_idx if i < len(all_conversations)]
    
    print(f"Filtered {len(filtered_conversations)} high-quality conversations out of {len(all_conversations)}")

    # Calculate token counts from conversations (approximate: 4 chars = 1 token)
    CHARS_PER_TOKEN = 4
    instruction_tokens_list = []
    output_tokens_list = []
    
    for conv_item in filtered_conversations:
        conversations = conv_item["conversations"]
        input_tokens = sum(
            len(msg["content"]) // CHARS_PER_TOKEN 
            for msg in conversations if msg["role"] == "user"
        )
        output_tokens = sum(
            len(msg["content"]) // CHARS_PER_TOKEN 
            for msg in conversations if msg["role"] == "assistant"
        )
        instruction_tokens_list.append(input_tokens)
        output_tokens_list.append(output_tokens)
    
    instruction_tokens = np.array(instruction_tokens_list)
    output_tokens = np.array(output_tokens_list)
    examples_count = len(instruction_tokens)
    
    # Write stats to markdown file
    stats_md_path = generated_dir / "alpaca_sampled_dataset_stats.md"
        
    with open(stats_md_path, 'w', encoding='utf-8') as f:
        f.write("# Sampled Alpaca Dataset Statistics\n\n")
        f.write("Token counts are approximated (4 characters = 1 token).\n\n")
            
        f.write(f"- **Total examples:** {examples_count} (out of original {len(all_conversations)} - {round(examples_count / len(all_conversations) * 100, 1)}% preserved)\n")

        f.write("\n## Trace Evaluation Distribution\n")
        f.write("| Dataset | Eval 0 | Eval 1 | Eval 2 | Eval 3 |\n")
        f.write("| :-------- | :----------- | :------------ | :----------- | :----------- |\n")

        stats = {
            "Original": {
                i: np.count_nonzero(scores == i) for i in range(4)
            },
            "Sampled": {
                i: np.count_nonzero(scores[quality_idx] == i) for i in range(4)
            }
        }

        for stat_key in list(stats.keys()):
            f.write(f"| **{stat_key}** | {format_num(stats[stat_key][0])} | {format_num(stats[stat_key][1])} | {format_num(stats[stat_key][2])} | {format_num(stats[stat_key][3])} |\n")

            
        f.write("\n## Maximum Token Lengths Per Example\n")
        f.write(f"- **Longest instruction:** {format_num(instruction_tokens.max())} tokens\n")
        f.write(f"- **Longest output:** {format_num(output_tokens.max())} tokens\n")
        f.write(f"- **Longest combined (instruction + output):** {format_num((instruction_tokens + output_tokens).max())} tokens\n")
            
        f.write("\n## Total Token Counts\n")
        f.write(f"- **Total instruction tokens:** {format_num(instruction_tokens.sum())} tokens\n")
        f.write(f"- **Total output tokens:** {format_num(output_tokens.sum())} tokens\n")
        f.write(f"- **Overall total (all instructions + all outputs):** {format_num((instruction_tokens + output_tokens).sum())} tokens\n")

        f.write("\n## Token Count Distribution Summary\n")
        f.write("| Statistic | Input Tokens | Output Tokens | Total Tokens |\n")
        f.write("| :-------- | :----------- | :------------ | :----------- |\n")
        stat_names = ["count", "min", "median", "mean", "max", "std"]
        display_names = {"count": "Count", "min": "Min", "median": "Median", "mean": "Mean", "max": "Max", "std": "Std Dev"}

        token_distribution_stats = {
            "input": {
                "count": examples_count,
                "min": instruction_tokens.min(),
                "median": np.median(instruction_tokens),
                "mean": instruction_tokens.mean(),
                "max": instruction_tokens.max(),
                "std": instruction_tokens.std()
            },
            "output": {
                "count": examples_count,
                "min": output_tokens.min(),
                "median": np.median(output_tokens),
                "mean": output_tokens.mean(),
                "max": output_tokens.max(),
                "std": output_tokens.std()
            },
            "total": {
                "count": examples_count,
                "min": (instruction_tokens + output_tokens).min(),
                "median": np.median((instruction_tokens + output_tokens)),
                "mean": (instruction_tokens + output_tokens).mean(),
                "max": (instruction_tokens + output_tokens).max(),
                "std": (instruction_tokens + output_tokens).std()
            }
        }

        for stat_key in stat_names:
            f.write(f"| **{display_names[stat_key]}** | {format_num(token_distribution_stats['input'][stat_key])} | {format_num(token_distribution_stats['output'][stat_key])} | {format_num(token_distribution_stats['total'][stat_key])} |\n")

    print(f"\nAlpaca dataset stats also written to {stats_md_path}")

    # Split data into train/valid/test sets
    random.seed(42)
    random.shuffle(filtered_conversations)
    data_size = len(filtered_conversations)
    train_size = int(data_size * 0.8)
    valid_size = int(data_size * 0.1)
    
    train_data = filtered_conversations[:train_size]
    valid_data = filtered_conversations[train_size:train_size+valid_size]
    test_data = filtered_conversations[train_size+valid_size:]
    
    # For Alpaca format, combine validation and test into a single eval set
    eval_data = filtered_conversations[train_size:]
    
    # # Write JSONL files
    # train_file = sft_data_dir / "train_sampled.jsonl"
    # valid_file = sft_data_dir / "valid_sampled.jsonl"
    # test_file = sft_data_dir / "test_sampled.jsonl"
    # write_jsonl(train_data, train_file)
    # write_jsonl(valid_data, valid_file)
    # write_jsonl(test_data, test_file)
        
    # Write Alpaca JSON files
    alpaca_train_file = alpaca_data_dir / "among_them_train.json"
    alpaca_eval_file = alpaca_data_dir / "among_them_eval.json"
    train_stats = write_alpaca_json(train_data, alpaca_train_file, True)
    eval_stats = write_alpaca_json(eval_data, alpaca_eval_file, True)
        
    # Create a single dataset_info.json file with both datasets
    dataset_info = {
        "among_them_train": {
            "file_name": "among_them_train.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations"
            }
        },
        "among_them_eval": {
            "file_name": "among_them_eval.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations"
            }
        }
    }
        
    info_file = os.path.join(alpaca_data_dir, "dataset_info.json")
    with open(info_file, 'w') as f:
        json.dump(dataset_info, f, indent=2)


if __name__ == "__main__":
    main()
