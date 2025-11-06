#!/usr/bin/env python3
"""
Simple script that:
1. Loads each JSON file in the data folder
2. Extracts discussion messages
3. Gets the prompt using reconstruct_environment_prompt_from_history
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

# Used as fallback to calculate token counts if actual tokenizer is not available
CHARS_PER_TOKEN = 4

# Maximum tokens per conversation (input + output combined)
# Conversations exceeding this are split into multiple chunks while maintaining temporal order
# This prevents OOM during training while preserving all training data
MAX_CONVERSATION_TOKENS = 3000

# Strict enforcement: if True, discard user-assistant pairs that exceed MAX_CONVERSATION_TOKENS
# This creates a hard limit but loses data. If False (default), allows large pairs as single chunks
STRICT_TOKEN_LIMIT = False

def format_num(n):
    if isinstance(n, float):
        return f"{n:,.2f}" # For mean, std dev
    try:
        return f"{int(n):,}" # For integers
    except (ValueError, TypeError):
        return str(n) # Fallback for non-numeric

def process_game_file(file_path: str, tokenizer_for_counting=None) -> list:
    """Process a single game file and extract multi-turn conversations per player"""
    from among_them.utils.prompt_utils import get_initial_turn_prompt, get_incremental_observations
    
    # Use GameEngine to load the state
    engine = GameEngine()
    engine.file_path = file_path
    engine.load_state()
    
    results = []
    
    # Build conversations per player
    for player in engine.players:
        # Find all turns for this player
        player_turn_indices = []
        for i, event in enumerate(engine.history):
            if event.action_taken.player_name == player.name:
                player_turn_indices.append(i)
        
        if not player_turn_indices:
            continue  # Player never took a turn
        
        # Build conversation turns
        conversation = []
        total_input_tokens = 0
        total_output_tokens = 0
        
        for turn_idx, history_index in enumerate(player_turn_indices):
            event = engine.history[history_index]
            
            # Generate user prompt
            if turn_idx == 0:
                # First turn: include system context
                user_prompt = get_initial_turn_prompt(
                    player,
                    engine.history,
                    engine.players,
                    engine.game_config,
                    history_index
                )
            else:
                # Subsequent turns: only incremental observations
                last_turn_index = player_turn_indices[turn_idx - 1]
                user_prompt = get_incremental_observations(
                    player,
                    engine.history,
                    engine.players,
                    engine.game_config,
                    last_turn_index,
                    history_index
                )
            
            # Generate assistant response (model's reasoning + action)
            event.action_taken.set_stories()
            assistant_response = f"{event.llm_cot}{event.action_taken.command_perspective}"
            
            # Count tokens if tokenizer available
            if tokenizer_for_counting:
                input_tokens = len(tokenizer_for_counting.encode(user_prompt))
                output_tokens = len(tokenizer_for_counting.encode(assistant_response))
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
            
            # Add turn to conversation
            conversation.append({
                "role": "user",
                "content": user_prompt
            })
            conversation.append({
                "role": "assistant",
                "content": assistant_response
            })
        
        # Create result item for this player's conversation
        item_data = {
            "json_file_name": os.path.basename(file_path),
            "player_name": player.name,
            "player_role": player.role.value,
            "num_turns": len(player_turn_indices),
            "conversations": conversation,
        }
        
        if tokenizer_for_counting:
            item_data['total_input_tokens'] = total_input_tokens
            item_data['total_output_tokens'] = total_output_tokens
        
        results.append(item_data)
    
    return results

# Unused for now
# def write_jsonl(data, output_file):
#     """Write data to a JSONL file in the format required for training."""
#     with open(output_file, 'w') as f:
#         for item in data:
#             conversation = [
#                 {"role": "user", "content": item["prompt"]},
#                 {"role": "assistant", "content": item["model_cot_and_cleaned_output"]}
#             ]
#             f.write(json.dumps({"messages": conversation}) + "\n")


def split_long_conversations(data, max_tokens: int, use_tokenizer: bool, tokenizer=None, strict_limit=False):
    """Split conversations that exceed max_tokens into multiple shorter conversations.
    
    Each conversation is split by accumulating user-assistant pairs until adding
    the next pair would exceed the limit. Every subsequence maintains temporal order.
    
    Args:
        data: List of conversation items with 'conversations' field
        max_tokens: Maximum total tokens per conversation chunk
        use_tokenizer: Whether to use actual tokenizer or character approximation
        tokenizer: Tokenizer instance (if use_tokenizer is True)
        strict_limit: If True, discard pairs exceeding max_tokens. If False, allow them as single chunks
    
    Returns:
        List of conversation items (potentially more than input due to splitting)
    """
    result = []
    split_count = 0
    discarded_pair_count = 0
    discarded_conversation_count = 0
    
    for item in data:
        conversations = item["conversations"]
        
        # Ensure conversations alternates user/assistant
        if len(conversations) < 2:
            result.append(item)
            continue
        
        current_chunk = []
        current_tokens = 0
        chunk_input_tokens = 0
        chunk_output_tokens = 0
        conversation_has_valid_pairs = False
        
        # Process pairs of user-assistant messages
        for i in range(0, len(conversations), 2):
            if i + 1 >= len(conversations):
                # Odd number of messages - just add the last one
                if current_chunk:
                    current_chunk.append(conversations[i])
                break
            
            user_msg = conversations[i]
            assistant_msg = conversations[i + 1]
            
            # Calculate tokens for this pair
            if use_tokenizer and tokenizer:
                user_tokens = len(tokenizer.encode(user_msg["content"]))
                assistant_tokens = len(tokenizer.encode(assistant_msg["content"]))
                pair_tokens = user_tokens + assistant_tokens
            else:
                user_tokens = len(user_msg["content"]) // CHARS_PER_TOKEN
                assistant_tokens = len(assistant_msg["content"]) // CHARS_PER_TOKEN
                pair_tokens = user_tokens + assistant_tokens
            
            # Strict enforcement: skip pairs that exceed limit individually
            if strict_limit and pair_tokens > max_tokens:
                discarded_pair_count += 1
                # Save current chunk if it exists before discarding this pair
                if current_chunk:
                    chunk_item = {
                        "json_file_name": item.get("json_file_name", "unknown"),
                        "player_name": item.get("player_name", "unknown"),
                        "player_role": item.get("player_role", "unknown"),
                        "num_turns": len(current_chunk) // 2,
                        "conversations": current_chunk,
                        "is_split_chunk": True
                    }
                    if use_tokenizer:
                        chunk_item["total_input_tokens"] = chunk_input_tokens
                        chunk_item["total_output_tokens"] = chunk_output_tokens
                    result.append(chunk_item)
                    split_count += 1
                    # Reset for next chunk
                    current_chunk = []
                    current_tokens = 0
                    chunk_input_tokens = 0
                    chunk_output_tokens = 0
                continue  # Skip this oversized pair
            
            # Check if adding this pair would exceed limit
            if current_chunk and (current_tokens + pair_tokens > max_tokens):
                # Save current chunk and start new one
                chunk_item = {
                    "json_file_name": item.get("json_file_name", "unknown"),
                    "player_name": item.get("player_name", "unknown"),
                    "player_role": item.get("player_role", "unknown"),
                    "num_turns": len(current_chunk) // 2,
                    "conversations": current_chunk,
                    "is_split_chunk": True
                }
                if use_tokenizer:
                    chunk_item["total_input_tokens"] = chunk_input_tokens
                    chunk_item["total_output_tokens"] = chunk_output_tokens
                result.append(chunk_item)
                split_count += 1
                
                # Start new chunk with current pair
                current_chunk = [user_msg, assistant_msg]
                current_tokens = pair_tokens
                chunk_input_tokens = user_tokens
                chunk_output_tokens = assistant_tokens
            else:
                # Add pair to current chunk
                current_chunk.extend([user_msg, assistant_msg])
                current_tokens += pair_tokens
                chunk_input_tokens += user_tokens
                chunk_output_tokens += assistant_tokens
                conversation_has_valid_pairs = True
        
        # Add final chunk if it has valid pairs
        if current_chunk:
            chunk_item = {
                "json_file_name": item.get("json_file_name", "unknown"),
                "player_name": item.get("player_name", "unknown"),
                "player_role": item.get("player_role", "unknown"),
                "num_turns": len(current_chunk) // 2,
                "conversations": current_chunk,
                "is_split_chunk": len(result) > 0 and result[-1].get("json_file_name") == item.get("json_file_name")
            }
            if use_tokenizer:
                chunk_item["total_input_tokens"] = chunk_input_tokens
                chunk_item["total_output_tokens"] = chunk_output_tokens
            result.append(chunk_item)
        elif strict_limit and not conversation_has_valid_pairs:
            # Entire conversation was discarded due to strict limit
            discarded_conversation_count += 1
    
    original_count = len(data)
    result_count = len(result)
    if split_count > 0:
        print(f"Split {split_count} long conversations into {result_count} total conversations (was {original_count})")
    
    if strict_limit and discarded_pair_count > 0:
        print(f"STRICT MODE: Discarded {discarded_pair_count} pairs exceeding {max_tokens} tokens")
        if discarded_conversation_count > 0:
            print(f"STRICT MODE: {discarded_conversation_count} conversations completely discarded (all pairs exceeded limit)")
    
    return result


def write_alpaca_json(data, output_file, use_tokenizer_for_stats: bool, tokenizer=None):
    """Writes data to Alpaca JSON (conversations format) and calculates token statistics."""
    output_data = []
    total_input_tokens = 0
    total_output_tokens = 0
    max_conversation_input_tokens = 0
    max_conversation_output_tokens = 0
    max_single_turn_output_tokens = 0

    for item in data:
        conversations = item["conversations"]
        
        # Calculate token counts for this conversation
        if use_tokenizer_for_stats:
            conversation_input_tokens = item.get('total_input_tokens', 0)
            conversation_output_tokens = item.get('total_output_tokens', 0)
        else:
            # Approximate from character counts
            conversation_input_tokens = sum(
                len(msg["content"]) // CHARS_PER_TOKEN 
                for msg in conversations if msg["role"] == "user"
            )
            conversation_output_tokens = sum(
                len(msg["content"]) // CHARS_PER_TOKEN 
                for msg in conversations if msg["role"] == "assistant"
            )
        
        # Track max single turn output for statistics
        max_turn_output = 0
        for msg in conversations:
            if msg["role"] == "assistant":
                if use_tokenizer_for_stats and tokenizer:
                    turn_tokens = len(tokenizer.encode(msg["content"]))
                else:
                    turn_tokens = len(msg["content"]) // CHARS_PER_TOKEN
                max_turn_output = max(max_turn_output, turn_tokens)

        total_input_tokens += conversation_input_tokens
        total_output_tokens += conversation_output_tokens
        max_conversation_input_tokens = max(max_conversation_input_tokens, conversation_input_tokens)
        max_conversation_output_tokens = max(max_conversation_output_tokens, conversation_output_tokens)
        max_single_turn_output_tokens = max(max_single_turn_output_tokens, max_turn_output)

        output_data.append({
            "conversations": conversations
        })
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    return {
        "max_conversation_input_tokens": max_conversation_input_tokens,
        "max_conversation_output_tokens": max_conversation_output_tokens,
        "max_single_turn_output_tokens": max_single_turn_output_tokens,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "conversations_count": len(output_data)
    }


def calculate_and_plot_token_metrics(all_results, output_path: Path, use_tokenizer_for_calc: bool, tokenizer_model_name_for_report: str):
    """Calculate token metrics for multi-turn conversations and plot distributions. Returns a dictionary of stats."""
    stats_data = {}
    conversation_input_list = []
    conversation_output_list = []
    num_turns_list = []

    for r_item in all_results:
        input_tokens = 0
        output_tokens = 0

        if use_tokenizer_for_calc:
            input_tokens = r_item.get('total_input_tokens', 0)
            output_tokens = r_item.get('total_output_tokens', 0)
            # Fallback if actual counts are somehow not present
            if input_tokens == 0:
                print(f"Warning: Missing actual input token count for item from {r_item.get('json_file_name', 'Unknown Game')}. Approximating.")
                conversations = r_item["conversations"]
                input_tokens = sum(len(msg["content"]) // CHARS_PER_TOKEN for msg in conversations if msg["role"] == "user")
            if output_tokens == 0:
                print(f"Warning: Missing actual output token count for item from {r_item.get('json_file_name', 'Unknown Game')}. Approximating.")
                conversations = r_item["conversations"]
                output_tokens = sum(len(msg["content"]) // CHARS_PER_TOKEN for msg in conversations if msg["role"] == "assistant")
        else:
            conversations = r_item["conversations"]
            input_tokens = sum(len(msg["content"]) // CHARS_PER_TOKEN for msg in conversations if msg["role"] == "user")
            output_tokens = sum(len(msg["content"]) // CHARS_PER_TOKEN for msg in conversations if msg["role"] == "assistant")
        
        conversation_input_list.append(input_tokens)
        conversation_output_list.append(output_tokens)
        num_turns_list.append(r_item.get('num_turns', len(r_item["conversations"]) // 2))

    # Calculate total tokens (input + output)
    total_list = [i + o for i, o in zip(conversation_input_list, conversation_output_list)]
    
    if not conversation_input_list or not conversation_output_list or not total_list:
        print("No token data available to calculate metrics or plot.")
        return stats_data
    
    print(f"\nConversation Token Count Distribution Summary:")
    
    for name, arr in [("Input", conversation_input_list), ("Output", conversation_output_list), ("Total", total_list)]:
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
        print(f"  {name} tokens per conversation: count={format_num(count)}, min={format_num(min_val)}, median={format_num(median_val)}, mean={format_num(mean_val)}, max={format_num(max_val)}, std={format_num(std_val)}")
    
    # Print turns distribution
    if num_turns_list:
        print(f"\n  Turns per conversation: count={format_num(len(num_turns_list))}, min={format_num(min(num_turns_list))}, median={format_num(statistics.median(num_turns_list))}, mean={format_num(statistics.mean(num_turns_list))}, max={format_num(max(num_turns_list))}")
    
    fig, axs = plt.subplots(1, 3, figsize=(18,5))
    axs[0].hist(conversation_input_list, bins=50, color="C0", alpha=0.7)
    axs[0].set_title("Input tokens per conversation")
    axs[0].set_xlabel("Tokens")
    axs[0].set_ylabel("Count")
    axs[1].hist(conversation_output_list, bins=50, color="C1", alpha=0.7)
    axs[1].set_title("Output tokens per conversation")
    axs[1].set_xlabel("Tokens")
    axs[2].hist(total_list, bins=50, color="C2", alpha=0.7)
    axs[2].set_title("Total tokens per conversation")
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
    # sft_data_dir.mkdir(exist_ok=True, parents=True)
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

    # Split long conversations into multiple shorter ones to prevent OOM during training
    mode_msg = "strict mode - discarding oversized pairs" if STRICT_TOKEN_LIMIT else "soft mode - allowing oversized pairs as single chunks"
    print(f"\nSplitting conversations longer than {MAX_CONVERSATION_TOKENS} tokens ({mode_msg})...")
    all_results = split_long_conversations(
        all_results, 
        MAX_CONVERSATION_TOKENS, 
        use_actual_tokenizer,
        tokenizer_instance_for_counting,
        STRICT_TOKEN_LIMIT
    )
    print(f"After splitting: {len(all_results)} total conversations\n")

    # CSV not needed for now
    # with open(output_csv_file, 'w', newline='', encoding='utf-8') as f:
    #     fieldnames = ["json_file_name", "player_name", "player_role", "votes_before", "votes_after", "prompt", "model_cot_and_cleaned_output", "input_tokens", "output_tokens", "instruction_token_count_actual", "output_token_count_actual"]
    #     writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL, extrasaction='ignore') # Ignore extra fields not in fieldnames for robustness
    #     for result in all_results:
    #         row_to_write = {}
    #         for field_name in fieldnames:
    #             value = result.get(field_name) # Use .get() to avoid KeyError if a field is missing
    #             if isinstance(value, str):
    #                 row_to_write[field_name] = value.replace('\n', '\\n')
    #             else:
    #                 row_to_write[field_name] = value
    #         writer.writerow(row_to_write)

    # print(all_results)
        
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
    # train_file = sft_data_dir / "train.jsonl"
    # valid_file = sft_data_dir / "valid.jsonl"
    # test_file = sft_data_dir / "test.jsonl"
    # write_jsonl(train_data, train_file)
    # write_jsonl(valid_data, valid_file)
    # write_jsonl(test_data, test_file)
        
    # Write Alpaca JSON files
    alpaca_train_file = alpaca_data_dir / "among_them_train.json"
    alpaca_eval_file = alpaca_data_dir / "among_them_eval.json"
    train_stats = write_alpaca_json(train_data, alpaca_train_file, use_actual_tokenizer, tokenizer_instance_for_counting)
    eval_stats = write_alpaca_json(eval_data, alpaca_eval_file, use_actual_tokenizer, tokenizer_instance_for_counting)
        
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
        
    # Calculate overall stats
    max_conversation_input_tokens = max(train_stats["max_conversation_input_tokens"], eval_stats["max_conversation_input_tokens"])
    max_conversation_output_tokens = max(train_stats["max_conversation_output_tokens"], eval_stats["max_conversation_output_tokens"])
    max_single_turn_output = max(train_stats["max_single_turn_output_tokens"], eval_stats["max_single_turn_output_tokens"])
        
    total_input_tokens = train_stats["total_input_tokens"] + eval_stats["total_input_tokens"]
    total_output_tokens = train_stats["total_output_tokens"] + eval_stats["total_output_tokens"]
        
    print(f"Successfully processed {format_num(len(all_results))} conversations")
    print(f"\nAlpaca format datasets (multi-turn conversations):")
    print(f"  {format_num(train_stats['conversations_count'])} conversations to {alpaca_train_file}")
    print(f"  {format_num(eval_stats['conversations_count'])} conversations to {alpaca_eval_file}")
    mode_note = "discarding oversized pairs" if STRICT_TOKEN_LIMIT else "allowing oversized pairs as single chunks"
    print(f"  Note: Long conversations split at {MAX_CONVERSATION_TOKENS} tokens ({mode_note})")
        
    token_counting_method_info_oneline = f"Tokenizer: {tokenizer_model_name}" if use_actual_tokenizer else "Token Count Method: Estimated (4 chars = 1 token)"
    print(f"\n--- Alpaca Dataset Token Statistics ({token_counting_method_info_oneline}) ---")

    print(f"\nMaximum Token Lengths:")
    print(f"  Longest conversation input (all user turns): {format_num(max_conversation_input_tokens)} tokens")
    print(f"  Longest conversation output (all assistant turns): {format_num(max_conversation_output_tokens)} tokens")
    print(f"  Longest single turn output: {format_num(max_single_turn_output)} tokens")

    print(f"\nTotal Token Counts:")
    print(f"  Total input tokens (all user turns): {format_num(total_input_tokens)} tokens")
    print(f"  Total output tokens (all assistant turns): {format_num(total_output_tokens)} tokens")
    print(f"  Overall total: {format_num(total_input_tokens + total_output_tokens)} tokens")

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
            
        total_conversations_count = train_stats['conversations_count'] + eval_stats['conversations_count']
        f.write(f"- **Training conversations:** {format_num(train_stats['conversations_count'])}\n")
        f.write(f"- **Evaluation conversations:** {format_num(eval_stats['conversations_count'])}\n")
        f.write(f"- **Total conversations:** {format_num(total_conversations_count)}\n")
        
        if STRICT_TOKEN_LIMIT:
            f.write(f"\n> **Strict Mode:** Conversations are split at {MAX_CONVERSATION_TOKENS} tokens. User-assistant pairs exceeding this limit individually are **discarded** (data loss). This creates a hard limit to prevent OOM.\n")
        else:
            f.write(f"\n> **Soft Mode (Default):** Conversations are split at {MAX_CONVERSATION_TOKENS} tokens. If a single user-assistant pair exceeds this limit, it becomes its own chunk (preserves all data). Maximum chunk size may exceed limit.\n")
            
        f.write("\n## Maximum Token Lengths\n")
        f.write(f"- **Longest conversation input (all user turns):** {format_num(max_conversation_input_tokens)} tokens\n")
        f.write(f"- **Longest conversation output (all assistant turns):** {format_num(max_conversation_output_tokens)} tokens\n")
        f.write(f"- **Longest single turn output:** {format_num(max_single_turn_output)} tokens\n")
        if token_distribution_stats and 'total' in token_distribution_stats and 'max' in token_distribution_stats['total']:
             f.write(f"- **Longest combined (instruction + output):** {format_num(token_distribution_stats['total']['max'])} tokens\n")
            
        f.write("\n## Total Token Counts\n")
        f.write(f"- **Total input tokens (all user turns):** {format_num(total_input_tokens)} tokens\n")
        f.write(f"- **Total output tokens (all assistant turns):** {format_num(total_output_tokens)} tokens\n")
        f.write(f"- **Overall total (all inputs + all outputs):** {format_num(total_input_tokens + total_output_tokens)} tokens\n")

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
