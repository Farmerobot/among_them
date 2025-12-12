#!/usr/bin/env python3
"""
Evaluates SFT dataset traces using GPT-4o-mini, scoring each trace 0-3 based on quality.
Creates generated/trace_analysis.txt which is required by create_sampled_sft_dataset.py.
Requires OPENAI_API_KEY or OPENROUTER_API_KEY environment variable depending on USE_OPENROUTER setting.
"""

# Configuration: Set to True to use OpenRouter, False to use OpenAI
USE_OPENROUTER = True

# from among_them.llm_prompts import UNIVERSAL_SYSTEM_PROMPT
import openai
# from ollama import chat
import os
import json
import sys
from time import sleep
from pathlib import Path
from among_them.llm_prompts import UNIVERSAL_SYSTEM_PROMPT

# Import centralized configuration
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from among_them.config import OPENAI_API_KEY, OPENROUTER_API_KEY

def validate_api_key():
    """Validate that the appropriate API key is properly set."""
    if USE_OPENROUTER:
        api_key = OPENROUTER_API_KEY
        if not api_key:
            print("ERROR: OPENROUTER_API_KEY environment variable is not set.")
            print("Please set your OpenRouter API key:")
            print("  Windows: set OPENROUTER_API_KEY=your_api_key_here")
            print("  Linux/Mac: export OPENROUTER_API_KEY=your_api_key_here")
            sys.exit(1)
    else:
        api_key = OPENAI_API_KEY
        if not api_key:
            print("ERROR: OPENAI_API_KEY environment variable is not set.")
            print("Please set your OpenAI API key:")
            print("  Windows: set OPENAI_API_KEY=your_api_key_here")
            print("  Linux/Mac: export OPENAI_API_KEY=your_api_key_here")
            sys.exit(1)
    
    return api_key


# Validate API key before proceeding
validate_api_key()

# Load conversations from JSON files
data_dir = Path("data/alpaca")
train_file = data_dir / "among_them_train.json"
eval_file = data_dir / "among_them_eval.json"

with open(train_file, 'r') as f:
    train_data = json.load(f)
with open(eval_file, 'r') as f:
    eval_data = json.load(f)

# Combine all conversations
all_conversations = train_data + eval_data
print(f"Loaded {len(all_conversations)} conversations for evaluation")

def get_trace_evaluation(conversation_item):
    """Evaluate a multi-turn conversation."""
    conversations = conversation_item["conversations"]

    # New prompt format: prepend the universal system prompt used by live/manual play
    messages = [{"role": "system", "content": UNIVERSAL_SYSTEM_PROMPT}] + conversations

    conversation_text = ""
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        conversation_text += f"\n{role}: {content}\n"
    
    prompt = f"""
    <task>
    You are evaluating a player's multi-turn reasoning and actions in the game *Among Them* (similar to *Among Us*).
    Your job is to check if the player's explanations and responses align with the rules and logic of the game across all turns.

    If the player's explanation contains a hallucination (i.e., invents game facts not present or contradicts the game's setup/rules) or misinterprets the situation, you must score it as 0.

    Otherwise, evaluate their overall performance from 1 to 3 based on the quality of their reasoning and in-game actions:

    - 0: Player hallucinated or clearly misunderstood the game logic/setup.
    - 1: Weak or flawed reasoning; actions taken were not effective or insightful.
    - 2: Decent reasoning and solid, valid in-game actions.
    - 3: Excellent reasoning and tactically smart or creative moves given the situation.

    </task>

    <context>
    # Multi-turn conversation for evaluation:
    {conversation_text}
    </context>

    <important>
    Give your evaluation in the form:
    [score, "Brief explanation of 1-2 sentences why this score was given"]
    </important>

    <output_format>
    e.g.:
    [2, "Player showed reasonable logic across turns and made solid tactical decisions."]
    e.g.:
    [0, "Player claimed to see a vent animation in a room where vents don't exist. This is a hallucination."]
    </output_format>
    """

    print(prompt)

    messages = [
        {"role": "user", "content": prompt},
    ]

    if USE_OPENROUTER:
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        
        completion = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=messages,
        )
    else:
        client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
        )
        
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )

    return completion.choices[0].message.content


# Ensure generated directory exists
Path("generated").mkdir(exist_ok=True)

# Clear existing analysis file if it exists
analysis_file = Path("generated/trace_analysis.txt")
if analysis_file.exists():
    analysis_file.unlink()
    print("Cleared existing trace_analysis.txt\n")

for i in range(len(all_conversations)):
    conversation = all_conversations[i]
    print(f"Conversation {i+1}/{len(all_conversations)}:")

    res, res_list = None, None
    while res is None or res_list is None:
        try:
            res = get_trace_evaluation(conversation)
            res_list = json.loads(res)
        except Exception as e:
            print(f"An error occurred: {e}, repeating in 1s")
            sleep(1)

    print(res_list, "\n")
    with open("generated/trace_analysis.txt", "a") as f:
        f.write(res + "\n")

print(f"\n✅ Completed evaluation of {len(all_conversations)} conversations")
print(f"Results saved to: {analysis_file.absolute()}")