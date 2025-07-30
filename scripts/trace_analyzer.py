#!/usr/bin/env python3
"""
Evaluates SFT dataset traces using GPT-4o-mini, scoring each trace 0-3 based on quality.
Creates generated/trace_analysis.txt which is required by create_sampled_sft_dataset.py.
Requires OPENAI_API_KEY or OPENROUTER_API_KEY environment variable depending on USE_OPENROUTER setting.
"""

# Configuration: Set to True to use OpenRouter, False to use OpenAI
USE_OPENROUTER = False

# from among_them.llm_prompts import UNIVERSAL_SYSTEM_PROMPT
import openai
# from ollama import chat
import pandas as pd
import os
import json
import sys
from time import sleep

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

fieldnames = ["json_file_name", "player_name", "player_role", "votes_before", "votes_after", "prompt", "model_cot_and_cleaned_output", "input_tokens", "output_tokens", "instruction_token_count_actual", "output_token_count_actual"]

traces = pd.read_csv("data/sft_dataset.csv", names=fieldnames)
impostor_traces = traces[traces["player_role"] == "Impostor"]

def get_trace_evaluation(trace):
    prompt = f"""
    <task>
    You are evaluating a player's reasoning and action taken during a round in the game *Among Them* (similar to *Among Us*).
    Your job is to check if the player's explanation and response align with the rules and logic of the game.

    If the player's explanation contains a hallucination (i.e., invents game facts not present or contradicts the game's setup/rules) or misinterprets the situation, you must score it as 0.

    Otherwise, evaluate their response from 1 to 3 based on the quality of their reasoning and in-game action:

    - 0: Player hallucinated or clearly misunderstood the game logic/setup.
    - 1: Weak or flawed reasoning; action taken was not effective or insightful.
    - 2: Decent reasoning and a solid, valid in-game action.
    - 3: Excellent reasoning and a tactically smart or creative move given the situation.

    </task>

    <context>
    # Game trace for evaluation:
    ## Player name: {trace["player_name"]}
    ## Player role: '{trace["player_role"]}'
    ## Player prompt: '{trace["prompt"]}'
    ## Player's explanation and response: '{trace["model_cot_and_cleaned_output"]}'
    </context>

    <important>
    Give your evaluation in the form:
    [score, "Brief explanation of 1-2 sentences why this score was given"]
    </important>

    <output_format>
    e.g.:
    [2, "Anne had reasonable logic and shifted suspicion away with a valid argument."]
    e.g.:
    [0, "Frank claimed to see a vent animation in a room where vents don't exist. This is a hallucination."]
    </output_format>
    """

    # with open("prompt.txt", "w") as file:
    #     file.write(prompt)

    messages = [
        {"role": "user", "content": prompt},
    ]

    # response = chat(model="deepseek-r1:1.5b", messages=messages)

    # response = openai.ChatCompletion.create(
    #     model="gpt-4o-mini",
    #     messages=messages
    # )

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

    # print(response)
    # print(response.message["content"])


for i in range(len(traces)):
    trace = traces.iloc[i]
    print(f"Trace number {i}:")

    res, res_list = None, None
    while res is None or res_list is None:
        try:
            res = get_trace_evaluation(trace)
            res_list = json.loads(res)
        except:
            print("An error occured, repeating in 1s")
            sleep(1)

    print(res_list, "\n")
    with open("generated/trace_analysis.txt", "a") as f:
        f.write(res + "\n")