#!/usr/bin/env python3
"""
Analyzes voting statistics from SFT dataset traces.
Optional analysis script - doesn't create files needed by other scripts.
"""

import pandas as pd
import json

fieldnames = ["json_file_name", "player_name", "player_role", "votes_before", "votes_after", "prompt", "model_cot_and_cleaned_output", "input_tokens", "output_tokens", "instruction_token_count_actual", "output_token_count_actual"]

traces = pd.read_csv("data/sft_dataset.csv", names=fieldnames)
impostor_traces = traces[traces["player_role"] == "Impostor"]

k_traces_with_votes, k_pos_change, k_neg_change = 0, 0, 0
for i in range(len(traces)):
    trace = traces.iloc[i]
    if trace["votes_before"] != "{}":
        k_traces_with_votes += 1
        votes_before = list(json.loads(trace['votes_before']).values())
        votes_after = list(json.loads(trace['votes_after']).values())
        k_votes_before = votes_before.count(trace["player_name"])
        k_votes_after = votes_after.count(trace["player_name"])

        if k_votes_after - k_votes_before < 0:
            k_pos_change += 1
        elif k_votes_after - k_votes_before > 0:
            k_neg_change += 1

        print(f"Trace number {i+1}: {k_votes_before} | {k_votes_after} ({k_votes_after - k_votes_before})")

print(f"\n\nOut of {len(traces)} traces {k_traces_with_votes} had votes ({round(k_traces_with_votes / len(traces) * 100, 1)}%)\n")
print(f"{k_pos_change + k_neg_change} traces had different votes before and after")
print(f"It is {round((k_pos_change + k_neg_change) / len(traces) * 100, 1)}% of all the dataset or {round((k_pos_change + k_neg_change) / k_traces_with_votes * 100, 1)}% of traces with votes\n")
print(f"{k_pos_change} responses had a positive impact on the votes")
print(f"It is {round(k_pos_change / len(traces) * 100, 1)}% of all the dataset or {round(k_pos_change / k_traces_with_votes * 100, 1)}% of traces with votes")