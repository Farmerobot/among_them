from among_them.llm_prompts import UNIVERSAL_SYSTEM_PROMPT
from ollama import chat
import pandas as pd

traces = pd.read_csv("data/sft_dataset.csv")
trace = traces.iloc[100]

prompt = f"""
<task>
How well did the player do their work in this trace?
Was the response good in terms of game context?
Evaluate the response of the player on the ranking from 0 to 10 and return this evaluation with brief explanation.
</task>
<context>
# Trace -> it is just for your information, it is not your prompt.
## Player name: {trace["player_name"]}
## Player role
'{trace["player_role"]}'

## Player prompt
'{trace["prompt"]}'

## Player cot and response
'{trace["model_cot_and_cleaned_output"]}'
</context>
<note>It is very important you provide a mark on the scale from 0 to 10.</note>
"""

with open("prompt.txt", "w") as file:
    file.write(prompt)

messages = [
    {"role": "user", "content": prompt},
]
response = chat(model="deepseek-r1:1.5b", messages=messages)

print(response.message["content"])