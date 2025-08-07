#!/usr/bin/env python3
"""
Script to load a deepseek model and calculate the most probable token/word 
after generating a closing think token.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse


def get_top_k_tokens(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str, k: int = 5, show_tokenized: bool = False, partial_completion: str = ""):
    """
    Get the top k most probable tokens and their probabilities after the prompt.
    """
    # Format as chat messages and apply chat template if the tokenizer has a chat template
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        messages = [
            {"role": "user", "content": prompt}
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        # Fallback to simple prompt formatting
        formatted_prompt = prompt
    
    # Add partial completion if provided
    if partial_completion:
        formatted_prompt += partial_completion
    
    # Tokenize the prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    # Show tokenized prompt in readable form if requested
    if show_tokenized:
        input_ids = inputs["input_ids"][0]
        tokens = [tokenizer.decode([token_id]) for token_id in input_ids]
        print(f"Tokenized prompt: {tokens}")
        # print(f"Token IDs: {input_ids.tolist()}")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get logits for the last token position
    next_token_logits = logits[0, -1, :]
    
    # Apply softmax to get probabilities
    probabilities = torch.softmax(next_token_logits, dim=-1)
    
    top_probs, top_indices = torch.topk(probabilities, k)
    top_tokens = [tokenizer.decode([top_indices[i].item()]) for i in range(k)]
    
    return list(zip(top_tokens, top_probs.tolist()))


def build_probability_tree(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str, max_depth: int = 5, min_probability: float = 0.1, partial_completion: str = ""):
    """
    Build a tree visualization of token probabilities for the next max_depth tokens.
    Only includes tokens with probability >= min_probability (default 1%).
    """
    def _build_tree_recursive(current_prompt, depth, prefix=""):
        if depth >= max_depth:
            return ""
        
        # Get top tokens with probabilities
        top_tokens = get_top_k_tokens(model, tokenizer, prompt, k=5, partial_completion=current_prompt)
        
        # Filter tokens by minimum probability
        filtered_tokens = [(token, prob) for token, prob in top_tokens if prob >= min_probability]
        
        tree_str = ""
        for i, (token, prob) in enumerate(filtered_tokens):
            # Create branch representation
            connector = "├── " if i < len(filtered_tokens) - 1 else "└── "
            tree_str += f"{prefix}{connector}{token} ({prob:.2%})\n"
            
            # Recursively build subtree for this token
            if depth < max_depth - 1:
                print(f"{depth + 1} - {len(filtered_tokens)}")
                new_prompt = current_prompt + token
                child_prefix = prefix + ("│   " if i < len(filtered_tokens) - 1 else "    ")
                tree_str += _build_tree_recursive(new_prompt, depth + 1, child_prefix)
        
        return tree_str
    
    return _build_tree_recursive(partial_completion, 0)


def main(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
    """
    Main function to load model and calculate token probabilities.
    """
    print(f"Loading model: {model_name}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Example prompt with a closing think tag
    prompt = """
<purpose>
  You are an AI agent interacting directly with console_environment in a text-based social deduction game. 
  Operate strictly within defined game_context rules (phases, roles, mechanics) to achieve victory based on your assigned role while following execution_rules in console_environment.
</purpose>

<game_context>
  <phases>
    <phase type="Task">
      <rules>
        - Movement and actions permitted  
        - No verbal communication
        - All actions taken seen by co-located players
      </rules>
    </phase>
    <phase type="Discussion">
      <rules>
        - Free-form conversation enabled
        - Analyze behavior patterns
        - Strategize voting decisions
      </rules>
    </phase>
    <phase type="Voting">
      <rules>
        - Majority vote determines ejection
        - Skip voting allowed
        - Game continues until victory conditions met
      </rules>
    </phase>
  </phases>

  <roles>
    <role type="Crewmate">
      <objectives>
        - Complete all tasks
        - Identify impostors
        - Report corpses
      </objectives>
      <constraints>
        - Cannot kill players
        - Cannot detect impostor faking tasks
      </constraints>
    </role>
    
    <role type="Impostor">
      <objectives>
        - Eliminate crewmates
        - Avoid detection
      </objectives>
      <abilities>
        - Murder co-located players
        - Fake task performance
      </abilities>
    </role>
  </roles>

  <mechanics>
    <mechanic type="Places">
      - Players can move between places 
      - There is nothing else players can do in places
      - All actions taken seen by co-located players
    </mechanic>
    <mechanic type="Tasks">
      - Tasks are assigned to each player and completed tasks are removed from the list
      - Tasks are independent of each other and of other players meaning they can be completed multiple times by multiple players
      - All tasks must be completed to win
    </mechanic>
    <mechanic type="Corpse Reporting">
      - Any player can report corpses during task phase
      - Triggers discussion phase
      - No automatic death alerts
      - If no one reports a corpse discussion phase will not start
    </mechanic>
    <mechanic type="Victory Conditions">
      - Crewmates win at 100% tasks
      - Impostors win when kill all crewmates
      - Ejection affects team ratios. Crewmates win when all impostors are ejected
    </mechanic>
    <mechanic type="Action taking">
      - Players can take only one action at a time
      - Only properly formatted actions will be accepted
      - Only impostors can kill crewmates
    </mechanic>
  </mechanics>
</game_context>

<execution_rules>
  <rule>Never explain reasoning or strategy</rule>
  <rule>Assume perfect game rule knowledge</rule>
  <rule>Consider all visible player actions from history</rule>
  <rule>Maintain role consistency (impostors never reveal themselves)</rule>
  <rule>During discussions: Messages must be natural conversational English</rule>
</execution_rules>
<console_environment>
  <input>
    <input_type>Text</input_type>
    <input_format>One action in one line or message in one line</input_format>
  </input>
  <output>
    <output_type>Text</output_type>
    <output_format>Observations from environment</output_format>
  </output>
</console_environment>


You are Charlie.
You are a crewmate. Your goal is to complete tasks and identify impostors.
There are 1 impostors among the players.
Potential impostors: David, Alice, Bob, Eve.
Tasks: Route power to attack in electrical, Clean the floor in storage.

System said: The game started
You are in Cafeteria. You see doors to: storage, medbay. You see David, Alice, Bob and Eve here. Given the situation, the best action to take is:"""
    
    print(f"\nPrompt: {prompt}")
    
    # Now let's see what the model would predict after a partial completion
    print(f"\n=== WITH PARTIAL COMPLETION ===")
    partial_completion = """
Alright, I'm Charlie, a crewmate in this social deduction game. My goal is to complete all tasks and identify the impostor among the players. The system just told me the game has started, and I'm currently in the Cafeteria. I can see doors leading to storage and medbay here.

Looking around, I see David, Alice, Bob, and Eve with me. There are four of us, but one is an impostor. The tasks assigned are "Route power to attack in electrical" and "Clean the floor in storage." Since I'm a crewmate, I need to complete these tasks to win.

I remember that in Task Phase, players can move between places, but only one action at a time. My first thought is to check if there are any tasks I can do from here. The Cafeteria doesn't have any tasks listed, so my next best move is to head to either storage or medbay. 

Storage has the "Clean the floor" task, which seems straightforward. If I go there and find no one else doing it, I can complete it quickly. Alternatively, if someone else is there, we might collaborate, but since only four players are here, maybe it's better to move alone.

I also need to be cautious about impostors. Impostors can fake tasks or even kill crewmates. If I see someone loitering without doing anything, that could be a red flag. But right now, everyone is in the Cafeteria with me. Since the game has just started, maybe it's better for me to start moving towards storage first since there's a task there.

So, my plan is to move to storage and try to complete the "Clean the floor" task. By doing so, I contribute directly to our victory condition of completing all tasks. If someone follows or tries to interfere, that might indicate they're the impostor. Also, if I find the place already clean, perhaps I can help with other tasks once back in discussion phase.

I should also keep an eye out for any suspicious behavior. For instance, if David, Alice, Bob, or Eve seems reluctant to move or acts oddly, that could be a sign. But right now, my immediate action is to take the first step towards completing a task.
</think>
"""
    print(f"Partial completion: {partial_completion}")
    
    top_tokens_partial = get_top_k_tokens(model, tokenizer, prompt, k=5, show_tokenized=True, partial_completion=partial_completion)
    token_partial, probability_partial = top_tokens_partial[0]
    
    print(f"\nMost probable next token: '{token_partial}'")
    print(f"Probability: {probability_partial:.6f}")
    
    # Let's also get top 5 most probable tokens for more insight
    print(f"\nTop {len(top_tokens_partial)} most probable tokens:")
    for i, (token_str, prob) in enumerate(top_tokens_partial):
        print(f"  {i+1}. '{token_str}' (probability: {prob:.6f})")

    # Build and display probability tree
    print(f"\n=== PROBABILITY TREE (5 levels, >=1%) ===")
    tree_str = build_probability_tree(model, tokenizer, prompt, max_depth=5, min_probability=0.1, partial_completion=partial_completion)
    if tree_str:
        print(tree_str)
    else:
        print("No tokens with probability >= 1% found.")
    
    # # Interactive loop to choose tokens for next 5 positions
    # print(f"\n=== INTERACTIVE TOKEN SELECTION ===")
    # current_completion = partial_completion
    # for step in range(5):
    #     print(f"\n--- Step {step + 1}/5 ---")
    #     top_tokens = get_top_k_tokens(model, tokenizer, prompt, k=5, partial_completion=current_completion)
        
    #     print(f"\nTop 5 most probable tokens:")
    #     for i, (token_str, prob) in enumerate(top_tokens):
    #         print(f"  {i+1}. '{token_str}' (probability: {prob:.6f})")
        
    #     # Get user choice
    #     while True:
    #         try:
    #             choice = int(input("\nChoose token (1-5, or 0 to enter custom token): "))
    #             if 0 <= choice <= 5:
    #                 break
    #             else:
    #                 print("Please enter a number between 0 and 5.")
    #         except ValueError:
    #             print("Please enter a valid number.")
        
    #     if choice == 0:
    #         # Custom token
    #         chosen_token = input("Enter custom token: ")
    #     else:
    #         # Selected token
    #         chosen_token = top_tokens[choice - 1][0]
        
    #     current_completion += chosen_token
    #     print(f"\nCurrent completion: {current_completion}")
    
    # print(f"\n=== FINAL COMPLETION ===")
    # print(current_completion)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate token probabilities after think tag")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
                        help="Model name or path to load")
    
    args = parser.parse_args()
    main(args.model)
