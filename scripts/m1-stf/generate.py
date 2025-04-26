#!/usr/bin/env python3
"""
Generate a game action using MLX LM Python API for Among Them.
"""
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from among_them.llm_prompts import UNIVERSAL_SYSTEM_PROMPT

def main():
    # Load the fine-tuned (or LoRA-quantized) model
    model_name = "mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit"
    model, tokenizer = load(model_name)

    # System prompt defining game rules and context
    system_prompt = UNIVERSAL_SYSTEM_PROMPT

    # Player-specific prompt
    user_prompt = "<player_info>\n<player_name>Charlie</player_name>\n<current_role>Crewmate</current_role>\n<current_location>Weapons</current_location>\n<impostor_count>1</impostor_count>\n<potential_impostors>\n<player>David</player>\n<player>Eve</player>\n<player>Alice</player>\n<player>Bob</player>\n</potential_impostors>\n<tasks_left>\n<task><name>Check catalyzer in lower engine</name><location>Lower Engine</location></task>\n</tasks_left>\n</player_info>\n\n<game_history>\n<action type=""observed"">The game started</action>\n<player_thought><thought>\nOkay, so I'm Charlie, a crewmate in this game. My goal is to complete all the tasks before the impostors can either kill enough of us or get voted out. Right now, I'm in the Cafeteria with David, Eve, Alice, and Bob. The available actions are wait, move to Medbay, Admin, or Weapons.\n\nI have four tasks left: Fix wiring in storage, Clear asteroids in weapons, Check catalyzer in lower engine, and Fix wiring in weapons. Since I'm a crewmate, completing these tasks is crucial for us to win.\n\nLooking at the available actions, moving might be necessary to get closer to completing tasks or to check if there are any dead bodies that could trigger discussion. Right now, no one has died yet, so it's all about tasks and blending in without suspicion.\n\nSince I'm in the Cafeteria with others, maybe moving somewhere else where I can work on my tasks would be better. Let me see which room is best for each task. Fixing wiring in storage might be in the Storage room, but that's not an option here. Clear asteroids in weapons probably requires me to go to the Weapons room because that's where the action is. So moving to Weapons could allow me to work on two tasks: Check catalyzer and Fix wiring in weapons.\n\nAlternatively, if I stay here, I can't do any tasks. So moving makes sense. Let me choose to move to Weapons.\n</thought></player_thought>\n<action type=""player_action"">You [Charlie] moved to location Weapons</action>\n</game_history>\n\n<current_state>\n<companions>none</companions>\n<location>Weapons</location>\n<phase>Task</phase>\n<note>You cannot speak during this phase.</note>\n</current_state>\n\n\n<available_actions>\n<action>wait</action>\n<action>move to location Medbay</action>\n<action>move to location Admin</action>\n<action>move to location Weapons</action>\n</available_actions>\n\n\nChoose one action. Respond in the following xml format: <action>action</action>"

    # Format as chat messages and apply chat template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    # Generate chat response
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=2000,
        sampler=make_sampler(temp=1.0),
        verbose=True
    )
    print(response)

if __name__ == "__main__":
    main()
