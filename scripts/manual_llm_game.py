import random
import tiktoken
from among_them.config import STATE_FILE
from among_them.game_engine import GameEngine
from among_them.utils.llm_utils import parse_llm_response_to_action, invoke_llm
from among_them.utils.ui_utils import prompt_manual_fallback_action
from among_them.game_config import GameConfig
import os

def main():
    """Runs the game with manual LLM control."""
    game_config = GameConfig(
        num_tasks=4,
        num_players=7,
        num_impostors=2,
        map_size=1,
        num_task_phase_actions_per_player=10,
        num_discuss_phase_actions_per_player=2,
        impostor_cooldown=1
    )
    engine = GameEngine(game_config)
    if os.path.exists(STATE_FILE):
        engine.load_state()
        print(f"Game loaded from state file with {len(engine.history)} history entries")

    for history_item in engine.history:
        print(history_item)

    while True:
        game_over, end_reason = engine.handle_automatic_transitions()
        if game_over:
            print(f"Game over! Reason: {end_reason}")
            break

        turn_context_history, actions_player_can_take, system_prompt, user_prompt, pre_discussion_vote_prompts = engine.get_turn_context()
        if not turn_context_history:
            continue

        pre_discussion_votes = {}
        if pre_discussion_vote_prompts:
            print("--- Collecting Pre-Discussion Votes ---")
            for vote_prompt in pre_discussion_vote_prompts:
                player = vote_prompt["player"]
                system_prompt_pd = vote_prompt["system_prompt"]
                user_prompt_pd = vote_prompt["user_prompt"]
                actions_pd = vote_prompt["actions"]

                try:
                    # Here, you would insert your custom LLM call and log probability logic.
                    llm_response, cot = invoke_llm(system_prompt_pd, user_prompt_pd, player.llm_model_name)
                except KeyboardInterrupt:
                    action_idx, llm_response, cot, _ = prompt_manual_fallback_action(actions_pd)

                print(f"Simulated LLM Response from {player.name}: {llm_response}")

                # Parse the response and get the action
                action_idx, _ = parse_llm_response_to_action(
                    actions_pd, llm_response, player.name
                )
                action_taken = actions_pd[action_idx]

                pre_discussion_votes[player.name] = {
                    "voted_player": action_taken.target_player_name,
                    "chain_of_thought": cot,
                }

        current_player_name = turn_context_history.action_taken.player_name
        current_player = next((p for p in engine.players if p.name == current_player_name), None)
        if current_player is None:
            raise ValueError(f"Current player {current_player_name} not found.")

        print(f"--- {current_player.name}'s turn ---")

        try:
            # Here, you would insert your custom LLM call and log probability logic.
            llm_response, cot = invoke_llm(system_prompt, user_prompt, current_player.llm_model_name)
        except KeyboardInterrupt:
            action_idx, llm_response, cot, _ = prompt_manual_fallback_action(actions_player_can_take)

        # Parse the response and get the action
        action_idx, response_text = parse_llm_response_to_action(
            actions_player_can_take, llm_response, current_player.name
        )
        action_taken = actions_player_can_take[action_idx]

        # Calculate token usage with tiktoken
        encoding = tiktoken.encoding_for_model("gpt-4o")
        input_tokens = len(encoding.encode(system_prompt + user_prompt))
        output_tokens = len(encoding.encode(response_text + (cot or "")))
        token_usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}

        print(f"Action taken: {action_taken} Token usage: {token_usage}")

        # Step the environment
        engine.step(turn_context_history, action_taken, response_text, cot, token_usage, pre_discussion_votes)

if __name__ == "__main__":
    main()
