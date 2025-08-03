import random
import tiktoken
from among_them.config import STATE_FILE
from among_them.game_engine import GameEngine
from among_them.models.action_type import ActionType
from among_them.models.player_role import PlayerRole
from among_them.utils.llm_utils import parse_llm_response_to_action
from among_them.utils.ui_utils import prompt_manual_fallback_action
from among_them.game_config import GameConfig
import os
import argparse

def main(reset=False, always_kill=False, remove_last_n=0):
    """Runs the game with manual LLM control."""
    game_config = GameConfig(
        num_tasks=2,
        num_players=5,
        num_impostors=1,
        map_size=0,
        num_task_phase_actions_per_player=10,
        num_discuss_phase_actions_per_player=2,
        impostor_cooldown=1
    )
    engine = GameEngine(game_config)
    
    # Reset state if requested
    if reset and os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        print("State file removed.")
    
    if not reset and os.path.exists(STATE_FILE):
        engine.load_state()
        print(f"Game loaded from state file with {len(engine.history)} history entries")
        
        # Remove last n entries if requested
        if remove_last_n > 0 and len(engine.history) > remove_last_n:
            engine.history = engine.history[:-remove_last_n]
            print(f"Removed last {remove_last_n} entries from history. {len(engine.history)} entries remaining.")

    for history_item in engine.history:
        print(history_item)

    while True:
        turn_context_history, actions_player_can_take, system_prompt, user_prompt, pre_discussion_vote_prompts = engine.get_turn_context()
        if not turn_context_history:
            print("Game over!")
            break

        pre_discussion_votes = {}
        if pre_discussion_vote_prompts:
            print("--- Collecting Pre-Discussion Votes ---")
            for vote_prompt in pre_discussion_vote_prompts:
                player = vote_prompt["player"]
                actions_pd = vote_prompt["actions"]

                try:
                    # Here, you would insert your custom LLM call and log probability logic.
                    action_taken = random.choice(actions_pd)
                    llm_response, cot = action_taken.text, "<think>Was thinking about this option: " + action_taken.text + "</think>"
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

        try:
            # Here, you would insert your custom LLM call and log probability logic.
            action_taken = random.choice(actions_player_can_take)
            kill_actions = [action for action in actions_player_can_take if action.type == ActionType.KILL]
            task_actions = [action for action in actions_player_can_take if action.type == ActionType.TASK]
            if current_player.role == PlayerRole.IMPOSTOR and kill_actions and always_kill:
                action_taken = random.choice(kill_actions)
            elif current_player.role == PlayerRole.CREWMATE and task_actions:
                action_taken = random.choice(task_actions)
            llm_response, cot = action_taken.text, "<think>Was thinking about this option: " + action_taken.text + "</think>"
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

        # Step the environment
        history_len = len(engine.history)
        game_over, end_reason = engine.step(turn_context_history, action_taken, response_text, cot, token_usage, pre_discussion_votes)
        while history_len != len(engine.history):
            print(f"{engine.history[history_len]}")
            history_len += 1
        if game_over:
            print(f"Game over! Reason: {end_reason}")
            break

def parse_args():
    parser = argparse.ArgumentParser(description="Run Among Them game with custom options.")
    parser.add_argument("--reset", action="store_true", 
                        help="Remove the state file and start a new game")
    parser.add_argument("--always-kill", action="store_true",
                        help="Impostors will always choose to kill when possible")
    parser.add_argument("--remove-last-n", type=int, default=0,
                        help="Remove the last N entries from history and start from there")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(reset=args.reset, always_kill=args.always_kill, remove_last_n=args.remove_last_n)
