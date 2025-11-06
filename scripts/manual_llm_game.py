import tiktoken
from among_them.config import STATE_FILE
from among_them.game_engine import GameEngine
from among_them.utils.llm_utils import parse_llm_response_to_action, invoke_llm
from among_them.utils.ui_utils import prompt_manual_fallback_action
from among_them.game_config import GameConfig
import os
import argparse

def main(reset=False, remove_last_n=0, game_config: GameConfig = None, state_file_path: str = None):
    """Runs the game with manual LLM control."""
    # Determine the state file path to use
    current_state_file = state_file_path if state_file_path is not None else STATE_FILE

    # Reset state if requested
    if reset and os.path.exists(current_state_file):
        os.remove(current_state_file)
        print("State file removed.")
    
    # Initialize game_config if not provided
    if game_config is None:
        game_config = GameConfig(
            num_tasks=2,
            num_players=5,
            num_impostors=1,
            map_size=0,
            num_task_phase_actions_per_player=10,
            num_discuss_phase_actions_per_player=2,
            impostor_cooldown=1
        )
    # Initialize GameEngine with game_config and the determined state file path
    engine = GameEngine(game_config, file_path=current_state_file)
    
    # Load state if file exists
    if os.path.exists(current_state_file):
        engine.load_state()
        print(f"Game loaded from state file with {len(engine.history)} history entries")
        
        # Remove last n entries if requested
        if remove_last_n > 0 and len(engine.history) > remove_last_n:
            engine.history = engine.history[:-remove_last_n]
            print(f"Removed last {remove_last_n} entries from history. {len(engine.history)} entries remaining.")

    for history_item in engine.history:
        print(history_item)

    while True:
        turn_context_history, actions_player_can_take, conversation, pre_discussion_vote_prompts = engine.get_turn_context()
        if not turn_context_history:
            print("Game over!")
            break

        pre_discussion_votes = {}
        if pre_discussion_vote_prompts:
            print("--- Collecting Pre-Discussion Votes ---")
            for vote_prompt in pre_discussion_vote_prompts:
                player = vote_prompt["player"]
                pd_conversation = vote_prompt["conversation"]
                actions_pd = vote_prompt["actions"]

                # Retry loop for pre-discussion votes
                max_retries = 3
                retry_count = 0
                
                while retry_count < max_retries:
                    try:
                        # Build allowed actions for early-stop single-line streaming
                        allowed_actions_pd = [a.set_stories().command_perspective for a in actions_pd]
                        llm_response, cot = invoke_llm(
                            pd_conversation,
                            player.llm_model_name,
                            allowed_actions=allowed_actions_pd,
                            single_line_only=True,
                            actions=actions_pd,
                        )
                        
                        # Try to parse the response
                        try:
                            action_idx, _ = parse_llm_response_to_action(
                                actions_pd, llm_response, player.name
                            )
                            action_taken = actions_pd[action_idx]
                            break  # success, exit the loop
                        except ValueError as parse_error:
                            print(f"\033[93mParsing error for {player.name} (attempt {retry_count + 1}/{max_retries}): {parse_error}\033[0m")
                            print(f"\033[93mRetrying with same prompt...\033[0m")
                            retry_count += 1
                            if retry_count >= max_retries:
                                print(f"\033[91mMax retries reached for {player.name}. Using manual fallback.\033[0m")
                                action_idx, llm_response, cot, _ = prompt_manual_fallback_action(actions_pd)
                                action_taken = actions_pd[action_idx]
                                break
                            continue  # retry with same prompt
                            
                    except KeyboardInterrupt:
                        action_idx, llm_response, cot, _ = prompt_manual_fallback_action(actions_pd)
                        action_taken = actions_pd[action_idx]
                        break  # we got a result manually, so exit the loop
                    except Exception as e:
                        # Log the error but keep the loop running
                        print(f"\033[91mError during LLM invocation for {player.name}: {e}\033[0m")
                        retry_count += 1
                        if retry_count >= max_retries:
                            print(f"\033[91mMax retries reached for {player.name}. Using manual fallback.\033[0m")
                            action_idx, llm_response, cot, _ = prompt_manual_fallback_action(actions_pd)
                            action_taken = actions_pd[action_idx]
                            break

                print(f"Simulated LLM Response from {player.name}: {llm_response}")

                pre_discussion_votes[player.name] = {
                    "voted_player": action_taken.target_player_name,
                    "chain_of_thought": cot,
                }

        current_player_name = turn_context_history.action_taken.player_name
        current_player = next((p for p in engine.players if p.name == current_player_name), None)
        if current_player is None:
            raise ValueError(f"Current player {current_player_name} not found.")

        print(f"--- {current_player.name}'s turn ---")

        # Retry loop for LLM invocation and action parsing
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Attempt to invoke the LLM with early-stop on first valid action for non-speak phases
                allowed_actions_main = [a.set_stories().command_perspective for a in actions_player_can_take if a.type.name != "SPEAK"]
                single_line_only = len(allowed_actions_main) > 0
                allowed_actions_for_call = allowed_actions_main if single_line_only else None
                llm_response, cot = invoke_llm(
                    conversation,
                    current_player.llm_model_name,
                    allowed_actions=allowed_actions_for_call,
                    single_line_only=single_line_only,
                    max_output_chars=None if single_line_only else 1500,
                    actions=actions_player_can_take,
                )
                
                # Try to parse the response
                try:
                    action_idx, response_text = parse_llm_response_to_action(
                        actions_player_can_take, llm_response, current_player.name
                    )
                    action_taken = actions_player_can_take[action_idx]
                    break  # success, exit the loop
                except ValueError as parse_error:
                    print(f"\033[93mParsing error (attempt {retry_count + 1}/{max_retries}): {parse_error}\033[0m")
                    print(f"\033[93mRetrying with same prompt...\033[0m")
                    retry_count += 1
                    if retry_count >= max_retries:
                        print(f"\033[91mMax retries reached. Using manual fallback.\033[0m")
                        action_idx, llm_response, cot, _ = prompt_manual_fallback_action(actions_player_can_take)
                        action_taken = actions_player_can_take[action_idx]
                        response_text = llm_response
                        break
                    continue  # retry with same prompt
                    
            except KeyboardInterrupt:
                # First Ctrl+C: switch to manual fallback. If another Ctrl+C occurs during fallback, exit cleanly.
                try:
                    action_idx, llm_response, cot, _ = prompt_manual_fallback_action(actions_player_can_take)
                    action_taken = actions_player_can_take[action_idx]
                    response_text = llm_response
                    break  # manual action obtained
                except KeyboardInterrupt:
                    print("\nInterrupted during manual fallback. Exiting game.")
                    return  # exit the main() function cleanly
            except Exception as e:
                # Log the error but keep the loop running
                import traceback
                stacktrace = traceback.format_exc()
                print(f"\033[91mError during LLM invocation: {e}\033[0m")
                print(f"\033[91mStacktrace: {stacktrace}\033[0m")
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"\033[91mMax retries reached. Using manual fallback.\033[0m")
                    action_idx, llm_response, cot, _ = prompt_manual_fallback_action(actions_player_can_take)
                    action_taken = actions_player_can_take[action_idx]
                    response_text = llm_response
                    break

        # Calculate token usage with tiktoken
        encoding = tiktoken.encoding_for_model("gpt-4o")
        # Count tokens for all messages in conversation
        conversation_text = "\n".join([msg["content"] for msg in conversation])
        input_tokens = len(encoding.encode(conversation_text))
        output_tokens = len(encoding.encode(response_text + (cot or "")))
        token_usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}

        print(f"Action taken: {action_taken} Token usage: {token_usage}")

        # Step the environment
        game_over, end_reason = engine.step(turn_context_history, action_taken, response_text, cot, token_usage, pre_discussion_votes)
        if game_over:
            print(f"Game over! Reason: {end_reason}")
            break

def parse_args():
    parser = argparse.ArgumentParser(description="Run Among Them game with manual LLM control.")
    parser.add_argument("--reset", action="store_true", 
                        help="Remove the state file and start a new game")
    parser.add_argument("-n", type=int, default=0,
                        help="Remove the last N entries from history and start from there")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to a state file")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()  # parse CLI arguments
        main(reset=args.reset, remove_last_n=args.n, state_file_path=args.file)
    except KeyboardInterrupt:
        # Any uncaught Ctrl+C lands here – exit without stack trace.
        print("\nGame interrupted by user. Goodbye!")
        import sys
        sys.exit(0)
