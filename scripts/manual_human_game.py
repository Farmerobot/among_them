import os
import random
from among_them.game_engine import GameEngine
from among_them.game_config import GameConfig
from among_them.models.player_role import PlayerRole
from among_them.models.phase import GamePhase

# ANSI color codes
PROMPT_COLOR = "\033[96m"  # Cyan for prompts
ASSISTANT_COLOR = "\033[92m"  # Green for assistant messages
RESET_COLOR = "\033[0m"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main() -> None:
    game_config = GameConfig(
        num_tasks=2,
        num_players=3,
        num_impostors=1,
        map_size=0,
        num_task_phase_actions_per_player=10,
        num_discuss_phase_actions_per_player=2,
        impostor_cooldown=1
    )
    engine = GameEngine(game_config)
    
    while True:
        clear_screen()
        
        turn_context_history, actions_player_can_take, conversation, pre_discussion_vote_prompts = engine.get_turn_context()
        
        if not turn_context_history:
            print("Game over!")
            break
        
        current_player_name = turn_context_history.action_taken.player_name
        current_player = next(p for p in engine.players if p.name == current_player_name)
        
        is_impostor = current_player.role == PlayerRole.IMPOSTOR
        
        print(f"\n{'='*60}")
        print(f"PLAYING AS: {current_player_name} ({'IMPOSTOR' if is_impostor else 'CREWMATE'})")
        print(f"{'='*60}\n")
        
        # Show conversation history
        print(f"{ASSISTANT_COLOR}--- Conversation History ---{RESET_COLOR}")
        for msg in conversation:
            if msg["role"] == "user":
                print(f"{ASSISTANT_COLOR}\"{msg['content']}\"{RESET_COLOR}\n")
            elif msg["role"] == "assistant":
                print(f"{PROMPT_COLOR}\"{msg['content']}\"{RESET_COLOR}\n")
        
        is_discussion = turn_context_history.phase == GamePhase.DISCUSS
        
        if is_discussion:
            # Discussion phase: just type a message
            if is_impostor:
                print(f"\n{PROMPT_COLOR}Enter your thoughts (chain of thought):{RESET_COLOR}")
                cot = input("> ")
                print(f"\n{PROMPT_COLOR}Enter your message:{RESET_COLOR}")
                response_text = input("> ")
            else:
                response_text = "I don't know who is sus"
                cot = "Auto-playing as crewmate"
                print(f"\n{ASSISTANT_COLOR}[AUTO] Saying: {response_text}{RESET_COLOR}")
            action_taken = actions_player_can_take[0]
        else:
            # Task/Voting phase: show actions and select
            print(f"\n{PROMPT_COLOR}--- Available Actions ---{RESET_COLOR}")
            for idx, action in enumerate(actions_player_can_take):
                print(f"{PROMPT_COLOR}{idx}: {action.set_stories().command_perspective}{RESET_COLOR}")
            
            if is_impostor:
                print(f"\n{PROMPT_COLOR}Enter your thoughts (chain of thought):{RESET_COLOR}")
                cot = input("> ")
                
                while True:
                    print(f"\n{PROMPT_COLOR}Select action number:{RESET_COLOR}")
                    try:
                        action_idx = int(input("> "))
                        if 0 <= action_idx < len(actions_player_can_take):
                            break
                        print(f"Invalid: enter 0-{len(actions_player_can_take)-1}")
                    except ValueError:
                        print("Invalid: enter a number")
            else:
                action_idx = random.randint(0, len(actions_player_can_take) - 1)
                cot = "Auto-playing as crewmate"
                print(f"\n{ASSISTANT_COLOR}[AUTO] Selected action {action_idx}{RESET_COLOR}")
            
            action_taken = actions_player_can_take[action_idx]
            response_text = action_taken.set_stories().command_perspective
        
        # Step the game
        game_over, end_reason = engine.step(
            turn_context_history, 
            action_taken, 
            response_text, 
            cot, 
            {"input_tokens": 0, "output_tokens": 0},
            {}
        )
        
        if game_over:
            clear_screen()
            print(f"\n{'='*60}")
            print(f"GAME OVER: {end_reason}")
            print(f"{'='*60}\n")
            break
        
        if is_impostor:
            input(f"\n{PROMPT_COLOR}Press ENTER to continue to next player...{RESET_COLOR}")
        else:
            input(f"\n{ASSISTANT_COLOR}[AUTO] Press ENTER to continue...{RESET_COLOR}")

if __name__ == "__main__":
    main()
