from typing import Dict, List, Tuple
from among_them.models.action import Action, ActionType

def prompt_human_action(
    actions: List[Action], history_str: str, player_name: str
) -> Tuple[int, str, str, Dict[str, int]]:
    """Handle action selection for a human-controlled player."""
    print(history_str)
    if actions[0].type == ActionType.SPEAK:
        return 0, input("Your message to others:"), "", {}
    else:
        action_prompt = "\n".join(
            [f"{i}: {action.text}" for i, action in enumerate(actions)]
        )
        prompt = "========================================\n"
        prompt += f"Your turn {player_name}: Choose an action\n{action_prompt}\n\n"
        prompt += "========================================\n"
        print(prompt)
        while True:
            try:
                chosen_action = int(input("Choose action (enter the number):"))
                if 0 <= chosen_action < len(actions):
                    return (
                        chosen_action,
                        actions[chosen_action].text,
                        "",
                        {},
                    )
                else:
                    print(f"Please enter a number between 0 and {len(actions) - 1}")
            except ValueError:
                print("Invalid input. Please enter a number.")

def prompt_manual_fallback_action(actions: List[Action]) -> Tuple[int, str, str, Dict[str, int]]:
    """Handles manual action selection when AI generation is interrupted."""
    print("\nAI action generation interrupted. Please choose an action manually:")
    for i, action in enumerate(actions):
        print(f"{i + 1}. {action.text}")

    if actions[0].type == ActionType.SPEAK:
        try:
            return 0, input("Your message to others:"), "", {}
        except EOFError:  # Handle Ctrl+D or similar EOF signals gracefully
            print("\nInput stream closed. Defaulting to \"Who did it?\"")
            return 0, "Who did it?", "", {}
    while True:
        try:
            choice = input(f"Enter the number of your choice (1-{len(actions)}):")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(actions):
                selected_action = actions[choice_idx]
                print(f"You chose: {selected_action.text}")
                return (
                    choice_idx,
                    selected_action.text,
                    "",
                    {},
                )
            else:
                print("Invalid choice. Please enter a number within the range.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except EOFError:  # Handle Ctrl+D or similar EOF signals gracefully
            print("\nInput stream closed. Defaulting to first action.")
            return (
                0,
                actions[0].text,
                "",
                {},
            )
