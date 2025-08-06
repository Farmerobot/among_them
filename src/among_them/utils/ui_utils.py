from typing import Dict, List, Tuple
from among_them.models.action import Action, ActionType


def prompt_manual_fallback_action(actions: List[Action]) -> Tuple[int, str, str, Dict[str, int]]:
    """Handles manual action selection when AI generation is interrupted."""
    print("\nAI action generation interrupted. Please choose an action manually:")
    for i, action in enumerate(actions):
        print(f"{i + 1}. {action.set_stories().command_perspective}")

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
                print(f"You chose: {selected_action.command_perspective}")
                return (
                    choice_idx,
                    selected_action.command_perspective,
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
                actions[0].command_perspective,
                "",
                {},
            )
