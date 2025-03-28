from typing import List

from among_them.players.player import Player


class HumanPlayer(Player):
    def prompt_action(self, actions: List[str], history_str: str) -> int:
        prompt = ""
        action_prompt = "\n".join(
            [f"{i}: {action}" for i, action in enumerate(actions)]
        )
        task_str = "\n".join([str(task) for task in self.state.tasks])
        prompt += f"Here are your tasks: \n{task_str}\n\n"
        prompt += "========================================\n"
        prompt += f"Your turn {self.name}: Choose an action\n{action_prompt}\n\n"
        prompt += "========================================\n"
        print(prompt)

        while True:
            try:
                chosen_action = int(input("Choose action (enter the number): "))
                if 0 <= chosen_action < len(actions):
                    return chosen_action
                else:
                    print(f"Please enter a number between 0 and {len(actions) - 1}")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def prompt_discussion(self, history_str: str) -> str:
        print(history_str)
        print(f"=============/n{self.name} it's your turn to discuss:\n=============")
        answer = input("")
        return answer

    def prompt_vote(self, voting_actions: List[str], history_str: str) -> int:
        voting_prompt = "\n".join(
            [f"{i}: {action}" for i, action in enumerate(voting_actions)]
        )
        print(history_str)
        print(voting_prompt)
        answer = input("Choose player to banish: ")
        if answer.isdigit():
            return int(answer)
        else:
            return self.prompt_vote(voting_actions)
