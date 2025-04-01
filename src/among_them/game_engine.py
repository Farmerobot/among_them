from typing import List

from among_them.consts import STATE_FILE
from among_them.game_jsonencoder import GameJSONEncoder, game_object_hook
from among_them.models import action
from among_them.models.history import History, get_action_history_str, initialize_history
from among_them.models.location import Location
from among_them.models.phase import GamePhase
from among_them.utils.phase_utils import get_phase_and_when_it_ends
from among_them.models.action_type import ActionType
from among_them.models.player_role import PlayerRole
from among_them.utils.player_utils import get_alive_players, get_next_random_player, get_players_in_room, get_last_player_action
from among_them.utils.action_utils import get_task_phase_actions, get_vote_actions
from among_them.models.player import Player
import json
import random
from among_them.models.end_game import get_end_game_reason
from among_them.models.action import Action
from typing import List


class GameEngine:
    """Manages
    - game logic,
    - including player actions,
    - game state transitions,and win conditions.
    """

    history: List[History] = []
    players: List[Player] = []
    file_path: str = STATE_FILE

    def __init__(self, players: List[Player], impostor_count: int = 1) -> None:
        self.players = players
        self.check_players_set_impostors(impostor_count)
        self.history = initialize_history(self.players)


    def perform_step(self) -> bool:
        """Executes a single step in the game, which is a player action.

        Only when player successfully completes the action, the result will be saved to history. 
        Otherwise, nothing will change and next perform_step() call will start from the same place.

        Returns:
            True and end game reason if the game is over or in MAIN_MENU stage, False otherwise
        """
        alive_players = get_alive_players(self.history, self.players)
        phase, actions_until_phase_ends = get_phase_and_when_it_ends(self.history, alive_players)
        end_game_reason = get_end_game_reason(self.history, self.players)
        if phase == GamePhase.MAIN_MENU or end_game_reason is not None:
            return True, end_game_reason

        current_player, next_players = get_next_random_player(self.history, alive_players)

        last_player_action = get_last_player_action(self.history, current_player)
        players_in_room = get_players_in_room(self.history, self.players, last_player_action.location)
        if phase == GamePhase.TASK:
            location = last_player_action.location
            actions_player_can_take = get_task_phase_actions(current_player, last_player_action.location, last_player_action.impostor_cooldown, self.history, self.players)
            spectators_who_saw = [p.name for p in players_in_room]
        elif phase == GamePhase.DISCUSS:
            # Set some variables
            location = Location.CAFETERIA
            actions_player_can_take = [Action(type=ActionType.SPEAK, player_name=current_player.name)]
            spectators_who_saw = [p.name for p in alive_players]
        elif phase == GamePhase.VOTE:
            location = Location.CAFETERIA
            actions_player_can_take = get_vote_actions(alive_players, current_player)
            spectators_who_saw = [p.name for p in alive_players]

        # Force a new vote BEFORE each discussion message.
        # It does not affect the game logic. It is just extra data.
        votes_before_this_discussion_message = {}
        if phase == GamePhase.DISCUSS:
            for player in alive_players:
                retry_count = 0
                while True:
                    retry_count += 1
                    try:
                        fake_voting_actions_player_can_take = get_vote_actions(alive_players, player)
                        fake_history_str = get_action_history_str(self.history, player, players_in_room=alive_players, alive_players=alive_players, location=Location.CAFETERIA, phase=GamePhase.VOTE)
                        fake_action_taken_idx, _, _, _ = player.prompt_action(fake_voting_actions_player_can_take, fake_history_str)
                        fake_action_taken = fake_voting_actions_player_can_take[fake_action_taken_idx]
                        votes_before_this_discussion_message[player.name] = fake_action_taken.target_player_name
                        print(f"Discussion phase fake voting: {player.name} voted for {fake_action_taken.target_player_name}")
                        break
                    except Exception as e:
                        if "LLM did" in str(e):
                            print(f"Error: {e}")
                            print(f"Model failed to vote. Retry count: {retry_count}")
                            continue
                        else:
                            raise e
            print(f"Votes before this discussion message: {votes_before_this_discussion_message}")

        history_str = get_action_history_str(self.history, current_player, players_in_room, alive_players, location, phase)
        action_taken_idx, response, cot, token_usage = current_player.prompt_action(actions_player_can_take, history_str)
        action_taken = actions_player_can_take[action_taken_idx]

        if action_taken.type == ActionType.REPORT:
            spectators_who_saw = [p.name for p in alive_players]

        agent_sees = f"{action_taken.result}"
        spectator_sees = action_taken.spectator
        if action_taken.type == ActionType.SPEAK:
            agent_sees = f"[{current_player.name}]: {response}"
            spectator_sees = f"[{current_player.name}]: {response}"

        tasks_left_to_do = self.history[-1].tasks_left_to_do
        if action_taken.type == ActionType.TASK:
            tasks_left_to_do[current_player.name].remove(action_taken.target_task)
        elif action_taken.type == ActionType.MOVE:
            location = action_taken.target_location
            new_players_in_room = get_players_in_room(self.history, self.players, location)
            spectators_who_saw = list(set([p.name for p in players_in_room] + [p.name for p in new_players_in_room]))

        new_history_item = History(
            player_names_to_play_next=next_players,
            phase=phase,
            actions_until_phase_ends=actions_until_phase_ends,
            location=location,
            impostor_cooldown=max(0, last_player_action.impostor_cooldown-1),
            actions_agent_could_take=[a.text for a in actions_player_can_take],
            spectators_who_saw=spectators_who_saw,
            llm_cot=cot,
            llm_response=response,
            token_usage=token_usage,
            action_taken=action_taken,
            tasks_left_to_do=tasks_left_to_do,
            votes_before_this_discussion_message=votes_before_this_discussion_message
        )

        self.history.append(new_history_item)
        self.save_state()
        return False, None

    def save_state(self):
        with open(self.file_path, 'w') as f:
            json_str = json.dumps((self.history, self.players), indent=2, cls=GameJSONEncoder)
            f.write(json_str)
            
    def load_state(self):
        with open(self.file_path, 'r') as f:
            json_str = f.read()
            self.history, self.players = json.loads(json_str, object_hook=game_object_hook)
            return True
    
    def check_players_set_impostors(
        self, impostor_count: int = 1
    ):
        """Checks if the players have already been assigned roles, checks balance and assigns roles (crewmate or impostor).

        Args:
            impostor_count: Expected number of impostors

        Returns:
            List of players with roles assigned

        Raises:
            ValueError: Inconsistent configuration
        """
        if len(self.players) < 3:
            raise ValueError("Minimum number of players is 3.")

        if impostor_count >= len(self.players) or impostor_count <= 0:
            raise ValueError("Invalid number of impostors")

        # Count existing impostors
        existing_impostors = sum(
            1 for player in self.players if player.role == PlayerRole.IMPOSTOR
        )

        # Assign impostors randomly, only if needed
        impostors_to_assign = impostor_count - existing_impostors
        while impostors_to_assign > 0:
            available_players = [p for p in self.players if p.role != PlayerRole.IMPOSTOR]
            if not available_players:
                break  # No more players to assign as impostors
            chosen_player = random.choice(available_players)
            chosen_player.role = PlayerRole.IMPOSTOR
            impostors_to_assign -= 1

        # Check for imbalanced team sizes AFTER role assignment
        crewmates_count = len(self.players) - impostor_count
        if impostor_count >= crewmates_count:
            raise ValueError(
                "Number of impostors cannot be greater than "
                "or equal to the number of crewmates."
            )
        random.shuffle(self.players)
        print("Players:", {p.name: p.role.value for p in self.players})