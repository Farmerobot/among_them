import json
import random
from typing import List, Optional

from among_them.config import STATE_FILE
from among_them.game_jsonencoder import GameJSONEncoder, game_object_hook
from among_them.models.action import Action
from among_them.models.action_type import ActionType
from among_them.models.end_game import EndGameReason
from among_them.models.history import History
from among_them.models.location import Location
from among_them.models.phase import GamePhase
from among_them.models.player import Player
from among_them.models.player_role import PlayerRole
from among_them.utils.action_utils import (get_task_phase_actions,
                                           get_vote_actions)
from among_them.utils.phase_utils import count_votes, determine_ejection_result, get_phase_and_when_it_ends
from among_them.utils.player_utils import (get_last_player_action,
                                           get_next_random_player,
                                           get_players_in_room)
from among_them.utils.end_utils import get_end_game_reason
from among_them.utils.history_utils import initialize_history, end_game_history, create_vote_history_entry, get_action_history_str
from among_them.game_config import GameConfig

class GameEngine:
    """Manages
    - game logic,
    - including player actions,
    - game state transitions,and win conditions.
    """

    history: List[History] = []
    players: List[Player] = []
    game_config: GameConfig = GameConfig()
    file_path: str = STATE_FILE

    def __init__(self, game_config: GameConfig = GameConfig()):
        player_names = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Hank", "Ivy", "Jack", "Jill", "Katie", "Liam", "Mia", "Nathan", "Olivia", "Pete", "Quinn", "Riley", "Samantha", "Tom", "Uma", "Victor", "Wendy", "Xander", "Yara", "Zack"]
        self.players = [Player(name) for name in player_names[:game_config.num_players]]
        self.check_players_set_impostors(game_config.num_impostors)
        self.history = initialize_history(self.players, game_config)
        self.game_config = game_config


    def perform_step(self) -> tuple[bool, Optional[EndGameReason]]:
        """Executes a single step in the game, which is a player action.

        Only when player successfully completes the action, the result will be saved to history. 
        Otherwise, nothing will change and next perform_step() call will start from the same place.

        Returns:
            True and end game reason if the game is over or in GAME_END phase, False otherwise
        """
        alive_player_names = self.history[-1].alive_player_names.copy()
        alive_players = [p for p in self.players if p.name in alive_player_names]
        phase, actions_until_phase_ends = get_phase_and_when_it_ends(self.history, self.game_config, self.players)

        # Handle phases where new history item is added
        if phase == GamePhase.GAME_END:
            reason = get_end_game_reason(self.history, self.players)
            if reason is None: 
                raise ValueError("Game ended with no reason")
            if self.history[-1].phase != GamePhase.GAME_END:
                self.history.append(end_game_history(self.history, self.players, self.game_config, reason)) # type: ignore
                self.save_state()
            return True, reason
        elif phase == GamePhase.VOTE_RESULTS:
            vote_counts, _ = count_votes(self.history)
            ejected_player, action_type = determine_ejection_result(vote_counts)

            self.history.append(create_vote_history_entry(
                history=self.history,
                alive_players=alive_players,
                ejected_player=ejected_player,
                action_result=f"{ejected_player} was voted out.",
                action_type=action_type,
                game_config=self.game_config
            ))
            self.save_state()
            return False, None

        current_player, next_players = get_next_random_player(self.history, self.players)
        last_player_action = get_last_player_action(self.history, current_player)
        players_in_room = get_players_in_room(self.history, self.players, current_player)
        if phase == GamePhase.TASK:
            location = last_player_action.location
            actions_player_can_take = get_task_phase_actions(current_player, self.history, self.players, self.game_config)
            spectators_who_saw = [p.name for p in players_in_room]
        elif phase == GamePhase.DISCUSS:
            # Set some variables
            location = Location.CAFETERIA
            actions_player_can_take = [Action(type=ActionType.SPEAK, player_name=current_player.name)]
            spectators_who_saw = alive_player_names
        elif phase == GamePhase.VOTING:
            location = Location.CAFETERIA
            actions_player_can_take = get_vote_actions(self.history, self.players, current_player)
            spectators_who_saw = alive_player_names

        # Force a new vote BEFORE each discussion message.
        # It does not affect the game logic. It is just extra data.
        votes_before_this_discussion_message: dict[str, dict] = {}
        if phase == GamePhase.DISCUSS:
            for player in alive_players:
                retry_count = 0
                while True:
                    retry_count += 1
                    try:
                        fake_voting_actions_player_can_take = get_vote_actions(self.history, self.players, player)
                        fake_history_str = get_action_history_str(self.history, self.players, player, self.game_config, GamePhase.VOTING)
                        fake_action_taken_idx, _, fake_action_chain_of_thought, _ = player.prompt_action(fake_voting_actions_player_can_take, fake_history_str)
                        fake_action_taken = fake_voting_actions_player_can_take[fake_action_taken_idx]
                        votes_before_this_discussion_message[player.name] = {"voted_player": fake_action_taken.target_player_name, "chain_of_thought": fake_action_chain_of_thought}
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

        history_str = get_action_history_str(self.history, self.players, current_player, self.game_config)
        if phase == GamePhase.DISCUSS:
            retry_count = 0
            while True:
                try:
                    action_taken_idx, response, cot, token_usage = current_player.prompt_action(actions_player_can_take, history_str)
                    action_taken = actions_player_can_take[action_taken_idx]
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count > 3:
                        raise e
                    if "LLM did" in str(e):
                        print(f"Error: {e}")
                        print(f"Model failed to respond. Retry count: {retry_count}")
                        continue
                    else:
                        raise e
        else:
            action_taken_idx, response, cot, token_usage = current_player.prompt_action(actions_player_can_take, history_str)
            action_taken = actions_player_can_take[action_taken_idx]

        if action_taken.type == ActionType.REPORT:
            spectators_who_saw = alive_player_names

        if action_taken.type == ActionType.SPEAK:
            action_taken.result = f"[{current_player.name}]: {response}"
            action_taken.spectator = f"[{current_player.name}]: {response}"

        tasks_left_to_do = self.history[-1].tasks_left_to_do.copy()
        if action_taken.type == ActionType.TASK:
            tasks_left_to_do[current_player.name].remove(action_taken.target_task) # type: ignore
        elif action_taken.type == ActionType.MOVE:
            location = action_taken.target_location # type: ignore
            new_players_in_room = get_players_in_room(self.history, self.players, current_player, location)
            spectators_who_saw = list(set([p.name for p in players_in_room] + [p.name for p in new_players_in_room]))

        if action_taken.type == ActionType.KILL:
            alive_player_names.remove(action_taken.target_player_name) # type: ignore

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
            votes_before_this_discussion_message=votes_before_this_discussion_message,
            alive_player_names=alive_player_names
        )

        self.history.append(new_history_item)
        self.save_state()
        return False, None

    def save_state(self):
        """Saves the current game state (history, players, config) to a JSON file."""
        with open(self.file_path, 'w') as f:
            # Save history, players, and game_config
            json_str = json.dumps((self.history, self.players, self.game_config), indent=2, cls=GameJSONEncoder)
            f.write(json_str)
            
    def load_state(self):
        """Loads the game state (history, players, config) from a JSON file."""
        with open(self.file_path, 'r') as f:
            json_str = f.read()
            # Load history, players, and game_config
            loaded_data = json.loads(json_str, object_hook=game_object_hook)
            if len(loaded_data) == 3:
                self.history, self.players, self.game_config = loaded_data
            elif len(loaded_data) == 2: # Handle old save files without config
                # print("\033[38;5;244mLoading old save file format. Using default GameConfig.\033[0m")
                self.history, self.players = loaded_data
                self.game_config = GameConfig() # Initialize with defaults
            else:
                raise ValueError("Invalid save file format")
                
            # Add alive_player_names to each history item if missing
            dead_players = []
            for h in self.history:
                if h.action_taken.type == ActionType.KILL:
                    dead_players.append(h.action_taken.target_player_name)
                if not hasattr(h, 'alive_player_names'):
                    h.alive_player_names = [p.name for p in self.players if p.name not in dead_players]
                    
            return True
        return False

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