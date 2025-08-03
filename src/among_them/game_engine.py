import json
import random
from typing import List, Optional
import copy

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
from among_them.utils.phase_utils import count_votes, determine_ejection_result
from among_them.utils.player_utils import (get_last_player_action,
                                           get_next_random_player,
                                           get_players_in_room)
from among_them.utils.end_utils import get_end_game_reason
from among_them.utils.history_utils import initialize_history, create_system_message
from among_them.utils.prompt_utils import reconstruct_environment_prompt_from_history
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

    def __init__(self, game_config: GameConfig = GameConfig(), file_path: str = None):
        player_names = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Hank", "Ivy", "Jack", "Jill", "Katie", "Liam", "Mia", "Nathan", "Olivia", "Pete", "Quinn", "Riley", "Samantha", "Tom", "Uma", "Victor", "Wendy", "Xander", "Yara", "Zack"]
        self.players = [Player(name) for name in player_names[:game_config.num_players]]
        self.check_players_set_impostors(game_config.num_impostors)
        self.history = initialize_history(self.players, game_config)
        self.game_config = game_config
        if file_path is not None:
            self.file_path = file_path

    def get_turn_context(self) -> Optional[tuple[History, List[Action], str, str, List[dict]]]:
        """Gathers all necessary context for the current player's turn.

        This method calculates everything needed for a turn up-front to avoid
        redundant calculations in the step function.

        Returns:
            A tuple containing:
            - An incomplete History object representing the current turn's context.
            - The system prompt for the LLM.
            - The user prompt for the LLM.
            - A list of pre-discussion vote prompts (empty if not in DISCUSS phase).
            Or None if the game is in a state that doesn't require player input.
        """
        alive_player_names = self.history[-1].alive_player_names.copy()
        alive_players = [p for p in self.players if p.name in alive_player_names]
        phase = self.history[-1].phase # valid traces should have system messages that dictate current phase
        actions_until_phase_ends = self.history[-1].actions_until_phase_ends - 1

        # If game ended no further player input is required.
        if get_end_game_reason(self.history, self.players) is not None:
            print("Game over! {}".format(get_end_game_reason(self.history, self.players)))
            return None, None, None, None, None

        current_player, next_players = get_next_random_player(self.history, self.players)
        last_player_action = get_last_player_action(self.history, current_player)
        players_in_room = get_players_in_room(self.history, self.players, current_player)

        # Initialize fields for the History object
        location = Location.CAFETERIA # Default location
        actions_player_can_take = []
        impostor_cooldown = 0
        if last_player_action:
            impostor_cooldown = max(0, last_player_action.impostor_cooldown - 1)

        if phase == GamePhase.TASKS:
            actions_player_can_take = get_task_phase_actions(current_player, self.history, self.players, self.game_config)
            if last_player_action:
                location = last_player_action.location
        elif phase == GamePhase.DISCUSS:
            actions_player_can_take = [Action(type=ActionType.SPEAK, player_name=current_player.name)]
            location = Location.CAFETERIA
        elif phase == GamePhase.VOTING:
            actions_player_can_take = get_vote_actions(self.history, self.players, current_player)
            location = Location.CAFETERIA
        else:
            return None, None, None, None, None

        # Create a placeholder action for the current player
        placeholder_action = Action(player_name=current_player.name, type=ActionType.SPEAK, text="")

        # Create the incomplete History object
        turn_history = History(
            player_names_to_play_next=next_players,
            phase=phase,
            actions_until_phase_ends=actions_until_phase_ends,
            location=location,
            impostor_cooldown=impostor_cooldown,
            actions_agent_could_take=[a.text for a in actions_player_can_take],
            tasks_left_to_do={k: v.copy() for k, v in self.history[-1].tasks_left_to_do.items()},
            alive_player_names=alive_player_names,
            action_taken=placeholder_action, # Placeholder for current player
            spectators_who_saw=[p.name for p in players_in_room], # Represents players in room
            llm_cot="",
            llm_response="",
            token_usage={},
            votes_before_this_discussion_message={},
        )

        # Build new environment-like prompt
        environment_prompt = reconstruct_environment_prompt_from_history(
            current_player, self.history, self.players, self.game_config
        )
        system_prompt = ""  # System context is included in environment_prompt
        user_prompt = environment_prompt

        pre_discussion_vote_prompts = []
        if phase == GamePhase.DISCUSS:
            for player in alive_players:
                fake_voting_actions = get_vote_actions(self.history, self.players, player)
                # Build environment prompt for voting phase
                voting_prompt = reconstruct_environment_prompt_from_history(
                    player, self.history, self.players, self.game_config
                )
                pre_discussion_vote_prompts.append({
                    "player": player,
                    "system_prompt": "",
                    "user_prompt": voting_prompt,
                    "actions": fake_voting_actions
                })

        return turn_history, actions_player_can_take, system_prompt, user_prompt, pre_discussion_vote_prompts

    def step(self, turn_context: History, action_taken: Action, llm_response: str = "", llm_cot: str = "", token_usage: dict = {}, pre_discussion_votes: Optional[dict] = None) -> tuple[bool, Optional[EndGameReason]]:
        """Executes a single player action and updates the game state using pre-calculated context.

        Args:
            turn_context: The incomplete History object representing the current turn's context.
            action_taken: The action to be executed.
            llm_response: The raw response from the LLM.
            llm_cot: The chain of thought from the LLM.
            token_usage: Token usage details.
            pre_discussion_votes: A dictionary of votes collected before the discussion phase.

        Returns:
            True and end game reason if the game is over, False otherwise.
        """
        # Unpack the pre-calculated context from the History object
        current_player_name = turn_context.action_taken.player_name # Get current player name from placeholder action
        current_player = next((p for p in self.players if p.name == current_player_name), None)
        if current_player is None:
            raise ValueError(f"Current player {current_player_name} not found.")

        alive_player_names = [p for p in turn_context.alive_player_names]
        players_in_room_names = turn_context.spectators_who_saw.copy()
        location = turn_context.location

        spectators_who_saw = players_in_room_names
        if action_taken.type == ActionType.REPORT:
            spectators_who_saw = alive_player_names

        if action_taken.type == ActionType.SPEAK:
            action_taken.result = f"[{current_player.name}]: {llm_response}"
            action_taken.spectator = f"[{current_player.name}]: {llm_response}"
            spectators_who_saw = alive_player_names

        tasks_left_to_do = {k: v.copy() for k, v in self.history[-1].tasks_left_to_do.items()}
        if action_taken.type == ActionType.TASK:
            tasks_left_to_do[current_player.name].remove(action_taken.target_task) # type: ignore
        elif action_taken.type == ActionType.MOVE:
            location = action_taken.target_location # type: ignore
            new_players_in_room = get_players_in_room(self.history, self.players, current_player, location)
            spectators_who_saw = list(set(players_in_room_names + [p.name for p in new_players_in_room]))

        cooldown = turn_context.impostor_cooldown
        if action_taken.type == ActionType.KILL:
            cooldown = self.game_config.impostor_cooldown
            alive_player_names.remove(action_taken.target_player_name) # type: ignore

        new_history_item = turn_context.copy(
            spectators_who_saw=spectators_who_saw,
            llm_cot=llm_cot,
            llm_response=llm_response,
            token_usage=token_usage,
            action_taken=action_taken,
            tasks_left_to_do=tasks_left_to_do,
            votes_before_this_discussion_message=pre_discussion_votes or {},
            alive_player_names=alive_player_names,
            location=location,
            impostor_cooldown=cooldown
        )

        self.history.append(new_history_item)

        # Handle phase change
        if action_taken.type == ActionType.REPORT: # if report start discussion
            self.history.append(
                create_system_message(
                    history=self.history,
                    phase=GamePhase.DISCUSS,
                    text="It is discussion phase now. Discuss who to eject from the game.",
                    game_config=self.game_config,
                    alive_player_names=alive_player_names,
                )
            )
        if new_history_item.phase == GamePhase.DISCUSS and new_history_item.actions_until_phase_ends == 0:
            self.history.append(
                create_system_message(
                    history=self.history,
                    phase=GamePhase.VOTING,
                    text="Discussion ended. Voting phase started. Vote who to eject from the game.",
                    game_config=self.game_config,
                    alive_player_names=alive_player_names,
                )
            )
        elif new_history_item.phase == GamePhase.VOTING and new_history_item.actions_until_phase_ends == 0:
            vote_counts, _ = count_votes(self.history)
            ejected_player, action_type = determine_ejection_result(vote_counts)

            # Message with vote outcome (still VOTING phase)
            self.history.append(
                create_system_message(
                    history=self.history,
                    phase=GamePhase.VOTING,
                    text=f"{ejected_player} was voted out.",
                    ejected_player=ejected_player,
                    game_config=self.game_config,
                    alive_player_names=[p for p in alive_player_names if p != ejected_player],
                    no_more_actions=True
                )
            )
            # Remove ejected player from alive list if needed
            if ejected_player != "nobody" and ejected_player in alive_player_names:
                alive_player_names.remove(ejected_player)

            # Message that tasks resume (TASK phase)
            self.history.append(
                create_system_message(
                    history=self.history,
                    phase=GamePhase.TASKS,
                    text="Everyone is in the cafeteria and start from there. It is task phase now.",
                    game_config=self.game_config,
                    alive_player_names=alive_player_names,
                )
            )

        self.save_state()

        end_reason = get_end_game_reason(self.history, self.players)
        if end_reason:
            self.history.append(
                create_system_message(
                    history=self.history,
                    phase=GamePhase.TASKS,
                    text=f"The game ended ({end_reason})",
                    game_config=self.game_config,
                    alive_player_names=self.history[-1].alive_player_names,
                    no_more_actions=True
                )
            )
            self.save_state()
            return True, end_reason

        return False, None

    def save_state(self):
        """Saves the current game state (history, players, config) to a JSON file."""
        try:
            # Ensure the directory exists
            import os
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            
            with open(self.file_path, 'w') as f:
                # Save history, players, and game_config
                json_str = json.dumps((self.history, self.players, self.game_config), indent=2, cls=GameJSONEncoder)
                f.write(json_str)
            
            print(f"Game state saved to: {self.file_path}")
        except Exception as e:
            print(f"ERROR saving game state to {self.file_path}: {e}")
            raise
            
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