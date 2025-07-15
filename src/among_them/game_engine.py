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
from among_them.utils.llm_utils import create_llm_prompts
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

    def get_turn_context(self) -> Optional[dict]:
        """Gathers all necessary context for the current player's turn.

        This method calculates everything needed for a turn up-front to avoid
        redundant calculations in the step function.

        Returns:
            A dictionary with all the information needed for a player to make a decision,
            or None if the game is in a state that doesn't require player input.
        """
        alive_player_names = self.history[-1].alive_player_names.copy()
        alive_players = [p for p in self.players if p.name in alive_player_names]
        phase, actions_until_phase_ends = get_phase_and_when_it_ends(self.history, self.game_config, self.players)

        if phase in [GamePhase.GAME_END, GamePhase.VOTE_RESULTS]:
            return None

        current_player, next_players = get_next_random_player(self.history, self.players)
        last_player_action = get_last_player_action(self.history, current_player)
        players_in_room = get_players_in_room(self.history, self.players, current_player)

        context = {
            "current_player": current_player,
            "next_players": next_players,
            "phase": phase,
            "actions_until_phase_ends": actions_until_phase_ends,
            "last_player_action": last_player_action,
            "alive_players": alive_players,
            "players_in_room": players_in_room,
        }

        if phase == GamePhase.TASK:
            actions_player_can_take = get_task_phase_actions(current_player, self.history, self.players, self.game_config)
            context["location"] = last_player_action.location
        elif phase == GamePhase.DISCUSS:
            actions_player_can_take = [Action(type=ActionType.SPEAK, player_name=current_player.name)]
            context["location"] = Location.CAFETERIA
            # Add prompts for pre-discussion voting
            pre_discussion_vote_prompts = []
            for player in alive_players:
                fake_voting_actions = get_vote_actions(self.history, self.players, player)
                fake_history_str = get_action_history_str(self.history, self.players, player, self.game_config, GamePhase.VOTING)
                system_prompt, user_prompt = create_llm_prompts(fake_voting_actions, fake_history_str)
                pre_discussion_vote_prompts.append({
                    "player": player,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "actions": fake_voting_actions
                })
            context["pre_discussion_vote_prompts"] = pre_discussion_vote_prompts
        elif phase == GamePhase.VOTING:
            actions_player_can_take = get_vote_actions(self.history, self.players, current_player)
            context["location"] = Location.CAFETERIA
        else:
            return None

        history_str = get_action_history_str(self.history, self.players, current_player, self.game_config)
        system_prompt, user_prompt = create_llm_prompts(actions_player_can_take, history_str)

        context["actions_player_can_take"] = actions_player_can_take
        context["history_str"] = history_str
        context["system_prompt"] = system_prompt
        context["user_prompt"] = user_prompt

        return context

    def step(self, turn_context: dict, action_taken: Action, llm_response: str = "", llm_cot: str = "", token_usage: dict = None, pre_discussion_votes: Optional[dict] = None) -> tuple[bool, Optional[EndGameReason]]:
        """Executes a single player action and updates the game state using pre-calculated context.

        Args:
            turn_context: The dictionary of pre-calculated turn data from get_turn_context.
            action_taken: The action to be executed.
            llm_response: The raw response from the LLM.
            llm_cot: The chain of thought from the LLM.
            token_usage: Token usage details.
            pre_discussion_votes: A dictionary of votes collected before the discussion phase.

        Returns:
            True and end game reason if the game is over, False otherwise.
        """
        # Unpack the pre-calculated context
        current_player = turn_context["current_player"]
        alive_player_names = [p.name for p in turn_context["alive_players"]]
        players_in_room = turn_context["players_in_room"]
        location = turn_context["location"]

        spectators_who_saw = [p.name for p in players_in_room]
        if action_taken.type == ActionType.REPORT:
            spectators_who_saw = alive_player_names

        if action_taken.type == ActionType.SPEAK:
            action_taken.result = f"[{current_player.name}]: {llm_response}"
            action_taken.spectator = f"[{current_player.name}]: {llm_response}"

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
            player_names_to_play_next=turn_context["next_players"],
            phase=turn_context["phase"],
            actions_until_phase_ends=turn_context["actions_until_phase_ends"],
            location=location,
            impostor_cooldown=max(0, turn_context["last_player_action"].impostor_cooldown-1),
            actions_agent_could_take=[a.text for a in turn_context["actions_player_can_take"]],
            spectators_who_saw=spectators_who_saw,
            llm_cot=llm_cot,
            llm_response=llm_response,
            token_usage=token_usage,
            action_taken=action_taken,
            tasks_left_to_do=tasks_left_to_do,
            votes_before_this_discussion_message=pre_discussion_votes or {},
            alive_player_names=alive_player_names
        )

        self.history.append(new_history_item)
        self.save_state()

        end_reason = get_end_game_reason(self.history, self.players)
        if end_reason:
            self.history.append(end_game_history(self.history, self.players, self.game_config, end_reason))
            self.save_state()
            return True, end_reason

        return False, None

    def handle_automatic_transitions(self) -> tuple[bool, Optional[EndGameReason]]:
        """Handles automatic game state transitions that don't require player input.

        This includes processing vote results and checking for game-end conditions.

        Returns:
            True and end game reason if the game is over, False otherwise.
        """
        alive_player_names = self.history[-1].alive_player_names.copy()
        alive_players = [p for p in self.players if p.name in alive_player_names]
        phase, _ = get_phase_and_when_it_ends(self.history, self.game_config, self.players)

        if phase == GamePhase.GAME_END:
            reason = get_end_game_reason(self.history, self.players)
            if reason is None:
                raise ValueError("Game ended with no reason")
            if self.history[-1].phase != GamePhase.GAME_END:
                self.history.append(end_game_history(self.history, self.players, self.game_config, reason))
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