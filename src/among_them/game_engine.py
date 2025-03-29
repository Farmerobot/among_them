from typing import List

from among_them.consts import STATE_FILE
from among_them.models.history import History
from among_them.models.location import Location
from among_them.models.game_phase import GamePhase
from among_them.models.action_type import ActionType
from among_them.models.player_role import PlayerRole
from among_them.players.player import Player
from among_them.engine.check_and_get_players import check_and_get_players
from among_them.engine.initialize_history import initialize_history
import jsonpickle
import pickle
from among_them.engine.player_filters import get_alive_players, get_next_random_player, get_players_in_room
from among_them.engine.phase_filters import get_phase_and_actions_until_phase_ends
from among_them.models.end_game import get_end_game_reason
from among_them.engine.player_filters import get_last_player_action
from among_them.engine.actions import get_action_history_str, get_task_phase_actions, get_vote_actions
from among_them.models.action import Action


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
        self.players = check_and_get_players(players, impostor_count)
        self.history = initialize_history(self.players)


    def perform_step(self) -> bool:
        """Executes a single step in the game, which is a player action.

        Only when player successfully completes the action, the result will be saved to history. 
        Otherwise, nothing will change and next perform_step() call will start from the same place.

        Returns:
            True and end game reason if the game is over or in MAIN_MENU stage, False otherwise
        """
        alive_players = get_alive_players(self.history, self.players)
        phase, actions_until_phase_ends = get_phase_and_actions_until_phase_ends(self.history, len(alive_players))
        end_game_reason = get_end_game_reason(self.history, self.players)
        if phase == GamePhase.MAIN_MENU or end_game_reason is not None:
            return True, end_game_reason

        current_player, next_players = get_next_random_player(self.history, self.players)

        last_player_action = get_last_player_action(self.history, current_player)
        players_in_room = get_players_in_room(self.history, self.players, last_player_action.location)
        if phase == GamePhase.TASK:
            location = last_player_action.location
            actions_player_can_take = get_task_phase_actions(current_player, last_player_action.location, last_player_action.impostor_cooldown, self.history, self.players)
            spectators_who_saw = [p.name for p in players_in_room]
        elif phase == GamePhase.DISCUSS:
            location = Location.CAFETERIA
            actions_player_can_take = [Action(type=ActionType.SPEAK, player_name=current_player.name)]
            spectators_who_saw = [p.name for p in alive_players]
        elif phase == GamePhase.VOTE:
            location = Location.CAFETERIA
            actions_player_can_take = get_vote_actions(self.history, self.players, current_player)
            spectators_who_saw = [p.name for p in alive_players]
        
        history_str = get_action_history_str(self.history, current_player, players_in_room, alive_players, location, phase)
        action_taken_idx, response, cot, token_usage = current_player.prompt_action(actions_player_can_take, history_str)
        action_taken = actions_player_can_take[action_taken_idx]

        killed_or_reported_player_name = None
        if action_taken.type == ActionType.KILL or action_taken.type == ActionType.REPORT:
            killed_or_reported_player_name = action_taken.target_player_name
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

        new_history_item = History(
            player_names_to_play_next=next_players,
            acted_by_player=current_player.name,
            phase=phase,
            actions_until_phase_ends=actions_until_phase_ends,
            location=location,
            impostor_cooldown=max(0, last_player_action.impostor_cooldown-1),
            actions_agent_could_take=[a.text for a in actions_player_can_take],
            spectators_who_saw=spectators_who_saw,
            action_type=action_taken.type,
            llm_cot=cot,
            llm_response=response,
            token_usage=token_usage,
            killed_or_reported_player_name=killed_or_reported_player_name,
            action_result_agent_sees=agent_sees,
            action_result_spectator_sees=spectator_sees,
            tasks_left_to_do=tasks_left_to_do,
        )
        self.history.append(new_history_item)
        self.save_state()
        return False, None

    def save_state(self):
        with open(self.file_path, 'wb') as f:
            # pickle.dump((self.history, self.players), f)
            obj_str = jsonpickle.encode((self.history, self.players), indent=2)
            f.write(obj_str.encode('utf-8'))
            
    def load_state(self):
        with open(self.file_path, 'rb') as f:
            # self.history, self.players = pickle.load(f)
            self.history, self.players = jsonpickle.decode(f.read())
            return True
