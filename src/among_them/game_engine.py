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
        current_player, next_players = get_next_random_player(self.history, self.players)
        alive_players = get_alive_players(self.history, self.players)
        phase, actions_until_phase_ends = get_phase_and_actions_until_phase_ends(self.history, len(alive_players))
        if phase == GamePhase.MAIN_MENU:
            return True, get_end_game_reason(self.history, self.players)


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
        
        history_str = ""
        history_str += f"You are {current_player.name} in a text-based social deduction game.\n You ({current_player.name}) are assigned the role of {current_player.role.value}.\n"
        if current_player.role == PlayerRole.IMPOSTOR:
            other_impostors = [p for p in alive_players if p.role == PlayerRole.IMPOSTOR and p.name != current_player.name]
            if other_impostors:
                history_str += ", ".join([p.name for p in other_impostors]) + " are other impostors. Work with them to eliminate all crewmates.\n"
            else:
                history_str += "You are the only impostor. Eliminate all crewmates.\n"
        history_str += get_action_history_str(self.history, current_player)
        player_tasks = self.history[-1].tasks_left_to_do[current_player.name]
        history_str += f"You ({current_player.name}) have {len(player_tasks)} tasks left:\n"
        for task in player_tasks:
            history_str += f"{task}\n"
        other_players = [p for p in players_in_room if p.name != current_player.name]
        history_str += (f"You ({current_player.name}) are currently alone" if len(other_players) == 0 else f"You ({current_player.name}) are currently with {', '.join([p.name for p in other_players])}") + " in " + location.value + "\n"
        if phase != GamePhase.DISCUSS:
            history_str += "Shhh... You can not speak now. It is against the rules\n"
        action_taken_idx, response, cot, token_usage = current_player.prompt_action(actions_player_can_take, history_str)
        action_taken = actions_player_can_take[action_taken_idx]

        killed_or_reported_player_name = None
        if action_taken.type == ActionType.KILL or action_taken.type == ActionType.REPORT:
            killed_or_reported_player_name = action_taken.target_player_name

        agent_sees = action_taken.result
        spectator_sees = action_taken.spectator
        if action_taken.type == ActionType.SPEAK:
            agent_sees = f"<think>{cot}</think>\n[{current_player.name}]: {response}"
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
            pickle.dump((self.history, self.players), f)
            
    def load_state(self):
        with open(self.file_path, 'rb') as f:
            self.history, self.players = pickle.load(f)
            return True
