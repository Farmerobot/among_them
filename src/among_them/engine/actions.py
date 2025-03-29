
from among_them.models.action import Action, ActionType
from among_them.models.game_phase import GamePhase
from among_them.models.location import DOORS, Location
from among_them.models.history import History
from among_them.models.tasks import get_impostor_pretend_tasks_at_location
from among_them.players.player import Player
from among_them.models.player_role import PlayerRole
from among_them.engine.player_filters import get_alive_players, get_dead_players, get_players_in_room
from typing import List

def get_task_phase_actions(player: Player, location: Location, cooldown: int, history: List[History], players: List[Player]) -> list[Action]:
        """Creates available actions based on the circumstances.

        Returns:
            A list of actions
        """
        actions = []

        # actions for WAIT
        actions.append(Action(type=ActionType.WAIT, player_name=player.name))

        # action for REPORT
        dead_players_in_room_names = [name for name, dead_location in get_dead_players(history, players).items() if dead_location == location]
        for dead_name in dead_players_in_room_names:
            actions.append(
                Action(
                    type=ActionType.REPORT,
                    player_name=player.name,
                    target_player_name=dead_name,
                )
            )

        # actions for MOVE
        for room in DOORS[location]:
            actions.append(
                Action(type=ActionType.MOVE, player_name=player.name, target_location=room)
            )

        # actions for tasks TASK
        for task in history[-1].tasks_left_to_do[player.name]:
            if task.location == location and not task.completed:
                actions.append(
                    Action(
                        type=ActionType.TASK, player_name=player.name, target_task=task
                    )
                )

        # actions for impostors KILL
        if player.role == PlayerRole.IMPOSTOR and cooldown == 0:
            targets = get_players_in_room(history, players, location)
            for target in targets:
                if target.name != player.name:
                    actions.append(
                        Action(type=ActionType.KILL, player_name=player.name, target_player_name=target.name)
                    )

        # actions for impostros PRETEND
        if player.role == PlayerRole.IMPOSTOR:
            for task in get_impostor_pretend_tasks_at_location(location):
                actions.append(
                    Action(type=ActionType.PRETEND, player_name=player.name, target_task=task)
                )

        return actions



def get_vote_actions(history: List[History], players: List[Player], player: Player) -> list[Action]:
    """Creates voting options.

    Returns:
        A list of game actions
    """
    actions = []
    actions.append(
        Action(type=ActionType.VOTE, player_name=player.name, target_player_name="nobody")
    )
    for other_player in get_alive_players(history, players):
        if other_player != player:
            actions.append(
                Action(
                    type=ActionType.VOTE, player_name=player.name, target_player_name=other_player.name
                )
            )
    return actions


def get_action_history_str(history: List[History], player: Player, players_in_room: List[Player], alive_players: List[Player], location: Location, phase: GamePhase) -> str:
    """Returns all actions seen by agent and actions that agent saw/spectated in history in order."""

    # Agent description
    history_str = "<player_info>\n"
    history_str += f"You are {player.name} in a text-based social deduction game.\n You ({player.name}) are assigned the role of {player.role.value}.\n"
    other_impostors = [p for p in alive_players if p.role == PlayerRole.IMPOSTOR and p.name != player.name]
    if player.role == PlayerRole.IMPOSTOR:
        other_crewmates = [p for p in alive_players if p.role == PlayerRole.CREWMATE and p.name != player.name]
        if other_impostors:
            if len(other_impostors) == 1:
                history_str += f"{other_impostors[0].name} is the only other impostor. Work with them to vote out or kill all crewmates.\n"
            else:
                history_str += ", ".join([p.name for p in other_impostors]) + " are other impostors. Work with them to vote out or kill all crewmates.\n"
        else:
            history_str += "You are the only impostor. Vote out or kill all crewmates.\n"
        if other_crewmates:
            history_str += ", ".join([p.name for p in other_crewmates]) + " are crewmates. Vote out or kill them to win.\n"
    else:
        other_players = [p.name for p in alive_players if p.name != player.name]
        history_str += f"You are a crewmate. Vote out all impostors to win. There is {len(other_impostors)} impostor(s) among ({', '.join(other_players)})\n"
    
    # Player tasks
    player_tasks = history[-1].tasks_left_to_do[player.name]
    history_str += f"You ({player.name}) have {len(player_tasks)} tasks left:\n"
    for task in player_tasks:
        history_str += f"{task}\n"
    history_str += "</player_info>\n"

    # History
    history_str += "\n<history>\n"
    for history_item in history:
        if history_item.acted_by_player == player.name:
            cot_without_think_tags = "At this point, you thought to yourself:" + history_item.llm_cot.replace("<think>", "\n").replace("</think>", "\n")
            history_str += cot_without_think_tags + "\n"
            history_str += history_item.action_result_agent_sees + "\n"
        else:
            if player.name in history_item.spectators_who_saw:
                if history_item.action_type == ActionType.SPEAK:
                    history_str += "Discussion: " + history_item.action_result_spectator_sees + "\n"
                else:
                    history_str += "You saw: " + history_item.action_result_spectator_sees + "\n"
    history_str += "</history>\n"
    
    # Player location and phase
    other_players = [p.name for p in players_in_room if p.name != player.name]
    if phase == GamePhase.TASK:
        history_str += (f"You ({player.name}) are currently alone in {location.value}" if len(other_players) == 0 else f"You ({player.name}) are currently with {', '.join(other_players)}") + " in " + location.value + "\n"
        history_str += "Shhh... You can not speak now. It is against the rules\n"
    elif phase == GamePhase.VOTE:
        history_str += "It is voting phase now. Vote out all impostors.\n"
    else:
        history_str += "It is discussion phase now. Respond to the crewmates.\n"
    return history_str
    