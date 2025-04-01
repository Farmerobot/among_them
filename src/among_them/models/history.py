from typing import List, Dict
from among_them.consts import IMPOSTOR_COOLDOWN
from among_them.models.phase import GamePhase
from among_them.models.location import Location
from among_them.models.action_type import ActionType
from among_them.models.player_role import PlayerRole
from among_them.models.tasks import Task, get_crewmate_tasks, get_impostor_tasks
from among_them.models.player import Player
from among_them.models.action import Action

class History:
    """Item in the sequence of actions that have occurred in the game."""

    def __init__(
        self,
        player_names_to_play_next: List[str],
        phase: GamePhase,
        actions_until_phase_ends: int,
        location: Location, # This is after the action
        impostor_cooldown: int,
        actions_agent_could_take: List[str],
        spectators_who_saw: List[str],
        llm_cot: str,
        llm_response: str,
        token_usage: Dict[str, int],
        action_taken: Action,
        tasks_left_to_do: Dict[str, List[Task]],
        votes_before_this_discussion_message: Dict[str, str],
    ):
        self.player_names_to_play_next: List[str] = player_names_to_play_next
        self.phase: GamePhase = phase
        self.actions_until_phase_ends: int = actions_until_phase_ends
        self.location: Location = location
        self.impostor_cooldown: int = impostor_cooldown

        self.actions_agent_could_take: List[str] = actions_agent_could_take
        self.llm_cot: str = llm_cot
        self.llm_response: str = llm_response
        self.token_usage: dict = token_usage
        self.action_taken: Action = action_taken

        self.spectators_who_saw: List[str] = spectators_who_saw

        self.tasks_left_to_do: Dict[str, List[Task]] = tasks_left_to_do
        self.votes_before_this_discussion_message: Dict[str, str] = votes_before_this_discussion_message

    def __repr__(self) -> str:
        # Format the token usage for better readability
        token_usage_str = "\n".join([f"{key}: {value}" for key, value in self.token_usage.items()])
        
        # Format task information with indentation
        tasks_str = ""
        for player, tasks in self.tasks_left_to_do.items():
            if player == self.action_taken.player_name:
                task_list = "\n    - ".join(str(task) for task in tasks)
                tasks_str += f"\n    - {player}:\n    - {task_list}"
        if not tasks_str:
            tasks_str = " None"
            
        # Format the lists for better readability
        players_next = ", ".join(self.player_names_to_play_next)
        spectators = ", ".join(self.spectators_who_saw)
            
        # Build the full representation with clear sections
        return f"""{self.phase}({self.actions_until_phase_ends}) [\033[33m{self.action_taken.spectator}\033[0m{"- "+self.action_taken.target_player_name if self.action_taken.type == ActionType.KILL else ""}({self.impostor_cooldown}), {self.location}] next: {players_next}||{spectators} saw it"""


def initialize_history(players: List[Player]) -> List[History]:
    tasks = {}
    for player in players:
        tasks[player.name] = get_impostor_tasks() if player.role == PlayerRole.IMPOSTOR else get_crewmate_tasks()

    first_entry = History(
        player_names_to_play_next = [p.name for p in players],
        phase = GamePhase.MAIN_MENU,
        actions_until_phase_ends = 0,
        location = Location.CAFETERIA,
        impostor_cooldown = IMPOSTOR_COOLDOWN,
        actions_agent_could_take = [],
        spectators_who_saw = [player.name for player in players],
        llm_cot = "",
        llm_response = "",
        token_usage = {},
        action_taken = Action(ActionType.WAIT, "System", spectator="The game started"),
        tasks_left_to_do = tasks,
        votes_before_this_discussion_message = {}
    )
    return [first_entry]
    
def create_vote_history_entry(
    history: List[History],
    alive_players: List[Player],
    ejected_player: str,
    action_result: str,
    action_type: ActionType,
    votes: Dict[str, str]
) -> History:
    """
    Create a new history entry for a vote action. 
    
    Args:
        votes: dictionary of player name to voted for. This is for votes_before_this_discussion_message variable so that it includes votes after discussion.
    
    Returns:
        A new history entry for the vote action.
    """
    return History(
        player_names_to_play_next = [], # handled automatically
        phase = GamePhase.MAIN_MENU,
        actions_until_phase_ends = 0,
        location = Location.CAFETERIA,
        impostor_cooldown = IMPOSTOR_COOLDOWN,
        actions_agent_could_take = [],
        spectators_who_saw = [p.name for p in alive_players],
        llm_cot = "",
        llm_response = "",
        token_usage = {},
        action_taken = Action(type=action_type, player_name="System", target_player_name=ejected_player, spectator=action_result),
        tasks_left_to_do = history[-1].tasks_left_to_do,
        votes_before_this_discussion_message = votes
    )

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
        if history_item.action_taken.player_name == player.name:
            cot_without_think_tags = "At this point, you thought to yourself:" + history_item.llm_cot.replace("<think>", "\n").replace("</think>", "\n")
            history_str += cot_without_think_tags + "\nAnd after thinking\n"
            history_str += history_item.action_taken.result + "\n"
        else:
            if player.name in history_item.spectators_who_saw:
                if history_item.action_taken.type == ActionType.SPEAK:
                    history_str += "Discussion: " + history_item.action_taken.spectator + "\n"
                else:
                    history_str += "You saw: " + history_item.action_taken.spectator + "\n"
    history_str += "</history>\n"
    
    # Player location and phase
    other_players = [p.name for p in players_in_room if p.name != player.name]
    if phase == GamePhase.TASK:
        history_str += (f"You ({player.name}) are currently alone in {location.value}" if len(other_players) == 0 else f"You ({player.name}) are currently with {', '.join(other_players)}") + " in " + location.value + "\n"
        history_str += "Shhh... You can not speak now. It is against the rules\n"
    elif phase == GamePhase.VOTE:
        history_str += "It is voting phase now.\n"
    else:
        history_str += "It is discussion phase now. You can speak now. Respond to the crewmates.\n"
    return history_str
