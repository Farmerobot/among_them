import pytest
from typing import Dict, List

from among_them.models.action import Action, ActionType
from among_them.models.game_config import GameConfig
from among_them.models.phase import GamePhase
from among_them.models.history import History, initialize_history
from among_them.models.location import Location
from among_them.models.player import Player
from among_them.models.player_role import PlayerRole
from among_them.models.tasks import Task
from among_them.utils.action_utils import get_task_phase_actions, get_vote_actions


# --- Tests for get_task_phase_actions ---

def test_get_task_phase_actions_base(crewmate_player: Player, cafeteria_location: Location,
                                     base_history_entry: List[History], generic_test_players: List[Player],
                                     game_config: GameConfig):
    """Test basic actions available to a crewmate."""
    acting_player = crewmate_player
    current_location = cafeteria_location  # For readability in the test

    # Ensure the base history reflects the player being in the location
    history = base_history_entry  # Assuming base_history_entry sets the initial location correctly

    actions = get_task_phase_actions(acting_player, current_location, 0, history, generic_test_players, game_config)
    action_types = {action.type for action in actions}

    assert ActionType.WAIT in action_types
    # Check move actions based on DOORS from CAFETERIA
    assert any(a.type == ActionType.MOVE and a.target_location == Location.WEAPONS for a in actions)
    assert any(a.type == ActionType.MOVE and a.target_location == Location.ADMIN for a in actions)
    assert any(a.type == ActionType.MOVE and a.target_location == Location.MEDBAY for a in actions)


def test_get_task_phase_actions_report(crewmate_player: Player, cafeteria_location: Location,
                                       base_history_entry: List[History], generic_test_players: List[Player],
                                       dead_player: Player, impostor_player: Player, kill_action: Action,
                                       game_config: GameConfig):
    """Test that REPORT action is available when a dead body is present."""
    acting_player = crewmate_player
    target_dead_player = dead_player  # For readability
    current_location = cafeteria_location
    players_list_with_dead = generic_test_players + [target_dead_player]

    # Start with the base history
    history = base_history_entry.copy()

    # Add kill action using the fixture (Impostor kills Crewmate)
    # We need to modify the kill_action fixture's target to be the dead_player
    specific_kill_action = Action(
        type=ActionType.KILL,
        player_name=impostor_player.name,
        target_player_name=target_dead_player.name,
        spectator=f"{impostor_player.name} killed {target_dead_player.name}"
    )

    # Simulate the state *after* the kill, before the report
    # Ensure impostor is in the location for the kill
    impostor_move_hist = history[-1].copy(
        location=current_location,
        action_taken=Action(type=ActionType.MOVE, player_name=impostor_player.name, target_location=current_location)
    )
    history.append(impostor_move_hist)

    # The kill happens
    kill_history = impostor_move_hist.copy(
        location=current_location,
        action_taken=specific_kill_action,
        player_names_to_play_next=[p for p in impostor_move_hist.player_names_to_play_next if p != impostor_player.name],
        impostor_cooldown=1  # Reset cooldown after kill
    )
    history.append(kill_history)

    # Add a subsequent action by the reporting player to place them in the room *after* the kill
    wait_action = Action(
        type=ActionType.WAIT,
        player_name=acting_player.name,
        spectator=f"{acting_player.name} waited"
    )
    wait_history = kill_history.copy(
        action_taken=wait_action,
        player_names_to_play_next=[p for p in kill_history.player_names_to_play_next if p != acting_player.name]
    )
    history.append(wait_history)

    # Call the function with the constructed history
    actions = get_task_phase_actions(acting_player, current_location, 0, history, players_list_with_dead, game_config)
    report_actions = [a for a in actions if a.type == ActionType.REPORT]
    assert len(report_actions) == 1, "REPORT action should be available"
    assert report_actions[0].target_player_name == target_dead_player.name  # Check it reports the correct player


def test_get_task_phase_actions_no_report_if_body_elsewhere(crewmate_player: Player, cafeteria_location: Location,
                                                           base_history_entry: List[History], generic_test_players: List[Player],
                                                           dead_player: Player, weapons_location: Location,
                                                           game_config: GameConfig):
    """Test REPORT is not available if the body is in a different location."""
    acting_player = crewmate_player
    current_location = cafeteria_location
    other_location = weapons_location  # For readability
    players_list_with_dead = generic_test_players + [dead_player]

    # Start with the base history
    history = base_history_entry.copy()

    # Dead player moves to other_location
    dead_move_action = Action(
        type=ActionType.MOVE,
        player_name=dead_player.name,
        target_location=other_location,
        spectator=f"{dead_player.name} moved to {other_location.value}"
    )
    
    next_players = [p for p in history[-1].player_names_to_play_next if p != dead_player.name]
    
    dead_move_history = history[-1].copy(
        location=other_location,
        action_taken=dead_move_action,
        player_names_to_play_next=next_players
    )
    history.append(dead_move_history)

    # Acting player moves to current_location
    player_move_action = Action(
        type=ActionType.MOVE,
        player_name=acting_player.name,
        target_location=current_location,
        spectator=f"{acting_player.name} moved to {current_location.value}"
    )
    move_history = history[-1].copy(
        location=current_location,
        action_taken=player_move_action,
        player_names_to_play_next=[p for p in history[-1].player_names_to_play_next if p != acting_player.name]
    )
    history.append(move_history)

    # Add history entry simulating dead player at weapons_location
    # We need impostor to kill dead player at weapons_location
    impostor_name = next(p.name for p in generic_test_players if p.role == PlayerRole.IMPOSTOR)
    kill_at_weapons = Action(type=ActionType.KILL, player_name=impostor_name, target_player_name=dead_player.name)
    hist_entry_dead_at_weapons = History(
        player_names_to_play_next=[], phase=GamePhase.TASK, actions_until_phase_ends=10,
        location=weapons_location,  # Location of kill
        impostor_cooldown=1,
        actions_agent_could_take=[], spectators_who_saw=[],
        llm_cot="", llm_response="", token_usage={},
        action_taken=kill_at_weapons, tasks_left_to_do={},
        votes_before_this_discussion_message={}
    )
    history_with_body_elsewhere = history + [hist_entry_dead_at_weapons, move_history]  # Add the kill event

    actions = get_task_phase_actions(acting_player, current_location, 0, history_with_body_elsewhere, players_list_with_dead, game_config)
    report_actions = [a for a in actions if a.type == ActionType.REPORT]
    assert len(report_actions) == 0, "REPORT action should NOT be available"


def test_get_task_phase_actions_task(crewmate_player: Player, cafeteria_location: Location,
                                     base_history_entry: List[History], generic_test_players: List[Player],
                                     task_in_cafeteria: Task, game_config: GameConfig):
    """Test DO_TASK action is available when a task is at the current location."""
    acting_player = crewmate_player
    current_location = cafeteria_location
    task_at_location = task_in_cafeteria  # For readability

    actions = get_task_phase_actions(acting_player, current_location, 0, base_history_entry, generic_test_players, game_config)
    task_actions = [a for a in actions if a.type == ActionType.TASK]

    assert len(task_actions) == 1
    assert task_actions[0].target_task.name == task_at_location.name


def test_get_task_phase_actions_no_task_if_elsewhere(crewmate_player: Player, weapons_location: Location,
                                                     base_history_entry: List[History], generic_test_players: List[Player],
                                                     game_config: GameConfig):
    """Test DO_TASK is not available if the task is elsewhere."""
    acting_player = crewmate_player
    other_location = weapons_location  # For readability

    # Start with the base history (player starts in Cafeteria)
    history = base_history_entry.copy()

    # Add action to move player to other_location
    move_action = Action(
        type=ActionType.MOVE,
        player_name=acting_player.name,
        target_location=other_location,
        spectator=f"{acting_player.name} moved to {other_location.value}"
    )
    move_history = history[-1].copy(
        location=other_location,
        action_taken=move_action,
        player_names_to_play_next=[p for p in history[-1].player_names_to_play_next if p != acting_player.name]
    )
    history.append(move_history)

    actions = get_task_phase_actions(acting_player, other_location, 0, history, generic_test_players, game_config)
    task_actions = [a for a in actions if a.type == ActionType.TASK]
    assert len(task_actions) == 0


def test_get_task_phase_actions_impostor_kill_available(impostor_player: Player, crewmate_player: Player,
                                                      cafeteria_location: Location, base_history_entry: List[History],
                                                      generic_test_players: List[Player], game_config: GameConfig):
    """Test KILL action is available for impostor when cooldown is 0 and target is present."""
    acting_player = impostor_player
    target_player = crewmate_player
    current_location = cafeteria_location

    # Start with the base history
    history = base_history_entry.copy()

    # Target moves to location
    target_move_action = Action(
        type=ActionType.MOVE,
        player_name=target_player.name,
        target_location=current_location,
        spectator=f"{target_player.name} moved to {current_location.value}"
    )
    
    next_players = [p for p in history[-1].player_names_to_play_next if p != target_player.name]
    
    target_move_history = history[-1].copy(
        location=current_location,
        action_taken=target_move_action,
        player_names_to_play_next=next_players
    )
    history.append(target_move_history)
    
    # Impostor moves to the same location
    impostor_move_action = Action(
        type=ActionType.MOVE,
        player_name=acting_player.name,
        target_location=current_location,
        spectator=f"{acting_player.name} moved to {current_location.value}"
    )
    
    next_players = [p for p in target_move_history.player_names_to_play_next if p != acting_player.name]
    
    impostor_move_history = target_move_history.copy(
        location=current_location,
        impostor_cooldown=0,  # Set cooldown to 0
        action_taken=impostor_move_action,
        player_names_to_play_next=next_players
    )
    history.append(impostor_move_history)

    # Check available actions
    actions = get_task_phase_actions(acting_player, current_location, 0, history, generic_test_players, game_config)
    kill_actions = [a for a in actions if a.type == ActionType.KILL]

    assert len(kill_actions) >= 1  # Should be at least one kill action
    # Find the specific action targeting the crewmate
    kill_target_action = next((a for a in kill_actions if a.target_player_name == target_player.name), None)
    assert kill_target_action is not None, f"Kill action targeting {target_player.name} not found"


def test_get_task_phase_actions_impostor_kill_cooldown(impostor_player: Player, crewmate_player: Player,
                                                      cafeteria_location: Location, base_history_entry: List[History],
                                                      generic_test_players: List[Player], game_config: GameConfig):
    """Test KILL action is NOT available for impostor when cooldown > 0."""
    acting_player = impostor_player
    current_location = cafeteria_location
    target_player = crewmate_player

    # Start with the base history
    history = base_history_entry.copy()

    # Target moves to location
    target_move_action = Action(
        type=ActionType.MOVE,
        player_name=target_player.name,
        target_location=current_location,
        spectator=f"{target_player.name} moved to {current_location.value}"
    )
    
    next_players = [p for p in history[-1].player_names_to_play_next if p != target_player.name]
    
    target_move_history = history[-1].copy(
        location=current_location,
        action_taken=target_move_action,
        player_names_to_play_next=next_players
    )
    history.append(target_move_history)
    
    # Impostor moves to the same location with cooldown > 0
    impostor_move_action = Action(
        type=ActionType.MOVE,
        player_name=acting_player.name,
        target_location=current_location,
        spectator=f"{acting_player.name} moved to {current_location.value}"
    )
    
    next_players = [p for p in target_move_history.player_names_to_play_next if p != acting_player.name]
    
    impostor_move_history = target_move_history.copy(
        location=current_location,
        impostor_cooldown=5,  # Set cooldown > 0
        action_taken=impostor_move_action,
        player_names_to_play_next=next_players
    )
    history.append(impostor_move_history)

    # Check available actions
    actions = get_task_phase_actions(acting_player, current_location, 5, history, generic_test_players, game_config)
    kill_actions = [a for a in actions if a.type == ActionType.KILL]
    
    assert len(kill_actions) == 0, "KILL action should not be available when cooldown > 0"


def test_get_task_phase_actions_impostor_kill_target_elsewhere(impostor_player: Player, crewmate_player: Player,
                                                               cafeteria_location: Location, weapons_location: Location,
                                                               base_history_entry: List[History], generic_test_players: List[Player],
                                                               game_config: GameConfig):
    """Test KILL action is NOT available for impostor if target is elsewhere."""
    acting_player = impostor_player
    target_player = crewmate_player
    impostor_location = cafeteria_location
    target_location = weapons_location

    # Start with the base history
    history = base_history_entry.copy()

    # Other players move to other_location
    for player in [p for p in generic_test_players if p != impostor_player]:
        move_action = Action(
            type=ActionType.MOVE,
            player_name=player.name,
            target_location=target_location,
            spectator=f"{player.name} moved to {target_location.value}"
        )
        
        next_players = [p for p in history[-1].player_names_to_play_next if p != player.name]
        
        move_history = history[-1].copy(
            location=target_location,
            action_taken=move_action,
            player_names_to_play_next=next_players
        )
        history.append(move_history)
    
    # Impostor moves to current_location
    impostor_move_action = Action(
        type=ActionType.MOVE,
        player_name=acting_player.name,
        target_location=impostor_location,
        spectator=f"{acting_player.name} moved to {impostor_location.value}"
    )
    
    next_players = [p for p in history[-1].player_names_to_play_next if p != acting_player.name]
    
    impostor_move_history = history[-1].copy(
        location=impostor_location,
        impostor_cooldown=0,  # Cooldown is 0
        action_taken=impostor_move_action,
        player_names_to_play_next=next_players
    )
    history.append(impostor_move_history)

    # Check available actions
    actions = get_task_phase_actions(acting_player, impostor_location, 0, history, generic_test_players, game_config)
    kill_actions = [a for a in actions if a.type == ActionType.KILL]
        
    assert len(kill_actions) == 0, "KILL action should not be available when no target is present"


def test_get_task_phase_actions_impostor_pretend(impostor_player: Player, weapons_location: Location, 
                                                base_history_entry: List[History], generic_test_players: List[Player],
                                                game_config: GameConfig):
    """Test PRETEND_TASK action is available for impostor."""
    acting_player = impostor_player
    
    actions = get_task_phase_actions(acting_player, weapons_location, 0, base_history_entry, generic_test_players, game_config)
    pretend_actions = [a for a in actions if a.type == ActionType.PRETEND]
    
    assert len(pretend_actions) >= 1, "PRETEND_TASK action should be available for impostor"


def test_get_task_phase_actions_crewmate_no_impostor_actions(crewmate_player: Player, weapons_location: Location, 
                                                           base_history_entry: List[History], generic_test_players: List[Player],
                                                           game_config: GameConfig):
    """Test that crewmates cannot KILL or PRETEND_TASK."""
    acting_player = crewmate_player

    actions = get_task_phase_actions(acting_player, weapons_location, 0, base_history_entry, generic_test_players, game_config)
    impostor_action_types = [ActionType.KILL, ActionType.PRETEND]

    for action in actions:
        assert action.type not in impostor_action_types, f"Crewmate should not have {action.type} action available"


# --- Tests for get_vote_actions ---

def test_get_vote_actions(crewmate_player: Player, generic_test_players: List[Player], dead_player: Player):
    """Test available VOTE actions, including voting for alive players and nobody."""
    acting_player = crewmate_player
    players_list_with_dead = generic_test_players
    # Filter out the dead player and the acting player for valid vote targets
    possible_targets = [p.name for p in players_list_with_dead if p.name != acting_player.name]

    actions = get_vote_actions(players_list_with_dead, acting_player)
    action_types = {action.type for action in actions}
    target_players = {action.target_player_name for action in actions if action.type == ActionType.VOTE}

    assert ActionType.VOTE in action_types
    assert ActionType.WAIT not in action_types  # Wait is not a vote option

    # Check if all possible alive targets are present
    for target in possible_targets:
        assert target in target_players

    # Check if voting for 'nobody' is an option
    assert "nobody" in target_players

    # Total vote actions should be number of alive players (excluding self) + 1 (for nobody)
    assert len(actions) == len(possible_targets) + 1


def test_get_vote_actions_only_self(crewmate_player: Player):
    alive_players = [crewmate_player]
    actions = get_vote_actions(alive_players, crewmate_player)
    vote_targets = {a.target_player_name for a in actions if a.type == ActionType.VOTE}

    assert len(actions) == 1
    assert "nobody" in vote_targets
