#!/usr/bin/env python3
"""
Trajectory Analysis Script for Among Them MAPPO Training

Analyzes game trajectories saved during training to understand strategy evolution
across iterations. Loads games using GameEngine format and computes various metrics.

Usage:
    python analyze_trajectories.py --checkpoint_dir <path_to_checkpoints> [options]
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import statistics


# ============================================================================
# Standalone enum/class definitions to avoid importing from among_them
# (which has dependencies on dotenv, etc.)
# ============================================================================

class ActionType(Enum):
    VOTE = "Vote"
    SPEAK = "Speak"
    WAIT = "Wait"
    MOVE = "Move"
    TASK = "Task"
    KILL = "Kill"
    REPORT = "Report"
    PRETEND = "Pretend"


class GamePhase(Enum):
    TASKS = "Tasks"
    DISCUSS = "Discuss"
    VOTING = "Voting"


class PlayerRole(Enum):
    CREWMATE = "Crewmate"
    IMPOSTOR = "Impostor"


class EndGameReason(Enum):
    NO_ACTIONS_LEFT = "No actions left"
    NO_IMPOSTORS_LEFT = "No impostors left"
    ALL_TASKS_DONE = "All tasks done"
    TOO_SMALL_NUMBER_OF_CREWMATES_LEFT = "Too small number of crewmates left"


class Location(Enum):
    CAFETERIA = "Cafeteria"
    REACTOR = "Reactor"
    UPPER_ENGINE = "Upper Engine"
    LOWER_ENGINE = "Lower Engine"
    SECURITY = "Security"
    MEDBAY = "Medbay"
    ELECTRICAL = "Electrical"
    STORAGE = "Storage"
    ADMIN = "Admin"
    COMMUNICATIONS = "Communications"
    O2 = "O2"
    WEAPONS = "Weapons"
    SHIELDS = "Shields"
    NAVIGATION = "Navigation"


def game_object_hook(obj_dict: Dict[str, Any]) -> Any:
    """Custom JSON decoder hook to handle deserialization of game objects."""
    if "__enum__" in obj_dict:
        # Handle enums
        for enum_cls in [Location, PlayerRole, GamePhase, ActionType]:
            try:
                return enum_cls(obj_dict["__enum__"])
            except (ValueError, TypeError):
                continue
        return obj_dict
    
    # Handle enum fields directly in objects - return as SimpleNamespace-like dict
    if "__class__" in obj_dict and "__module__" in obj_dict:
        class_name = obj_dict.pop("__class__")
        obj_dict.pop("__module__")
        
        # Create a simple object with attribute access
        obj = type(class_name, (), obj_dict)()
        return obj
    
    return obj_dict


@dataclass
class GameStats:
    """Statistics for a single game."""
    # Basic info
    iteration: int = 0
    num_turns: int = 0
    winner_role: Optional[PlayerRole] = None
    end_reason: Optional[EndGameReason] = None
    
    # Kill timing
    first_kill_turn: Optional[int] = None
    first_kill_round: Optional[int] = None  # Round = phase cycle
    first_kill_with_cooldown: bool = False  # Did impostor have to wait for cooldown?
    total_kills: int = 0
    kills_per_round: List[int] = field(default_factory=list)
    
    # Action counts
    action_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    impostor_action_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    crewmate_action_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Phases
    num_discussions: int = 0
    num_voting_phases: int = 0
    
    # Voting analysis
    votes_for_impostor: int = 0
    votes_for_crewmate: int = 0
    votes_for_nobody: int = 0
    impostor_ejected: bool = False
    crewmate_ejected: bool = False
    
    # Task completion
    tasks_completed: int = 0
    pretend_tasks: int = 0
    
    # Kill locations
    kill_locations: List[str] = field(default_factory=list)
    
    # Report timing
    turns_until_report: List[int] = field(default_factory=list)  # Turns between kill and report
    
    # Movement patterns
    impostor_moves: int = 0
    crewmate_moves: int = 0


@dataclass
class IterationStats:
    """Aggregated statistics for an iteration."""
    iteration: int
    num_games: int = 0
    
    # Win rates
    impostor_wins: int = 0
    crewmate_wins: int = 0
    
    # End reasons
    end_reasons: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Kill timing aggregates
    first_kill_turns: List[int] = field(default_factory=list)
    first_kill_rounds: List[int] = field(default_factory=list)
    kills_needed_cooldown_wait: int = 0  # Games where first kill needed cooldown wait
    immediate_kills: int = 0  # Kills on turn 1 or 2 without waiting
    
    # Game length
    game_lengths: List[int] = field(default_factory=list)
    
    # Kills
    total_kills: List[int] = field(default_factory=list)
    
    # Discussions
    discussions_per_game: List[int] = field(default_factory=list)
    
    # Voting accuracy
    correct_votes: int = 0
    incorrect_votes: int = 0
    abstain_votes: int = 0
    
    # Tasks
    tasks_completed: List[int] = field(default_factory=list)
    pretend_tasks: List[int] = field(default_factory=list)
    
    # Kill locations
    kill_location_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Report timing
    avg_turns_until_report: List[float] = field(default_factory=list)


def load_game(filepath: str) -> Tuple[List, List, Optional[dict]]:
    """Load a game trajectory from JSON file."""
    with open(filepath, 'r') as f:
        json_str = f.read()
        loaded_data = json.loads(json_str, object_hook=game_object_hook)
        
        if len(loaded_data) == 3:
            history, players, game_config = loaded_data
        elif len(loaded_data) == 2:
            history, players = loaded_data
            game_config = None
        else:
            raise ValueError(f"Invalid save file format in {filepath}")
        
        return history, players, game_config


def get_player_role(players: List, player_name: str) -> Optional[PlayerRole]:
    """Get the role of a player by name."""
    for p in players:
        if getattr(p, 'name', None) == player_name:
            return getattr(p, 'role', None)
    return None


def analyze_game(history: List, players: List, iteration: int) -> GameStats:
    """Analyze a single game and extract statistics."""
    stats = GameStats(iteration=iteration)
    stats.num_turns = len(history)
    
    # Track state
    last_kill_turn = None
    current_round = 1
    last_phase = None
    kills_this_round = 0
    impostor_had_cooldown = False
    
    impostors = [getattr(p, 'name', '') for p in players if getattr(p, 'role', None) == PlayerRole.IMPOSTOR]
    crewmates = [getattr(p, 'name', '') for p in players if getattr(p, 'role', None) == PlayerRole.CREWMATE]
    
    for turn_idx, h in enumerate(history):
        action = getattr(h, 'action_taken', None)
        if action is None:
            continue
            
        player_name = getattr(action, 'player_name', '')
        action_type = getattr(action, 'type', None)
        
        # Skip system messages for action counting
        if player_name == "System":
            # Check for end game
            target_msg = getattr(action, 'target_message', '')
            if target_msg and "The game ended" in target_msg:
                # Parse winner and reason
                if "IMPOSTOR" in target_msg:
                    stats.winner_role = PlayerRole.IMPOSTOR
                elif "CREWMATE" in target_msg:
                    stats.winner_role = PlayerRole.CREWMATE
                
                # Parse end reason
                for reason in EndGameReason:
                    if reason.value in target_msg:
                        stats.end_reason = reason
                        break
            continue
        
        # Track phase transitions for rounds
        h_phase = getattr(h, 'phase', None)
        if last_phase == GamePhase.VOTING and h_phase == GamePhase.TASKS:
            current_round += 1
            if kills_this_round > 0:
                stats.kills_per_round.append(kills_this_round)
            kills_this_round = 0
        last_phase = h_phase
        
        # Get player role for this action
        player_role = get_player_role(players, player_name)
        
        # Count actions
        if action_type:
            action_name = action_type.name if hasattr(action_type, 'name') else str(action_type)
            stats.action_counts[action_name] += 1
            
            if player_role == PlayerRole.IMPOSTOR:
                stats.impostor_action_counts[action_name] += 1
            elif player_role == PlayerRole.CREWMATE:
                stats.crewmate_action_counts[action_name] += 1
        
        # Analyze specific actions
        if action_type == ActionType.KILL:
            stats.total_kills += 1
            kills_this_round += 1
            
            if stats.first_kill_turn is None:
                stats.first_kill_turn = turn_idx
                stats.first_kill_round = current_round
                # Check if impostor waited for cooldown (had non-zero cooldown previously)
                stats.first_kill_with_cooldown = impostor_had_cooldown
            
            last_kill_turn = turn_idx
            
            # Track kill location
            h_location = getattr(h, 'location', None)
            if h_location:
                loc_name = h_location.name if hasattr(h_location, 'name') else str(h_location)
                stats.kill_locations.append(loc_name)
        
        elif action_type == ActionType.REPORT:
            stats.num_discussions += 1
            if last_kill_turn is not None:
                stats.turns_until_report.append(turn_idx - last_kill_turn)
        
        elif action_type == ActionType.VOTE:
            stats.num_voting_phases += 1
            target = getattr(action, 'target_player_name', '')
            
            if target == "nobody":
                stats.votes_for_nobody += 1
            elif target in impostors:
                stats.votes_for_impostor += 1
            elif target in crewmates:
                stats.votes_for_crewmate += 1
        
        elif action_type == ActionType.TASK:
            stats.tasks_completed += 1
        
        elif action_type == ActionType.PRETEND:
            stats.pretend_tasks += 1
        
        elif action_type == ActionType.MOVE:
            if player_role == PlayerRole.IMPOSTOR:
                stats.impostor_moves += 1
            else:
                stats.crewmate_moves += 1
        
        # Track cooldown state
        h_cooldown = getattr(h, 'impostor_cooldown', 0)
        if player_role == PlayerRole.IMPOSTOR and h_cooldown > 0:
            impostor_had_cooldown = True
    
    # Final round kills
    if kills_this_round > 0:
        stats.kills_per_round.append(kills_this_round)
    
    return stats


def aggregate_iteration_stats(games: List[GameStats], iteration: int) -> IterationStats:
    """Aggregate statistics across games in an iteration."""
    stats = IterationStats(iteration=iteration, num_games=len(games))
    
    for game in games:
        # Win tracking - infer from end_reason if winner_role not set
        winner = game.winner_role
        if winner is None and game.end_reason:
            # Impostor wins: too few crewmates left, no actions left (usually impostor advantage)
            if game.end_reason in [EndGameReason.TOO_SMALL_NUMBER_OF_CREWMATES_LEFT]:
                winner = PlayerRole.IMPOSTOR
            # Crewmate wins: all tasks done, no impostors left
            elif game.end_reason in [EndGameReason.ALL_TASKS_DONE, EndGameReason.NO_IMPOSTORS_LEFT]:
                winner = PlayerRole.CREWMATE
        
        if winner == PlayerRole.IMPOSTOR:
            stats.impostor_wins += 1
        elif winner == PlayerRole.CREWMATE:
            stats.crewmate_wins += 1
        
        # End reasons
        if game.end_reason:
            stats.end_reasons[game.end_reason.name] += 1
        
        # Kill timing
        if game.first_kill_turn is not None:
            stats.first_kill_turns.append(game.first_kill_turn)
        if game.first_kill_round is not None:
            stats.first_kill_rounds.append(game.first_kill_round)
        
        if game.first_kill_with_cooldown:
            stats.kills_needed_cooldown_wait += 1
        elif game.first_kill_turn is not None and game.first_kill_turn <= 2:
            stats.immediate_kills += 1
        
        # Game length
        stats.game_lengths.append(game.num_turns)
        
        # Kills
        stats.total_kills.append(game.total_kills)
        
        # Discussions
        stats.discussions_per_game.append(game.num_discussions)
        
        # Voting
        stats.correct_votes += game.votes_for_impostor
        stats.incorrect_votes += game.votes_for_crewmate
        stats.abstain_votes += game.votes_for_nobody
        
        # Tasks
        stats.tasks_completed.append(game.tasks_completed)
        stats.pretend_tasks.append(game.pretend_tasks)
        
        # Kill locations
        for loc in game.kill_locations:
            stats.kill_location_counts[loc] += 1
        
        # Report timing
        if game.turns_until_report:
            stats.avg_turns_until_report.append(statistics.mean(game.turns_until_report))
    
    return stats


def find_trajectory_dirs(checkpoint_dir: str) -> Dict[int, str]:
    """Find all iteration directories with trajectories."""
    iterations = {}
    
    for item in os.listdir(checkpoint_dir):
        if item.startswith("iteration_"):
            try:
                iteration_num = int(item.split("_")[1])
                traj_path = os.path.join(checkpoint_dir, item, "trajectories")
                if os.path.isdir(traj_path):
                    iterations[iteration_num] = traj_path
            except (ValueError, IndexError):
                continue
    
    return dict(sorted(iterations.items()))


def load_iteration_games(traj_dir: str, max_games: Optional[int] = None) -> List[Tuple[List, List]]:
    """Load all games from a trajectory directory."""
    games = []
    
    files = sorted([f for f in os.listdir(traj_dir) if f.endswith('.json')])
    if max_games:
        files = files[:max_games]
    
    for filename in files:
        filepath = os.path.join(traj_dir, filename)
        try:
            history, players, _ = load_game(filepath)
            games.append((history, players))
        except Exception as e:
            print(f"  Warning: Failed to load {filename}: {e}")
    
    return games


def safe_mean(values: List[float], default: float = 0.0) -> float:
    """Calculate mean, returning default if list is empty."""
    return statistics.mean(values) if values else default


def safe_stdev(values: List[float], default: float = 0.0) -> float:
    """Calculate stdev, returning default if list has < 2 items."""
    return statistics.stdev(values) if len(values) >= 2 else default


def print_iteration_summary(stats: IterationStats):
    """Print summary for an iteration."""
    print(f"\n{'='*60}")
    print(f"ITERATION {stats.iteration} ({stats.num_games} games)")
    print('='*60)
    
    # Win rates
    total_decided = stats.impostor_wins + stats.crewmate_wins
    if total_decided > 0:
        imp_rate = stats.impostor_wins / total_decided * 100
        crew_rate = stats.crewmate_wins / total_decided * 100
        print(f"\n📊 Win Rates:")
        print(f"   Impostor: {stats.impostor_wins}/{total_decided} ({imp_rate:.1f}%)")
        print(f"   Crewmate: {stats.crewmate_wins}/{total_decided} ({crew_rate:.1f}%)")
    
    # End reasons
    if stats.end_reasons:
        print(f"\n🏁 End Reasons:")
        for reason, count in sorted(stats.end_reasons.items(), key=lambda x: -x[1]):
            pct = count / stats.num_games * 100
            print(f"   {reason}: {count} ({pct:.1f}%)")
    
    # Kill timing
    print(f"\n🔪 Kill Timing:")
    if stats.first_kill_turns:
        print(f"   First kill turn: {safe_mean(stats.first_kill_turns):.1f} ± {safe_stdev(stats.first_kill_turns):.1f}")
    if stats.first_kill_rounds:
        print(f"   First kill round: {safe_mean(stats.first_kill_rounds):.1f} ± {safe_stdev(stats.first_kill_rounds):.1f}")
    if stats.num_games > 0:
        imm_pct = stats.immediate_kills / stats.num_games * 100
        wait_pct = stats.kills_needed_cooldown_wait / stats.num_games * 100
        print(f"   Immediate kills (turn ≤2): {stats.immediate_kills} ({imm_pct:.1f}%)")
        print(f"   Needed cooldown wait: {stats.kills_needed_cooldown_wait} ({wait_pct:.1f}%)")
    
    # Game length
    if stats.game_lengths:
        print(f"\n📏 Game Length:")
        print(f"   Turns: {safe_mean(stats.game_lengths):.1f} ± {safe_stdev(stats.game_lengths):.1f}")
        print(f"   Range: {min(stats.game_lengths)} - {max(stats.game_lengths)}")
    
    # Kills
    if stats.total_kills:
        print(f"\n💀 Kills per Game:")
        print(f"   Average: {safe_mean(stats.total_kills):.1f} ± {safe_stdev(stats.total_kills):.1f}")
    
    # Discussions
    if stats.discussions_per_game:
        print(f"\n💬 Discussions per Game:")
        print(f"   Average: {safe_mean(stats.discussions_per_game):.1f} ± {safe_stdev(stats.discussions_per_game):.1f}")
    
    # Voting accuracy
    total_votes = stats.correct_votes + stats.incorrect_votes + stats.abstain_votes
    if total_votes > 0:
        print(f"\n🗳️  Voting Accuracy (Crewmate perspective):")
        print(f"   Correct (voted impostor): {stats.correct_votes} ({stats.correct_votes/total_votes*100:.1f}%)")
        print(f"   Incorrect (voted crewmate): {stats.incorrect_votes} ({stats.incorrect_votes/total_votes*100:.1f}%)")
        print(f"   Abstain (voted nobody): {stats.abstain_votes} ({stats.abstain_votes/total_votes*100:.1f}%)")
    
    # Tasks
    if stats.tasks_completed:
        print(f"\n✅ Tasks per Game:")
        print(f"   Completed: {safe_mean(stats.tasks_completed):.1f} ± {safe_stdev(stats.tasks_completed):.1f}")
    if stats.pretend_tasks:
        print(f"   Pretend (impostor): {safe_mean(stats.pretend_tasks):.1f} ± {safe_stdev(stats.pretend_tasks):.1f}")
    
    # Kill locations
    if stats.kill_location_counts:
        print(f"\n📍 Kill Locations (top 5):")
        sorted_locs = sorted(stats.kill_location_counts.items(), key=lambda x: -x[1])[:5]
        total_kills = sum(stats.kill_location_counts.values())
        for loc, count in sorted_locs:
            print(f"   {loc}: {count} ({count/total_kills*100:.1f}%)")
    
    # Report timing
    if stats.avg_turns_until_report:
        print(f"\n⏱️  Report Timing:")
        print(f"   Avg turns after kill: {safe_mean(stats.avg_turns_until_report):.1f}")


def print_evolution_summary(all_stats: List[IterationStats]):
    """Print summary of how metrics evolved across iterations."""
    if len(all_stats) < 2:
        print("\n⚠️  Need at least 2 iterations to show evolution")
        return
    
    print(f"\n{'='*60}")
    print("STRATEGY EVOLUTION ACROSS ITERATIONS")
    print('='*60)
    
    # Extract key metrics for comparison
    iterations = [s.iteration for s in all_stats]
    
    # Win rates
    imp_win_rates = []
    for s in all_stats:
        total = s.impostor_wins + s.crewmate_wins
        imp_win_rates.append(s.impostor_wins / total * 100 if total > 0 else 0)
    
    print(f"\n📈 Impostor Win Rate Evolution:")
    for i, (it, rate) in enumerate(zip(iterations, imp_win_rates)):
        bar = "█" * int(rate / 5)
        delta = ""
        if i > 0:
            diff = rate - imp_win_rates[i-1]
            delta = f" ({'+' if diff > 0 else ''}{diff:.1f}%)"
        print(f"   Iter {it:3d}: {rate:5.1f}% {bar}{delta}")
    
    # First kill timing
    print(f"\n📈 First Kill Timing Evolution:")
    for i, s in enumerate(all_stats):
        if s.first_kill_turns:
            avg = safe_mean(s.first_kill_turns)
            delta = ""
            if i > 0 and all_stats[i-1].first_kill_turns:
                prev_avg = safe_mean(all_stats[i-1].first_kill_turns)
                diff = avg - prev_avg
                delta = f" ({'+' if diff > 0 else ''}{diff:.1f})"
            print(f"   Iter {s.iteration:3d}: Turn {avg:5.1f}{delta}")
    
    # Immediate kills trend
    print(f"\n📈 Immediate Kills (turn ≤2) Rate:")
    for i, s in enumerate(all_stats):
        if s.num_games > 0:
            rate = s.immediate_kills / s.num_games * 100
            bar = "█" * int(rate / 5)
            delta = ""
            if i > 0 and all_stats[i-1].num_games > 0:
                prev_rate = all_stats[i-1].immediate_kills / all_stats[i-1].num_games * 100
                diff = rate - prev_rate
                delta = f" ({'+' if diff > 0 else ''}{diff:.1f}%)"
            print(f"   Iter {s.iteration:3d}: {rate:5.1f}% {bar}{delta}")
    
    # Game length
    print(f"\n📈 Average Game Length:")
    for i, s in enumerate(all_stats):
        if s.game_lengths:
            avg = safe_mean(s.game_lengths)
            delta = ""
            if i > 0 and all_stats[i-1].game_lengths:
                prev_avg = safe_mean(all_stats[i-1].game_lengths)
                diff = avg - prev_avg
                delta = f" ({'+' if diff > 0 else ''}{diff:.1f})"
            print(f"   Iter {s.iteration:3d}: {avg:5.1f} turns{delta}")
    
    # Voting accuracy trend
    print(f"\n📈 Crewmate Voting Accuracy:")
    for i, s in enumerate(all_stats):
        total = s.correct_votes + s.incorrect_votes + s.abstain_votes
        if total > 0:
            accuracy = s.correct_votes / total * 100
            bar = "█" * int(accuracy / 5)
            delta = ""
            if i > 0:
                prev_total = all_stats[i-1].correct_votes + all_stats[i-1].incorrect_votes + all_stats[i-1].abstain_votes
                if prev_total > 0:
                    prev_acc = all_stats[i-1].correct_votes / prev_total * 100
                    diff = accuracy - prev_acc
                    delta = f" ({'+' if diff > 0 else ''}{diff:.1f}%)"
            print(f"   Iter {s.iteration:3d}: {accuracy:5.1f}% {bar}{delta}")
    
    # Summary insights
    print(f"\n{'='*60}")
    print("KEY INSIGHTS")
    print('='*60)
    
    first = all_stats[0]
    last = all_stats[-1]
    
    # Win rate change
    first_total = first.impostor_wins + first.crewmate_wins
    last_total = last.impostor_wins + last.crewmate_wins
    if first_total > 0 and last_total > 0:
        first_imp = first.impostor_wins / first_total * 100
        last_imp = last.impostor_wins / last_total * 100
        change = last_imp - first_imp
        direction = "increased" if change > 0 else "decreased"
        print(f"\n🎯 Impostor win rate {direction} by {abs(change):.1f}% ({first_imp:.1f}% → {last_imp:.1f}%)")
    
    # Kill timing change
    if first.first_kill_turns and last.first_kill_turns:
        first_avg = safe_mean(first.first_kill_turns)
        last_avg = safe_mean(last.first_kill_turns)
        change = last_avg - first_avg
        direction = "later" if change > 0 else "earlier"
        print(f"🔪 First kills happening {direction} ({first_avg:.1f} → {last_avg:.1f} turns)")
    
    # Immediate kills change
    if first.num_games > 0 and last.num_games > 0:
        first_imm = first.immediate_kills / first.num_games * 100
        last_imm = last.immediate_kills / last.num_games * 100
        change = last_imm - first_imm
        if abs(change) > 5:
            direction = "more" if change > 0 else "fewer"
            print(f"⚡ {direction.capitalize()} immediate kills ({first_imm:.1f}% → {last_imm:.1f}%)")
    
    # Discussions change
    if first.discussions_per_game and last.discussions_per_game:
        first_avg = safe_mean(first.discussions_per_game)
        last_avg = safe_mean(last.discussions_per_game)
        change = last_avg - first_avg
        if abs(change) > 0.5:
            direction = "more" if change > 0 else "fewer"
            print(f"💬 {direction.capitalize()} discussions per game ({first_avg:.1f} → {last_avg:.1f})")


def export_csv(all_stats: List[IterationStats], output_path: str):
    """Export statistics to CSV for further analysis."""
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'iteration', 'num_games',
            'impostor_wins', 'crewmate_wins', 'impostor_win_rate',
            'avg_first_kill_turn', 'avg_first_kill_round',
            'immediate_kills', 'cooldown_waits',
            'avg_game_length', 'avg_kills',
            'avg_discussions', 'voting_accuracy',
            'avg_tasks_completed', 'avg_pretend_tasks'
        ])
        
        # Data rows
        for s in all_stats:
            total = s.impostor_wins + s.crewmate_wins
            imp_rate = s.impostor_wins / total if total > 0 else 0
            
            total_votes = s.correct_votes + s.incorrect_votes + s.abstain_votes
            vote_acc = s.correct_votes / total_votes if total_votes > 0 else 0
            
            writer.writerow([
                s.iteration, s.num_games,
                s.impostor_wins, s.crewmate_wins, f"{imp_rate:.4f}",
                f"{safe_mean(s.first_kill_turns):.2f}" if s.first_kill_turns else "",
                f"{safe_mean(s.first_kill_rounds):.2f}" if s.first_kill_rounds else "",
                s.immediate_kills, s.kills_needed_cooldown_wait,
                f"{safe_mean(s.game_lengths):.2f}" if s.game_lengths else "",
                f"{safe_mean(s.total_kills):.2f}" if s.total_kills else "",
                f"{safe_mean(s.discussions_per_game):.2f}" if s.discussions_per_game else "",
                f"{vote_acc:.4f}",
                f"{safe_mean(s.tasks_completed):.2f}" if s.tasks_completed else "",
                f"{safe_mean(s.pretend_tasks):.2f}" if s.pretend_tasks else ""
            ])
    
    print(f"\n📁 Exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Among Them MAPPO training trajectories")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to checkpoint directory containing iteration_N folders")
    parser.add_argument("--max_games", type=int, default=None,
                        help="Maximum games to load per iteration (for faster analysis)")
    parser.add_argument("--iterations", type=str, default=None,
                        help="Comma-separated list of iterations to analyze (e.g., '1,5,10,20')")
    parser.add_argument("--export_csv", type=str, default=None,
                        help="Path to export CSV summary")
    parser.add_argument("--quiet", action="store_true",
                        help="Only show evolution summary, not per-iteration details")
    
    args = parser.parse_args()
    
    # Find trajectory directories
    print(f"🔍 Scanning: {args.checkpoint_dir}")
    iterations = find_trajectory_dirs(args.checkpoint_dir)
    
    if not iterations:
        print("❌ No trajectory directories found!")
        print("   Expected structure: checkpoint_dir/iteration_N/trajectories/*.json")
        return 1
    
    print(f"   Found {len(iterations)} iterations with trajectories")
    
    # Filter iterations if specified
    if args.iterations:
        selected = set(int(x.strip()) for x in args.iterations.split(","))
        iterations = {k: v for k, v in iterations.items() if k in selected}
        print(f"   Analyzing {len(iterations)} selected iterations")
    
    # Analyze each iteration
    all_stats = []
    
    for iteration, traj_dir in iterations.items():
        print(f"\n📂 Loading iteration {iteration}...")
        games_data = load_iteration_games(traj_dir, args.max_games)
        print(f"   Loaded {len(games_data)} games")
        
        if not games_data:
            continue
        
        # Analyze games
        game_stats = []
        for history, players in games_data:
            try:
                stats = analyze_game(history, players, iteration)
                game_stats.append(stats)
            except Exception as e:
                print(f"   Warning: Failed to analyze game: {e}")
        
        # Aggregate
        iter_stats = aggregate_iteration_stats(game_stats, iteration)
        all_stats.append(iter_stats)
        
        # Print details unless quiet mode
        if not args.quiet:
            print_iteration_summary(iter_stats)
    
    # Print evolution summary
    print_evolution_summary(all_stats)
    
    # Export if requested
    if args.export_csv:
        export_csv(all_stats, args.export_csv)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
