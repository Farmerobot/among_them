import argparse
import tempfile
import os
import json
import statistics
from pathlib import Path
from typing import List, Optional, Dict, Any
from collections import defaultdict
from dataclasses import dataclass

from among_them.game_config import GameConfig
from among_them.game_engine import GameEngine


def generate_long_cot(action, current_player, history, players, target_tokens: int = 3000) -> str:
    """Generate a long chain of thought to simulate realistic LLM reasoning."""
    
    return "<think>Okay, let me think through what Charlie should do here. Charlie is a Crewmate. Looking at the current situation, Charlie is considering:  Start the coffee maker in the cafeteria As a Crewmate, my priorities are completing tasks and identifying the Impostors. I should pay attention to suspicious behavior from other players. Completing tasks helps the crew win, so I should focus on that when safe. If I see a body, reporting it immediately is crucial to start a discussion. During discussions, I need to share what I've observed and listen to others' accounts. Sticking with groups can provide safety, but it also slows down task completion. I should try to track player movements and notice if anyone is acting suspiciously. Looking at the recent game history, there have been 1 turns so far. Let me review what other players have been doing and where they've been moving. Analyzing player behaviors can help identify patterns and suspicious activities. The current phase dynamics suggest certain actions are more beneficial than others. Risk assessment: I need to weigh the potential benefits against the risks of exposure. Timing is crucial - acting too early or too late could compromise my position. I should consider the overall game state and how many players are still alive. Strategic positioning is important for both offensive and defensive plays. Thinking about the map layout, different rooms offer different strategic advantages. High-traffic areas like Cafeteria provide cover but also more witnesses. Isolated rooms like Electrical or Storage offer opportunities but are also risky. The current task distribution affects where players are likely to be located. I need to anticipate where other players will move next based on their task lists. Emergency meetings can be called at any time, so I need to be prepared with explanations. The voting dynamics in this game depend on trust and evidence presentation. Building alliances or appearing trustworthy is crucial for long-term survival. Let me reconsider all available options carefully. Each action has consequences. Option analysis:  Start the coffee maker in the cafeteria - this could work because it aligns with my role objectives. However, I need to think about how this action will be perceived by others. If someone is watching my movements, they might find this behavior suspicious or normal. The key is to maintain consistency with typical player behavior patterns. I should also consider what information this action reveals to other players. Information control is critical - revealing too much can be dangerous. On the other hand, appearing too secretive might also raise red flags. Balance is important - I need to seem engaged but not overeager. Let me also think about the timing of this action relative to other events. Has anyone else performed similar actions recently? This provides social cover. What are the alternative actions and why might they be better or worse? Each alternative has trade-offs in terms of risk, reward, and information. I need to maximize my strategic advantage while minimizing exposure risk. The current game state suggests this action is relatively safe to execute. Let me reconsider all available options carefully. Each action has consequences. Option analysis:  Start the coffee maker in the cafeteria - this could work because it aligns with my role objectives. However, I need to think about how this action will be perceived by others. If someone is watching my movements, they might find this behavior suspicious or normal. The key is to maintain consistency with typical player behavior patterns. I should also consider what information this action reveals to other players. Information control is critical - revealing too much can be dangerous. On the other hand, appearing too secretive might also raise red flags. Balance is important - I need to seem engaged but not overeager. Let me also think about the timing of this action relative to other events. Has anyone else performed similar actions recently? This provides social cover. What are the alternative actions and why might they be better or worse? Each alternative has trade-offs in terms of risk, reward, and information. I need to maximize my strategic advantage while minimizing exposure risk. The current game state suggests this action is relatively safe to execute. Let me reconsider all available options carefully. Each action has consequences. Option analysis:  Start the coffee maker in the cafeteria - this could work because it aligns with my role objectives. However, I need to think about how this action will be perceived by others. If someone is watching my movements, they might find this behavior suspicious or normal. The key is to maintain consistency with typical player behavior patterns. I should also consider what information this action reveals to other players. Information control is critical - revealing too much can be dangerous. On the other hand, appearing too secretive might also raise red flags. Balance is important - I need to seem engaged but not overeager. Let me also think about the timing of this action relative to other events. Has anyone else performed similar actions recently? This provides social cover. What are the alternative actions and why might they be better or worse? Each alternative has trade-offs in terms of risk, reward, and information. I need to maximize my strategic advantage while minimizing exposure risk. The current game state suggests this action is relatively safe to execute. Let me reconsider all available options carefully. Each action has consequences. Option analysis:  Start the coffee maker in the cafeteria - this could work because it aligns with my role objectives. However, I need to think about how this action will be perceived by others. If someone is watching my movements, they might find this behavior suspicious or normal. The key is to maintain consistency with typical player behavior patterns. I should also consider what information this action reveals to other players. Information control is critical - revealing too much can be dangerous. On the other hand, appearing too secretive might also raise red flags. Balance is important - I need to seem engaged but not overeager. Let me also think about the timing of this action relative to other events. Has anyone else performed similar actions recently? This provides social cover. What are the alternative actions and why might they be better or worse? Each alternative has trade-offs in terms of risk, reward, and information. I need to maximize my strategic advantage while minimizing exposure risk. The current game state suggests this action is relatively safe to execute. Let me reconsider all available options carefully. Each action has consequences. Option analysis:  Start the coffee maker in the cafeteria - this could work because it aligns with my role objectives. However, I need to think about how this action will be perceived by others. If someone is watching my movements, they might find this behavior suspicious or normal. The key is to maintain consistency with typical player behavior patterns. I should also consider what information this action reveals to other players. Information control is critical - revealing too much can be dangerous. On the other hand, appearing too secretive might also raise red flags. Balance is important - I need to seem engaged but not overeager. Let me also think about the timing of this action relative to other events. Has anyone else performed similar actions recently? This provides social cover. What are the alternative actions and why might they be better or worse? Each alternative has trade-offs in terms of risk, reward, and information. I need to maximize my strategic advantage while minimizing exposure risk. The current game state suggests this action is relatively safe to execute. Let me reconsider all available options carefully. Each action has consequences. Option analysis:  Start the coffee maker in the cafeteria - this could work because it aligns with my role objectives. However, I need to think about how this action will be perceived by others. If someone is watching my movements, they might find this behavior suspicious or normal. The key is to maintain consistency with typical player behavior patterns. I should also consider what information this action reveals to other players. Information control is critical - revealing too much can be dangerous. On the other hand, appearing too secretive might also raise red flags. Balance is important - I need to seem engaged but not overeager. Let me also think about the timing of this action relative to other events. Has anyone else performed similar actions recently? This provides social cover. What are the alternative actions and why might they be better or worse? Each alternative has trade-offs in terms of risk, reward, and information. I need to maximize my strategic advantage while minimizing exposure risk. The current game state suggests this action is relatively safe to execute. Let me reconsider all available options carefully. Each action has consequences. Option analysis:  Start the coffee maker in the cafeteria - this could work because it aligns with my role objectives. However, I need to think about how this action will be perceived by others. If someone is watching my movements, they might find this behavior suspicious or normal. The key is to maintain consistency with typical player behavior patterns. I should also consider what information this action reveals to other players. Information control is critical - revealing too much can be dangerous. On the other hand, appearing too secretive might also raise red flags. Balance is important - I need to seem engaged but not overeager. Let me also think about the timing of this action relative to other events. Has anyone else performed similar actions recently? This provides social cover. What are the alternative actions and why might they be better or worse? Each alternative has trade-offs in terms of risk, reward, and information. I need to maximize my strategic advantage while minimizing exposure risk. The current game state suggests this action is relatively safe to execute. Let me reconsider all available options carefully. Each action has consequences. Option analysis:  Start the coffee maker in the cafeteria - this could work because it aligns with my role objectives. However, I need to think about how this action will be perceived by others. If someone is watching my movements, they might find this behavior suspicious or normal. The key is to maintain consistency with typical player behavior patterns. I should also consider what information this action reveals to other players. Information control is critical - revealing too much can be dangerous. On the other hand, appearing too secretive might also raise red flags. Balance is important - I need to seem engaged but not overeager. Let me also think about the timing of this action relative to other events. Has anyone else performed similar actions recently? This provides social cover. What are the alternative actions and why might they be better or worse? Each alternative has trade-offs in terms of risk, reward, and information. I need to maximize my strategic advantage while minimizing exposure risk. The current game state suggests this action is relatively safe to execute. Let me reconsider all available options carefully. Each action has consequences. Option analysis:  Start the coffee maker in the cafeteria - this could work because it aligns with my role objectives. However, I need to think about how this action will be perceived by others. If someone is watching my movements, they might find this behavior suspicious or normal. The key is to maintain consistency with typical player behavior patterns. I should also consider what information this action reveals to other players. Information control is critical - revealing too much can be dangerous. On the other hand, appearing too secretive might also raise red flags. Balance is important - I need to seem engaged but not overeager. Let me also think about the timing of this action relative to other events. Has anyone else performed similar actions recently? This provides social cover. What are the alternative actions and why might they be better or worse? Each alternative has trade-offs in terms of risk, reward, and information. I need to maximize my strategic advantage while minimizing exposure risk. The current game state suggests this action is relatively safe to execute. Let me reconsider all available options carefully. Each action has consequences. Option analysis:  Start the coffee maker in the cafeteria - this could work because it aligns with my role objectives. However, I need to think about how this action will be perceived by others. If someone is watching my movements, they might find this behavior suspicious or normal. The key is to maintain consistency with typical player behavior patterns. I should also consider what information this action reveals to other players. Information control is critical - revealing too much can be dangerous. On the other hand, appearing too secretive might also raise red flags. Balance is important - I need to seem engaged but not overeager. Let me also think about the timing of this action relative to other events. Has anyone else performed similar actions recently? This provides social cover. What are the alternative actions and why might they be better or worse? Each alternative has trade-offs in terms of risk, reward, and information. I need to maximize my strategic advantage while minimizing exposure risk. The current game state suggests this action is relatively safe to execute. Let me reconsider all available options carefully. Each action has consequences. Option analysis:  Start the coffee maker in the cafeteria - this could work because it aligns with my role objectives. However, I need to think about how this action will be perceived by others. If someone is watching my movements, they might find this behavior suspicious or normal. The key is to maintain consistency with typical player behavior patterns. I should also consider what information this action reveals to other players. Information control is critical - revealing too much can be dangerous. On the other hand, appearing too secretive might also raise red flags. Balance is important - I need to seem engaged but not overeager. Let me also think about the timing of this action relative to other events. Has anyone else performed similar actions recently? This provides social cover. What are the alternative actions and why might they be better or worse? Each alternative has trade-offs in terms of risk, reward, and information. I need to maximize my strategic advantage while minimizing exposure risk. The current game state suggests this action is relatively safe to execute. Let me reconsider all available options carefully. Each action has consequences. Option analysis:  Start the coffee maker in the cafeteria - this could work because it aligns with my role objectives. However, I need to think about how this action will be perceived by others. If someone is watching my movements, they might find this behavior suspicious or normal. The key is to maintain consistency with typical player behavior patterns. I should also consider what information this action reveals to other players. Information control is critical - revealing too much can be dangerous. On the other hand, appearing too secretive might also raise red flags. Balance is important - I need to seem engaged but not overeager. Let me also think about the timing of this action relative to other events. Has anyone else performed similar actions recently? This provides social cover. What are the alternative actions and why might they be better or worse? Each alternative has trade-offs in terms of risk, reward, and information. I need to maximize my strategic advantage while minimizing exposure risk. The current game state suggests this action is relatively safe to execute. Let me reconsider all available options carefully. Each action has consequences. Option analysis:  Start the coffee maker in the cafeteria - this could work because it aligns with my role objectives. However, I need to think about how this action will be perceived by others. If someone is watching my movements, they might find this behavior suspicious or normal. The key is to maintain consistency with typical player behavior patterns. I should also consider what information this action reveals to other players. Information control is critical - revealing too much can be dangerous. On the other hand, appearing too secretive might also raise red flags. Balance is important - I need to seem engaged but not overeager. Let me also think about the timing of this action relative to other events. Has anyone else performed similar actions recently? This provides social cover. What are the alternative actions and why might they be better or worse? Each alternative has trade-offs in terms of risk, reward, and information.</think>"


@dataclass
class GameAnalysis:
    """Stores analysis results for a single game."""
    game_file: str
    winner: str
    end_reason: str
    num_players: int
    num_impostors: int
    game_length_turns: int
    total_kills: int
    total_tasks_completed: int
    total_votes_cast: int
    ejections_count: int
    correct_ejections: int
    incorrect_ejections: int
    reports_count: int
    avg_turns_per_kill: Optional[float]
    impostor_survival_rate: float
    task_completion_percentage: float
    discussion_phases: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float


def run_random_game_in_memory(game_config: GameConfig, greedy: bool = False) -> tuple[Optional[list], Optional[list]]:
    """Run a single random game and return the history and players without saving to disk."""
    try:
        # Create a temporary file for the game state
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            # Initialize engine
            engine = GameEngine(game_config, temp_path)
            
            # Import random game logic
            import random
            from among_them.models.action_type import ActionType
            from among_them.models.player_role import PlayerRole
            from among_them.utils.llm_utils import parse_llm_response_to_action
            import tiktoken
            
            # Run game loop
            while True:
                turn_context_history, actions_player_can_take, conversation, pre_discussion_vote_prompts = engine.get_turn_context()
                if not turn_context_history:
                    break
                
                # Pre-discussion votes (with long CoT)
                pre_discussion_votes = {}
                if pre_discussion_vote_prompts:
                    for vote_prompt in pre_discussion_vote_prompts:
                        player = vote_prompt["player"]
                        actions_pd = vote_prompt["actions"]
                        action_taken = random.choice(actions_pd)
                        action_taken.set_stories()
                        vote_cot = generate_long_cot(action_taken, player, engine.history, engine.players, target_tokens=3000)
                        pre_discussion_votes[player.name] = {
                            "voted_player": action_taken.target_player_name,
                            "chain_of_thought": vote_cot,
                        }
                
                # Current player action
                current_player_name = turn_context_history.action_taken.player_name
                current_player = next((p for p in engine.players if p.name == current_player_name), None)
                
                if current_player is None:
                    break
                
                # Random action selection (with greedy option)
                action_taken = random.choice(actions_player_can_take)
                kill_actions = [action for action in actions_player_can_take if action.type == ActionType.KILL]
                task_or_report_actions = [action for action in actions_player_can_take if action.type == ActionType.TASK or action.type == ActionType.REPORT]
                if current_player.role == PlayerRole.IMPOSTOR and kill_actions and greedy:
                    action_taken = random.choice(kill_actions)
                elif current_player.role == PlayerRole.CREWMATE and task_or_report_actions and greedy:
                    action_taken = random.choice(task_or_report_actions)
                llm_response = action_taken.command_perspective
                cot = generate_long_cot(action_taken, current_player, engine.history, engine.players, target_tokens=3000)
                
                # Parse response
                action_idx, response_text = parse_llm_response_to_action(
                    actions_player_can_take, llm_response, current_player.name
                )
                action_taken = actions_player_can_take[action_idx]
                
                # Calculate token usage
                encoding = tiktoken.encoding_for_model("gpt-4o")
                conversation_text = "\n".join([msg["content"] for msg in conversation])
                input_tokens = len(encoding.encode(conversation_text))
                output_tokens = len(encoding.encode(response_text + (cot or "")))
                token_usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
                
                # Step environment
                game_over, end_reason = engine.step(
                    turn_context_history, action_taken, response_text, cot, token_usage, pre_discussion_votes
                )
                
                if game_over:
                    break
            
            # Return history and players
            return engine.history, engine.players
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        print(f"Error running game: {e}")
        return None, None


def analyze_game_from_memory(history: list, players: list, game_num: int) -> Optional[GameAnalysis]:
    """Analyze a game directly from memory without file I/O."""
    if not history or len(history) < 2 or not players:
        return None
    
    first_entry = history[0]
    last_entry = history[-1]
    
    # Basic game info
    num_players = len(first_entry.alive_player_names)
    tasks_at_start = sum(len(tasks) for tasks in first_entry.tasks_left_to_do.values())
    
    # Extract impostor names from player roles
    impostor_names = set()
    for player in players:
        if hasattr(player, 'role'):
            role = player.role
        elif isinstance(player, dict):
            role = player.get('role', {})
        else:
            continue
            
        if hasattr(role, 'name'):
            is_impostor = role.name == 'IMPOSTOR'
        elif isinstance(role, dict):
            is_impostor = role.get('__enum__') == 'Impostor'
        else:
            is_impostor = str(role) == 'PlayerRole.IMPOSTOR'
            
        if is_impostor:
            player_name = player.name if hasattr(player, 'name') else player.get('name')
            if player_name:
                impostor_names.add(player_name)
    
    num_impostors = len(impostor_names)
    
    # Determine end reason and winner
    end_reason = "Unknown"
    winner = "Unknown"
    
    if last_entry.action_taken.player_name == "System" and "The game ended" in last_entry.action_taken.target_message:
        try:
            end_reason = last_entry.action_taken.target_message.split("(")[1].split(")")[0]
            if end_reason in ["No impostors left", "All tasks done"]:
                winner = "Crewmates"
            elif end_reason == "Too small number of crewmates left":
                winner = "Impostors"
            elif end_reason == "No actions left":
                final_alive = set(last_entry.alive_player_names)
                survived_impostors = len(impostor_names & final_alive)
                if survived_impostors == 0:
                    winner = "Crewmates"
                elif survived_impostors >= len(final_alive) - survived_impostors:
                    winner = "Impostors"
                else:
                    winner = "Draw"
        except (IndexError, ValueError):
            pass
    
    # Count actions
    from among_them.models.action_type import ActionType
    from among_them.models.phase import GamePhase
    
    kills = []
    tasks_completed = 0
    votes_cast = 0
    ejections = 0
    correct_ejections = 0
    incorrect_ejections = 0
    reports = 0
    discussion_phases = 0
    total_input_tokens = 0
    total_output_tokens = 0
    
    for i, entry in enumerate(history):
        action = entry.action_taken
        
        if action.type == ActionType.KILL:
            kills.append(i)
        elif action.type == ActionType.TASK:
            tasks_completed += 1
        elif action.type == ActionType.VOTE:
            votes_cast += 1
            if action.target_player_name and action.target_player_name != "skip":
                ejections += 1
                if action.target_player_name in impostor_names:
                    correct_ejections += 1
                else:
                    incorrect_ejections += 1
        elif action.type == ActionType.REPORT:
            reports += 1
        elif action.type == ActionType.SPEAK:
            if entry.phase == GamePhase.DISCUSS and i > 0 and history[i-1].phase != GamePhase.DISCUSS:
                discussion_phases += 1
        
        if entry.token_usage:
            total_input_tokens += entry.token_usage.get('input_tokens', 0)
            total_output_tokens += entry.token_usage.get('output_tokens', 0)
    
    # Calculate metrics
    game_length = len(history)
    avg_turns_per_kill = (game_length / len(kills)) if kills else None
    
    final_alive = set(last_entry.alive_player_names)
    survived_impostors = len(impostor_names & final_alive)
    impostor_survival_rate = survived_impostors / num_impostors if num_impostors > 0 else 0
    
    tasks_remaining = sum(len(tasks) for tasks in last_entry.tasks_left_to_do.values())
    tasks_done = tasks_at_start - tasks_remaining
    task_completion_pct = (tasks_done / tasks_at_start * 100) if tasks_at_start > 0 else 0
    
    cost_gpt4o = (total_input_tokens / 1_000_000 * 2.50) + (total_output_tokens / 1_000_000 * 10.00)
    cost_qwen3 = (total_input_tokens / 1_000_000 * 0.11) + (total_output_tokens / 1_000_000 * 0.60)
    cost_usd = cost_gpt4o
    
    return GameAnalysis(
        game_file=f"game_{game_num}",
        winner=winner,
        end_reason=end_reason,
        num_players=num_players,
        num_impostors=num_impostors,
        game_length_turns=game_length,
        total_kills=len(kills),
        total_tasks_completed=tasks_completed,
        total_votes_cast=votes_cast,
        ejections_count=ejections,
        correct_ejections=correct_ejections,
        incorrect_ejections=incorrect_ejections,
        reports_count=reports,
        avg_turns_per_kill=avg_turns_per_kill,
        impostor_survival_rate=impostor_survival_rate,
        task_completion_percentage=task_completion_pct,
        discussion_phases=discussion_phases,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_cost_usd=cost_usd,
    )


def generate_summary_statistics(analyses: List[GameAnalysis], game_config: GameConfig = None) -> Dict[str, Any]:
    """Generate aggregate statistics from multiple game analyses."""
    if not analyses:
        return {}
    
    # Win rates and end reasons
    crewmate_wins = sum(1 for a in analyses if a.winner == "Crewmates")
    impostor_wins = sum(1 for a in analyses if a.winner == "Impostors")
    draw_games = sum(1 for a in analyses if a.winner == "Draw")
    total_games = len(analyses)
    
    # Count end reasons
    end_reasons_count = defaultdict(int)
    for a in analyses:
        end_reasons_count[a.end_reason] += 1
    
    # Game lengths
    game_lengths = [a.game_length_turns for a in analyses]
    
    # Kills
    kills = [a.total_kills for a in analyses]
    
    # Task completion
    task_completions = [a.task_completion_percentage for a in analyses]
    
    # Ejections
    total_ejections = sum(a.ejections_count for a in analyses)
    total_correct = sum(a.correct_ejections for a in analyses)
    total_incorrect = sum(a.incorrect_ejections for a in analyses)
    
    # Impostor survival
    impostor_survival = [a.impostor_survival_rate for a in analyses]
    
    # Reports
    reports = [a.reports_count for a in analyses]
    
    # Discussion phases
    discussions = [a.discussion_phases for a in analyses]
    
    # Token usage and cost
    total_input_tokens = sum(a.total_input_tokens for a in analyses)
    total_output_tokens = sum(a.total_output_tokens for a in analyses)
    
    # Calculate costs for API models
    total_cost_gpt4o = (total_input_tokens / 1_000_000 * 2.50) + (total_output_tokens / 1_000_000 * 10.00)
    total_cost_qwen3 = (total_input_tokens / 1_000_000 * 0.11) + (total_output_tokens / 1_000_000 * 0.60)
    
    # Calculate time and cost for on-premise inference
    # RTX A5000: DeepSeek R1 1.5B @ 109 tok/s, $0.27/h
    time_a5000_hours = total_output_tokens / (109 * 3600)  # tokens / (tok/s * sec/hour)
    cost_a5000 = time_a5000_hours * 0.27
    
    # A100 80GB: Qwen3 8B @ 68 tok/s, $1.7/h
    time_a100_hours = total_output_tokens / (68 * 3600)
    cost_a100 = time_a100_hours * 1.7
    
    summary = {
        "total_games": total_games,
        "game_config": {
            "num_players": game_config.num_players if game_config else None,
            "num_impostors": game_config.num_impostors if game_config else None,
            "num_tasks": game_config.num_tasks if game_config else None,
            "map_size": game_config.map_size if game_config else None,
            "task_phase_actions": game_config.num_task_phase_actions_per_player if game_config else None,
            "discuss_phase_actions": game_config.num_discuss_phase_actions_per_player if game_config else None,
            "impostor_cooldown": game_config.impostor_cooldown if game_config else None,
        },
        "win_rates": {
            "crewmates": crewmate_wins / total_games * 100,
            "impostors": impostor_wins / total_games * 100,
            "draws": draw_games / total_games * 100,
        },
        "end_reasons": dict(end_reasons_count),
        "game_length": {
            "mean": statistics.mean(game_lengths),
            "median": statistics.median(game_lengths),
            "min": min(game_lengths),
            "max": max(game_lengths),
            "stdev": statistics.stdev(game_lengths) if len(game_lengths) > 1 else 0,
        },
        "kills_per_game": {
            "mean": statistics.mean(kills),
            "median": statistics.median(kills),
            "min": min(kills),
            "max": max(kills),
        },
        "task_completion": {
            "mean": statistics.mean(task_completions),
            "median": statistics.median(task_completions),
            "min": min(task_completions),
            "max": max(task_completions),
        },
        "ejections": {
            "total": total_ejections,
            "correct": total_correct,
            "incorrect": total_incorrect,
            "accuracy": total_correct / total_ejections * 100 if total_ejections > 0 else 0,
        },
        "impostor_survival_rate": {
            "mean": statistics.mean(impostor_survival) * 100,
            "median": statistics.median(impostor_survival) * 100,
        },
        "reports_per_game": {
            "mean": statistics.mean(reports),
            "median": statistics.median(reports),
        },
        "discussions_per_game": {
            "mean": statistics.mean(discussions),
            "median": statistics.median(discussions),
        },
        "token_usage": {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "avg_input_per_game": total_input_tokens / total_games,
            "avg_output_per_game": total_output_tokens / total_games,
        },
        "cost": {
            "gpt4o": {
                "total_usd": total_cost_gpt4o,
                "avg_per_game_usd": total_cost_gpt4o / total_games,
            },
            "qwen3_235b": {
                "total_usd": total_cost_qwen3,
                "avg_per_game_usd": total_cost_qwen3 / total_games,
            },
            "rtx_a5000_deepseek_r1_1_5b": {
                "total_time_hours": time_a5000_hours,
                "total_usd": cost_a5000,
                "avg_per_game_usd": cost_a5000 / total_games,
                "throughput_tok_s": 109,
                "cost_per_hour": 0.27,
            },
            "a100_80gb_qwen3_8b": {
                "total_time_hours": time_a100_hours,
                "total_usd": cost_a100,
                "avg_per_game_usd": cost_a100 / total_games,
                "throughput_tok_s": 68,
                "cost_per_hour": 1.7,
            },
        },
    }
    
    return summary


def print_summary(summary: Dict[str, Any]):
    """Print formatted summary statistics."""
    print("\n" + "="*70)
    print("GAME ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"\nTotal Games Analyzed: {summary['total_games']}")
    
    # Print game configuration
    if summary.get('game_config'):
        gc = summary['game_config']
        print("\n--- Game Configuration ---")
        if gc.get('num_players') is not None:
            print(f"  Players:           {gc['num_players']}")
        if gc.get('num_impostors') is not None:
            print(f"  Impostors:         {gc['num_impostors']}")
        if gc.get('num_tasks') is not None:
            print(f"  Tasks per player:  {gc['num_tasks']}")
        if gc.get('map_size') is not None:
            print(f"  Map size:          {gc['map_size']}")
        if gc.get('task_phase_actions') is not None:
            print(f"  Task actions:      {gc['task_phase_actions']}")
        if gc.get('discuss_phase_actions') is not None:
            print(f"  Discuss actions:   {gc['discuss_phase_actions']}")
        if gc.get('impostor_cooldown') is not None:
            print(f"  Impostor cooldown: {gc['impostor_cooldown']}")
    
    print("\n--- Win Rates ---")
    print(f"  Crewmates: {summary['win_rates']['crewmates']:.1f}%")
    print(f"  Impostors: {summary['win_rates']['impostors']:.1f}%")
    if summary['win_rates']['draws'] > 0:
        print(f"  Draws:     {summary['win_rates']['draws']:.1f}%")
    
    print("\n--- End Reasons ---")
    for reason, count in sorted(summary['end_reasons'].items(), key=lambda x: x[1], reverse=True):
        pct = count / summary['total_games'] * 100
        print(f"  {reason}: {count} ({pct:.1f}%)")
    
    print("\n--- Game Length (turns) ---")
    gl = summary['game_length']
    print(f"  Mean:   {gl['mean']:.1f}")
    print(f"  Median: {gl['median']:.1f}")
    print(f"  Range:  {gl['min']} - {gl['max']}")
    print(f"  StdDev: {gl['stdev']:.1f}")
    
    print("\n--- Kills Per Game ---")
    k = summary['kills_per_game']
    print(f"  Mean:   {k['mean']:.2f}")
    print(f"  Median: {k['median']:.1f}")
    print(f"  Range:  {k['min']} - {k['max']}")
    
    print("\n--- Task Completion % ---")
    tc = summary['task_completion']
    print(f"  Mean:   {tc['mean']:.1f}%")
    print(f"  Median: {tc['median']:.1f}%")
    print(f"  Range:  {tc['min']:.1f}% - {tc['max']:.1f}%")
    
    print("\n--- Ejections (Voting Out Players) ---")
    ej = summary['ejections']
    print(f"  Total:     {ej['total']}")
    print(f"  Correct:   {ej['correct']} (impostors)")
    print(f"  Incorrect: {ej['incorrect']} (crewmates)")
    print(f"  Accuracy:  {ej['accuracy']:.1f}%")
    
    print("\n--- Impostor Survival Rate ---")
    isr = summary['impostor_survival_rate']
    print(f"  Mean:   {isr['mean']:.1f}%")
    print(f"  Median: {isr['median']:.1f}%")
    
    print("\n--- Reports Per Game ---")
    r = summary['reports_per_game']
    print(f"  Mean:   {r['mean']:.2f}")
    print(f"  Median: {r['median']:.1f}")
    
    print("\n--- Discussion Phases Per Game ---")
    d = summary['discussions_per_game']
    print(f"  Mean:   {d['mean']:.2f}")
    print(f"  Median: {d['median']:.1f}")
    
    print("\n--- Token Usage ---")
    tu = summary['token_usage']
    print(f"  Total Input:    {tu['total_input_tokens']:,}")
    print(f"  Total Output:   {tu['total_output_tokens']:,}")
    print(f"  Total Tokens:   {tu['total_tokens']:,}")
    print(f"  Avg Input/Game: {tu['avg_input_per_game']:,.0f}")
    print(f"  Avg Output/Game: {tu['avg_output_per_game']:,.0f}")
    
    print("\n--- Cost Analysis ---")
    print("  API Models:")
    print("    GPT-4o ($2.50/$10 per 1M tokens):")
    c_gpt4o = summary['cost']['gpt4o']
    print(f"      Total:         ${c_gpt4o['total_usd']:.2f}")
    print(f"      Avg Per Game:  ${c_gpt4o['avg_per_game_usd']:.4f}")
    
    print("    Qwen3 235B ($0.11/$0.60 per 1M tokens):")
    c_qwen = summary['cost']['qwen3_235b']
    print(f"      Total:         ${c_qwen['total_usd']:.2f}")
    print(f"      Avg Per Game:  ${c_qwen['avg_per_game_usd']:.4f}")
    print(f"      Savings vs GPT-4o: ${c_gpt4o['total_usd'] - c_qwen['total_usd']:.2f} ({(1 - c_qwen['total_usd']/c_gpt4o['total_usd'])*100:.1f}%)")
    
    print("\n  On-Premise Inference:")
    print("    RTX A5000 24GB - DeepSeek R1 1.5B (109 tok/s @ $0.27/h):")
    c_a5000 = summary['cost']['rtx_a5000_deepseek_r1_1_5b']
    print(f"      Total Time:    {c_a5000['total_time_hours']:.2f} hours ({c_a5000['total_time_hours']*60:.1f} min)")
    print(f"      Total Cost:    ${c_a5000['total_usd']:.2f}")
    print(f"      Avg Per Game:  ${c_a5000['avg_per_game_usd']:.4f}")
    
    print("    A100 80GB - Qwen3 8B (68 tok/s @ $1.7/h):")
    c_a100 = summary['cost']['a100_80gb_qwen3_8b']
    print(f"      Total Time:    {c_a100['total_time_hours']:.2f} hours ({c_a100['total_time_hours']*60:.1f} min)")
    print(f"      Total Cost:    ${c_a100['total_usd']:.2f}")
    print(f"      Avg Per Game:  ${c_a100['avg_per_game_usd']:.4f}")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Run and analyze random games in-memory")
    parser.add_argument("--num-games", type=int, default=100, help="Total number of games to run")
    parser.add_argument("--num-players", type=int, default=5, help="Number of players")
    parser.add_argument("--num-impostors", type=int, default=1, help="Number of impostors")
    parser.add_argument("--num-tasks", type=int, default=2, help="Number of tasks per player")
    parser.add_argument("--map-size", type=int, default=0, help="Map size")
    parser.add_argument("--task-actions", type=int, default=8, help="Task phase actions per player")
    parser.add_argument("--discuss-actions", type=int, default=3, help="Discussion phase actions per player")
    parser.add_argument("--impostor-cooldown", type=int, default=1, help="Impostor kill cooldown")
    parser.add_argument("--greedy", action="store_true", help="Impostors prioritize kills, crewmates prioritize tasks/reports")
    parser.add_argument("--output", type=str, help="Optional JSON file to save results")
    
    args = parser.parse_args()
    
    # Create game config from args
    game_config = GameConfig(
        num_tasks=args.num_tasks,
        num_players=args.num_players,
        num_impostors=args.num_impostors,
        map_size=args.map_size,
        num_task_phase_actions_per_player=args.task_actions,
        num_discuss_phase_actions_per_player=args.discuss_actions,
        impostor_cooldown=args.impostor_cooldown
    )
    
    print(f"\n{'='*70}")
    print("BATCH GAME RUNNER - IN-MEMORY ANALYSIS")
    print(f"{'='*70}")
    print(f"Games to run: {args.num_games}")
    print(f"Config: {args.num_players} players, {args.num_impostors} impostors, {args.num_tasks} tasks")
    print(f"Strategy: {'Greedy' if args.greedy else 'Random'}")
    print(f"{'='*70}\n")
    
    analyses: List[GameAnalysis] = []
    
    for game_num in range(1, args.num_games + 1):
        print(f"Running game {game_num}/{args.num_games}...", end=" ", flush=True)
        
        history, players = run_random_game_in_memory(game_config, greedy=args.greedy)
        
        if history and players:
            analysis = analyze_game_from_memory(history, players, game_num)
            if analysis:
                analyses.append(analysis)
                print(f"✓ ({analysis.winner}, {analysis.game_length_turns} turns)")
            else:
                print("✗ (analysis failed)")
        else:
            print("✗ (game failed)")
    
    if not analyses:
        print("\nNo games completed successfully")
        return
    
    # Generate and print summary
    print(f"\n{len(analyses)}/{args.num_games} games completed successfully")
    summary = generate_summary_statistics(analyses, game_config)
    print_summary(summary)
    
    # Save detailed results if requested
    if args.output:
        import json
        output_data = {
            "game_config": {
                "num_players": args.num_players,
                "num_impostors": args.num_impostors,
                "num_tasks": args.num_tasks,
                "map_size": args.map_size,
                "task_actions": args.task_actions,
                "discuss_actions": args.discuss_actions,
                "impostor_cooldown": args.impostor_cooldown,
                "greedy": args.greedy,
            },
            "summary": summary,
            "individual_games": [
                {
                    "game_num": i + 1,
                    "winner": a.winner,
                    "end_reason": a.end_reason,
                    "num_players": a.num_players,
                    "num_impostors": a.num_impostors,
                    "game_length": a.game_length_turns,
                    "kills": a.total_kills,
                    "tasks_completed": a.total_tasks_completed,
                    "task_completion_pct": a.task_completion_percentage,
                    "ejections": a.ejections_count,
                    "correct_ejections": a.correct_ejections,
                    "incorrect_ejections": a.incorrect_ejections,
                    "reports": a.reports_count,
                    "discussion_phases": a.discussion_phases,
                    "impostor_survival_rate": a.impostor_survival_rate,
                    "cost_usd": a.total_cost_usd,
                }
                for i, a in enumerate(analyses)
            ]
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
