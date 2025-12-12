#!/usr/bin/env python3
"""
Plot MAPPO training statistics from checkpoint stats.json

Usage:
    python scripts/plot_training_stats.py [checkpoint_dir]
    
If no checkpoint_dir is provided, reads from MAPPO_CHECKPOINT_DIR env var
or defaults to looking for the latest checkpoint.
"""

import csv
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# Style configuration
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'crewmate': '#4CAF50',  # Green
    'impostor': '#F44336',  # Red
    'policy': '#2196F3',    # Blue
    'value': '#FF9800',     # Orange
    'kl': '#9C27B0',        # Purple
    'entropy': '#00BCD4',   # Cyan
    'actor': '#3F51B5',     # Indigo
    'critic': '#E91E63',    # Pink
    'primary': '#1976D2',   # Primary blue
    'secondary': '#757575', # Gray
}


def find_latest_checkpoint(base_dir: str = None) -> Path:
    """Find the latest checkpoint directory with stats.json"""
    if base_dir is None:
        base_dir = os.environ.get(
            "MAPPO_CHECKPOINT_DIR", 
            "/content/drive/MyDrive/among_them/outputs/mappo_checkpoints"
        )
    
    base_path = Path(base_dir)
    
    # Check if stats.json exists directly in base_dir
    if (base_path / "stats.json").exists():
        return base_path
    
    # Look for iteration subdirectories
    iter_dirs = sorted(
        [d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda x: int(x.name),
        reverse=True
    )
    
    for iter_dir in iter_dirs:
        if (iter_dir / "stats.json").exists():
            return iter_dir
    
    # Return base if nothing found
    return base_path


def load_stats(checkpoint_dir: Path) -> List[Dict[str, Any]]:
    """Load stats.json from checkpoint directory"""
    stats_file = checkpoint_dir / "stats.json"
    
    if not stats_file.exists():
        raise FileNotFoundError(f"stats.json not found in {checkpoint_dir}")
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    print(f"✅ Loaded {len(stats)} iterations from {stats_file}")
    return stats


def extract_metric(stats: List[Dict], key: str, default: float = 0.0) -> np.ndarray:
    """Extract a metric across all iterations"""
    return np.array([s.get(key, default) for s in stats])


def create_training_plots(stats: List[Dict[str, Any]], output_path: str = None):
    """Create comprehensive training visualization"""
    iterations = extract_metric(stats, 'iteration')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    # ========== Row 1: Win Rates, Losses, PPO Metrics ==========
    
    # 1. Win Rates
    ax1 = fig.add_subplot(gs[0, 0])
    crewmate_wr = extract_metric(stats, 'CREWMATE_win_rate') * 100
    impostor_wr = extract_metric(stats, 'IMPOSTOR_win_rate') * 100
    ax1.plot(iterations, crewmate_wr, 'o-', color=COLORS['crewmate'], 
             label='Crewmate', linewidth=2, markersize=6)
    ax1.plot(iterations, impostor_wr, 's-', color=COLORS['impostor'], 
             label='Impostor', linewidth=2, markersize=6)
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50%')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Win Rate (%)')
    ax1.set_title('Win Rates by Role', fontweight='bold')
    ax1.legend(loc='best')
    ax1.set_ylim(0, 100)
    
    # 2. Losses
    ax2 = fig.add_subplot(gs[0, 1])
    policy_loss = extract_metric(stats, 'policy_loss')
    value_loss = extract_metric(stats, 'value_loss')
    kl_loss = extract_metric(stats, 'kl_loss')
    ax2.plot(iterations, policy_loss, 'o-', color=COLORS['policy'], 
             label='Policy', linewidth=2, markersize=5)
    ax2.plot(iterations, value_loss, 's-', color=COLORS['value'], 
             label='Value', linewidth=2, markersize=5)
    ax2.plot(iterations, kl_loss, '^-', color=COLORS['kl'], 
             label='KL', linewidth=2, markersize=5)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Losses', fontweight='bold')
    ax2.legend(loc='best')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # 3. PPO Metrics (Ratio & Clip Fraction)
    ax3 = fig.add_subplot(gs[0, 2])
    avg_ratio = extract_metric(stats, 'avg_ratio')
    clip_frac = extract_metric(stats, 'clip_fraction') * 100
    ax3_twin = ax3.twinx()
    l1, = ax3.plot(iterations, avg_ratio, 'o-', color=COLORS['primary'], 
                   label='Avg Ratio', linewidth=2, markersize=5)
    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    l2, = ax3_twin.plot(iterations, clip_frac, 's-', color=COLORS['impostor'], 
                        label='Clip %', linewidth=2, markersize=5)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Importance Ratio', color=COLORS['primary'])
    ax3_twin.set_ylabel('Clip Fraction (%)', color=COLORS['impostor'])
    ax3.set_title('PPO Metrics', fontweight='bold')
    ax3.legend(handles=[l1, l2], loc='best')
    
    # ========== Row 2: Entropy, Gradient Norms, Explained Variance ==========
    
    # 4. Entropy
    ax4 = fig.add_subplot(gs[1, 0])
    entropy = extract_metric(stats, 'avg_entropy')
    ax4.plot(iterations, entropy, 'o-', color=COLORS['entropy'], 
             linewidth=2, markersize=6)
    ax4.fill_between(iterations, 0, entropy, color=COLORS['entropy'], alpha=0.2)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Entropy')
    ax4.set_title('Policy Entropy', fontweight='bold')
    
    # 5. Gradient Norms
    ax5 = fig.add_subplot(gs[1, 1])
    actor_grad = extract_metric(stats, 'actor_grad_norm')
    critic_grad = extract_metric(stats, 'critic_grad_norm')
    ax5.plot(iterations, actor_grad, 'o-', color=COLORS['actor'], 
             label='Actor', linewidth=2, markersize=5)
    ax5.plot(iterations, critic_grad, 's-', color=COLORS['critic'], 
             label='Critic', linewidth=2, markersize=5)
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Gradient Norm')
    ax5.set_title('Gradient Norms', fontweight='bold')
    ax5.legend(loc='best')
    ax5.set_yscale('log')
    
    # 6. Explained Variance
    ax6 = fig.add_subplot(gs[1, 2])
    exp_var = extract_metric(stats, 'explained_variance')
    colors = [COLORS['crewmate'] if v > 0 else COLORS['impostor'] for v in exp_var]
    ax6.bar(iterations, exp_var, color=colors, alpha=0.7, edgecolor='black')
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax6.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Perfect')
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Explained Variance')
    ax6.set_title('Value Function Quality', fontweight='bold')
    ax6.set_ylim(-1.5, 1.5)
    
    # ========== Row 3: Advantage Stats, Game Length, Training Efficiency ==========
    
    # 7. Advantage Distribution
    ax7 = fig.add_subplot(gs[2, 0])
    adv_mean = extract_metric(stats, 'advantage_mean')
    adv_std = extract_metric(stats, 'advantage_std')
    ax7.plot(iterations, adv_mean, 'o-', color=COLORS['primary'], 
             label='Mean', linewidth=2, markersize=5)
    ax7.fill_between(iterations, adv_mean - adv_std, adv_mean + adv_std, 
                     color=COLORS['primary'], alpha=0.2, label='±1 Std')
    ax7.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax7.set_xlabel('Iteration')
    ax7.set_ylabel('Advantage')
    ax7.set_title('Advantage Distribution', fontweight='bold')
    ax7.legend(loc='best')
    
    # 8. Game Length / Turns
    ax8 = fig.add_subplot(gs[2, 1])
    num_turns = extract_metric(stats, 'num_turns')
    avg_len = extract_metric(stats, 'avg_trajectory_length')
    ax8_twin = ax8.twinx()
    bars = ax8.bar(iterations, num_turns, width=0.5, color=COLORS['primary'], 
                   alpha=0.7, label='Total Turns')
    line, = ax8_twin.plot(iterations, avg_len, 's-', color=COLORS['impostor'], 
                          linewidth=2, markersize=6, label='Avg Length')
    ax8.set_xlabel('Iteration')
    ax8.set_ylabel('Total Turns', color=COLORS['primary'])
    ax8_twin.set_ylabel('Avg Trajectory Length', color=COLORS['impostor'])
    ax8.set_title('Game Statistics', fontweight='bold')
    ax8.legend(handles=[bars, line], loc='best')
    
    # 9. Training Throughput
    ax9 = fig.add_subplot(gs[2, 2])
    tokens_per_sec = extract_metric(stats, 'training/tokens_per_second')
    ax9.bar(iterations, tokens_per_sec, color=COLORS['crewmate'], alpha=0.7, 
            edgecolor='black')
    ax9.set_xlabel('Iteration')
    ax9.set_ylabel('Tokens/Second')
    ax9.set_title('Training Throughput', fontweight='bold')
    
    # ========== Row 4: Time Breakdown, Token Stats ==========
    
    # 10. Time Breakdown (Stacked Bar)
    ax10 = fig.add_subplot(gs[3, 0])
    collection_time = extract_metric(stats, 'collection/total_time') / 60  # minutes
    update_time = extract_metric(stats, 'training/update_time') / 60  # minutes
    ax10.bar(iterations, collection_time, label='Collection', 
             color=COLORS['crewmate'], alpha=0.7)
    ax10.bar(iterations, update_time, bottom=collection_time, label='Training', 
             color=COLORS['policy'], alpha=0.7)
    ax10.set_xlabel('Iteration')
    ax10.set_ylabel('Time (minutes)')
    ax10.set_title('Time Breakdown', fontweight='bold')
    ax10.legend(loc='best')
    
    # 11. Token Statistics
    ax11 = fig.add_subplot(gs[3, 1])
    input_mean = extract_metric(stats, 'turn/input_tokens_mean')
    output_mean = extract_metric(stats, 'turn/output_tokens_mean')
    # Filter out zeros for visualization
    valid_mask = input_mean > 0
    if valid_mask.any():
        ax11.plot(iterations[valid_mask], input_mean[valid_mask], 'o-', 
                  color=COLORS['primary'], label='Input', linewidth=2, markersize=5)
    ax11.plot(iterations, output_mean, 's-', color=COLORS['impostor'], 
              label='Output', linewidth=2, markersize=5)
    ax11.set_xlabel('Iteration')
    ax11.set_ylabel('Tokens per Turn')
    ax11.set_title('Token Statistics', fontweight='bold')
    ax11.legend(loc='best')
    
    # 12. Summary Stats (Text)
    ax12 = fig.add_subplot(gs[3, 2])
    ax12.axis('off')
    
    # Calculate summary statistics
    latest = stats[-1]
    summary_text = f"""
    Training Summary (Iter {int(latest.get('iteration', 0))})
    
    Win Rates:
      - Crewmate: {latest.get('CREWMATE_win_rate', 0)*100:.1f}%
      - Impostor: {latest.get('IMPOSTOR_win_rate', 0)*100:.1f}%
    
    Losses:
      - Policy: {latest.get('policy_loss', 0):.4f}
      - Value: {latest.get('value_loss', 0):.4f}
      - KL: {latest.get('kl_loss', 0):.4f}
    
    PPO Health:
      - Clip Frac: {latest.get('clip_fraction', 0)*100:.1f}%
      - Avg Ratio: {latest.get('avg_ratio', 0):.3f}
      - Entropy: {latest.get('avg_entropy', 0):.3f}
    
    Efficiency:
      - Tokens/s: {latest.get('training/tokens_per_second', 0):.0f}
      - Turns: {latest.get('num_turns', 0)}
    """
    ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes, 
              fontsize=10, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax12.set_title('Latest Iteration Summary', fontweight='bold')
    
    # Main title
    fig.suptitle('MAPPO Training Dashboard', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✅ Saved plot to {output_path}")
    else:
        plt.show()
    
    return fig


def load_game_stats(checkpoint_dir: Path) -> pd.DataFrame:
    """Load stats.csv from checkpoint directory"""
    stats_file = checkpoint_dir / "stats.csv"
    
    if not stats_file.exists():
        raise FileNotFoundError(f"stats.csv not found in {checkpoint_dir}")
    
    df = pd.read_csv(stats_file)
    print(f"✅ Loaded {len(df)} iterations from {stats_file}")
    return df


def create_game_stats_plots(df: pd.DataFrame, output_path: str = None):
    """Create comprehensive game statistics visualization from CSV data"""
    iterations = df['iteration'].values
    
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    # ========== Row 1: Win Rates, Kill Timing, Game Length ==========
    
    # 1. Win Rates (from CSV)
    ax1 = fig.add_subplot(gs[0, 0])
    impostor_wr = df['impostor_win_rate'].values * 100
    crewmate_wr = (1 - df['impostor_win_rate'].values) * 100
    ax1.plot(iterations, crewmate_wr, 'o-', color=COLORS['crewmate'], 
             label='Crewmate', linewidth=2, markersize=6)
    ax1.plot(iterations, impostor_wr, 's-', color=COLORS['impostor'], 
             label='Impostor', linewidth=2, markersize=6)
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50%')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Win Rate (%)')
    ax1.set_title('Win Rates (Game Stats)', fontweight='bold')
    ax1.legend(loc='best')
    ax1.set_ylim(0, 100)
    
    # 2. First Kill Timing
    ax2 = fig.add_subplot(gs[0, 1])
    first_kill_turn = df['avg_first_kill_turn'].values
    ax2.plot(iterations, first_kill_turn, 'o-', color=COLORS['impostor'], 
             linewidth=2, markersize=6)
    ax2.fill_between(iterations, 0, first_kill_turn, color=COLORS['impostor'], alpha=0.2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Turn')
    ax2.set_title('Avg First Kill Turn', fontweight='bold')
    ax2.set_ylim(0, max(first_kill_turn) * 1.2)
    
    # 3. Game Length
    ax3 = fig.add_subplot(gs[0, 2])
    game_length = df['avg_game_length'].values
    ax3.bar(iterations, game_length, color=COLORS['primary'], alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Turns')
    ax3.set_title('Avg Game Length', fontweight='bold')
    
    # ========== Row 2: Kills, Discussions, Voting Accuracy ==========
    
    # 4. Average Kills per Game
    ax4 = fig.add_subplot(gs[1, 0])
    avg_kills = df['avg_kills'].values
    ax4.plot(iterations, avg_kills, 'o-', color=COLORS['impostor'], 
             linewidth=2, markersize=6)
    ax4.fill_between(iterations, 0, avg_kills, color=COLORS['impostor'], alpha=0.2)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Kills')
    ax4.set_title('Avg Kills per Game', fontweight='bold')
    
    # 5. Average Discussions per Game
    ax5 = fig.add_subplot(gs[1, 1])
    avg_discussions = df['avg_discussions'].values
    ax5.bar(iterations, avg_discussions, color=COLORS['policy'], alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Discussions')
    ax5.set_title('Avg Discussions per Game', fontweight='bold')
    
    # 6. Voting Accuracy
    ax6 = fig.add_subplot(gs[1, 2])
    voting_acc = df['voting_accuracy'].values * 100
    colors = [COLORS['crewmate'] if v > 25 else COLORS['impostor'] for v in voting_acc]
    ax6.bar(iterations, voting_acc, color=colors, alpha=0.7, edgecolor='black')
    ax6.axhline(y=25, color='gray', linestyle='--', alpha=0.5, label='Random (25%)')
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Accuracy (%)')
    ax6.set_title('Voting Accuracy', fontweight='bold')
    ax6.legend(loc='best')
    ax6.set_ylim(0, 50)
    
    # ========== Row 3: Tasks, Pretend Tasks, Task Ratio ==========
    
    # 7. Tasks Completed
    ax7 = fig.add_subplot(gs[2, 0])
    tasks = df['avg_tasks_completed'].values
    ax7.plot(iterations, tasks, 'o-', color=COLORS['crewmate'], 
             linewidth=2, markersize=6)
    ax7.fill_between(iterations, 0, tasks, color=COLORS['crewmate'], alpha=0.2)
    ax7.set_xlabel('Iteration')
    ax7.set_ylabel('Tasks')
    ax7.set_title('Avg Tasks Completed', fontweight='bold')
    
    # 8. Pretend Tasks (Impostor behavior)
    ax8 = fig.add_subplot(gs[2, 1])
    pretend_tasks = df['avg_pretend_tasks'].values
    ax8.plot(iterations, pretend_tasks, 's-', color=COLORS['kl'], 
             linewidth=2, markersize=6)
    ax8.fill_between(iterations, 0, pretend_tasks, color=COLORS['kl'], alpha=0.2)
    ax8.set_xlabel('Iteration')
    ax8.set_ylabel('Pretend Tasks')
    ax8.set_title('Avg Impostor Pretend Tasks', fontweight='bold')
    
    # 9. Real vs Pretend Tasks Comparison
    ax9 = fig.add_subplot(gs[2, 2])
    width = 0.35
    x = np.arange(len(iterations))
    ax9.bar(x - width/2, tasks, width, label='Real Tasks', color=COLORS['crewmate'], alpha=0.7)
    ax9.bar(x + width/2, pretend_tasks, width, label='Pretend Tasks', color=COLORS['kl'], alpha=0.7)
    ax9.set_xlabel('Iteration')
    ax9.set_ylabel('Tasks')
    ax9.set_title('Tasks Comparison', fontweight='bold')
    ax9.set_xticks(x[::3])  # Show every 3rd iteration
    ax9.set_xticklabels(iterations[::3])
    ax9.legend(loc='best')
    
    # ========== Row 4: Win Counts, Trends, Summary ==========
    
    # 10. Win Counts (Stacked Bar)
    ax10 = fig.add_subplot(gs[3, 0])
    impostor_wins = df['impostor_wins'].values
    crewmate_wins = df['crewmate_wins'].values
    ax10.bar(iterations, crewmate_wins, label='Crewmate Wins', 
             color=COLORS['crewmate'], alpha=0.7)
    ax10.bar(iterations, impostor_wins, bottom=crewmate_wins, label='Impostor Wins', 
             color=COLORS['impostor'], alpha=0.7)
    ax10.set_xlabel('Iteration')
    ax10.set_ylabel('Games')
    ax10.set_title('Win Distribution', fontweight='bold')
    ax10.legend(loc='best')
    
    # 11. Key Metrics Trends (normalized)
    ax11 = fig.add_subplot(gs[3, 1])
    # Normalize metrics to 0-1 range for comparison
    def normalize(arr):
        arr = np.array(arr)
        if arr.max() == arr.min():
            return np.zeros_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min())
    
    ax11.plot(iterations, normalize(game_length), 'o-', label='Game Length', 
              linewidth=2, markersize=4, color=COLORS['primary'])
    ax11.plot(iterations, normalize(first_kill_turn), 's-', label='First Kill Turn', 
              linewidth=2, markersize=4, color=COLORS['impostor'])
    ax11.plot(iterations, normalize(pretend_tasks), '^-', label='Pretend Tasks', 
              linewidth=2, markersize=4, color=COLORS['kl'])
    ax11.set_xlabel('Iteration')
    ax11.set_ylabel('Normalized Value')
    ax11.set_title('Metric Trends (Normalized)', fontweight='bold')
    ax11.legend(loc='best', fontsize=8)
    
    # 12. Summary Stats (Text)
    ax12 = fig.add_subplot(gs[3, 2])
    ax12.axis('off')
    
    latest = df.iloc[-1]
    summary_text = f"""
    Game Stats Summary (Iter {int(latest['iteration'])})
    
    Win Rates:
      - Crewmate: {(1-latest['impostor_win_rate'])*100:.1f}%
      - Impostor: {latest['impostor_win_rate']*100:.1f}%
    
    Combat:
      - Avg First Kill: Turn {latest['avg_first_kill_turn']:.1f}
      - Avg Kills/Game: {latest['avg_kills']:.2f}
    
    Gameplay:
      - Game Length: {latest['avg_game_length']:.1f} turns
      - Discussions: {latest['avg_discussions']:.2f}/game
      - Voting Acc: {latest['voting_accuracy']*100:.1f}%
    
    Tasks:
      - Completed: {latest['avg_tasks_completed']:.2f}
      - Pretend: {latest['avg_pretend_tasks']:.2f}
    """
    ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes, 
              fontsize=10, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax12.set_title('Latest Iteration Summary', fontweight='bold')
    
    fig.suptitle('Among Them - Game Statistics Dashboard', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✅ Saved game stats plot to {output_path}")
    else:
        plt.show()
    
    return fig


def main():
    # Parse arguments
    if len(sys.argv) > 1:
        checkpoint_dir = Path(sys.argv[1])
    else:
        checkpoint_dir = find_latest_checkpoint()
    
    print(f"📂 Using checkpoint directory: {checkpoint_dir}")
    
    # Load and plot JSON stats
    try:
        stats = load_stats(checkpoint_dir)
        if len(stats) == 0:
            print("❌ No iterations found in stats.json")
        else:
            output_path = checkpoint_dir / "training_dashboard.png"
            create_training_plots(stats, str(output_path))
            print(f"✅ Training dashboard saved to {output_path}")
    except FileNotFoundError as e:
        print(f"⚠️ stats.json not found: {e}")
    
    # Load and plot CSV stats
    try:
        game_stats = load_game_stats(checkpoint_dir)
        if len(game_stats) == 0:
            print("❌ No iterations found in stats.csv")
        else:
            output_path = checkpoint_dir / "game_stats_dashboard.png"
            create_game_stats_plots(game_stats, str(output_path))
            print(f"✅ Game stats dashboard saved to {output_path}")
    except FileNotFoundError as e:
        print(f"⚠️ stats.csv not found: {e}")
    
    print("\n🎉 Done!")


if __name__ == "__main__":
    main()
