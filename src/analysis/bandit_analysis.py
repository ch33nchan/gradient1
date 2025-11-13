"""
Analysis tools for bandit experiments.
Reads metrics and generates clean, labeled plots.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional


def load_metrics(run_dir: Path) -> pd.DataFrame:
    """Load metrics CSV from run directory."""
    metrics_path = run_dir / 'metrics.csv'
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics.csv found in {run_dir}")
    return pd.read_csv(metrics_path)


def plot_rewards(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plot average rewards over episodes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    window = 100

    # Plot self-gradient agent
    if 'our_reward' in df.columns:
        rolling = df['our_reward'].rolling(window=window, min_periods=1).mean()
        ax.plot(df['episode'], rolling, label='Self-Gradient', linewidth=2)

    # Plot baselines
    baseline_cols = [col for col in df.columns if col.endswith('_reward') and col != 'our_reward']
    for col in baseline_cols:
        name = col.replace('_reward', '').replace('_', ' ').title()
        rolling = df[col].rolling(window=window, min_periods=1).mean()
        ax.plot(df['episode'], rolling, label=name, alpha=0.7)

    ax.set_xlabel('Episode')
    ax.set_ylabel(f'Average Reward (rolling {window})')
    ax.set_title('Reward vs Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_cumulative_regret(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plot cumulative regret over episodes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot self-gradient agent
    if 'our_regret' in df.columns:
        cum_regret = df['our_regret'].cumsum()
        ax.plot(df['episode'], cum_regret, label='Self-Gradient', linewidth=2)

    # Plot baselines
    baseline_cols = [col for col in df.columns if col.endswith('_regret') and col != 'our_regret']
    for col in baseline_cols:
        name = col.replace('_regret', '').replace('_', ' ').title()
        cum_regret = df[col].cumsum()
        ax.plot(df['episode'], cum_regret, label=name, alpha=0.7)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Regret')
    ax.set_title('Cumulative Regret vs Episode (lower is better)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_gradient_error(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plot gradient prediction error over training."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if 'gradient_error' in df.columns:
        valid_data = df[df['gradient_error'].notna()]

        if len(valid_data) > 0:
            ax.plot(valid_data['episode'], valid_data['gradient_error'], alpha=0.5, label='Instantaneous')

            # Also plot EMA if available
            if 'grad_error_ema' in df.columns:
                valid_ema = df[df['grad_error_ema'].notna()]
                if len(valid_ema) > 0:
                    ax.plot(valid_ema['episode'], valid_ema['grad_error_ema'],
                           linewidth=2, label='EMA')

            ax.set_xlabel('Episode')
            ax.set_ylabel('Gradient Prediction MSE')
            ax.set_title('Gradient Prediction Error')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_planning_diagnostics(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plot planning-specific diagnostics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Planning weight over time
    ax = axes[0, 0]
    if 'planning_weight' in df.columns:
        ax.plot(df['episode'], df['planning_weight'])
        ax.set_xlabel('Episode')
        ax.set_ylabel('Planning Weight')
        ax.set_title('Planning Weight Over Time')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    # 2. Planning enabled (fraction over time)
    ax = axes[0, 1]
    if 'planning_enabled' in df.columns:
        window = 100
        rolling = df['planning_enabled'].rolling(window=window, min_periods=1).mean()
        ax.plot(df['episode'], rolling)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Fraction of Episodes with Planning')
        ax.set_title(f'Planning Enabled (rolling {window})')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    # 3. Planning score gap (when planning is enabled)
    ax = axes[1, 0]
    if 'planning_score_gap' in df.columns and 'planning_enabled' in df.columns:
        planning_episodes = df[df['planning_enabled'] > 0]
        if len(planning_episodes) > 0:
            ax.plot(planning_episodes['episode'], planning_episodes['planning_score_gap'])
            ax.set_xlabel('Episode')
            ax.set_ylabel('Score Gap (Best - Chosen)')
            ax.set_title('Planning Score Gap (when planning enabled)')
            ax.grid(True, alpha=0.3)

    # 4. Gradient error EMA with threshold
    ax = axes[1, 1]
    if 'grad_error_ema' in df.columns:
        ax.plot(df['episode'], df['grad_error_ema'], label='Grad Error EMA')

        # Try to infer threshold from when planning starts
        if 'planning_enabled' in df.columns:
            first_planning = df[df['planning_enabled'] > 0]
            if len(first_planning) > 0:
                first_ep = first_planning['episode'].iloc[0]
                threshold = df.loc[df['episode'] == first_ep, 'grad_error_ema'].values
                if len(threshold) > 0:
                    ax.axhline(threshold[0], color='r', linestyle='--',
                             label=f'Threshold ~{threshold[0]:.3f}')

        ax.set_xlabel('Episode')
        ax.set_ylabel('Gradient Error EMA')
        ax.set_title('Gradient Error EMA vs Threshold')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_all(run_dir: Path):
    """Generate all plots for a run."""
    df = load_metrics(run_dir)

    print(f"Generating plots for {run_dir.name}")

    plot_rewards(df, run_dir / 'rewards.png')
    print(f"  Saved rewards.png")

    plot_cumulative_regret(df, run_dir / 'cumulative_regret.png')
    print(f"  Saved cumulative_regret.png")

    if 'gradient_error' in df.columns and df['gradient_error'].notna().any():
        plot_gradient_error(df, run_dir / 'gradient_error.png')
        print(f"  Saved gradient_error.png")

    if 'planning_weight' in df.columns:
        plot_planning_diagnostics(df, run_dir / 'planning_diagnostics.png')
        print(f"  Saved planning_diagnostics.png")


def print_summary(run_dir: Path):
    """Print summary statistics for a run."""
    df = load_metrics(run_dir)

    print(f"\nSummary for {run_dir.name}")
    print("=" * 60)

    # Final 500 episodes
    window = min(500, len(df))

    if 'our_reward' in df.columns:
        final_reward = df['our_reward'].iloc[-window:].mean()
        final_std = df['our_reward'].iloc[-window:].std()
        cum_regret = df['our_regret'].sum()
        print(f"Self-Gradient Agent:")
        print(f"  Final {window} episodes reward: {final_reward:.3f} +/- {final_std:.3f}")
        print(f"  Cumulative regret: {cum_regret:.1f}")

    baseline_cols = [col for col in df.columns if col.endswith('_reward') and col != 'our_reward']
    for col in baseline_cols:
        name = col.replace('_reward', '').replace('_', ' ').title()
        final_reward = df[col].iloc[-window:].mean()
        final_std = df[col].iloc[-window:].std()

        regret_col = col.replace('_reward', '_regret')
        cum_regret = df[regret_col].sum()

        print(f"{name}:")
        print(f"  Final {window} episodes reward: {final_reward:.3f} +/- {final_std:.3f}")
        print(f"  Cumulative regret: {cum_regret:.1f}")

    # Gradient prediction
    if 'gradient_error' in df.columns:
        valid_errors = df[df['gradient_error'].notna()]['gradient_error']
        if len(valid_errors) > 0:
            final_error = valid_errors.iloc[-min(100, len(valid_errors)):].mean()
            print(f"\nGradient Prediction:")
            print(f"  Final error: {final_error:.6f}")

            if 'grad_error_ema' in df.columns:
                final_ema = df['grad_error_ema'].iloc[-1]
                print(f"  Final EMA: {final_ema:.6f}")

    # Planning statistics
    if 'planning_enabled' in df.columns:
        total_episodes = len(df)
        planning_episodes = df['planning_enabled'].sum()
        fraction_planning = planning_episodes / total_episodes

        print(f"\nPlanning Statistics:")
        print(f"  Episodes with planning: {int(planning_episodes)}/{total_episodes} ({fraction_planning:.1%})")

        if 'planning_weight' in df.columns:
            avg_weight = df[df['planning_enabled'] > 0]['planning_weight'].mean()
            if not np.isnan(avg_weight):
                print(f"  Average planning weight (when enabled): {avg_weight:.3f}")

        if 'planning_score_gap' in df.columns:
            planning_df = df[df['planning_enabled'] > 0]
            if len(planning_df) > 0:
                avg_gap = planning_df['planning_score_gap'].mean()
                print(f"  Average score gap (best - chosen): {avg_gap:.4f}")
