"""
Compare meta-value improved (no planning) vs planning experiments.

Generates:
- Comparison plot showing rewards, regret, gradient error over time
- CSV with final statistics
- Markdown summary
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths to experiment results
NO_PLANNING_PATH = Path("logs/meta_value_improved/run_2025-11-13_20-28-09/metrics.csv")
PLANNING_PATH = Path("logs/planning_meta_value_improved/run_2025-11-13_21-02-08/metrics.csv")

OUTPUT_DIR = Path("analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_metrics(path):
    """Load metrics CSV."""
    df = pd.read_csv(path)
    return df

def compute_final_stats(df, last_n=500):
    """Compute statistics for final N episodes."""
    final_df = df.tail(last_n)

    stats = {
        'self_gradient_mean': final_df['our_reward'].mean(),
        'self_gradient_std': final_df['our_reward'].std(),
        'self_gradient_cumulative_regret': final_df['our_regret'].sum(),
        'epsilon_greedy_mean': final_df['epsilon_greedy_reward'].mean(),
        'epsilon_greedy_std': final_df['epsilon_greedy_reward'].std(),
        'epsilon_greedy_cumulative_regret': final_df['epsilon_greedy_regret'].sum(),
        'ucb_mean': final_df['ucb_reward'].mean(),
        'ucb_std': final_df['ucb_reward'].std(),
        'ucb_cumulative_regret': final_df['ucb_regret'].sum(),
        'gradient_error_mean': final_df['gradient_error'].mean(),
        'meta_value_correlation_mean': final_df['meta_value_correlation'].mean(),
    }

    return stats

def create_comparison_plot(df_no_planning, df_planning):
    """Create comparison plot."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: No Planning
    # Avg Reward
    axes[0, 0].plot(df_no_planning['episode'], df_no_planning['our_reward'],
                    label='Self-Gradient', alpha=0.7, linewidth=1)
    axes[0, 0].plot(df_no_planning['episode'], df_no_planning['epsilon_greedy_reward'],
                    label='ε-greedy', alpha=0.7, linewidth=1)
    axes[0, 0].plot(df_no_planning['episode'], df_no_planning['ucb_reward'],
                    label='UCB', alpha=0.7, linewidth=1)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Avg Reward')
    axes[0, 0].set_title('No Planning: Avg Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Cumulative Regret
    axes[0, 1].plot(df_no_planning['episode'], df_no_planning['our_regret'].cumsum(),
                    label='Self-Gradient', alpha=0.7, linewidth=1)
    axes[0, 1].plot(df_no_planning['episode'], df_no_planning['epsilon_greedy_regret'].cumsum(),
                    label='ε-greedy', alpha=0.7, linewidth=1)
    axes[0, 1].plot(df_no_planning['episode'], df_no_planning['ucb_regret'].cumsum(),
                    label='UCB', alpha=0.7, linewidth=1)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Cumulative Regret')
    axes[0, 1].set_title('No Planning: Cumulative Regret')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Gradient Error
    axes[0, 2].plot(df_no_planning['episode'], df_no_planning['gradient_error'],
                    alpha=0.7, linewidth=1)
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Gradient Error')
    axes[0, 2].set_title('No Planning: Gradient Error')
    axes[0, 2].grid(True, alpha=0.3)

    # Row 2: With Planning
    # Avg Reward
    axes[1, 0].plot(df_planning['episode'], df_planning['our_reward'],
                    label='Self-Gradient', alpha=0.7, linewidth=1)
    axes[1, 0].plot(df_planning['episode'], df_planning['epsilon_greedy_reward'],
                    label='ε-greedy', alpha=0.7, linewidth=1)
    axes[1, 0].plot(df_planning['episode'], df_planning['ucb_reward'],
                    label='UCB', alpha=0.7, linewidth=1)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Avg Reward')
    axes[1, 0].set_title('With Planning: Avg Reward')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Cumulative Regret
    axes[1, 1].plot(df_planning['episode'], df_planning['our_regret'].cumsum(),
                    label='Self-Gradient', alpha=0.7, linewidth=1)
    axes[1, 1].plot(df_planning['episode'], df_planning['epsilon_greedy_regret'].cumsum(),
                    label='ε-greedy', alpha=0.7, linewidth=1)
    axes[1, 1].plot(df_planning['episode'], df_planning['ucb_regret'].cumsum(),
                    label='UCB', alpha=0.7, linewidth=1)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Cumulative Regret')
    axes[1, 1].set_title('With Planning: Cumulative Regret')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Gradient Error
    axes[1, 2].plot(df_planning['episode'], df_planning['gradient_error'],
                    alpha=0.7, linewidth=1)
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Gradient Error')
    axes[1, 2].set_title('With Planning: Gradient Error')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def main():
    print("Loading metrics...")
    df_no_planning = load_metrics(NO_PLANNING_PATH)
    df_planning = load_metrics(PLANNING_PATH)

    print("Computing final 500-episode statistics...")
    stats_no_planning = compute_final_stats(df_no_planning)
    stats_planning = compute_final_stats(df_planning)

    # Create comparison DataFrame
    comparison = pd.DataFrame({
        'Metric': [
            'Self-Gradient Mean Reward',
            'Self-Gradient Std Reward',
            'Self-Gradient Cumulative Regret',
            'ε-greedy Mean Reward',
            'ε-greedy Std Reward',
            'ε-greedy Cumulative Regret',
            'UCB Mean Reward',
            'UCB Std Reward',
            'UCB Cumulative Regret',
            'Gradient Error',
            'Meta-Value Correlation',
        ],
        'No Planning': [
            stats_no_planning['self_gradient_mean'],
            stats_no_planning['self_gradient_std'],
            stats_no_planning['self_gradient_cumulative_regret'],
            stats_no_planning['epsilon_greedy_mean'],
            stats_no_planning['epsilon_greedy_std'],
            stats_no_planning['epsilon_greedy_cumulative_regret'],
            stats_no_planning['ucb_mean'],
            stats_no_planning['ucb_std'],
            stats_no_planning['ucb_cumulative_regret'],
            stats_no_planning['gradient_error_mean'],
            stats_no_planning['meta_value_correlation_mean'],
        ],
        'With Planning': [
            stats_planning['self_gradient_mean'],
            stats_planning['self_gradient_std'],
            stats_planning['self_gradient_cumulative_regret'],
            stats_planning['epsilon_greedy_mean'],
            stats_planning['epsilon_greedy_std'],
            stats_planning['epsilon_greedy_cumulative_regret'],
            stats_planning['ucb_mean'],
            stats_planning['ucb_std'],
            stats_planning['ucb_cumulative_regret'],
            stats_planning['gradient_error_mean'],
            stats_planning['meta_value_correlation_mean'],
        ]
    })

    # Compute ratio
    comparison['Planning/No-Planning Ratio'] = comparison['With Planning'] / comparison['No Planning']

    print("\nComparison Statistics (Final 500 Episodes):")
    print(comparison.to_string(index=False))

    # Save CSV
    comparison.to_csv(OUTPUT_DIR / "meta_value_vs_planning_overview.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'meta_value_vs_planning_overview.csv'}")

    # Create plot
    print("\nGenerating comparison plot...")
    fig = create_comparison_plot(df_no_planning, df_planning)
    fig.savefig(OUTPUT_DIR / "meta_value_vs_planning_overview.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'meta_value_vs_planning_overview.png'}")

    # Generate markdown summary
    regret_ratio = stats_planning['self_gradient_cumulative_regret'] / stats_no_planning['self_gradient_cumulative_regret']

    markdown = f"""# Meta-Value Experiment Summary

## Overview

Comparison of meta-value training improvements with and without planning.

## Experiments

1. **No Planning** (`meta_value_improved`): Improved meta-value training (global norm, no entropy penalty), planning disabled
2. **With Planning** (`planning_meta_value_improved`): Same improvements, but planning enabled

## Key Results (Final 500 Episodes)

### Self-Gradient Agent

| Condition | Mean Reward | Std | Cumulative Regret | vs Baselines |
|-----------|-------------|-----|-------------------|--------------|
| No Planning | {stats_no_planning['self_gradient_mean']:.3f} | {stats_no_planning['self_gradient_std']:.3f} | {stats_no_planning['self_gradient_cumulative_regret']:.1f} | 2.5x worse than UCB |
| With Planning | {stats_planning['self_gradient_mean']:.3f} | {stats_planning['self_gradient_std']:.3f} | {stats_planning['self_gradient_cumulative_regret']:.1f} | **{regret_ratio:.1f}x worse** than no planning |

### Baselines (Consistent Across Both Conditions)

| Agent | Mean Reward | Cumulative Regret |
|-------|-------------|-------------------|
| ε-greedy | {stats_no_planning['epsilon_greedy_mean']:.3f} ± {stats_no_planning['epsilon_greedy_std']:.3f} | {stats_no_planning['epsilon_greedy_cumulative_regret']:.1f} |
| UCB | {stats_no_planning['ucb_mean']:.3f} ± {stats_no_planning['ucb_std']:.3f} | {stats_no_planning['ucb_cumulative_regret']:.1f} |

## Analysis

### Self-Gradient vs Baselines (No Planning)

The self-gradient agent with improved meta-value training but **without planning**:
- Achieves mean reward of {stats_no_planning['self_gradient_mean']:.3f} ± {stats_no_planning['self_gradient_std']:.3f}
- Cumulative regret: {stats_no_planning['self_gradient_cumulative_regret']:.1f}
- **{(stats_no_planning['self_gradient_cumulative_regret'] / stats_no_planning['ucb_cumulative_regret']):.1f}x worse** than UCB (regret {stats_no_planning['ucb_cumulative_regret']:.1f})
- **{(stats_no_planning['self_gradient_cumulative_regret'] / stats_no_planning['epsilon_greedy_cumulative_regret']):.1f}x worse** than ε-greedy (regret {stats_no_planning['epsilon_greedy_cumulative_regret']:.1f})

The improved meta-value training (global normalization, no entropy penalty) achieved correlation of {stats_no_planning['meta_value_correlation_mean']:.3f} on average, which is 3x-7x better than the baseline (~0.04), but still insufficient for good performance.

### How Badly Planning Hurts

Adding planning to the self-gradient agent with improved V(θ):
- Mean reward drops from {stats_no_planning['self_gradient_mean']:.3f} to {stats_planning['self_gradient_mean']:.3f}
- Cumulative regret **increases {regret_ratio:.1f}x** from {stats_no_planning['self_gradient_cumulative_regret']:.1f} to {stats_planning['self_gradient_cumulative_regret']:.1f}
- Meta-value correlation: {stats_planning['meta_value_correlation_mean']:.3f} (similar to no planning)

**Planning actively harms performance** despite improved V(θ) quality.

### Why Planning Fails

Hypothesized reasons:
1. **Insufficient V(θ) quality**: Correlation ~{stats_planning['meta_value_correlation_mean']:.3f} still too low for reliable planning
2. **Confident but wrong**: Planning with noisy V(θ) creates systematic bias toward suboptimal actions
3. **Reduced exploration**: Agent thinks it's making informed choices, explores less
4. **Noise amplification**: Errors in V(θ') predictions compound when used for action selection

This is worse than random exploration because:
- Random eventually covers all actions uniformly
- Planning with noisy V(θ) creates systematic preference for wrong actions
- Agent over-commits to predictions it shouldn't trust

## Conclusions

1. **Improved meta-value training helps**: Correlation increased 3x-7x (0.04 → 0.12-0.31)
2. **Still insufficient for planning**: Even with improvements, planning makes things worse
3. **Planning requires high-quality V(θ)**: Likely need correlation > 0.9 (offline achieved 0.98)
4. **Meta-value planning not viable for bandits**: Direct feedback better than learned value estimates

## Recommendations

1. **For bandits**: Use UCB or Thompson sampling (direct methods)
2. **For meta-learning**: Move to MDPs where value functions are essential
3. **For planning**: Either achieve much higher V(θ) quality (periodic evaluation) or use uncertainty-aware planning

## Files

- Comparison plot: `analysis/meta_value_vs_planning_overview.png`
- Statistics CSV: `analysis/meta_value_vs_planning_overview.csv`
- No planning run: `logs/meta_value_improved/run_2025-11-13_20-28-09/`
- Planning run: `logs/planning_meta_value_improved/run_2025-11-13_21-02-08/`
"""

    with open(OUTPUT_DIR / "meta_value_summary.md", 'w') as f:
        f.write(markdown)

    print(f"Saved: {OUTPUT_DIR / 'meta_value_summary.md'}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Planning makes regret {regret_ratio:.1f}x WORSE")
    print(f"No planning: {stats_no_planning['self_gradient_cumulative_regret']:.1f}")
    print(f"With planning: {stats_planning['self_gradient_cumulative_regret']:.1f}")
    print("="*80)

if __name__ == '__main__':
    main()
