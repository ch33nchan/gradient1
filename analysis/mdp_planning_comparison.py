"""Compare MDP baseline vs planning performance.

This script:
1. Loads baseline and planning run metrics
2. Compares learning curves (returns, success rates)
3. Generates comparison plots and summary report

Usage:
    python -m analysis.mdp_planning_comparison \
        --baseline-dir logs/chain_mdp_baseline/run_2025-11-19_06-18-51 \
        --planning-dir logs/chain_mdp_planning/run_2025-11-19_12-34-56 \
        --output-dir analysis/planning_comparison
"""

import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_run_metrics(run_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load metrics and config from training run.

    Args:
        run_dir: Path to training run directory

    Returns:
        metrics_df: Training metrics DataFrame
        config: Experiment configuration
    """
    # Load metrics
    metrics_path = run_dir / 'metrics.csv'
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics not found: {metrics_path}")

    metrics_df = pd.read_csv(metrics_path)

    # Load config
    config_path = run_dir / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Load planning metrics if available
    planning_metrics_path = run_dir / 'planning_metrics.yaml'
    if planning_metrics_path.exists():
        with open(planning_metrics_path, 'r') as f:
            planning_metrics = yaml.load(f, Loader=yaml.UnsafeLoader)
            config['planning_metrics'] = planning_metrics

    return metrics_df, config


def compute_summary_statistics(metrics_df: pd.DataFrame, window: int = 20) -> Dict[str, float]:
    """Compute summary statistics from metrics.

    Args:
        metrics_df: Training metrics DataFrame
        window: Window for computing final performance

    Returns:
        stats: Dictionary with summary statistics
    """
    # Final performance (last window episodes)
    final_returns = metrics_df['episode_returns'].iloc[-window:].values
    final_success = metrics_df['success_rate'].iloc[-window:].values

    # Learning speed (episode to reach threshold)
    success_threshold = 0.8
    success_ma = metrics_df['success_rate'].rolling(window=10).mean()
    episodes_to_threshold = None
    for i, val in enumerate(success_ma):
        if val >= success_threshold:
            episodes_to_threshold = i + 1
            break

    stats = {
        'final_mean_return': np.mean(final_returns),
        'final_std_return': np.std(final_returns),
        'final_success_rate': np.mean(final_success),
        'best_return': np.max(metrics_df['episode_returns']),
        'total_episodes': len(metrics_df),
        'episodes_to_80pct_success': episodes_to_threshold,
    }

    return stats


def plot_comparison(
    baseline_df: pd.DataFrame,
    planning_df: pd.DataFrame,
    output_path: Path,
    window: int = 20
):
    """Plot comparison of baseline vs planning.

    Args:
        baseline_df: Baseline metrics
        planning_df: Planning metrics
        output_path: Path to save plot
        window: Window for moving average
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Helper for moving average
    def moving_average(x, w):
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w) / w, mode='valid')

    # Plot 1: Episode returns
    baseline_returns = baseline_df['episode_returns'].values
    planning_returns = planning_df['episode_returns'].values

    baseline_ma = moving_average(baseline_returns, window)
    planning_ma = moving_average(planning_returns, window)

    axes[0, 0].plot(baseline_returns, alpha=0.2, color='blue')
    axes[0, 0].plot(range(window - 1, len(baseline_returns)), baseline_ma,
                    label='Baseline', color='blue', linewidth=2)

    axes[0, 0].plot(planning_returns, alpha=0.2, color='red')
    axes[0, 0].plot(range(window - 1, len(planning_returns)), planning_ma,
                    label='Planning', color='red', linewidth=2)

    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Episode Return')
    axes[0, 0].set_title(f'Training Returns ({window}-episode MA)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Success rate
    baseline_success = baseline_df['success_rate'].values
    planning_success = planning_df['success_rate'].values

    baseline_success_ma = moving_average(baseline_success, window)
    planning_success_ma = moving_average(planning_success, window)

    axes[0, 1].plot(baseline_success, alpha=0.2, color='blue')
    axes[0, 1].plot(range(window - 1, len(baseline_success)), baseline_success_ma,
                    label='Baseline', color='blue', linewidth=2)

    axes[0, 1].plot(planning_success, alpha=0.2, color='red')
    axes[0, 1].plot(range(window - 1, len(planning_success)), planning_success_ma,
                    label='Planning', color='red', linewidth=2)

    axes[0, 1].axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='80% threshold')

    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Success Rate')
    axes[0, 1].set_title(f'Success Rate ({window}-episode MA)')
    axes[0, 1].set_ylim([-0.05, 1.05])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Episode lengths
    baseline_lengths = baseline_df['episode_lengths'].values
    planning_lengths = planning_df['episode_lengths'].values

    baseline_lengths_ma = moving_average(baseline_lengths, window)
    planning_lengths_ma = moving_average(planning_lengths, window)

    axes[1, 0].plot(baseline_lengths, alpha=0.2, color='blue')
    axes[1, 0].plot(range(window - 1, len(baseline_lengths)), baseline_lengths_ma,
                    label='Baseline', color='blue', linewidth=2)

    axes[1, 0].plot(planning_lengths, alpha=0.2, color='red')
    axes[1, 0].plot(range(window - 1, len(planning_lengths)), planning_lengths_ma,
                    label='Planning', color='red', linewidth=2)

    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Episode Length')
    axes[1, 0].set_title(f'Episode Lengths ({window}-episode MA)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Cumulative returns
    baseline_cumulative = np.cumsum(baseline_returns)
    planning_cumulative = np.cumsum(planning_returns)

    axes[1, 1].plot(baseline_cumulative, label='Baseline', color='blue', linewidth=2)
    axes[1, 1].plot(planning_cumulative, label='Planning', color='red', linewidth=2)

    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Cumulative Return')
    axes[1, 1].set_title('Cumulative Returns')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot: {output_path}")
    plt.close(fig)


def generate_summary_report(
    baseline_stats: Dict[str, float],
    planning_stats: Dict[str, float],
    baseline_config: Dict[str, Any],
    planning_config: Dict[str, Any],
    output_path: Path
):
    """Generate markdown summary report.

    Args:
        baseline_stats: Baseline summary statistics
        planning_stats: Planning summary statistics
        baseline_config: Baseline configuration
        planning_config: Planning configuration
        output_path: Path to save report
    """
    report = f"""# MDP Planning vs Baseline Comparison

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Experiment Setup

### Baseline
- **Type**: {baseline_config.get('experiment', {}).get('type', 'mdp')}
- **Agent**: REINFORCE
- **Learning rate**: {baseline_config.get('agent', {}).get('learning_rate', 'N/A')}
- **Hidden dim**: {baseline_config.get('agent', {}).get('hidden_dim', 'N/A')}
- **Entropy bonus**: {baseline_config.get('agent', {}).get('entropy_bonus', 'N/A')}

### Planning
- **Type**: {planning_config.get('experiment', {}).get('type', 'mdp_planning')}
- **Agent**: REINFORCE + Meta-Value Planning
- **Learning rate**: {planning_config.get('agent', {}).get('learning_rate', 'N/A')}
- **Hidden dim**: {planning_config.get('agent', {}).get('hidden_dim', 'N/A')}
- **Entropy bonus**: {planning_config.get('agent', {}).get('entropy_bonus', 'N/A')}
- **Planning weight**: {planning_config.get('agent', {}).get('planning_weight', 'N/A')}
- **Meta-value model**: {planning_config.get('agent', {}).get('meta_value_model_path', 'N/A')}

## Results Summary

### Final Performance (last 20 episodes)

| Metric | Baseline | Planning | Difference | Improvement |
|--------|----------|----------|------------|-------------|
"""

    # Return
    baseline_return = baseline_stats['final_mean_return']
    planning_return = planning_stats['final_mean_return']
    return_diff = planning_return - baseline_return
    return_improvement = (return_diff / baseline_return * 100) if baseline_return != 0 else 0

    report += f"| Mean Return | {baseline_return:.3f} ± {baseline_stats['final_std_return']:.3f} | "
    report += f"{planning_return:.3f} ± {planning_stats['final_std_return']:.3f} | "
    report += f"{return_diff:+.3f} | {return_improvement:+.1f}% |\n"

    # Success rate
    baseline_success = baseline_stats['final_success_rate']
    planning_success = planning_stats['final_success_rate']
    success_diff = planning_success - baseline_success

    report += f"| Success Rate | {baseline_success:.1%} | {planning_success:.1%} | "
    report += f"{success_diff:+.1%} | - |\n"

    # Best return
    baseline_best = baseline_stats['best_return']
    planning_best = planning_stats['best_return']

    report += f"| Best Return | {baseline_best:.3f} | {planning_best:.3f} | "
    report += f"{planning_best - baseline_best:+.3f} | - |\n"

    report += f"""
### Learning Speed

| Metric | Baseline | Planning |
|--------|----------|----------|
"""

    baseline_to_threshold = baseline_stats['episodes_to_80pct_success']
    planning_to_threshold = planning_stats['episodes_to_80pct_success']

    report += f"| Episodes to 80% success | "
    report += f"{baseline_to_threshold if baseline_to_threshold else 'N/A'} | "
    report += f"{planning_to_threshold if planning_to_threshold else 'N/A'} |\n"

    report += f"""
## Planning Metrics

"""

    if 'planning_metrics' in planning_config:
        pm = planning_config['planning_metrics']
        report += f"- **Mean meta-value score**: {pm.get('mean_meta_value', 'N/A'):.3f}\n"
        report += f"- **Final meta-value score**: {pm.get('final_meta_value', 'N/A'):.3f}\n"
        report += f"- **Mean base entropy**: {pm.get('mean_base_entropy', 'N/A'):.3f}\n"
        report += f"- **Mean planned entropy**: {pm.get('mean_planned_entropy', 'N/A'):.3f}\n"
        report += f"- **Planning weight**: {pm.get('planning_weight', 'N/A')}\n"
    else:
        report += "No planning metrics available.\n"

    report += f"""
## Interpretation

### Performance Comparison
"""

    if return_improvement > 5:
        report += f"- ✓ **Planning improves performance**: {return_improvement:+.1f}% higher final return\n"
    elif return_improvement < -5:
        report += f"- ✗ **Planning hurts performance**: {return_improvement:+.1f}% lower final return\n"
    else:
        report += f"- ~ **Neutral effect**: Performance difference within ±5%\n"

    if planning_success > baseline_success:
        report += f"- ✓ Planning achieves higher success rate ({planning_success:.1%} vs {baseline_success:.1%})\n"
    else:
        report += f"- ~ Similar success rates ({planning_success:.1%} vs {baseline_success:.1%})\n"

    report += """
### Learning Speed
"""

    if baseline_to_threshold and planning_to_threshold:
        if planning_to_threshold < baseline_to_threshold:
            speedup = baseline_to_threshold / planning_to_threshold
            report += f"- ✓ **Planning accelerates learning**: {speedup:.1f}x faster to 80% success\n"
        else:
            report += f"- ~ Similar learning speeds\n"
    else:
        report += "- Unable to compare learning speeds (threshold not reached)\n"

    report += """
## Conclusion

"""

    if return_improvement > 5 or (planning_to_threshold and baseline_to_threshold and planning_to_threshold < baseline_to_threshold):
        report += "**Meta-value planning shows positive results**: It either improves final performance or accelerates learning.\n"
    elif abs(return_improvement) <= 5:
        report += "**Neutral result**: Planning has minimal impact on performance. This could indicate:\n"
        report += "- The baseline is already near-optimal for this task\n"
        report += "- Planning weight may need tuning\n"
        report += "- Meta-value model may need more training data\n"
    else:
        report += "**Negative result**: Planning hurts performance. Possible causes:\n"
        report += "- Meta-value model predictions are inaccurate\n"
        report += "- Planning weight is too high\n"
        report += "- Planning strategy needs refinement\n"

    report += """
## Files Generated
- `comparison.png`: Learning curves comparison
- `summary.md`: This report
"""

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Saved summary report: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare MDP baseline vs planning performance'
    )
    parser.add_argument(
        '--baseline-dir',
        type=str,
        required=True,
        help='Path to baseline training run directory'
    )
    parser.add_argument(
        '--planning-dir',
        type=str,
        required=True,
        help='Path to planning training run directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='analysis/planning_comparison',
        help='Output directory for comparison results'
    )
    parser.add_argument(
        '--window',
        type=int,
        default=20,
        help='Window size for moving averages (default: 20)'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MDP Planning vs Baseline Comparison")
    print("=" * 80)

    # Load metrics
    print(f"\nLoading baseline from: {args.baseline_dir}")
    baseline_df, baseline_config = load_run_metrics(Path(args.baseline_dir))
    print(f"  Episodes: {len(baseline_df)}")

    print(f"\nLoading planning from: {args.planning_dir}")
    planning_df, planning_config = load_run_metrics(Path(args.planning_dir))
    print(f"  Episodes: {len(planning_df)}")

    # Compute statistics
    print("\nComputing summary statistics...")
    baseline_stats = compute_summary_statistics(baseline_df, window=args.window)
    planning_stats = compute_summary_statistics(planning_df, window=args.window)

    print("\nBaseline final performance:")
    print(f"  Mean return: {baseline_stats['final_mean_return']:.3f} ± {baseline_stats['final_std_return']:.3f}")
    print(f"  Success rate: {baseline_stats['final_success_rate']:.1%}")

    print("\nPlanning final performance:")
    print(f"  Mean return: {planning_stats['final_mean_return']:.3f} ± {planning_stats['final_std_return']:.3f}")
    print(f"  Success rate: {planning_stats['final_success_rate']:.1%}")

    # Generate plots
    print("\nGenerating comparison plots...")
    plot_comparison(
        baseline_df, planning_df,
        output_dir / 'comparison.png',
        window=args.window
    )

    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(
        baseline_stats, planning_stats,
        baseline_config, planning_config,
        output_dir / 'summary.md'
    )

    print(f"\n{'=' * 80}")
    print(f"Comparison complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
