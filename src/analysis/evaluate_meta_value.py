"""
Evaluate meta-value network quality.

This script sanity-checks whether V_meta(θ) actually predicts parameter quality.
Uses logged data from training runs to correlate predicted values with actual rewards.

Usage:
    python -m src.analysis.evaluate_meta_value <run_directory>

Example:
    python -m src.analysis.evaluate_meta_value logs/planning_meta_value_reward_only/run_2025-11-13_*/
"""

import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

from ..models.gradient_world_model import MetaValueNetwork


def load_checkpoint(checkpoint_path: Path) -> Dict:
    """Load agent checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location='cpu')


def evaluate_meta_value_from_logs(run_dir: Path):
    """
    Evaluate meta-value network quality using logged training data.

    For each episode with logged parameters, we evaluate V_meta(θ) and
    correlate it with the actual reward observed in that episode.
    """
    print(f"\nEvaluating meta-value network for: {run_dir.name}")
    print("=" * 70)

    # Load metrics
    metrics_path = run_dir / 'metrics.csv'
    if not metrics_path.exists():
        print(f"Error: No metrics.csv found in {run_dir}")
        return

    df = pd.read_csv(metrics_path)

    # Check if we have meta-value data
    if 'meta_value_current' not in df.columns:
        print("Warning: No meta_value_current column found. Run may not have meta-value logging.")
        return

    # Extract meta-value predictions and rewards
    meta_vals = df['meta_value_current'].values
    rewards = df['our_reward'].values

    print(f"\nDataset:")
    print(f"  Total episodes: {len(df)}")
    print(f"  Meta-value range: [{meta_vals.min():.4f}, {meta_vals.max():.4f}]")
    print(f"  Reward range: [{rewards.min():.4f}, {rewards.max():.4f}]")

    # Overall correlation
    corr_all = np.corrcoef(meta_vals, rewards)[0, 1]
    print(f"\nOverall Correlation:")
    print(f"  V(θ) vs reward (all episodes): {corr_all:.4f}")

    # Correlation during planning episodes only
    if 'planning_enabled' in df.columns:
        planning_mask = df['planning_enabled'] > 0.5
        if planning_mask.sum() > 0:
            corr_planning = np.corrcoef(
                meta_vals[planning_mask],
                rewards[planning_mask]
            )[0, 1]
            print(f"  V(θ) vs reward (planning episodes): {corr_planning:.4f}")
            print(f"  Planning episodes: {planning_mask.sum()}/{len(df)}")

    # Calibration: bin by predicted value and show average reward
    n_bins = 5
    bins = np.percentile(meta_vals, np.linspace(0, 100, n_bins + 1))
    bin_indices = np.digitize(meta_vals, bins[1:-1])

    print(f"\nCalibration (binned by V(θ)):")
    print(f"  {'Bin':<5} {'V(θ) Range':<25} {'Avg Reward':<15} {'Std':<10} {'N':<10}")
    print(f"  {'-'*70}")

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_meta_min = meta_vals[mask].min()
            bin_meta_max = meta_vals[mask].max()
            bin_reward_mean = rewards[mask].mean()
            bin_reward_std = rewards[mask].std()
            bin_count = mask.sum()

            print(f"  {i+1:<5} [{bin_meta_min:>7.4f}, {bin_meta_max:>7.4f}] "
                  f"{bin_reward_mean:>10.4f}     {bin_reward_std:>8.4f}   {bin_count:>8}")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot
    axes[0].scatter(meta_vals, rewards, alpha=0.3, s=10)
    z = np.polyfit(meta_vals, rewards, 1)
    p = np.poly1d(z)
    x_line = np.linspace(meta_vals.min(), meta_vals.max(), 100)
    axes[0].plot(x_line, p(x_line), "r--", linewidth=2,
                label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    axes[0].set_xlabel('Predicted V(θ)')
    axes[0].set_ylabel('Actual Reward')
    axes[0].set_title(f'Meta-Value vs Reward (corr={corr_all:.3f})')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Calibration plot
    bin_means_meta = []
    bin_means_reward = []
    bin_stds_reward = []

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_means_meta.append(meta_vals[mask].mean())
            bin_means_reward.append(rewards[mask].mean())
            bin_stds_reward.append(rewards[mask].std())

    axes[1].errorbar(bin_means_meta, bin_means_reward,
                    yerr=bin_stds_reward, fmt='o-', linewidth=2,
                    capsize=5, capthick=2, markersize=8)
    axes[1].set_xlabel('Meta-value (binned)')
    axes[1].set_ylabel('Average Reward in Bin')
    axes[1].set_title('Meta-Value Calibration')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = run_dir / 'meta_value_evaluation.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {save_path}")

    # Interpretation
    print(f"\nInterpretation:")
    if abs(corr_all) < 0.1:
        print("  POOR: Meta-value shows almost no correlation with actual rewards.")
        print("  The meta-value network is not learning a meaningful signal.")
    elif abs(corr_all) < 0.3:
        print("  WEAK: Meta-value shows weak correlation with actual rewards.")
        print("  Planning based on V(θ) may provide limited benefit.")
    elif abs(corr_all) < 0.5:
        print("  MODERATE: Meta-value shows moderate correlation with actual rewards.")
        print("  V(θ) captures some signal but is noisy.")
    else:
        print("  STRONG: Meta-value shows strong correlation with actual rewards.")
        print("  V(θ) is a good predictor of parameter quality.")

    # Check if correlation is positive or negative
    if corr_all < 0:
        print("  WARNING: Negative correlation! V(θ) predicts OPPOSITE of reward.")
        print("  This suggests the meta-value network is learning the wrong signal.")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python -m src.analysis.evaluate_meta_value <run_directory>")
        print("\nExample:")
        print("  python -m src.analysis.evaluate_meta_value logs/planning_meta_value_reward_only/run_2025-11-13_14-02-55/")
        sys.exit(1)

    run_dir = Path(sys.argv[1])

    if not run_dir.exists():
        print(f"Error: Directory does not exist: {run_dir}")
        sys.exit(1)

    evaluate_meta_value_from_logs(run_dir)


if __name__ == '__main__':
    main()
