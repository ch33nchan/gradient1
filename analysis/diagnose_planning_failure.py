"""
Diagnose why planning hurts performance.

Analyzes planning trace to identify systematic errors:
1. Does planning pick wrong arms despite high meta-values?
2. Are meta-values inversely correlated with actual rewards?
3. Does planning have bugs (off-by-one, wrong normalization)?
4. What's the relationship between meta-value estimates and arm quality?

Usage:
    python analysis/diagnose_planning_failure.py \
        --trace-path logs/planning_test_instrumented/run_XXX/planning_trace.csv \
        --output-dir analysis
"""

import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_trace(path):
    """Load planning trace."""
    return pd.read_csv(path)

def analyze_arm_selection(df, optimal_arm=3):
    """Analyze which arms are being selected vs optimal."""
    print("="*80)
    print("ARM SELECTION ANALYSIS")
    print("="*80)

    # Count how often each arm is chosen
    arm_counts = df['chosen_arm'].value_counts().sort_index()
    print(f"\nArm selection frequency (out of {len(df)} planning decisions):")
    for arm, count in arm_counts.items():
        is_optimal = " [OPTIMAL]" if arm == optimal_arm else ""
        print(f"  Arm {arm}: {count} ({count/len(df)*100:.1f}%){is_optimal}")

    # How often does planning choose optimal arm?
    optimal_chosen = (df['chosen_arm'] == optimal_arm).sum()
    print(f"\nOptimal arm chosen: {optimal_chosen}/{len(df)} ({optimal_chosen/len(df)*100:.1f}%)")

    # How often does planning choose the arm with best meta-value?
    chose_best_mv = df['chose_best_meta_value'].sum()
    print(f"Chose arm with best meta-value: {chose_best_mv}/{len(df)} ({chose_best_mv/len(df)*100:.1f}%)")

    # Which arm has best meta-value most often?
    best_mv_arm_counts = df['best_meta_value_arm'].value_counts().sort_index()
    print(f"\nArm with highest meta-value frequency:")
    for arm, count in best_mv_arm_counts.items():
        is_optimal = " [OPTIMAL]" if arm == optimal_arm else ""
        print(f"  Arm {arm}: {count} ({count/len(df)*100:.1f}%){is_optimal}")

def analyze_meta_value_quality(df, optimal_arm=3):
    """Analyze quality of meta-value estimates."""
    print("\n" + "="*80)
    print("META-VALUE QUALITY ANALYSIS")
    print("="*80)

    # Get meta-value estimates for each arm
    meta_values = {}
    for arm in range(5):
        meta_values[arm] = df[f'meta_value_arm_{arm}'].values

    # Print average meta-value for each arm
    print(f"\nAverage meta-value by arm:")
    for arm in range(5):
        is_optimal = " [OPTIMAL]" if arm == optimal_arm else ""
        mean_mv = np.mean(meta_values[arm])
        std_mv = np.std(meta_values[arm])
        print(f"  Arm {arm}: {mean_mv:.6f} ± {std_mv:.6f}{is_optimal}")

    # Does optimal arm have highest average meta-value?
    avg_meta_values = [np.mean(meta_values[arm]) for arm in range(5)]
    best_avg_arm = np.argmax(avg_meta_values)
    print(f"\nArm with highest average meta-value: {best_avg_arm}")
    if best_avg_arm == optimal_arm:
        print("  ✓ Correct! Meta-value correctly identifies optimal arm on average")
    else:
        print(f"  ✗ Wrong! Optimal arm is {optimal_arm}, but meta-value prefers {best_avg_arm}")
        print(f"    Meta-value gap: {avg_meta_values[best_avg_arm] - avg_meta_values[optimal_arm]:.6f}")

    # Correlation between arms (do meta-values vary consistently?)
    print("\nMeta-value variation:")
    for arm in range(5):
        print(f"  Arm {arm} std: {np.std(meta_values[arm]):.6f}")

def analyze_planning_probabilities(df, optimal_arm=3):
    """Analyze how planning probabilities are computed."""
    print("\n" + "="*80)
    print("PLANNING PROBABILITY ANALYSIS")
    print("="*80)

    # Check if planning probabilities sum to 1
    planning_probs_sum = sum([df[f'planning_prob_arm_{i}'].mean() for i in range(5)])
    print(f"\nSum of planning probabilities: {planning_probs_sum:.6f} (should be ~1.0)")

    # Average planning probability per arm
    print(f"\nAverage planning probability by arm:")
    for arm in range(5):
        is_optimal = " [OPTIMAL]" if arm == optimal_arm else ""
        mean_prob = df[f'planning_prob_arm_{arm}'].mean()
        print(f"  Arm {arm}: {mean_prob:.6f}{is_optimal}")

    # Does optimal arm get highest planning probability?
    avg_planning_probs = [df[f'planning_prob_arm_{arm}'].mean() for arm in range(5)]
    best_prob_arm = np.argmax(avg_planning_probs)
    print(f"\nArm with highest average planning probability: {best_prob_arm}")
    if best_prob_arm == optimal_arm:
        print("  ✓ Correct!")
    else:
        print(f"  ✗ Wrong! Optimal is {optimal_arm}")

def analyze_blending(df):
    """Analyze how greedy and planning probabilities are blended."""
    print("\n" + "="*80)
    print("BLENDING ANALYSIS")
    print("="*80)

    avg_weight = df['planning_weight'].mean()
    print(f"\nAverage planning weight: {avg_weight:.3f}")
    print(f"Planning weight range: [{df['planning_weight'].min():.3f}, {df['planning_weight'].max():.3f}]")

    # Check if blending is done correctly
    # blended = (1-w) * greedy + w * planning
    sample_idx = len(df) // 2
    row = df.iloc[sample_idx]

    print(f"\nSample blending check (episode {row['episode']}):")
    print(f"  Planning weight: {row['planning_weight']:.3f}")
    for arm in range(5):
        greedy = row[f'greedy_prob_arm_{arm}']
        planning = row[f'planning_prob_arm_{arm}']
        blended = row[f'blended_prob_arm_{arm}']
        expected = (1 - row['planning_weight']) * greedy + row['planning_weight'] * planning

        print(f"  Arm {arm}: greedy={greedy:.4f}, planning={planning:.4f}, "
              f"blended={blended:.4f}, expected={expected:.4f}, "
              f"error={abs(blended-expected):.6f}")

def create_diagnostic_plots(df, optimal_arm=3):
    """Create diagnostic visualizations."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Meta-value over time for each arm
    ax = axes[0, 0]
    for arm in range(5):
        label = f"Arm {arm}" + (" [OPT]" if arm == optimal_arm else "")
        ax.plot(df['episode'], df[f'meta_value_arm_{arm}'], label=label, alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Meta-Value')
    ax.set_title('Meta-Value Estimates Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Chosen arm over time
    ax = axes[0, 1]
    ax.scatter(df['episode'], df['chosen_arm'], alpha=0.5, s=10)
    ax.axhline(y=optimal_arm, color='r', linestyle='--', label=f'Optimal (arm {optimal_arm})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Chosen Arm')
    ax.set_title('Arm Selection Over Time')
    ax.set_yticks(range(5))
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Meta-value gap (best - chosen)
    ax = axes[0, 2]
    ax.plot(df['episode'], df['meta_value_gap'], alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Meta-Value Gap')
    ax.set_title('Meta-Value Gap (Best - Chosen)')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)

    # 4. Planning weight over time
    ax = axes[1, 0]
    ax.plot(df['episode'], df['planning_weight'], alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Planning Weight')
    ax.set_title('Planning Weight Over Time')
    ax.grid(True, alpha=0.3)

    # 5. Distribution of chosen arms
    ax = axes[1, 1]
    arm_counts = df['chosen_arm'].value_counts().sort_index()
    colors = ['green' if arm == optimal_arm else 'blue' for arm in arm_counts.index]
    ax.bar(arm_counts.index, arm_counts.values, color=colors, alpha=0.7)
    ax.set_xlabel('Arm')
    ax.set_ylabel('Count')
    ax.set_title('Arm Selection Distribution')
    ax.set_xticks(range(5))
    ax.grid(True, alpha=0.3, axis='y')

    # 6. Average meta-value per arm
    ax = axes[1, 2]
    avg_mvs = [df[f'meta_value_arm_{arm}'].mean() for arm in range(5)]
    colors = ['green' if arm == optimal_arm else 'blue' for arm in range(5)]
    ax.bar(range(5), avg_mvs, color=colors, alpha=0.7)
    ax.set_xlabel('Arm')
    ax.set_ylabel('Average Meta-Value')
    ax.set_title('Average Meta-Value by Arm')
    ax.set_xticks(range(5))
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig

def main():
    """Main diagnostic routine."""
    parser = argparse.ArgumentParser(
        description='Diagnose planning failures by analyzing planning trace'
    )
    parser.add_argument(
        '--trace-path',
        type=str,
        required=True,
        help='Path to planning_trace.csv file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='analysis',
        help='Output directory for diagnostic plots (default: analysis)'
    )
    parser.add_argument(
        '--optimal-arm',
        type=int,
        default=3,
        help='Index of optimal arm (default: 3)'
    )

    args = parser.parse_args()

    trace_path = Path(args.trace_path)
    output_dir = Path(args.output_dir)

    # Check path exists
    if not trace_path.exists():
        print(f"Error: Planning trace not found: {trace_path}", file=sys.stderr)
        print(f"Expected to find planning_trace.csv at the specified path", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading planning trace...")
    df = load_trace(trace_path)

    print(f"Loaded {len(df)} planning decisions")
    print(f"Episodes: {df['episode'].min()} to {df['episode'].max()}")

    # Optimal arm from CLI argument
    optimal_arm = args.optimal_arm

    # Run analyses
    analyze_arm_selection(df, optimal_arm)
    analyze_meta_value_quality(df, optimal_arm)
    analyze_planning_probabilities(df, optimal_arm)
    analyze_blending(df)

    # Create plots
    print("\nGenerating diagnostic plots...")
    fig = create_diagnostic_plots(df, optimal_arm)
    output_path = output_dir / "planning_diagnosis.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Summary
    print("\n" + "="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)

    optimal_chosen_pct = (df['chosen_arm'] == optimal_arm).sum() / len(df) * 100
    avg_meta_values = [df[f'meta_value_arm_{arm}'].mean() for arm in range(5)]
    best_mv_arm = np.argmax(avg_meta_values)

    print(f"\n1. Arm Selection:")
    print(f"   - Optimal arm ({optimal_arm}) chosen {optimal_chosen_pct:.1f}% of the time")
    if optimal_chosen_pct < 40:
        print(f"   ✗ PROBLEM: Planning rarely chooses optimal arm")

    print(f"\n2. Meta-Value Quality:")
    print(f"   - Arm with highest avg meta-value: {best_mv_arm}")
    if best_mv_arm != optimal_arm:
        print(f"   ✗ PROBLEM: Meta-value prefers wrong arm ({best_mv_arm} vs {optimal_arm})")
        print(f"   - Meta-value is giving wrong guidance to planning")
    else:
        print(f"   ✓ Meta-value correctly identifies optimal arm on average")

    print(f"\n3. Meta-Value Magnitudes:")
    for arm in range(5):
        print(f"   Arm {arm}: {avg_meta_values[arm]:.6f}")

    # Check if meta-values are all negative (common issue)
    if all(mv < 0 for mv in avg_meta_values):
        print(f"\n   ⚠ All meta-values are negative")
        print(f"   This might indicate normalization issues")

    # Check variance
    meta_value_std = [df[f'meta_value_arm_{arm}'].std() for arm in range(5)]
    avg_std = np.mean(meta_value_std)
    print(f"\n4. Meta-Value Variation:")
    print(f"   - Average std across arms: {avg_std:.6f}")
    if avg_std < 0.01:
        print(f"   ⚠ Very low variation - meta-values might not be learning")

if __name__ == '__main__':
    main()
