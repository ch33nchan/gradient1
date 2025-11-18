"""
Sample Efficiency Analysis for Meta-Value Learning

Analyzes why single-episode rewards are insufficient for meta-value learning:
1. Estimates per-arm return variance for different averaging horizons
2. Computes confidence intervals for arm ranking
3. Calculates sample complexity needed for correct ranking
4. Shows that single-episode labels cannot reliably rank arms

Usage:
    python analysis/sample_efficiency_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Environment parameters (from experiments)
N_ARMS = 5
ARM_MEANS = np.array([0.5, 0.3, 1.0, 1.9, 0.8])
ARM_STD = 1.0  # Reward standard deviation (Gaussian rewards)
OPTIMAL_ARM = 3

OUTPUT_DIR = Path("analysis")

def estimate_ranking_confidence(
    arm_means: np.ndarray,
    arm_std: float,
    n_samples: int,
    n_trials: int = 1000
) -> dict:
    """
    Estimate confidence in correctly ranking arms.

    Args:
        arm_means: True mean rewards
        arm_std: Reward standard deviation
        n_samples: Number of samples to average
        n_trials: Monte Carlo trials

    Returns:
        Dictionary with ranking statistics
    """
    n_arms = len(arm_means)
    correct_ranking_count = 0
    optimal_is_best_count = 0

    for _ in range(n_trials):
        # Sample rewards and average
        samples = np.random.normal(
            arm_means.reshape(-1, 1),
            arm_std,
            size=(n_arms, n_samples)
        )
        estimated_means = samples.mean(axis=1)

        # Check if ranking is correct
        true_ranking = np.argsort(-arm_means)  # Descending order
        estimated_ranking = np.argsort(-estimated_means)

        if np.array_equal(true_ranking, estimated_ranking):
            correct_ranking_count += 1

        # Check if optimal arm is ranked best
        if estimated_ranking[0] == OPTIMAL_ARM:
            optimal_is_best_count += 1

    return {
        'n_samples': n_samples,
        'correct_ranking_prob': correct_ranking_count / n_trials,
        'optimal_ranked_best_prob': optimal_is_best_count / n_trials,
        'std_per_arm': arm_std / np.sqrt(n_samples)
    }

def compute_sample_complexity(
    arm_means: np.ndarray,
    arm_std: float,
    confidence: float = 0.95
) -> dict:
    """
    Compute sample complexity for reliable arm ranking.

    Args:
        arm_means: True mean rewards
        arm_std: Reward standard deviation
        confidence: Desired confidence level

    Returns:
        Dictionary with sample complexity estimates
    """
    # Find minimum gap between consecutive arms in ranking
    sorted_means = np.sort(arm_means)[::-1]
    gaps = sorted_means[:-1] - sorted_means[1:]
    min_gap = gaps.min()

    # For two arms with gap Δ, to distinguish with confidence p,
    # need n ≥ (z * σ / Δ)² where z = Φ^{-1}((1+p)/2)
    z = stats.norm.ppf((1 + confidence) / 2)
    n_required = (2 * z * arm_std / min_gap) ** 2

    # Gap between optimal and second-best
    optimal_gap = sorted_means[0] - sorted_means[1]
    n_optimal = (2 * z * arm_std / optimal_gap) ** 2

    return {
        'min_gap': min_gap,
        'optimal_gap': optimal_gap,
        'n_required_full_ranking': int(np.ceil(n_required)),
        'n_required_optimal': int(np.ceil(n_optimal)),
        'confidence': confidence
    }

def run_sample_efficiency_study():
    """Run comprehensive sample efficiency analysis."""
    print("="*80)
    print("SAMPLE EFFICIENCY ANALYSIS")
    print("="*80)
    print()

    print(f"Environment:")
    print(f"  N arms: {N_ARMS}")
    print(f"  Arm means: {ARM_MEANS}")
    print(f"  Reward std: {ARM_STD}")
    print(f"  Optimal arm: {OPTIMAL_ARM} (mean: {ARM_MEANS[OPTIMAL_ARM]})")
    print()

    # 1. Analyze different averaging horizons
    horizons = [1, 10, 50, 100, 200, 300, 500]

    results = []
    print("Ranking confidence vs averaging horizon:")
    print("-" * 80)
    print(f"{'Horizon':>8} | {'Std/Arm':>10} | {'Correct Rank %':>15} | {'Optimal Best %':>15}")
    print("-" * 80)

    for horizon in horizons:
        result = estimate_ranking_confidence(ARM_MEANS, ARM_STD, horizon)
        results.append(result)

        print(f"{horizon:8d} | {result['std_per_arm']:10.4f} | "
              f"{result['correct_ranking_prob']*100:14.1f}% | "
              f"{result['optimal_ranked_best_prob']*100:14.1f}%")

    print()

    # 2. Compute sample complexity
    complexity = compute_sample_complexity(ARM_MEANS, ARM_STD, confidence=0.95)

    print("Sample Complexity (95% confidence):")
    print("-" * 80)
    print(f"  Minimum gap between arms: {complexity['min_gap']:.3f}")
    print(f"  Gap (optimal vs 2nd): {complexity['optimal_gap']:.3f}")
    print(f"  Episodes needed (full ranking): {complexity['n_required_full_ranking']}")
    print(f"  Episodes needed (identify optimal): {complexity['n_required_optimal']}")
    print()

    # 3. Compute confidence intervals
    print("Confidence Intervals (1-episode vs 300-episode averages):")
    print("-" * 80)

    for arm in range(N_ARMS):
        is_optimal = " [OPTIMAL]" if arm == OPTIMAL_ARM else ""

        # 1-episode: CI = mean ± 1.96*std
        ci_1ep_low = ARM_MEANS[arm] - 1.96 * ARM_STD
        ci_1ep_high = ARM_MEANS[arm] + 1.96 * ARM_STD

        # 300-episode: CI = mean ± 1.96*std/√300
        ci_300ep_low = ARM_MEANS[arm] - 1.96 * ARM_STD / np.sqrt(300)
        ci_300ep_high = ARM_MEANS[arm] + 1.96 * ARM_STD / np.sqrt(300)

        print(f"  Arm {arm} (μ={ARM_MEANS[arm]:.1f}){is_optimal}:")
        print(f"    1-episode:   [{ci_1ep_low:5.2f}, {ci_1ep_high:5.2f}] (width: {ci_1ep_high-ci_1ep_low:.2f})")
        print(f"    300-episode: [{ci_300ep_low:5.2f}, {ci_300ep_high:5.2f}] (width: {ci_300ep_high-ci_300ep_low:.2f})")

    return pd.DataFrame(results), complexity

def plot_sample_efficiency(df, complexity):
    """Create sample efficiency plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Standard error vs horizon
    ax = axes[0]
    ax.plot(df['n_samples'], df['std_per_arm'], 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Averaging Horizon (episodes)', fontsize=12)
    ax.set_ylabel('Standard Error per Arm', fontsize=12)
    ax.set_title('Estimation Error vs Averaging Horizon', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.axvline(1, color='red', linestyle='--', alpha=0.5, label='Single episode')
    ax.axvline(300, color='green', linestyle='--', alpha=0.5, label='Offline (300 eps)')
    ax.legend()

    # 2. Correct ranking probability
    ax = axes[1]
    ax.plot(df['n_samples'], df['correct_ranking_prob'] * 100,
            'o-', linewidth=2, markersize=8, label='Full ranking correct')
    ax.plot(df['n_samples'], df['optimal_ranked_best_prob'] * 100,
            's-', linewidth=2, markersize=8, label='Optimal ranked #1')
    ax.axhline(95, color='gray', linestyle=':', alpha=0.7, label='95% confidence')
    ax.set_xlabel('Averaging Horizon (episodes)', fontsize=12)
    ax.set_ylabel('Probability (%)', fontsize=12)
    ax.set_title('Ranking Confidence vs Averaging Horizon', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axvline(1, color='red', linestyle='--', alpha=0.5)
    ax.axvline(300, color='green', linestyle='--', alpha=0.5)

    # 3. Confidence interval widths
    ax = axes[2]
    horizons = df['n_samples'].values
    ci_widths = 2 * 1.96 * ARM_STD / np.sqrt(horizons)
    ax.plot(horizons, ci_widths, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Averaging Horizon (episodes)', fontsize=12)
    ax.set_ylabel('95% CI Width', fontsize=12)
    ax.set_title('Confidence Interval Width vs Averaging', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.axvline(1, color='red', linestyle='--', alpha=0.5, label='Single episode')
    ax.axvline(300, color='green', linestyle='--', alpha=0.5, label='Offline (300 eps)')
    ax.axhline(complexity['min_gap'], color='orange', linestyle=':', alpha=0.7,
               label=f'Min gap ({complexity["min_gap"]:.2f})')
    ax.legend()

    plt.tight_layout()
    return fig

def append_to_diagnosis_report(df, complexity):
    """Append findings to diagnosis report."""
    report_path = OUTPUT_DIR / "planning_failure_diagnosis_report.md"

    if not report_path.exists():
        print(f"Warning: {report_path} not found, skipping append")
        return

    # Get key statistics
    single_ep = df[df['n_samples'] == 1].iloc[0]
    ep_300 = df[df['n_samples'] == 300].iloc[0]

    with open(report_path, 'a') as f:
        f.write("\n\n## Sample Efficiency Analysis\n\n")
        f.write("### Why Single-Episode Rewards Are Insufficient\n\n")

        f.write("Quantitative analysis of reward estimation variance:\n\n")

        f.write("| Averaging Horizon | Std Error | Correct Ranking | Optimal Ranked #1 |\n")
        f.write("|-------------------|-----------|-----------------|-------------------|\n")
        for _, row in df.iterrows():
            f.write(f"| {int(row['n_samples']):3d} episodes | "
                   f"{row['std_per_arm']:.4f} | "
                   f"{row['correct_ranking_prob']*100:5.1f}% | "
                   f"{row['optimal_ranked_best_prob']*100:5.1f}% |\n")

        f.write(f"\n**Key Finding**: Single-episode rewards have **{single_ep['std_per_arm']/ep_300['std_per_arm']:.1f}x higher** ")
        f.write(f"variance than 300-episode averages.\n\n")

        f.write("### Sample Complexity\n\n")
        f.write(f"To reliably distinguish arms with 95% confidence:\n\n")
        f.write(f"- **Minimum gap between arms**: {complexity['min_gap']:.3f}\n")
        f.write(f"- **Gap (optimal vs 2nd-best)**: {complexity['optimal_gap']:.3f}\n")
        f.write(f"- **Episodes needed for correct full ranking**: {complexity['n_required_full_ranking']}\n")
        f.write(f"- **Episodes needed to identify optimal**: {complexity['n_required_optimal']}\n\n")

        f.write("**Conclusion**: Single-episode rewards are **{:.0f}x below** the required sample size ".format(
            complexity['n_required_optimal']))
        f.write("for reliable meta-value learning.\n\n")

        f.write("This explains why:\n")
        f.write("1. Offline training (300-ep averages) achieves 0.98 correlation\n")
        f.write("2. Online training (1-ep rewards) achieves only 0.018-0.070 correlation\n")
        f.write(f"3. The gap is **structural**, not a hyperparameter issue\n")

    print(f"\nAppended sample efficiency analysis to: {report_path}")

def main():
    """Run sample efficiency analysis."""
    df, complexity = run_sample_efficiency_study()

    # Save results
    output_csv = OUTPUT_DIR / "sample_efficiency_analysis.csv"
    df.to_csv(output_csv, index=False)
    print(f"Saved results to: {output_csv}")

    # Plot
    fig = plot_sample_efficiency(df, complexity)
    output_png = OUTPUT_DIR / "sample_efficiency_analysis.png"
    fig.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_png}")

    # Append to report
    append_to_diagnosis_report(df, complexity)

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print(f"Single-episode rewards have {df.iloc[0]['std_per_arm']:.2f} standard error")
    print(f"300-episode averages have {df[df['n_samples']==300].iloc[0]['std_per_arm']:.4f} standard error")
    print(f"Reduction: {df.iloc[0]['std_per_arm']/df[df['n_samples']==300].iloc[0]['std_per_arm']:.1f}x")
    print()
    print(f"Sample complexity for reliable ranking: {complexity['n_required_full_ranking']} episodes")
    print(f"Current online training uses: 1 episode")
    print(f"Gap: {complexity['n_required_full_ranking']}x insufficient")
    print("="*80)

if __name__ == '__main__':
    main()
