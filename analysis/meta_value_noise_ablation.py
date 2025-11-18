"""
Meta-Value Noise Ablation Study

Investigates how meta-value correlation affects planning performance by:
1. Starting with perfect offline meta-values (0.98 correlation)
2. Adding controlled Gaussian noise to simulate different quality levels
3. Running planning with degraded meta-values
4. Measuring regret as function of correlation

This identifies the empirical correlation threshold below which planning hurts.

Usage:
    python analysis/meta_value_noise_ablation.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple

# Environment parameters (from actual experiments)
N_ARMS = 5
OPTIMAL_ARM = 3
ARM_MEANS = np.array([0.5, 0.3, 1.0, 1.9, 0.8])  # Estimated from experiments
N_EPISODES = 500
N_TRIALS = 10  # Monte Carlo trials per correlation level

OUTPUT_DIR = Path("analysis")

def add_noise_to_correlation(
    true_values: np.ndarray,
    target_correlation: float,
    seed: int = 42
) -> np.ndarray:
    """
    Add Gaussian noise to achieve target correlation.

    Given true values y, we want noisy values y' such that corr(y, y') = target.

    Solution: y' = ρ*y + √(1-ρ²)*ε where ε ~ N(0, σ²_y)
    This gives corr(y, y') = ρ

    Args:
        true_values: True values (n,)
        target_correlation: Desired correlation (0-1)
        seed: Random seed

    Returns:
        Noisy values with target correlation
    """
    np.random.seed(seed)

    # Standardize true values
    y_mean = true_values.mean()
    y_std = true_values.std()
    y_standardized = (true_values - y_mean) / (y_std + 1e-8)

    # Generate noise
    noise = np.random.randn(len(true_values))

    # Mix: y' = ρ*y + √(1-ρ²)*noise
    rho = target_correlation
    noise_weight = np.sqrt(max(0, 1 - rho**2))

    y_noisy_standardized = rho * y_standardized + noise_weight * noise

    # Denormalize
    y_noisy = y_noisy_standardized * y_std + y_mean

    # Verify correlation
    actual_corr = np.corrcoef(true_values, y_noisy)[0, 1]

    return y_noisy

def simulate_planning_episode(
    arm_means: np.ndarray,
    meta_values: np.ndarray,
    planning_weight: float = 0.8,
    greedy_prob: np.ndarray = None
) -> Tuple[int, float, float]:
    """
    Simulate one planning episode.

    Args:
        arm_means: True mean rewards per arm
        meta_values: Estimated meta-values per arm
        planning_weight: Weight for planning vs greedy
        greedy_prob: Greedy policy probabilities (if None, uniform)

    Returns:
        (chosen_arm, reward, regret)
    """
    # Default greedy to uniform
    if greedy_prob is None:
        greedy_prob = np.ones(len(arm_means)) / len(arm_means)

    # Planning probabilities from meta-values (softmax)
    planning_scores = meta_values - meta_values.min()  # Shift to positive
    planning_probs = np.exp(planning_scores / 0.1)  # Temperature 0.1
    planning_probs /= planning_probs.sum()

    # Blend
    blended_probs = (1 - planning_weight) * greedy_prob + planning_weight * planning_probs

    # Sample action
    chosen_arm = np.random.choice(len(arm_means), p=blended_probs)

    # Get reward (sample from Gaussian)
    reward = np.random.normal(arm_means[chosen_arm], 1.0)

    # Compute regret
    optimal_mean = arm_means.max()
    regret = optimal_mean - arm_means[chosen_arm]

    return chosen_arm, reward, regret

def run_planning_trial(
    arm_means: np.ndarray,
    meta_values: np.ndarray,
    n_episodes: int,
    planning_weight: float = 0.8,
    seed: int = 42
) -> dict:
    """
    Run one planning trial.

    Args:
        arm_means: True arm means
        meta_values: Meta-value estimates per arm
        n_episodes: Number of episodes
        planning_weight: Planning weight
        seed: Random seed

    Returns:
        Dictionary with results
    """
    np.random.seed(seed)

    rewards = []
    regrets = []
    arm_counts = np.zeros(len(arm_means))

    for _ in range(n_episodes):
        arm, reward, regret = simulate_planning_episode(
            arm_means, meta_values, planning_weight
        )
        rewards.append(reward)
        regrets.append(regret)
        arm_counts[arm] += 1

    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'cumulative_regret': np.sum(regrets),
        'arm_counts': arm_counts,
        'optimal_arm_frequency': arm_counts[OPTIMAL_ARM] / n_episodes
    }

def run_ablation_study():
    """Run full noise ablation study."""
    print("="*80)
    print("META-VALUE NOISE ABLATION STUDY")
    print("="*80)
    print()

    # True meta-values (proportional to arm means)
    true_meta_values = ARM_MEANS.copy()

    print(f"Environment:")
    print(f"  Arms: {N_ARMS}")
    print(f"  Optimal arm: {OPTIMAL_ARM}")
    print(f"  Arm means: {ARM_MEANS}")
    print(f"  True meta-values: {true_meta_values}")
    print()

    # Correlation levels to test
    correlation_levels = [0.99, 0.95, 0.90, 0.80, 0.70, 0.50, 0.30, 0.10, 0.05]

    results = []

    print("Running ablation...")
    for correlation in correlation_levels:
        print(f"\nCorrelation: {correlation:.2f}")

        trial_results = []
        for trial in range(N_TRIALS):
            # Add noise to meta-values
            noisy_meta_values = add_noise_to_correlation(
                true_meta_values,
                correlation,
                seed=42 + trial
            )

            # Verify correlation
            actual_corr = np.corrcoef(true_meta_values, noisy_meta_values)[0, 1]

            # Run planning trial
            result = run_planning_trial(
                ARM_MEANS,
                noisy_meta_values,
                N_EPISODES,
                planning_weight=0.8,
                seed=1000 + trial
            )
            result['target_correlation'] = correlation
            result['actual_correlation'] = actual_corr
            trial_results.append(result)

        # Aggregate trial results
        avg_regret = np.mean([r['cumulative_regret'] for r in trial_results])
        std_regret = np.std([r['cumulative_regret'] for r in trial_results])
        avg_reward = np.mean([r['mean_reward'] for r in trial_results])
        avg_optimal_freq = np.mean([r['optimal_arm_frequency'] for r in trial_results])
        actual_corr = np.mean([r['actual_correlation'] for r in trial_results])

        print(f"  Actual correlation: {actual_corr:.3f}")
        print(f"  Cumulative regret: {avg_regret:.1f} ± {std_regret:.1f}")
        print(f"  Mean reward: {avg_reward:.3f}")
        print(f"  Optimal arm frequency: {avg_optimal_freq:.1%}")

        results.append({
            'target_correlation': correlation,
            'actual_correlation': actual_corr,
            'mean_regret': avg_regret,
            'std_regret': std_regret,
            'mean_reward': avg_reward,
            'optimal_arm_frequency': avg_optimal_freq
        })

    # Add baseline: random selection
    print(f"\nBaseline: Random Selection")
    random_results = []
    for trial in range(N_TRIALS):
        result = run_planning_trial(
            ARM_MEANS,
            np.ones(N_ARMS),  # Uniform meta-values
            N_EPISODES,
            planning_weight=0.0,  # Pure random
            seed=2000 + trial
        )
        random_results.append(result)

    random_regret = np.mean([r['cumulative_regret'] for r in random_results])
    random_reward = np.mean([r['mean_reward'] for r in random_results])
    print(f"  Cumulative regret: {random_regret:.1f}")
    print(f"  Mean reward: {random_reward:.3f}")

    results.append({
        'target_correlation': 0.0,
        'actual_correlation': 0.0,
        'mean_regret': random_regret,
        'std_regret': np.std([r['cumulative_regret'] for r in random_results]),
        'mean_reward': random_reward,
        'optimal_arm_frequency': 1.0 / N_ARMS
    })

    return pd.DataFrame(results), random_regret

def plot_ablation_results(df, random_regret):
    """Create ablation study plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Sort by correlation
    df = df.sort_values('actual_correlation')

    # 1. Correlation vs Regret
    ax = axes[0]
    ax.errorbar(df['actual_correlation'], df['mean_regret'],
                yerr=df['std_regret'], fmt='o-', capsize=5, linewidth=2,
                markersize=8, label='Planning')
    ax.axhline(random_regret, color='red', linestyle='--', linewidth=2,
               label='Random baseline')
    ax.set_xlabel('Meta-Value Correlation', fontsize=12)
    ax.set_ylabel('Cumulative Regret (500 episodes)', fontsize=12)
    ax.set_title('Impact of Meta-Value Quality on Planning', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Find threshold where planning becomes worse than random
    worse_than_random = df[df['mean_regret'] > random_regret]
    if not worse_than_random.empty:
        threshold = worse_than_random['actual_correlation'].max()
        ax.axvline(threshold, color='orange', linestyle=':', linewidth=2,
                   label=f'Threshold ≈ {threshold:.2f}')
        ax.legend(fontsize=11)

    # 2. Correlation vs Mean Reward
    ax = axes[1]
    ax.plot(df['actual_correlation'], df['mean_reward'], 'o-',
            linewidth=2, markersize=8)
    ax.set_xlabel('Meta-Value Correlation', fontsize=12)
    ax.set_ylabel('Mean Reward', fontsize=12)
    ax.set_title('Reward vs Meta-Value Quality', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 3. Correlation vs Optimal Arm Frequency
    ax = axes[2]
    ax.plot(df['actual_correlation'], df['optimal_arm_frequency'] * 100,
            'o-', linewidth=2, markersize=8)
    ax.axhline(20, color='red', linestyle='--', linewidth=2,
               label='Random (20%)')
    ax.set_xlabel('Meta-Value Correlation', fontsize=12)
    ax.set_ylabel('Optimal Arm Selection (%)', fontsize=12)
    ax.set_title('Optimal Arm Selection vs Meta-Value Quality', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def append_to_summary(threshold_correlation):
    """Append findings to summary markdown."""
    summary_path = OUTPUT_DIR / "meta_value_summary.md"

    if not summary_path.exists():
        print(f"Warning: {summary_path} not found, skipping append")
        return

    with open(summary_path, 'a') as f:
        f.write("\n\n## Meta-Value Quality Threshold (Noise Ablation)\n\n")
        f.write(f"Synthetic noise ablation study identified the empirical correlation threshold:\n\n")
        f.write(f"- **Threshold: ≈{threshold_correlation:.2f}**\n")
        f.write(f"- Below this correlation, planning performs worse than random selection\n")
        f.write(f"- Current online training achieves 0.018-0.070 correlation\n")
        f.write(f"- This is **{(threshold_correlation/0.07):.0f}x below** the minimum required quality\n\n")
        f.write(f"Conclusion: Meta-value correlation must be > {threshold_correlation:.2f} for planning to help.\n")

    print(f"\nAppended findings to: {summary_path}")

def main():
    """Run ablation study."""
    # Run study
    df, random_regret = run_ablation_study()

    # Save results
    output_csv = OUTPUT_DIR / "meta_value_noise_ablation.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nSaved results to: {output_csv}")

    # Plot
    fig = plot_ablation_results(df, random_regret)
    output_png = OUTPUT_DIR / "meta_value_noise_ablation.png"
    fig.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_png}")

    # Find threshold
    worse_than_random = df[df['mean_regret'] > random_regret]
    if not worse_than_random.empty:
        threshold = worse_than_random['actual_correlation'].max()
    else:
        threshold = 0.0

    print("\n" + "="*80)
    print("FINDINGS")
    print("="*80)
    print(f"Random baseline regret: {random_regret:.1f}")
    print(f"Threshold correlation: ≈{threshold:.2f}")
    print(f"  - Above {threshold:.2f}: Planning helps")
    print(f"  - Below {threshold:.2f}: Planning hurts (worse than random)")
    print(f"\nCurrent online correlation: 0.018-0.070")
    print(f"Required improvement: {threshold/0.07:.1f}x")
    print("="*80)

    # Append to summary
    append_to_summary(threshold)

if __name__ == '__main__':
    main()
