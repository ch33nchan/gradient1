"""
Bandit Story Driver Script - Reproduces all analysis for final writeup.

This script regenerates all figures, tables, and statistics for the meta-value
planning failure analysis in bandits.

Usage:
    python analysis/run_bandit_story.py

Outputs:
    - All comparison plots and diagnostics
    - Key statistics table
    - Summary of findings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Ensure analysis modules are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.compare_meta_value_experiments import (
    load_metrics, compute_final_stats, create_comparison_plot
)

# Paths
OUTPUT_DIR = Path("analysis")
NO_PLANNING_PATH = Path("logs/meta_value_improved/run_2025-11-13_20-28-09/metrics.csv")
PLANNING_PATH = Path("logs/planning_meta_value_improved/run_2025-11-13_21-02-08/metrics.csv")

def print_header(title):
    """Print formatted header."""
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80 + "\n")

def print_key_statistics():
    """Print key statistics for the bandit story."""
    print_header("BANDIT META-VALUE PLANNING: KEY STATISTICS")

    # Load metrics
    df_no_planning = pd.read_csv(NO_PLANNING_PATH)
    df_planning = pd.read_csv(PLANNING_PATH)

    # Compute stats
    stats_no_planning = compute_final_stats(df_no_planning)
    stats_planning = compute_final_stats(df_planning)

    # Meta-value correlations
    print("1. META-VALUE QUALITY")
    print("-" * 80)
    print(f"   Offline (300-episode averages):  0.98 correlation  ✓")
    print(f"   Online no-planning:              {stats_no_planning['meta_value_correlation_mean']:.3f} correlation")
    print(f"   Online with planning:            {stats_planning['meta_value_correlation_mean']:.3f} correlation")
    print(f"   Online degradation:              {(1 - stats_planning['meta_value_correlation_mean']/0.98)*100:.1f}% loss")

    # Performance impact
    print("\n2. PERFORMANCE IMPACT (Final 500 Episodes)")
    print("-" * 80)
    print(f"   No Planning:    {stats_no_planning['self_gradient_mean']:6.3f} ± {stats_no_planning['self_gradient_std']:.3f} reward, "
          f"regret {stats_no_planning['self_gradient_cumulative_regret']:6.1f}")
    print(f"   With Planning:  {stats_planning['self_gradient_mean']:6.3f} ± {stats_planning['self_gradient_std']:.3f} reward, "
          f"regret {stats_planning['self_gradient_cumulative_regret']:6.1f}")

    regret_multiplier = stats_planning['self_gradient_cumulative_regret'] / stats_no_planning['self_gradient_cumulative_regret']
    print(f"\n   Planning regret multiplier: {regret_multiplier:.2f}x WORSE")

    # Baseline comparison
    print("\n3. BASELINE COMPARISON")
    print("-" * 80)
    print(f"   ε-greedy:       {stats_no_planning['epsilon_greedy_mean']:6.3f} ± {stats_no_planning['epsilon_greedy_std']:.3f} reward, "
          f"regret {stats_no_planning['epsilon_greedy_cumulative_regret']:6.1f}")
    print(f"   UCB:            {stats_no_planning['ucb_mean']:6.3f} ± {stats_no_planning['ucb_std']:.3f} reward, "
          f"regret {stats_no_planning['ucb_cumulative_regret']:6.1f}")

    ucb_multiplier = stats_planning['self_gradient_cumulative_regret'] / stats_no_planning['ucb_cumulative_regret']
    print(f"\n   Planning vs UCB: {ucb_multiplier:.1f}x WORSE")

    # Arm selection analysis (if trace exists)
    trace_path = Path("logs/planning_test_instrumented/run_2025-11-18_06-53-11/planning_trace.csv")
    if trace_path.exists():
        df_trace = pd.read_csv(trace_path)
        optimal_arm = 3

        print("\n4. ARM SELECTION ANALYSIS (Planning Trace)")
        print("-" * 80)

        # Meta-value estimates
        avg_mvs = [df_trace[f'meta_value_arm_{arm}'].mean() for arm in range(5)]
        best_mv_arm = np.argmax(avg_mvs)

        print(f"   Average meta-value by arm:")
        for arm in range(5):
            marker = " [OPTIMAL]" if arm == optimal_arm else ""
            marker += " [HIGHEST MV]" if arm == best_mv_arm else ""
            print(f"     Arm {arm}: {avg_mvs[arm]:7.4f}{marker}")

        # Selection frequency
        arm_counts = df_trace['chosen_arm'].value_counts().sort_index()
        print(f"\n   Arm selection frequency:")
        for arm, count in arm_counts.items():
            pct = count / len(df_trace) * 100
            marker = " [OPTIMAL]" if arm == optimal_arm else ""
            print(f"     Arm {arm}: {count:3d} ({pct:5.1f}%){marker}")

        optimal_chosen_pct = (df_trace['chosen_arm'] == optimal_arm).sum() / len(df_trace) * 100
        print(f"\n   Optimal arm chosen: {optimal_chosen_pct:.1f}% of time")

    # Summary
    print("\n5. DIAGNOSIS")
    print("-" * 80)
    print("   ✗ Meta-value assigns LOWEST score to OPTIMAL arm")
    print("   ✗ Meta-value assigns HIGHEST score to SUBOPTIMAL arm")
    print("   ✓ Planning mechanism is correct (follows meta-value)")
    print("   ✗ Confident wrong decisions worse than random")
    print("   → Single-episode rewards insufficient for meta-value learning")

    return {
        'offline_correlation': 0.98,
        'online_no_planning_correlation': stats_no_planning['meta_value_correlation_mean'],
        'online_planning_correlation': stats_planning['meta_value_correlation_mean'],
        'regret_multiplier': regret_multiplier,
        'ucb_multiplier': ucb_multiplier,
    }

def regenerate_figures():
    """Regenerate all analysis figures."""
    print_header("REGENERATING ANALYSIS FIGURES")

    # 1. Comparison plot
    print("Generating comparison plot...")
    df_no_planning = pd.read_csv(NO_PLANNING_PATH)
    df_planning = pd.read_csv(PLANNING_PATH)

    fig = create_comparison_plot(df_no_planning, df_planning)
    output_path = OUTPUT_DIR / "meta_value_vs_planning_overview.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close(fig)

    # 2. Diagnostic plot (if trace exists)
    trace_path = Path("logs/planning_test_instrumented/run_2025-11-18_06-53-11/planning_trace.csv")
    if trace_path.exists():
        print("Generating diagnostic plot...")
        from analysis.diagnose_planning_failure import create_diagnostic_plots

        df_trace = pd.read_csv(trace_path)
        fig = create_diagnostic_plots(df_trace, optimal_arm=3)
        output_path = OUTPUT_DIR / "planning_diagnosis.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close(fig)

def create_summary_table():
    """Create summary table of key results."""
    print_header("SUMMARY TABLE")

    df_no_planning = pd.read_csv(NO_PLANNING_PATH)
    df_planning = pd.read_csv(PLANNING_PATH)

    stats_no_planning = compute_final_stats(df_no_planning)
    stats_planning = compute_final_stats(df_planning)

    # Create summary table
    summary = pd.DataFrame({
        'Condition': ['Offline Training', 'Online No-Planning', 'Online With Planning', 'ε-greedy', 'UCB'],
        'Correlation': [
            0.98,
            stats_no_planning['meta_value_correlation_mean'],
            stats_planning['meta_value_correlation_mean'],
            np.nan,
            np.nan
        ],
        'Mean Reward': [
            np.nan,
            stats_no_planning['self_gradient_mean'],
            stats_planning['self_gradient_mean'],
            stats_no_planning['epsilon_greedy_mean'],
            stats_no_planning['ucb_mean']
        ],
        'Cumulative Regret': [
            np.nan,
            stats_no_planning['self_gradient_cumulative_regret'],
            stats_planning['self_gradient_cumulative_regret'],
            stats_no_planning['epsilon_greedy_cumulative_regret'],
            stats_no_planning['ucb_cumulative_regret']
        ],
        'vs No-Planning': [
            np.nan,
            1.0,
            stats_planning['self_gradient_cumulative_regret'] / stats_no_planning['self_gradient_cumulative_regret'],
            stats_no_planning['epsilon_greedy_cumulative_regret'] / stats_no_planning['self_gradient_cumulative_regret'],
            stats_no_planning['ucb_cumulative_regret'] / stats_no_planning['self_gradient_cumulative_regret']
        ]
    })

    print(summary.to_string(index=False))

    # Save table
    output_path = OUTPUT_DIR / "bandit_final_summary.csv"
    summary.to_csv(output_path, index=False)
    print(f"\n  ✓ Saved: {output_path}")

    return summary

def print_file_inventory():
    """Print inventory of all generated files."""
    print_header("FILE INVENTORY")

    files = {
        'Analysis Scripts': [
            'analysis/run_bandit_story.py',
            'analysis/compare_meta_value_experiments.py',
            'analysis/diagnose_planning_failure.py',
        ],
        'Figures': [
            'analysis/meta_value_vs_planning_overview.png',
            'analysis/planning_diagnosis.png',
        ],
        'Data': [
            'analysis/meta_value_vs_planning_overview.csv',
            'analysis/bandit_final_summary.csv',
        ],
        'Reports': [
            'analysis/meta_value_summary.md',
            'analysis/planning_failure_diagnosis_report.md',
        ],
        'Experiment Logs': [
            'logs/meta_value_improved/run_2025-11-13_20-28-09/',
            'logs/planning_meta_value_improved/run_2025-11-13_21-02-08/',
            'logs/planning_test_instrumented/run_2025-11-18_06-53-11/',
        ]
    }

    for category, file_list in files.items():
        print(f"{category}:")
        for f in file_list:
            path = Path(f)
            exists = "✓" if path.exists() else "✗"
            print(f"  {exists} {f}")
        print()

def main():
    """Main driver function."""
    print_header("BANDIT META-VALUE PLANNING STORY")
    print("Comprehensive analysis of why meta-value planning fails in bandits")
    print()

    # Print key statistics
    stats = print_key_statistics()

    # Regenerate figures
    regenerate_figures()

    # Create summary table
    summary = create_summary_table()

    # Print file inventory
    print_file_inventory()

    # Final summary
    print_header("CONCLUSION")
    print("Meta-value-based planning fails in bandits because:")
    print()
    print("  1. Single-episode rewards too noisy for meta-value learning")
    print(f"     - Offline: 0.98 correlation (300-ep averages)")
    print(f"     - Online: {stats['online_planning_correlation']:.3f} correlation (single-ep)")
    print()
    print("  2. Meta-value provides inverted guidance")
    print("     - Lowest score to optimal arm")
    print("     - Highest score to suboptimal arm")
    print()
    print(f"  3. Planning makes performance {stats['regret_multiplier']:.1f}x WORSE")
    print("     - Confident wrong decisions > random exploration")
    print()
    print("  4. Recommendation: Use UCB or Thompson Sampling for bandits")
    print(f"     - {stats['ucb_multiplier']:.1f}x better than planning")
    print()
    print("="*80)

if __name__ == '__main__':
    main()
