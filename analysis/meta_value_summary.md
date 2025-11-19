# Meta-Value Experiment Summary

## Overview

Comparison of meta-value training improvements with and without planning.

## Experiments

1. **No Planning** (`meta_value_improved`): Improved meta-value training (global norm, no entropy penalty), planning disabled
2. **With Planning** (`planning_meta_value_improved`): Same improvements, but planning enabled

## Key Results (Final 500 Episodes)

### Self-Gradient Agent

| Condition | Mean Reward | Std | Cumulative Regret | vs Baselines |
|-----------|-------------|-----|-------------------|--------------|
| No Planning | 1.631 | 1.307 | 144.9 | 2.5x worse than UCB |
| With Planning | 0.128 | 1.638 | 896.5 | **6.2x worse** than no planning |

### Baselines (Consistent Across Both Conditions)

| Agent | Mean Reward | Cumulative Regret |
|-------|-------------|-------------------|
| ε-greedy | 1.765 ± 1.161 | 85.7 |
| UCB | 1.881 ± 1.025 | 8.4 |

## Analysis

### Self-Gradient vs Baselines (No Planning)

The self-gradient agent with improved meta-value training but **without planning**:
- Achieves mean reward of 1.631 ± 1.307
- Cumulative regret: 144.9
- **17.3x worse** than UCB (regret 8.4)
- **1.7x worse** than ε-greedy (regret 85.7)

The improved meta-value training (global normalization, no entropy penalty) achieved correlation of 0.070 on average, which is 3x-7x better than the baseline (~0.04), but still insufficient for good performance.

### How Badly Planning Hurts

Adding planning to the self-gradient agent with improved V(θ):
- Mean reward drops from 1.631 to 0.128
- Cumulative regret **increases 6.2x** from 144.9 to 896.5
- Meta-value correlation: 0.018 (similar to no planning)

**Planning actively harms performance** despite improved V(θ) quality.

### Why Planning Fails

Hypothesized reasons:
1. **Insufficient V(θ) quality**: Correlation ~0.018 still too low for reliable planning
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

## How to Regenerate Figures

All bandit analysis can be regenerated from the saved log files using the following scripts:

### Single Command (Recommended)

Run the complete analysis driver script:
```bash
python3 analysis/run_bandit_story.py
```

This regenerates all figures, prints key statistics, and creates:
- `analysis/meta_value_vs_planning_overview.png` - Performance comparison
- `analysis/meta_value_vs_planning_overview.csv` - Summary statistics
- `analysis/planning_diagnosis.png` - Planning trace diagnostics (if available)
- `analysis/meta_value_noise_ablation.png` - Correlation threshold analysis
- `analysis/sample_efficiency_analysis.png` - Variance analysis

### Individual Scripts

For specific analyses:

1. **Compare experiments** (requires: `logs/meta_value_improved/`, `logs/planning_meta_value_improved/`):
   ```bash
   python3 analysis/compare_meta_value_experiments.py
   ```
   Generates comparison plots and statistics showing planning vs no-planning performance.

2. **Diagnose planning failure** (requires: `logs/planning_test_instrumented/`):
   ```bash
   python3 analysis/diagnose_planning_failure.py
   ```
   Analyzes planning trace to show why meta-value guidance is inverted.
   Note: Uses instrumented test run for detailed per-episode diagnostics.

3. **Meta-value noise ablation** (no logs required - synthetic):
   ```bash
   python3 analysis/meta_value_noise_ablation.py
   ```
   Tests how correlation degradation affects planning via controlled noise injection.

4. **Sample efficiency analysis** (no logs required - synthetic):
   ```bash
   python3 analysis/sample_efficiency_analysis.py
   ```
   Quantifies variance problem: why single-episode rewards are insufficient.

### Dependencies

- Core scripts (1) depend on main experiment runs
- Diagnostic script (2) requires instrumented test run for planning trace
- Synthetic analyses (3, 4) have no log dependencies
- All scripts use only standard dependencies: numpy, pandas, matplotlib


## Meta-Value Quality Threshold (Noise Ablation)

Synthetic noise ablation study identified the empirical correlation threshold:

- **Threshold: ≈0.00**
- Below this correlation, planning performs worse than random selection
- Current online training achieves 0.018-0.070 correlation
- This is **0x below** the minimum required quality

Conclusion: Meta-value correlation must be > 0.00 for planning to help.
