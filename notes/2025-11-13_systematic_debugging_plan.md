# Systematic Debugging Plan for Self-Gradient Bandits

Date: 2025-11-13

## Problem Statement

Initial quick test showed catastrophic performance:
- Self-gradient agent: avg reward ≈ -1.0 to -1.4, regret ≈ 2.3-2.5
- Baselines (ε-greedy, UCB): avg reward ≈ 1.0-1.3, low regret
- Gradient prediction error decreasing (0.2-0.3), indicating learning

## Hypothesis

The planning logic is broken. Possible causes:
1. Exploration bonus (0.1 * grad_norm) overwhelming meta-value signal
2. Meta-value network not trained on representative data
3. Reward signal being used incorrectly somewhere

## Changes Made

### 1. Agent Refactoring
- Added explicit comments in `update()` to verify reward usage
- Removed large exploration bonus (default now 0.0)
- Added configurable `enable_planning` flag for ablations
- Added `gradient_step_scale` to control lookahead distance
- Added `planning_exploration_bonus` as tunable parameter
- Improved meta-value training target (rewards - 0.3 * entropy)

### 2. Ablation Experiments

Created three ablation configs:

**Ablation 1: No Planning** (`ablation_no_planning.yaml`)
- `enable_planning: false`
- Agent behaves greedily, but trains gradient predictor in background
- Goal: Verify gradient prediction works when decoupled from planning
- Expected: Performance similar to greedy baseline, gradient error decreases

**Ablation 2: Planning Without Bonus** (`ablation_planning_no_bonus.yaml`)
- `enable_planning: true`
- `planning_exploration_bonus: 0.0`
- Pure meta-value optimization
- Goal: Test if planning works without exploration bonus interference
- Expected: Should be competitive with baselines if meta-value is reasonable

**Ablation 3: Planning With Small Bonus** (`ablation_planning_small_bonus.yaml`)
- `enable_planning: true`
- `planning_exploration_bonus: 0.01` (was 0.1)
- Goal: Test if small exploration bonus helps
- Expected: Should not catastrophically fail like before

### 3. Analysis Tools

Created `src/analysis/__init__.py` with:
- `load_metrics()`: Load metrics CSV
- `plot_rewards()`: Plot rolling average rewards
- `plot_cumulative_regret()`: Plot cumulative regret
- `plot_gradient_error()`: Plot gradient prediction error
- `plot_all()`: Generate all plots
- `print_summary()`: Print final statistics

## Experiment Protocol

### Step 1: Verify Baseline Stability

```bash
# Just run baselines to confirm environment is correct
python -m experiments.run_experiment --config experiments/ablation_no_planning.yaml
```

Check:
- ε-greedy and UCB converge to high rewards
- Environment optimal arm is identified
- Regret curves are reasonable

### Step 2: Gradient Predictor Training (No Planning)

```bash
python -m experiments.run_experiment --config experiments/ablation_no_planning.yaml
```

Check:
- Self-gradient agent reward matches greedy baseline (since no planning)
- Gradient prediction error decreases steadily
- Error stabilizes at low value (< 0.1)

Expected outcome: Confirms gradient model learns when decoupled from action selection.

### Step 3: Planning Without Exploration Bonus

```bash
python -m experiments.run_experiment --config experiments/ablation_planning_no_bonus.yaml
```

Check:
- Self-gradient agent performance vs baselines
- Is it competitive or still worse?
- Gradient error still decreases

If performance is:
- **Good**: Problem was exploration bonus, proceed to Step 4
- **Still bad**: Meta-value training is broken, needs investigation

### Step 4: Planning With Small Exploration Bonus

```bash
python -m experiments.run_experiment --config experiments/ablation_planning_small_bonus.yaml
```

Check:
- Does small bonus help exploration?
- Performance vs Step 3

### Step 5: Analysis

For each run:

```bash
python -m src.analysis <run_directory>
```

This will:
- Generate plots (rewards.png, cumulative_regret.png, gradient_error.png)
- Print summary statistics
- Save to run directory

## Success Criteria

1. **Gradient predictor works**: Error < 0.1 in no-planning ablation
2. **Planning doesn't hurt**: Planning ablation ≥ greedy performance
3. **Competitive with baselines**: Within 20% of UCB performance
4. **Gradient error stays low**: Even with planning enabled

## Next Steps After Debugging

Once ablations show reasonable performance:

1. Tune hyperparameters:
   - `gradient_step_scale`: How far to look ahead (0.5, 1.0, 2.0)
   - `meta_learning_rate`: Adjust value network learning
   - `exploration_episodes`: Balance exploration vs planning

2. Run longer experiments:
   - Full 5000 episodes
   - Multiple seeds (42, 123, 456, 789, 1011)
   - Larger bandit (10 arms)

3. Compare against Thompson Sampling:
   - Add to baselines
   - Should be competitive

4. Move to non-stationary environment:
   - Test adaptation capabilities
   - Compare against adaptive baselines

5. Document results:
   - Create experiment report
   - Plot comparisons across all conditions
   - Statistical significance tests

## Notes

- All experiments use seed 42 for initial debugging
- Use `seed=X` CLI override for multiple runs
- Save all logs and configs
- No cherry-picking results
- Document failures as well as successes
