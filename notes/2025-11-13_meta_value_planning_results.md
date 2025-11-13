# Meta-Value-Based Planning - Experimental Results

Date: 2025-11-13
Commit: (to be updated)

## Summary

Implemented and tested meta-value-based planning to replace logit-based reward prediction. **Result: Meta-value-based planning performs NO BETTER than logit-based planning** because the meta-value network itself fails to learn a meaningful signal about parameter quality.

## Problem Being Addressed

Previous experiments (documented in `2025-11-13_gated_planning_results.md`) showed that logit-based reward prediction was fundamentally flawed:
- Predicted reward: `E[r] ≈ E_π[θ_a]` (logits weighted by policy)
- Problem: Treats logits as rewards, but logits represent relative preferences, not absolute values
- Result: Planning was coherent but optimized the wrong signal

## Hypothesis

Replace logit-based prediction with meta-value network V(θ):
- V(θ) is trained on **actual returns**, not logits
- Should provide better estimates of parameter quality
- Planning objective: For each arm a, score = V_meta(θ'(a)) where θ' is predicted future parameters

## Implementation

### Code Changes

1. **Refactored planning objective** (`src/agents/self_gradient_agent.py:170-189`):
   ```python
   def _predict_future_reward(self, future_theta: torch.Tensor) -> float:
       """Use meta-value network to predict parameter quality."""
       with torch.no_grad():
           predicted_value = self.meta_value(future_theta).item()
           return predicted_value
   ```

2. **Added meta-value diagnostics** (`src/agents/self_gradient_agent.py:429-435`):
   - Track `meta_value_current = V(θ)` every episode
   - Export planning scores (which are now meta-value predictions)

3. **Extended analysis tools** (`src/analysis/bandit_analysis.py`):
   - New function: `plot_meta_value_diagnostics()` with 4-panel visualization
   - Meta-value vs reward scatter plot with correlation
   - Planning score vs reward correlation
   - Calibration plot (binned meta-value vs average reward)

4. **Created sanity-check evaluation** (`src/analysis/evaluate_meta_value.py`):
   - Standalone script to evaluate V_meta(θ) quality
   - Computes correlation between predictions and actual rewards
   - Binned calibration analysis
   - Clear interpretation of results

5. **Updated trainer** (`src/training/bandit_trainer.py:261,127`):
   - Store `meta_value_current` in metrics
   - Export to CSV for analysis

### Experiment Configs

Created two new configs:

1. **planning_meta_value_reward_only.yaml**: Pure meta-value objective
   - `planning_objective: "reward"` (now uses V_meta)
   - `planning_beta_self_change: 0.0` (no exploration bonus)

2. **planning_meta_value_reward_plus_update_norm.yaml**: With exploration
   - `planning_objective: "reward_plus_update_norm"`
   - `planning_beta_self_change: 0.1` (small exploration bonus)

## Experimental Results

### Experiment 1: Meta-Value Reward-Only

**Configuration:**
- Gated planning (warmup=150, threshold=0.25, max_weight=0.7)
- Objective: Pure V_meta(θ')
- No exploration bonus

**Results:**
- Final 500 episodes reward: **0.529 ± 1.754**
- Cumulative regret: **1579.5**
- Planning enabled: **83.3%** of episodes
- Planning score gap: **0.0108** (coherent)

**Baselines:**
- Epsilon-greedy: 1.735 ± 1.186, regret 203.8
- UCB: 1.881 ± 1.025, regret 50.5

**Analysis:**
- Performance: 31x worse regret than UCB, 7.7x worse than ε-greedy
- Planning is coherent (small score gaps)
- But performance is poor

### Experiment 2: Meta-Value + Exploration Bonus

**Configuration:**
- Same as Experiment 1 but with:
- `planning_beta_self_change: 0.1`

**Results:**
- Final 500 episodes reward: **0.176 ± 1.992**
- Cumulative regret: **1829.4**

**Analysis:**
- Even WORSE than pure meta-value
- Exploration bonus doesn't help

### Meta-Value Network Quality Evaluation

**Critical Finding:**

```
Correlation V(θ) vs reward: 0.0489
```

**Interpretation: The meta-value network has ZERO predictive power.**

Detailed calibration analysis:
```
Bin   V(θ) Range          Avg Reward    Std       N
1     [-0.98, -0.31]      0.10          1.66      200
2     [-0.31, -0.12]      0.40          1.77      200
3     [-0.12,  0.02]      0.33          1.71      200
4     [ 0.02,  0.10]      0.40          1.59      200
5     [ 0.10,  0.41]      0.35          1.64      200
```

All bins have nearly identical average rewards (0.1 to 0.4), showing **V(θ) cannot discriminate between good and bad parameters**.

## Root Cause Analysis

### Why Does the Meta-Value Network Fail?

Looking at the meta-value training code (`src/agents/self_gradient_agent.py:378-408`):

```python
def train_meta_value(self) -> float:
    theta_batch, _, _, reward_batch = self.buffer.sample(self.batch_size)

    with torch.no_grad():
        # Normalize rewards
        target_values = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-8)

        # Subtract entropy penalty
        policies = F.softmax(theta_batch, dim=1)
        entropies = -(policies * (policies + 1e-8).log()).sum(dim=1)
        target_values = target_values - 0.3 * entropies

    predicted_values = self.meta_value(theta_batch).squeeze()
    loss = F.mse_loss(predicted_values, target_values)
```

**Problem 1: Instant rewards, not returns**
- Trains on single-step rewards, not cumulative returns
- No temporal credit assignment
- V(θ) should predict future performance, but it only sees immediate reward

**Problem 2: Batch normalization destroys signal**
- Target is `(reward - mean) / std` **within each batch**
- Different batches have different normalizations
- V(θ) output is relative to batch, not absolute

**Problem 3: Entropy penalty mixes signals**
- Subtracts 0.3 * entropy from normalized reward
- High-entropy = bad, low-entropy = good (exploration vs exploitation)
- But this conflicts with reward signal (sometimes exploration is needed)

**Problem 4: No temporal structure**
- Buffer stores (θ, a, gradient, reward) independently
- No connection between consecutive parameters
- Can't learn "this θ led to good future returns"

### What Would Fix It?

For meta-value to work, need:

1. **Temporal targets**: Train on n-step returns or Monte Carlo returns
   - `target(θ_t) = sum of rewards from episode t onward`
   - Requires tracking episode boundaries and computing returns

2. **Consistent normalization**: Use global statistics, not batch statistics
   - Track running mean/std of returns
   - Or use raw returns if scale is reasonable

3. **Remove entropy penalty**: Keep it separate or remove entirely
   - Entropy is a property of policy, not value of parameters
   - Mixing signals hurts both

4. **Episode-based training**: Sample full episodes, not individual transitions
   - Enables proper return calculation
   - Allows temporal credit assignment

## Comparison to Previous Results

| Experiment | Objective | Final Reward | Regret | V(θ) Correlation |
|------------|-----------|--------------|--------|------------------|
| Logit-based reward-only | E_π[θ] | 0.605 | 1501.5 | N/A |
| Logit-based reward+bonus | E_π[θ] + bonus | 0.505 | 1545.5 | N/A |
| **Meta-value reward-only** | **V_meta(θ')** | **0.529** | **1579.5** | **0.049** |
| **Meta-value reward+bonus** | **V_meta(θ') + bonus** | **0.176** | **1829.4** | N/A |
| Update-norm control | ║Δθ║ | -0.191 | 2125.9 | N/A |

**Key Insights:**
1. Meta-value-based ≈ logit-based performance
2. Both worse than baselines (ε-greedy, UCB)
3. Meta-value has near-zero correlation with rewards
4. Planning is coherent in all cases (small score gaps)
5. The problem is NOT the planning mechanism, but the VALUE FUNCTION

## Conclusions

### What Worked

1. **Implementation**:
   - Clean refactor to use V_meta(θ')
   - Comprehensive diagnostics and analysis tools
   - Proper gating and blending

2. **Experimental methodology**:
   - Systematic comparison (reward-only, reward+bonus)
   - Sanity-check evaluation of meta-value quality
   - Clear diagnosis of failure mode

3. **Analysis**:
   - Identified that V(θ) has zero predictive power
   - Pinpointed specific issues in meta-value training
   - Proposed concrete fixes

### What Failed

1. **Meta-value network**:
   - Correlation with rewards: 0.05 (essentially zero)
   - Cannot distinguish good from bad parameters
   - Calibration shows no relationship

2. **Planning performance**:
   - No improvement over logit-based planning
   - Still 31x worse regret than UCB
   - Exploration bonus makes it worse

### Why This Happened

Meta-value training uses:
- Single-step rewards (not returns)
- Batch normalization (inconsistent targets)
- Entropy penalty (mixed signals)
- No temporal structure

Result: V_meta(θ) learns noise, not signal.

## Recommendations

### Option 1: Fix Meta-Value Training (HIGH EFFORT)

Implement proper value learning:
- Track full episode trajectories
- Compute Monte Carlo returns
- Use global normalization
- Remove entropy penalty
- Train end-of-episode

Challenges:
- Requires significant refactoring
- May still not help in bandits (too simple)
- Uncertain if it will work

### Option 2: Abandon Value-Based Planning in Bandits (RECOMMENDED)

Accept that bandits are too simple for gradient-based planning:
- Single-step rewards, no temporal structure
- Direct reward feedback is better than planning
- UCB/Thompson sampling are near-optimal

Instead:
- Move to sequential decision tasks (MDPs, control)
- Test planning where multi-step reasoning has value
- Bandits were just a testbed; real goal is richer environments

### Option 3: Use Oracle Value for Planning

Test planning mechanism in isolation:
- Create synthetic V_oracle(θ) that actually predicts reward
- Check if planning helps given perfect value estimates
- If not, planning itself is broken; if yes, confirms value is the issue

## Next Steps

**Recommended path forward:**

1. **Document this as a negative result** ✓ (this document)
2. **Commit all changes** (pending)
3. **Do NOT spend more time on bandit planning**
4. **Move to MDP/control tasks**:
   - CartPole, MountainCar, or similar
   - Where temporal credit assignment matters
   - Where planning multi-step ahead has value
5. **Revisit meta-value training** in that context
   - Episode-based returns
   - Proper value targets
   - Test if V(θ) can learn meaningful signal

## Files Generated

### Code
- `src/agents/self_gradient_agent.py` (modified)
  - Uses V_meta(θ') for planning instead of logits
  - Tracks meta_value_current diagnostic

- `src/training/bandit_trainer.py` (modified)
  - Logs and exports meta_value_current

- `src/analysis/bandit_analysis.py` (modified)
  - `plot_meta_value_diagnostics()`: 4-panel visualization
  - `print_summary()`: Added meta-value statistics

- `src/analysis/evaluate_meta_value.py` (new)
  - Standalone meta-value quality evaluation
  - Correlation and calibration analysis

- `experiments/planning_meta_value_reward_only.yaml` (new)
- `experiments/planning_meta_value_reward_plus_update_norm.yaml` (new)

### Results
- `logs/planning_meta_value_reward_only/run_2025-11-13_19-44-48/`
  - metrics.csv
  - rewards.png, cumulative_regret.png, gradient_error.png
  - planning_diagnostics.png
  - meta_value_diagnostics.png
  - meta_value_evaluation.png

## Research Standards Checklist

- ✓ No dummy scripts
- ✓ All experiments config-driven
- ✓ Proper logging (CSV, plots)
- ✓ Reproducible (seeds, configs saved)
- ✓ Systematic comparison
- ✓ Honest reporting of failures
- ✓ Root cause analysis
- ✓ Clear diagnosis with evidence
- ✓ Concrete proposals for fixes
- ✓ Documented what worked and what didn't

## Acknowledgment

This is another **valuable negative result**. We:
1. Systematically tested the meta-value hypothesis
2. Discovered the meta-value network doesn't learn meaningful signal
3. Diagnosed why (instant rewards, batch norm, no temporal structure)
4. Proposed clear fixes
5. Recommended path forward (move to MDPs)

The research process worked correctly: we had a hypothesis, tested it rigorously, found it didn't work, and understood why. This is progress.
