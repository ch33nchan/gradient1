# Chain MDP: Canonical Sanity-Check Environment

## Purpose

Chain MDP is the **canonical sanity-check environment** for all MDP experiments in this project.

Any new algorithm, meta-value architecture, or planning mechanism must:
1. Be tested on Chain MDP first
2. At minimum match REINFORCE baseline performance
3. Show clear, reproducible behavior (success or failure)

## Why Chain MDP?

### Simplicity
- 1D state space (linear chain)
- 2 actions (left/right)
- Deterministic transitions
- Sparse rewards (0 everywhere, 1 at goal)
- Optimal policy is trivial: always go right

### Diagnostic Value
- **Known optimal**: Return = γ^8 ≈ 0.923, length = 9 steps
- **Easy to verify**: Success = reached goal state
- **Fast training**: REINFORCE learns optimal in ~100 episodes
- **No confounds**: No stochasticity, no exploration-exploitation tradeoff complexity

### Multi-Step Credit Assignment
Unlike bandits:
- Requires multi-step planning (9 steps to reward)
- Tests value function learning
- Returns have lower variance than single rewards
- Agent must learn temporal dependencies

## Baseline Performance (REINFORCE)

**Required performance** for any new algorithm:
- Success rate: ≥ 95% by episode 200
- Episode length: ≤ 12 steps (average over last 100 episodes)
- Episode return: ≥ 0.85 (average over last 100 episodes)

**REINFORCE achieves**:
- Success rate: 100%
- Episode length: 9.0 steps (optimal)
- Episode return: 1.000 (maximum)
- Training time: < 2 seconds for 500 episodes

## Configuration

Standard Chain MDP config:
```yaml
environment:
  env_type: "chain_mdp"
  n_states: 10
  max_steps: 100
  discount_factor: 0.99
  seed: 42
```

Standard REINFORCE config:
```yaml
agent:
  type: "reinforce"
  learning_rate: 0.01
  hidden_dim: 64
  use_baseline: true
  entropy_bonus: 0.01
```

## Usage Guidelines

### For Baseline Experiments
- Use Chain MDP to verify new agent implementations work
- Run `experiments/chain_mdp_baseline.yaml` as-is
- Expected training time: 1-2 seconds
- If agent fails to learn, debug before moving to harder environments

### For Meta-Value Experiments
- Chain MDP is ideal for testing meta-value quality:
  - Multi-episode returns have low variance
  - Policy snapshots are easy to evaluate
  - Correlation should be very high (> 0.95) offline
- Any meta-value planning mechanism must at least match REINFORCE baseline

### For Planning Experiments
- If planning helps: Should beat REINFORCE (faster learning or higher return)
- If planning hurts: Must diagnose root cause (as we did for bandits)
- Chain MDP planning should be easier than bandits because:
  - Returns are less noisy (multi-step averaging)
  - Value functions are necessary (not optional)
  - Optimal policy is stationary

## Failure Modes to Watch For

### Agent Never Reaches Goal
- Check policy initialization (should be roughly uniform)
- Check learning rate (0.01 works well)
- Check entropy bonus (0.01 encourages exploration)
- Verify gradient updates are non-zero

### Agent Reaches Goal But Doesn't Converge to Optimal
- Increase training episodes (try 1000 instead of 500)
- Check baseline subtraction is working
- Verify return calculation is correct (discounted sum)

### Agent Performance Degrades Over Time
- Check for gradient explosion (add gradient clipping)
- Verify optimizer state is stable
- Check for numerical issues in log probabilities

### Planning Makes Things Worse
- **This is the key test for meta-value learning**
- If planning hurts on Chain MDP (even with perfect offline meta-value):
  - Check distribution shift (training policy ≠ evaluation policy)
  - Check meta-value input features (are they informative?)
  - Check planning mechanism (is it using meta-value correctly?)
- Chain MDP is simple enough that planning should work if meta-value is good

## Comparison to Bandits

| Aspect | Bandits | Chain MDP |
|--------|---------|-----------|
| Steps per episode | 1 | 9 (optimal) |
| Return variance | High (1.0 std) | Lower (multi-step averaging) |
| Optimal policy | Choose best arm | Always right |
| Credit assignment | Trivial | Multi-step |
| Meta-value target | Single reward | Multi-episode return |
| Sample complexity | 385 episodes/arm | ~100 episodes total |

**Key advantage for meta-value learning**: Multi-episode returns are much less noisy than single rewards.

## Expected Meta-Value Behavior

Based on bandit analysis, we expect:

### Offline (Good Conditions)
- Train meta-value on policy snapshots with multi-episode returns
- Expected correlation: > 0.95 (should be very high)
- Variance in targets: Much lower than bandits

### Online (Planning in the Loop)
- Meta-value quality may degrade under distribution shift
- But should degrade less than bandits because:
  - Returns are less noisy
  - More samples per policy (multi-step episodes)
  - Value functions are structurally necessary

### Success Criteria
- Offline correlation > 0.95: Meta-value model is working
- Online correlation > 0.80: Planning should help
- Performance ≥ REINFORCE: Planning doesn't hurt
- Performance > REINFORCE: Planning helps (ideal outcome)

## Files

- Environment: `src/envs/mdp_envs.py` (ChainMDP class)
- Baseline config: `experiments/chain_mdp_baseline.yaml`
- Baseline agent: `src/agents/reinforce_agent.py`
- Trainer: `src/training/mdp_trainer.py`
- Tests: `tests/test_chain_mdp.py`

## References

- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3-4), 229-256.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. (Chapter 13: Policy Gradient Methods)
