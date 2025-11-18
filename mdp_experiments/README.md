# MDP Experiments

## Overview

This directory contains specifications and experiments for applying meta-value learning to **multi-step decision problems (MDPs)**, where value functions are structurally necessary.

## Rationale: Why MDPs?

Bandit experiments showed meta-value planning fails due to:
- Single-step rewards too noisy for meta-value learning (17.3x higher variance)
- Sample complexity: need 385 episodes/arm, have only 1
- Direct methods (UCB) are optimal for bandits

**MDPs offer better conditions**:
1. **Multi-step returns**: Sum of discounted rewards → lower variance per parameter estimate
2. **Credit assignment needed**: Can't directly optimize without value functions
3. **Planning essential**: Must look ahead to solve optimally
4. **Natural temporal structure**: Episodes provide clear boundaries for meta-learning

## Proposed Environments

### 1. Chain MDP (Simple Diagnostic)

**Description**: Linear chain of N states, agent chooses left/right at each step.

```
States: S0 -- S1 -- S2 -- ... -- SN
         |     |     |           |
      r=0.0  r=0.1  r=0.2     r=1.0 (goal)
```

**Properties**:
- **States**: N (e.g., 10-20)
- **Actions**: 2 (left, right)
- **Rewards**: Sparse (only at goal)
- **Optimal policy**: Always go right
- **Challenge**: Credit assignment over N steps

**Why good for meta-value**:
- Clear correct policy (parameters)
- Multi-step return quality depends on how close to optimal
- Can test if meta-value V(θ) learns to predict return

**Spec**: See `envs/chain_mdp.yaml`

### 2. Gridworld (Spatial Navigation)

**Description**: 2D grid, agent navigates from start to goal, avoiding walls.

```
┌─────────────┐
│ S . . . . . │
│ . # # # . . │
│ . . . . . . │
│ . # # . . G │
└─────────────┘

S = Start
G = Goal
# = Wall
. = Empty
```

**Properties**:
- **States**: Grid positions (e.g., 8×8 = 64)
- **Actions**: 4 (up, down, left, right)
- **Rewards**: -0.01 per step, +1.0 at goal
- **Optimal policy**: Shortest path to goal
- **Challenge**: Spatial planning, obstacle avoidance

**Why good for meta-value**:
- Multiple paths with different qualities
- Policy parameters affect trajectory efficiency
- Meta-value should prefer parameters that reach goal faster

**Spec**: See `envs/gridworld.yaml`

### 3. CartPole (Control)

**Description**: Balance pole on cart, continuous state, discrete actions.

**Properties**:
- **States**: (cart position, velocity, pole angle, angular velocity)
- **Actions**: 2 (push left, push right)
- **Rewards**: +1 per timestep while balanced
- **Episode length**: Until pole falls or 200 steps
- **Challenge**: Continuous state, dynamic balancing

**Why good for meta-value**:
- Standard benchmark for policy gradient methods
- Clear performance metric (episode length)
- Policy quality varies smoothly with parameters

**Spec**: See `envs/cartpole.yaml`

## Architecture Adaptation

### Current Bandit Architecture

```python
class SelfGradientBanditAgent:
    - policy: SoftmaxBanditPolicy(n_arms)
    - gradient_predictor: GradientPredictor(n_arms, hidden_dim)
    - meta_value: MetaValueNetwork(n_arms, hidden_dim)
```

### Proposed MDP Architecture

```python
class SelfGradientMDPAgent:
    - policy: PolicyNetwork(state_dim, action_dim, hidden_dim)
    - value_network: ValueNetwork(state_dim, hidden_dim)
    - gradient_predictor: GradientPredictor(param_dim, state_dim, action_dim)
    - meta_value: MetaValueNetwork(param_dim, hidden_dim)
```

**Key differences**:
1. **Policy**: Maps states → actions (not just action probabilities)
2. **Value network**: V(s) for critic in actor-critic
3. **Gradient predictor**: Input includes state, output is parameter gradient
4. **Meta-value**: V(θ) predicts expected return of parameter vector

### Training Loop

```python
for episode in range(n_episodes):
    # 1. Rollout episode with current policy
    trajectory = rollout_episode(policy)

    # 2. Compute actual return
    G = compute_discounted_return(trajectory)

    # 3. Compute actual parameter gradient (REINFORCE or actor-critic)
    actual_grad = compute_policy_gradient(trajectory)

    # 4. Update gradient predictor
    predicted_grad = gradient_predictor(theta, states, actions)
    train_gradient_predictor(predicted_grad, actual_grad)

    # 5. Update meta-value on episode return
    train_meta_value(theta, G)  # Multi-step return!

    # 6. Planning (if enabled)
    if planning_enabled:
        # Predict gradient for each action
        # Simulate parameter update
        # Estimate return with meta-value
        # Choose action based on estimated future return
        action = plan_action(state, theta)

    # 7. Update policy parameters
    theta = theta - learning_rate * actual_grad
```

**Key improvement**: Meta-value trained on **episode returns G**, not single-step rewards!

## Reusable Components

From bandit experiments, we can reuse:

### 1. Meta-Value Network
```python
# src/models/gradient_world_model.py
class MetaValueNetwork(nn.Module):
    # Already parameter-agnostic, works for any θ
```

### 2. Gradient Predictor (adapt input)
```python
class GradientPredictor(nn.Module):
    # Need to modify to accept states/actions
    # But core architecture reusable
```

### 3. Experience Buffer
```python
class ExperienceBuffer:
    # Modify to store (θ, trajectory, return)
    # Instead of (θ, action, gradient, reward)
```

### 4. Training Infrastructure
```python
# src/training/bandit_trainer.py → mdp_trainer.py
# Core logging, metrics, plotting reusable
# Adapt for episode-level statistics
```

### 5. Analysis Tools
```python
# All analysis/* scripts
# Comparison plots, diagnostics, ablations
# Templates reusable for MDP results
```

## Interface Specification

### Environment Interface

```python
class MDPEnvironment:
    def reset(self) -> state:
        """Reset to initial state."""

    def step(self, action) -> (next_state, reward, done, info):
        """Take action, get next state and reward."""

    def get_state_dim(self) -> int:
        """State dimensionality."""

    def get_action_dim(self) -> int:
        """Number of discrete actions."""
```

### Agent Interface

```python
class SelfGradientMDPAgent:
    def act(self, state, exploration_mode='auto') -> action:
        """Select action given state."""

    def update(self, trajectory):
        """Update after episode completes."""

    def get_policy_parameters(self) -> theta:
        """Get current policy parameters."""
```

### Trainer Interface

```python
class MDPTrainer:
    def train(self, n_episodes) -> metrics:
        """Train for n episodes."""

    def evaluate(self, n_episodes) -> mean_return:
        """Evaluate current policy."""
```

## Expected Improvements Over Bandits

### 1. Better Meta-Value Targets

| Bandit | MDP |
|--------|-----|
| Single reward r | Episode return G = Σ γ^t r_t |
| High variance (σ=1.0) | Lower variance (averaging over trajectory) |
| No temporal structure | Clear episode boundaries |
| Need 385 samples for ranking | May need only 10-50 episodes |

### 2. Value Functions Necessary

- Bandits: Can directly estimate Q(a) from rewards
- MDPs: Need V(s) or Q(s,a) for multi-step planning
- Meta-value V(θ) serves role in parameter space

### 3. Natural Planning Horizon

- Bandits: Planning one step ahead
- MDPs: Planning over full episode trajectory
- Meta-value can look further into future

### 4. Richer Feedback

- Bandits: Binary success/failure at arm level
- MDPs: Continuous feedback through trajectory
- Better signal for learning which parameters lead to good behavior

## Implementation Plan

1. **Phase 1**: Chain MDP (simplest)
   - Implement environment
   - Adapt agent for state input
   - Train without planning (baseline)
   - Verify meta-value quality (correlation with returns)

2. **Phase 2**: Meta-value with planning
   - Enable planning in Chain MDP
   - Measure if planning helps (unlike bandits)
   - Tune planning weight, objective

3. **Phase 3**: Gridworld
   - More complex environment
   - Test generalization of approach
   - Compare to standard RL baselines (DQN, A2C)

4. **Phase 4**: CartPole (if successful)
   - Standard benchmark
   - Publishable results
   - Compare to meta-learning baselines (MAML, etc.)

## Success Criteria

**Minimum viable result**:
- Meta-value correlation > 0.5 with episode returns
- Planning improves performance vs no-planning baseline
- Competitive with standard RL on toy environments

**Strong result**:
- Meta-value correlation > 0.8
- Planning provides 2x+ speedup in sample efficiency
- Outperforms standard RL on at least one benchmark

**Publishable result**:
- Meta-value + planning achieves SOTA on some metric
- Clear ablations showing each component's contribution
- Theoretical analysis of when/why it works

## Open Questions

1. **How to compute policy gradients?**
   - REINFORCE (high variance)
   - Actor-critic (needs value network)
   - Trust region methods (TRPO/PPO)

2. **How to handle continuous states?**
   - Neural network policy π(a|s; θ)
   - Feature extraction before meta-value
   - Policy parameters θ are network weights

3. **How to do planning in continuous spaces?**
   - Sample-based planning (simulate trajectories)
   - Model-based planning (learn dynamics)
   - Gradient-based planning (differentiate through policy)

4. **How to ensure meta-value learns from returns not rewards?**
   - Only train on complete episodes
   - Use Monte Carlo returns (no bootstrapping)
   - Validate correlation with actual returns

## References

- **Chain MDP**: Classic diagnostic for credit assignment
- **Gridworld**: Sutton & Barto textbook
- **CartPole**: OpenAI Gym benchmark
- **Meta-learning**: MAML (Finn et al. 2017), Meta-SGD, etc.
- **Policy gradients**: REINFORCE (Williams 1992), PPO (Schulman et al. 2017)

---

**Status**: Planning phase
**Next**: Implement Chain MDP environment
**Previous**: Bandit experiments → `analysis/planning_failure_diagnosis_report.md`
