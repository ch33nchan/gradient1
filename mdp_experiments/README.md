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

## Meta-Value Target Specification

### Overview

The meta-value network V(θ) predicts the quality of policy parameters θ. For MDPs, quality is measured by **multi-episode average return**.

### Target Definition

For a policy parameter snapshot θ_k:

```
Target(θ_k) = (1/K_eval) * Σ_{i=1}^{K_eval} G_i(θ_k)

where:
  G_i(θ_k) = discounted return of episode i using policy π(·|·; θ_k)
  G_i = Σ_{t=0}^{T-1} γ^t r_t
  K_eval = number of evaluation episodes (100-300 recommended)
```

### Rationale

**Why multi-episode average?**
- Single-episode returns have high variance (stochastic policy, environment)
- Multi-episode average reduces noise by factor of √K_eval
- Bandit analysis showed single rewards insufficient (need 385 samples)
- For Chain MDP: K_eval=100 should give ~10x lower std than single episode

**Why not use training episodes?**
- Training episodes use exploration (entropy bonus, stochastic sampling)
- Evaluation should measure true policy quality (greedy or low-temperature)
- Avoids confounding exploration noise with policy quality

**Why discounted returns?**
- Standard RL objective
- Encourages reaching goal quickly (γ < 1 penalizes longer episodes)
- Matches what REINFORCE optimizes

### Evaluation Protocol

#### Environment Configuration
- **Use same environment** as training (same seed, same config)
- **Reset between episodes** to initial state distribution
- **No time limits** beyond natural episode termination
- **Deterministic transitions** (if possible) or fixed seed per evaluation

#### Policy Configuration
For each snapshot θ_k:
- **Mode**: Greedy (argmax) or low-temperature (τ=0.1)
- **No exploration**: Disable entropy bonus, ε-greedy, etc.
- **Deterministic**: If policy has randomness, use mode or mean
- **Fixed θ_k**: No parameter updates during evaluation

#### Discount Factor
- **Use training γ**: Same discount factor as REINFORCE training
- **Default**: γ = 0.99 (standard for episodic tasks)
- **Must match**: Training and evaluation discount must be identical

### Dataset Schema

Each sample in the meta-value dataset contains:

#### Required Fields

```python
{
  # Identification
  'policy_id': str,           # Unique ID (e.g., 'run_001_ep_0050')
  'run_id': str,              # Training run ID
  'snapshot_episode': int,    # Training episode when snapshot taken

  # Target
  'target_return': float,     # Multi-episode average return
  'target_std': float,        # Standard deviation across episodes
  'n_eval_episodes': int,     # K_eval (how many episodes averaged)

  # Features (inputs to V(θ))
  'policy_params': np.ndarray,  # Flattened θ (or θ_hash if too large)
  'param_norm': float,          # ||θ||_2
  'param_mean': float,          # mean(θ)
  'param_std': float,           # std(θ)

  # Training context (optional but recommended)
  'training_return_ema': float,  # EMA of training returns at snapshot
  'training_episodes_seen': int, # Total episodes trained so far
  'gradient_norm': float,        # ||∇θ|| at this snapshot
  'policy_entropy': float,       # Entropy of π(·|s₀; θ)

  # Environment metadata
  'env_name': str,            # 'chain_mdp', 'gridworld', etc.
  'env_config': dict,         # Environment configuration
  'discount_factor': float,   # γ used for returns
}
```

#### Optional Fields (for richer features)

```python
{
  # Episode statistics
  'eval_episode_lengths': List[int],    # Length of each eval episode
  'eval_success_rate': float,           # Fraction reaching goal
  'eval_min_return': float,             # Min across eval episodes
  'eval_max_return': float,             # Max across eval episodes

  # Policy analysis
  'initial_state_action_probs': np.ndarray,  # π(·|s₀; θ)
  'kl_from_uniform': float,                  # KL(π || uniform)
  'max_action_prob': float,                  # max_a π(a|s₀; θ)

  # Parameter statistics
  'param_percentiles': np.ndarray,  # [p25, p50, p75] of θ values
  'param_sparsity': float,          # Fraction |θ| < 1e-3
  'param_change_from_init': float,  # ||θ - θ_init||
}
```

### Storage Format

**Option 1: PyTorch .pt file (Recommended)**
```python
{
  'samples': List[Dict],  # List of sample dicts (schema above)
  'metadata': {
    'created_at': str,
    'n_samples': int,
    'snapshot_interval': int,
    'eval_episodes_per_snapshot': int,
    'source_runs': List[str],
  }
}
```

**Option 2: Separate arrays + CSV index**
```
mdp_meta_value_dataset/
  features.npy          # (N, D) feature matrix
  targets.npy           # (N,) target returns
  metadata.csv          # Index with policy_id, snapshot_episode, etc.
  config.yaml           # Dataset configuration
```

**Recommendation**: Use PyTorch .pt for simplicity. Can convert to numpy/CSV later if needed.

### Snapshot Selection Strategy

**Uniform spacing (recommended for exploration)**:
- Take snapshots every N episodes (e.g., N=20)
- Captures full training trajectory
- Good diversity in parameter space
- Example: 500 episodes, snapshot every 20 → 25 snapshots

**Performance-based sampling (for focused learning)**:
- Take more snapshots when performance is improving
- Skip plateaus to save evaluation budget
- Example: Snapshot if ΔR > threshold in last 10 episodes

**Random sampling (for unbiased distribution)**:
- Randomly sample episodes to snapshot
- Ensures no correlation between snapshots
- Good for train/test split

**Default recommendation**: Uniform spacing with N=20 for 500-episode runs.

### Train/Test Split

**Temporal split (recommended)**:
- Train: First 70% of snapshots (by episode number)
- Test: Last 30% of snapshots
- Simulates real use case (predict future performance)
- Example: Episodes 0-350 train, 350-500 test

**Random split (for i.i.d. assumption)**:
- Randomly assign 70% train, 30% test
- Better if snapshots are independent
- Less realistic for online meta-learning

**Cross-run split (for generalization)**:
- Train: Snapshots from runs 1-3
- Test: Snapshots from run 4
- Tests generalization across different training trajectories
- Most realistic but requires multiple runs

### Expected Dataset Size

**For Chain MDP baseline**:
- Training run: 500 episodes
- Snapshot interval: 20 episodes
- Snapshots per run: 25
- Eval episodes per snapshot: 100
- Total evaluations: 25 × 100 = 2,500 episodes
- Training time: ~10 seconds per snapshot @ 200 eps/s = ~250 seconds = 4 minutes
- Dataset samples: 25 per run

**For multiple runs**:
- 4 runs × 25 snapshots = 100 samples
- Sufficient for training small MLP meta-value network
- Larger datasets better for generalization

### Quality Checks

Before using dataset, verify:

1. **Target variance is reasonable**:
   - Mean target_std should be 10-30% of mean target_return
   - If > 50%, increase K_eval

2. **Snapshots span parameter space**:
   - Check param_norm varies across snapshots
   - If all similar, training might have converged too quickly

3. **Returns are not all identical**:
   - If all returns ≈ optimal, no learning signal
   - May need earlier snapshots or harder environment

4. **No data leakage**:
   - Evaluation uses different episodes than training
   - Test set is temporally later or from different runs

### Example Usage

```python
# Load dataset
data = torch.load('mdp_meta_value_dataset.pt')
samples = data['samples']

# Extract features and targets
X = np.array([s['policy_params'] for s in samples])
y = np.array([s['target_return'] for s in samples])

# Train/test split (temporal)
split_idx = int(0.7 * len(samples))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train meta-value network
meta_value_net = MLPRegressor(input_dim=X.shape[1], hidden_dim=64)
meta_value_net.fit(X_train, y_train)

# Evaluate
y_pred = meta_value_net.predict(X_test)
correlation = np.corrcoef(y_test, y_pred)[0, 1]
print(f"Test correlation: {correlation:.3f}")
```

### Building the Dataset

**Implementation**: [`src/analysis/build_mdp_meta_value_dataset.py`](../src/analysis/build_mdp_meta_value_dataset.py)

**Command-line usage**:
```bash
python -m src.analysis.build_mdp_meta_value_dataset \
  --run-dir logs/chain_mdp_baseline/run_2025-11-19_06-18-51 \
  --config experiments/chain_mdp_baseline.yaml \
  --snapshot-interval 20 \
  --n-eval-episodes 200 \
  --output analysis/mdp_meta_value_dataset.pt
```

**Arguments**:
- `--run-dir`: Path to training run directory (or parent logs dir)
- `--config`: Experiment config (defines environment and agent architecture)
- `--snapshot-interval`: Episode interval for snapshots (default: 20)
- `--n-eval-episodes`: Evaluation episodes per snapshot (default: 200)
- `--output`: Output dataset path (default: analysis/mdp_meta_value_dataset.pt)

**Expected output for Chain MDP baseline (500 episodes)**:
- Snapshots: 25 (episodes 20, 40, 60, ..., 500)
- Eval episodes per snapshot: 200
- Total evaluations: 5,000 episodes (~25 seconds @ 200 eps/s)
- Dataset samples: 25 per run
- Dataset size: ~1 MB for single run

**Multi-run dataset** (recommended for generalization):
```bash
# Train 4 runs with different seeds
for seed in 42 43 44 45; do
  python -m experiments.run_experiment \
    --config experiments/chain_mdp_baseline.yaml \
    environment.seed=$seed
done

# Build dataset from all runs
python -m src.analysis.build_mdp_meta_value_dataset \
  --run-dir logs/chain_mdp_baseline \
  --config experiments/chain_mdp_baseline.yaml \
  --snapshot-interval 20 \
  --n-eval-episodes 200 \
  --output analysis/mdp_meta_value_dataset_4runs.pt
```

This produces 100 samples (4 runs × 25 snapshots), sufficient for training meta-value network.

**Output structure**:
```python
{
  'samples': [
    {
      'policy_id': 'run_001_ep_0020',
      'snapshot_episode': 20,
      'target_return': 0.923,
      'target_std': 0.012,
      'policy_params': array([...]),  # 4930-D
      # ... other fields
    },
    # ... more samples
  ],
  'metadata': {
    'created_at': '2025-11-19T12:00:00',
    'n_samples': 100,
    'snapshot_interval': 20,
    'eval_episodes_per_snapshot': 200,
    'source_runs': ['run_001', 'run_002', 'run_003', 'run_004'],
  }
}
```

**Requirements**:
- Training runs must have periodic checkpoints (set `training.save_interval` in config)
- Agent architecture (hidden_dim) must match between training and dataset building
- Environment config must be identical for training and evaluation

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
