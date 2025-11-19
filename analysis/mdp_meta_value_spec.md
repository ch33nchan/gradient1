# MDP Meta-Value Dataset Specification

## Overview

This document specifies the exact format and schema for MDP meta-value datasets used to train V(θ) networks.

## Purpose

Meta-value network V(θ) predicts policy quality from parameters θ. For MDPs, quality = multi-episode average return.

## Dataset Schema

### Sample Structure

Each sample represents one policy parameter snapshot θ_k evaluated over K_eval episodes.

```python
{
  # ============================================================================
  # IDENTIFICATION
  # ============================================================================
  'policy_id': str,              # Unique identifier (e.g., 'run_001_ep_0100')
  'run_id': str,                 # Training run identifier
  'snapshot_episode': int,       # Episode number when snapshot was taken

  # ============================================================================
  # TARGET (what we're predicting)
  # ============================================================================
  'target_return': float,        # Mean return over K_eval episodes
  'target_std': float,           # Std dev of returns
  'n_eval_episodes': int,        # K_eval (100-300 recommended)

  # ============================================================================
  # FEATURES (inputs to V(θ))
  # ============================================================================

  ## Core Features (REQUIRED)
  'policy_params': np.ndarray,   # Flattened θ vector (float32)
  'param_norm': float,           # L2 norm ||θ||_2
  'param_mean': float,           # Mean of all parameters
  'param_std': float,            # Std dev of all parameters

  ## Training Context (REQUIRED)
  'training_return_ema': float,  # EMA of training returns at snapshot
  'training_episodes_seen': int, # Total episodes seen during training
  'gradient_norm': float,        # ||∇θ|| at snapshot (if available)
  'policy_entropy': float,       # H[π(·|s₀; θ)] at initial state

  ## Environment Metadata (REQUIRED)
  'env_name': str,               # 'chain_mdp', 'gridworld', 'cartpole'
  'env_config': dict,            # Full environment configuration
  'discount_factor': float,      # γ used in return computation

  # ============================================================================
  # OPTIONAL FIELDS (for richer analysis)
  # ============================================================================

  ## Episode Statistics
  'eval_episode_lengths': List[int],  # Length of each eval episode
  'eval_success_rate': float,         # Fraction of episodes reaching goal
  'eval_min_return': float,           # Min return across eval episodes
  'eval_max_return': float,           # Max return across eval episodes

  ## Policy Analysis
  'initial_state_action_probs': np.ndarray,  # π(·|s₀; θ)
  'kl_from_uniform': float,                  # KL divergence from uniform
  'max_action_prob': float,                  # max_a π(a|s₀; θ)

  ## Parameter Statistics
  'param_percentiles': np.ndarray,   # [p25, p50, p75] of θ values
  'param_sparsity': float,           # Fraction where |θ_i| < 1e-3
  'param_change_from_init': float,   # ||θ - θ_init||_2
}
```

### Data Types

| Field | Type | Shape | Notes |
|-------|------|-------|-------|
| `policy_id` | str | - | Must be unique |
| `target_return` | float32 | - | -∞ to +∞ |
| `target_std` | float32 | - | ≥ 0 |
| `policy_params` | float32 | (D,) | D = total parameters |
| `param_norm` | float32 | - | > 0 |
| `env_config` | dict | - | JSON-serializable |

## Target Computation

### Formula

```
Target(θ_k) = (1/K_eval) × Σ_{i=1}^{K_eval} G_i(θ_k)

where:
  G_i(θ_k) = Σ_{t=0}^{T-1} γ^t r_t
```

### Evaluation Protocol

**For each snapshot θ_k**:

1. **Load policy parameters**: π(a|s; θ_k)
2. **Set evaluation mode**:
   - Greedy action selection (argmax)
   - OR low-temperature sampling (τ = 0.1)
   - Disable exploration bonuses
3. **Run K_eval episodes**:
   - Reset environment to initial state
   - Roll out until termination
   - Compute discounted return G_i
4. **Compute statistics**:
   - target_return = mean(G_1, ..., G_K)
   - target_std = std(G_1, ..., G_K)

### Recommended K_eval

| Environment | K_eval | Rationale |
|-------------|--------|-----------|
| Chain MDP | 100 | Low stochasticity, 10x variance reduction |
| Gridworld | 200 | Medium stochasticity |
| CartPole | 300 | High stochasticity |

**Rule of thumb**: K_eval should make target_std < 30% of target_return.

## Feature Engineering

### Policy Parameters

**Raw parameters (recommended)**:
```python
params = agent.get_policy_parameters()  # Flattened θ
```

**Challenges**:
- High dimensionality (10K-100K parameters for neural nets)
- May need dimensionality reduction (PCA, random projection)
- Or use parameter statistics instead

**Parameter statistics (alternative)**:
- Mean, std, norm, percentiles
- Lower dimensional but loses information
- May be sufficient if V(θ) learns parameter-agnostic patterns

### Training Context

**Why include training context?**
- Early vs late training has different parameter distributions
- Training return gives signal about trajectory quality
- Gradient norm indicates learning dynamics
- Helps V(θ) learn "where in training we are"

**Computing gradient norm**:
```python
# After computing policy gradient
grad_norm = 0.0
for param in policy.parameters():
    if param.grad is not None:
        grad_norm += param.grad.norm().item() ** 2
grad_norm = grad_norm ** 0.5
```

**Computing policy entropy**:
```python
# At initial state s₀
state = torch.tensor([0], dtype=torch.long)
probs = policy.get_action_probs(state)
entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
```

## Storage Format

### PyTorch .pt File (Recommended)

```python
dataset = {
  'samples': [
    {
      'policy_id': 'run_001_ep_0020',
      'target_return': 0.856,
      'policy_params': np.array([...]),
      # ... all other fields
    },
    # ... more samples
  ],
  'metadata': {
    'created_at': '2025-11-19T12:34:56',
    'n_samples': 100,
    'snapshot_interval': 20,
    'eval_episodes_per_snapshot': 100,
    'source_runs': ['run_001', 'run_002', 'run_003', 'run_004'],
    'builder_version': '1.0.0',
  }
}

torch.save(dataset, 'mdp_meta_value_dataset.pt')
```

### Separate Files (Alternative)

```
mdp_meta_value_dataset/
  features.npy           # (N, D) matrix of policy_params
  targets.npy            # (N,) vector of target_returns
  target_stds.npy        # (N,) vector of target_stds
  metadata.csv           # Index with policy_id, snapshot_episode, etc.
  training_context.npy   # (N, K) matrix of training features
  config.yaml            # Dataset configuration
```

## Quality Checks

### Pre-Processing Validation

Run these checks before using dataset:

```python
def validate_dataset(samples):
    """Validate dataset quality."""

    # 1. Check for duplicates
    policy_ids = [s['policy_id'] for s in samples]
    assert len(policy_ids) == len(set(policy_ids)), "Duplicate policy_ids!"

    # 2. Check target variance
    target_returns = [s['target_return'] for s in samples]
    target_stds = [s['target_std'] for s in samples]
    mean_return = np.mean(target_returns)
    mean_std = np.mean(target_stds)
    cv = mean_std / mean_return if mean_return > 0 else float('inf')
    assert cv < 0.5, f"High coefficient of variation: {cv:.2f} > 0.5"

    # 3. Check parameter diversity
    param_norms = [s['param_norm'] for s in samples]
    assert np.std(param_norms) > 0.1 * np.mean(param_norms), "Low param diversity"

    # 4. Check for NaNs
    for i, s in enumerate(samples):
        assert np.isfinite(s['target_return']), f"NaN target at index {i}"
        assert np.all(np.isfinite(s['policy_params'])), f"NaN params at index {i}"

    # 5. Check temporal ordering (if using temporal split)
    episodes = [s['snapshot_episode'] for s in samples]
    assert episodes == sorted(episodes), "Samples not in temporal order"

    print(f"✓ Validation passed for {len(samples)} samples")
    print(f"  Mean return: {mean_return:.3f} ± {mean_std:.3f}")
    print(f"  Coefficient of variation: {cv:.3f}")
    print(f"  Parameter norm range: {min(param_norms):.2f} - {max(param_norms):.2f}")
```

### Post-Training Validation

After training meta-value network:

```python
def evaluate_meta_value(V_theta, X_test, y_test):
    """Evaluate meta-value network quality."""

    y_pred = V_theta.predict(X_test)

    # 1. Pearson correlation
    pearson = np.corrcoef(y_test, y_pred)[0, 1]
    print(f"Pearson correlation: {pearson:.3f}")
    assert pearson > 0.5, f"Low correlation: {pearson:.3f}"

    # 2. Spearman rank correlation
    from scipy.stats import spearmanr
    spearman, _ = spearmanr(y_test, y_pred)
    print(f"Spearman rank correlation: {spearman:.3f}")

    # 3. Mean absolute error
    mae = np.mean(np.abs(y_test - y_pred))
    print(f"MAE: {mae:.3f}")

    # 4. Check calibration (bucketed)
    n_buckets = 5
    for i in range(n_buckets):
        mask = (y_pred >= np.percentile(y_pred, i * 20)) & \
               (y_pred < np.percentile(y_pred, (i + 1) * 20))
        bucket_mean_pred = y_pred[mask].mean()
        bucket_mean_true = y_test[mask].mean()
        print(f"  Bucket {i}: pred {bucket_mean_pred:.3f}, true {bucket_mean_true:.3f}")

    return pearson
```

## Example Build Script

```python
def build_dataset(run_paths, snapshot_interval=20, k_eval=100):
    """Build meta-value dataset from training runs.

    Args:
        run_paths: List of paths to training run directories
        snapshot_interval: Episode interval for snapshots
        k_eval: Number of evaluation episodes per snapshot

    Returns:
        dataset: Dictionary with samples and metadata
    """
    samples = []

    for run_path in run_paths:
        run_id = Path(run_path).name

        # Load training run
        metrics = pd.read_csv(Path(run_path) / 'metrics.csv')
        n_episodes = len(metrics)

        # Determine snapshot episodes
        snapshot_episodes = list(range(snapshot_interval, n_episodes + 1, snapshot_interval))

        for ep in snapshot_episodes:
            # Load policy checkpoint
            checkpoint_path = Path(run_path) / f'checkpoint_ep_{ep}.pt'
            if not checkpoint_path.exists():
                continue

            agent = load_agent(checkpoint_path)

            # Evaluate policy
            returns = []
            lengths = []
            for _ in range(k_eval):
                G, length = evaluate_policy(agent, env, greedy=True)
                returns.append(G)
                lengths.append(length)

            # Create sample
            sample = {
                'policy_id': f'{run_id}_ep_{ep:04d}',
                'run_id': run_id,
                'snapshot_episode': ep,
                'target_return': np.mean(returns),
                'target_std': np.std(returns),
                'n_eval_episodes': k_eval,
                'policy_params': agent.get_policy_parameters(),
                'param_norm': np.linalg.norm(agent.get_policy_parameters()),
                # ... compute other features
            }

            samples.append(sample)

    dataset = {
        'samples': samples,
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'n_samples': len(samples),
            'snapshot_interval': snapshot_interval,
            'eval_episodes_per_snapshot': k_eval,
            'source_runs': [Path(p).name for p in run_paths],
        }
    }

    return dataset
```

## References

- Bandit sample complexity analysis: `analysis/sample_efficiency_analysis.py`
- REINFORCE agent: `src/agents/reinforce_agent.py`
- MDP trainer: `src/training/mdp_trainer.py`
- Chain MDP baseline: `experiments/chain_mdp_baseline.yaml`
