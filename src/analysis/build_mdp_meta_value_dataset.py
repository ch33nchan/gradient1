"""Build MDP meta-value dataset from training runs.

This script:
1. Loads one or more MDP training runs
2. Extracts policy parameter snapshots at regular intervals
3. Evaluates each snapshot with K_eval episodes (greedy policy)
4. Computes multi-episode average return as meta-value target
5. Saves dataset in format specified by analysis/mdp_meta_value_spec.md

Usage:
    python -m src.analysis.build_mdp_meta_value_dataset \
        --run-dir logs/chain_mdp_baseline/run_2025-11-19_06-18-51 \
        --config experiments/chain_mdp_baseline.yaml \
        --snapshot-interval 20 \
        --n-eval-episodes 200 \
        --output analysis/mdp_meta_value_dataset.pt
"""

import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.envs.mdp_envs import ChainMDP
from src.agents.reinforce_agent import REINFORCEAgent


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_environment(config: dict):
    """Create MDP environment from config."""
    env_config = config['environment']
    env_type = env_config.get('env_type', 'chain_mdp')

    if env_type == 'chain_mdp':
        env = ChainMDP(
            n_states=env_config['n_states'],
            max_steps=env_config.get('max_steps', 100),
            discount_factor=env_config.get('discount_factor', 0.99),
            seed=env_config['seed']
        )
    else:
        raise ValueError(f"Unknown environment type: {env_type}")

    return env


def evaluate_policy(agent: REINFORCEAgent, env, n_episodes: int, greedy: bool = True) -> Dict[str, Any]:
    """Evaluate policy over multiple episodes.

    Args:
        agent: REINFORCE agent with loaded parameters
        env: MDP environment
        n_episodes: Number of evaluation episodes
        greedy: Use greedy policy (default: True)

    Returns:
        stats: Dictionary with evaluation statistics
    """
    returns = []
    lengths = []
    successes = []

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        episode_return = 0.0
        episode_length = 0
        discount = 1.0

        while not done:
            action = agent.act(state, greedy=greedy)
            next_state, reward, done, info = env.step(action)

            episode_return += discount * reward
            discount *= env.get_discount_factor()
            episode_length += 1
            state = next_state

        returns.append(episode_return)
        lengths.append(episode_length)
        successes.append(1.0 if info.get('at_goal', False) else 0.0)

    stats = {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'min_return': np.min(returns),
        'max_return': np.max(returns),
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'success_rate': np.mean(successes),
        'returns': returns,
        'lengths': lengths,
    }

    return stats


def compute_policy_entropy(agent: REINFORCEAgent, initial_state: int = 0) -> float:
    """Compute policy entropy at initial state."""
    state = torch.tensor([initial_state], dtype=torch.long)
    with torch.no_grad():
        probs = agent.policy.get_action_probs(state)
        entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
    return entropy


def extract_features(agent: REINFORCEAgent, training_metrics: pd.DataFrame, snapshot_episode: int, env_config: dict) -> Dict[str, Any]:
    """Extract features for meta-value input.

    Args:
        agent: REINFORCE agent
        training_metrics: Training metrics DataFrame
        snapshot_episode: Episode number of snapshot
        env_config: Environment configuration

    Returns:
        features: Dictionary of features
    """
    # Get policy parameters
    params = agent.get_policy_parameters()

    # Compute parameter statistics
    param_norm = float(np.linalg.norm(params))
    param_mean = float(np.mean(params))
    param_std = float(np.std(params))

    # Get training context
    if snapshot_episode <= len(training_metrics):
        # Get training return at this episode (or EMA of last few)
        window = min(10, snapshot_episode)
        training_return_ema = training_metrics['episode_returns'].iloc[max(0, snapshot_episode - window):snapshot_episode].mean()
    else:
        training_return_ema = 0.0

    # Compute policy entropy at initial state
    policy_entropy = compute_policy_entropy(agent, initial_state=0)

    features = {
        'policy_params': params,
        'param_norm': param_norm,
        'param_mean': param_mean,
        'param_std': param_std,
        'training_return_ema': float(training_return_ema),
        'training_episodes_seen': snapshot_episode,
        'policy_entropy': policy_entropy,
    }

    return features


def build_dataset_from_run(
    run_dir: Path,
    config: dict,
    env,
    snapshot_interval: int,
    n_eval_episodes: int
) -> List[Dict[str, Any]]:
    """Build dataset samples from a single training run.

    Args:
        run_dir: Path to training run directory
        config: Experiment configuration
        env: MDP environment
        snapshot_interval: Episode interval for snapshots
        n_eval_episodes: Number of evaluation episodes per snapshot

    Returns:
        samples: List of dataset samples
    """
    run_id = run_dir.name
    print(f"\nProcessing run: {run_id}")

    # Load training metrics
    metrics_path = run_dir / 'metrics.csv'
    if not metrics_path.exists():
        print(f"  Warning: No metrics.csv found in {run_dir}")
        return []

    metrics = pd.read_csv(metrics_path)
    n_episodes = len(metrics)
    print(f"  Training episodes: {n_episodes}")

    # Determine snapshot episodes
    snapshot_episodes = list(range(snapshot_interval, n_episodes + 1, snapshot_interval))
    print(f"  Snapshot episodes: {snapshot_episodes}")

    samples = []

    for ep in snapshot_episodes:
        # Check if we have a checkpoint for this episode
        # Try both formats: checkpoint saved during training or final checkpoint
        checkpoint_path = run_dir / f'checkpoint_ep_{ep}.pt'
        if not checkpoint_path.exists() and ep == n_episodes:
            checkpoint_path = run_dir / 'final_checkpoint.pt'

        if not checkpoint_path.exists():
            print(f"  Skipping episode {ep}: No checkpoint found")
            continue

        # Create agent and load checkpoint
        agent = REINFORCEAgent(
            state_dim=env.get_state_dim(),
            action_dim=env.get_action_dim(),
            learning_rate=config['agent'].get('learning_rate', 0.01),
            discount_factor=env.get_discount_factor(),
            hidden_dim=config['agent'].get('hidden_dim', 64),
            seed=config['environment']['seed']
        )

        try:
            agent.load(str(checkpoint_path))
        except Exception as e:
            print(f"  Skipping episode {ep}: Failed to load checkpoint ({e})")
            continue

        # Evaluate policy
        print(f"  Evaluating episode {ep} with {n_eval_episodes} episodes...", end=' ')
        eval_stats = evaluate_policy(agent, env, n_eval_episodes, greedy=True)
        print(f"Return: {eval_stats['mean_return']:.3f} ± {eval_stats['std_return']:.3f}")

        # Extract features
        features = extract_features(agent, metrics, ep, config['environment'])

        # Create sample
        sample = {
            # Identification
            'policy_id': f'{run_id}_ep_{ep:04d}',
            'run_id': run_id,
            'snapshot_episode': ep,

            # Target
            'target_return': eval_stats['mean_return'],
            'target_std': eval_stats['std_return'],
            'n_eval_episodes': n_eval_episodes,

            # Features
            'policy_params': features['policy_params'],
            'param_norm': features['param_norm'],
            'param_mean': features['param_mean'],
            'param_std': features['param_std'],
            'training_return_ema': features['training_return_ema'],
            'training_episodes_seen': features['training_episodes_seen'],
            'policy_entropy': features['policy_entropy'],

            # Environment metadata
            'env_name': config['environment'].get('env_type', 'chain_mdp'),
            'env_config': config['environment'],
            'discount_factor': env.get_discount_factor(),

            # Optional: Episode statistics
            'eval_episode_lengths': eval_stats['lengths'],
            'eval_success_rate': eval_stats['success_rate'],
            'eval_min_return': eval_stats['min_return'],
            'eval_max_return': eval_stats['max_return'],
        }

        samples.append(sample)

    print(f"  Created {len(samples)} samples from this run")
    return samples


def main():
    parser = argparse.ArgumentParser(
        description='Build MDP meta-value dataset from training runs'
    )
    parser.add_argument(
        '--run-dir',
        type=str,
        required=True,
        help='Path to training run directory (or parent logs dir)'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to MDP experiment config (e.g., experiments/chain_mdp_baseline.yaml)'
    )
    parser.add_argument(
        '--snapshot-interval',
        type=int,
        default=20,
        help='Episode interval for snapshots (default: 20)'
    )
    parser.add_argument(
        '--n-eval-episodes',
        type=int,
        default=200,
        help='Number of evaluation episodes per snapshot (default: 200)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='analysis/mdp_meta_value_dataset.pt',
        help='Output dataset path (default: analysis/mdp_meta_value_dataset.pt)'
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config: {args.config}")
    print(f"Environment: {config['environment'].get('env_type', 'chain_mdp')}")

    # Create environment
    env = create_environment(config)
    print(f"Environment: {env.__class__.__name__} ({env.get_state_dim()} states, {env.get_action_dim()} actions)")

    # Find training runs
    run_dir = Path(args.run_dir)

    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)

    # Check if this is a single run or a parent directory
    if (run_dir / 'metrics.csv').exists():
        # Single run
        run_dirs = [run_dir]
    else:
        # Parent directory - find all subdirectories with metrics.csv
        run_dirs = [d for d in run_dir.iterdir() if d.is_dir() and (d / 'metrics.csv').exists()]

    if not run_dirs:
        print(f"Error: No training runs found in {run_dir}")
        print("Expected to find directories with metrics.csv files")
        sys.exit(1)

    print(f"\nFound {len(run_dirs)} training run(s)")

    # Build dataset from all runs
    all_samples = []
    for run_dir in sorted(run_dirs):
        samples = build_dataset_from_run(
            run_dir=run_dir,
            config=config,
            env=env,
            snapshot_interval=args.snapshot_interval,
            n_eval_episodes=args.n_eval_episodes
        )
        all_samples.extend(samples)

    if not all_samples:
        print("\nError: No samples collected!")
        print("Make sure training runs have checkpoints saved")
        sys.exit(1)

    # Sort samples by (run_id, snapshot_episode) for temporal ordering
    all_samples.sort(key=lambda s: (s['run_id'], s['snapshot_episode']))

    # Create dataset
    dataset = {
        'samples': all_samples,
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'n_samples': len(all_samples),
            'snapshot_interval': args.snapshot_interval,
            'eval_episodes_per_snapshot': args.n_eval_episodes,
            'source_runs': [run_dir.name for run_dir in run_dirs],
            'config_path': args.config,
            'environment': config['environment'].get('env_type', 'chain_mdp'),
        }
    }

    # Save dataset
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)

    print(f"\n{'=' * 80}")
    print(f"Dataset saved: {output_path}")
    print(f"Total samples: {len(all_samples)}")
    print(f"Source runs: {len(run_dirs)}")
    print(f"Snapshot interval: {args.snapshot_interval} episodes")
    print(f"Eval episodes per snapshot: {args.n_eval_episodes}")

    # Print statistics
    target_returns = [s['target_return'] for s in all_samples]
    target_stds = [s['target_std'] for s in all_samples]
    param_norms = [s['param_norm'] for s in all_samples]

    print(f"\nDataset statistics:")
    print(f"  Target return: {np.mean(target_returns):.3f} ± {np.std(target_returns):.3f}")
    print(f"  Target std (mean): {np.mean(target_stds):.3f}")
    print(f"  Param norm range: [{np.min(param_norms):.2f}, {np.max(param_norms):.2f}]")
    print(f"  Coefficient of variation: {np.mean(target_stds) / np.mean(target_returns):.3f}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
