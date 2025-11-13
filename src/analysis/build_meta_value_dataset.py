"""
Build offline meta-value evaluation dataset.

This utility generates clean (θ, avg_reward) pairs for supervised meta-value training.
During a training run, it periodically:
1. Snapshots current policy parameters θ
2. Freezes θ and runs evaluation episodes
3. Computes average reward over evaluation
4. Stores (θ, avg_reward) for offline learning

Usage:
    python -m src.analysis.build_meta_value_dataset --config experiments/ablation_no_planning.yaml
    python -m src.analysis.build_meta_value_dataset --config experiments/ablation_no_planning.yaml --snapshot-interval 20 --eval-episodes 300

Output:
    Saves meta_value_dataset.pt containing:
    - 'parameters': tensor of shape (N, n_arms) - policy logits
    - 'avg_rewards': tensor of shape (N,) - evaluation rewards
    - 'metadata': dict with collection settings
"""

import argparse
import torch
import numpy as np
import yaml
from pathlib import Path
from typing import List, Tuple
import torch.nn.functional as F

from ..envs.bandits import BanditEnvironment
from ..agents.self_gradient_agent import SelfGradientBanditAgent
from ..utils import setup_logger


def evaluate_policy(
    theta: torch.Tensor,
    env: BanditEnvironment,
    n_episodes: int,
    seed: int
) -> float:
    """
    Evaluate a frozen policy for n_episodes.

    Args:
        theta: Policy parameters (logits)
        env: Bandit environment
        n_episodes: Number of evaluation episodes
        seed: Random seed for reproducibility

    Returns:
        Average reward over evaluation episodes
    """
    np.random.seed(seed)

    rewards = []
    with torch.no_grad():
        policy = F.softmax(theta, dim=0).numpy()

        for _ in range(n_episodes):
            # Sample action from frozen policy
            action = np.random.choice(len(policy), p=policy)
            reward = env.pull(action)
            rewards.append(reward)

    return np.mean(rewards)


def build_dataset(
    config_path: Path,
    snapshot_interval: int = 20,
    eval_episodes: int = 300,
    output_path: Path = Path("meta_value_dataset.pt"),
    seed: int = 42
):
    """
    Build offline meta-value dataset from a training run.

    Args:
        config_path: Path to experiment config
        snapshot_interval: Snapshot θ every N episodes
        eval_episodes: Number of episodes for evaluation
        output_path: Where to save dataset
        seed: Random seed
    """
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"Building meta-value dataset from config: {config_path.name}")
    print(f"Snapshot interval: {snapshot_interval} episodes")
    print(f"Evaluation episodes per snapshot: {eval_episodes}")

    # Setup environment
    env_config = config['environment']
    env = BanditEnvironment(
        n_arms=env_config['n_arms'],
        non_stationary=env_config.get('non_stationary', False),
        drift_rate=env_config.get('drift_rate', 0.1),
        seed=env_config.get('seed', seed)
    )

    # Setup agent
    agent_config = config['agent']
    agent = SelfGradientBanditAgent(
        n_arms=env.n_arms,
        learning_rate=agent_config['learning_rate'],
        meta_learning_rate=agent_config['meta_learning_rate'],
        gradient_clip=agent_config.get('gradient_clip', 1.0),
        buffer_size=agent_config.get('buffer_size', 1000),
        batch_size=agent_config.get('batch_size', 16),
        exploration_episodes=agent_config.get('exploration_episodes', 50),
        hidden_dim=agent_config.get('hidden_dim', 32),
        enable_planning=False,  # Disable planning for clean data collection
    )

    # Training config
    training_config = config['training']
    n_episodes = training_config['n_episodes']

    # Storage for dataset
    snapshots: List[Tuple[torch.Tensor, float]] = []

    print(f"\nRunning {n_episodes} training episodes...")
    print(f"Expected snapshots: ~{n_episodes // snapshot_interval}")

    # Training loop with periodic snapshots
    for episode in range(n_episodes):
        # Normal training step
        action = agent.act(exploration_mode='auto')
        reward = env.pull(action)
        agent.update(action, reward)

        # Snapshot and evaluate
        if (episode + 1) % snapshot_interval == 0:
            # Get current policy parameters
            theta_snapshot = agent.get_parameters().detach().clone()

            # Evaluate frozen policy
            avg_reward = evaluate_policy(
                theta_snapshot,
                env,
                eval_episodes,
                seed=seed + episode  # Different seed per evaluation
            )

            snapshots.append((theta_snapshot, avg_reward))

            if (episode + 1) % (snapshot_interval * 5) == 0:
                print(f"  Episode {episode + 1}/{n_episodes}: "
                      f"{len(snapshots)} snapshots collected")

    print(f"\nCollection complete: {len(snapshots)} total snapshots")

    # Convert to tensors
    parameters = torch.stack([s[0] for s in snapshots])
    avg_rewards = torch.tensor([s[1] for s in snapshots], dtype=torch.float32)

    # Statistics
    print(f"\nDataset statistics:")
    print(f"  Shape: {parameters.shape}")
    print(f"  Reward range: [{avg_rewards.min():.3f}, {avg_rewards.max():.3f}]")
    print(f"  Reward mean: {avg_rewards.mean():.3f} ± {avg_rewards.std():.3f}")

    # Save dataset
    dataset = {
        'parameters': parameters,
        'avg_rewards': avg_rewards,
        'metadata': {
            'config': str(config_path),
            'snapshot_interval': snapshot_interval,
            'eval_episodes': eval_episodes,
            'n_samples': len(snapshots),
            'n_arms': env.n_arms,
            'seed': seed
        }
    }

    torch.save(dataset, output_path)
    print(f"\nDataset saved to: {output_path}")
    print(f"Load with: dataset = torch.load('{output_path}')")

    return dataset


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build offline meta-value evaluation dataset"
    )
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to experiment config (e.g., experiments/ablation_no_planning.yaml)'
    )
    parser.add_argument(
        '--snapshot-interval',
        type=int,
        default=20,
        help='Snapshot θ every N episodes (default: 20)'
    )
    parser.add_argument(
        '--eval-episodes',
        type=int,
        default=300,
        help='Number of episodes for evaluation per snapshot (default: 300)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('meta_value_dataset.pt'),
        help='Output path for dataset (default: meta_value_dataset.pt)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        return

    build_dataset(
        config_path=args.config,
        snapshot_interval=args.snapshot_interval,
        eval_episodes=args.eval_episodes,
        output_path=args.output,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
