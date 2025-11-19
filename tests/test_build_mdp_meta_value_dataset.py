"""Tests for MDP meta-value dataset builder."""

import pytest
import torch
import numpy as np
import yaml
from pathlib import Path
import subprocess
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.mdp_envs import ChainMDP
from src.agents.reinforce_agent import REINFORCEAgent
from src.training.mdp_trainer import MDPTrainer


@pytest.fixture
def tiny_training_run(tmp_path):
    """Create a tiny training run for testing dataset builder.

    Returns:
        Path to training run directory with checkpoints
    """
    # Create config
    config = {
        'experiment': {
            'name': 'test_run',
            'type': 'mdp',
            'description': 'Test run for dataset builder',
        },
        'environment': {
            'env_type': 'chain_mdp',
            'n_states': 5,  # Small for speed
            'max_steps': 50,
            'discount_factor': 0.99,
            'seed': 42,
        },
        'agent': {
            'type': 'reinforce',
            'learning_rate': 0.01,
            'hidden_dim': 32,  # Small for speed
            'use_baseline': True,
            'entropy_bonus': 0.01,
        },
        'training': {
            'n_episodes': 50,
            'log_interval': 10,
            'save_interval': 10,  # Save every 10 episodes
        },
        'logging': {
            'log_dir': str(tmp_path),
            'save_metrics': True,
            'save_plots': False,
        }
    }

    # Save config
    config_path = tmp_path / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    # Create and train agent
    env = ChainMDP(
        n_states=5,
        max_steps=50,
        discount_factor=0.99,
        seed=42
    )

    agent = REINFORCEAgent(
        state_dim=env.get_state_dim(),
        action_dim=env.get_action_dim(),
        learning_rate=0.01,
        discount_factor=0.99,
        hidden_dim=32,
        use_baseline=True,
        entropy_bonus=0.01,
        seed=42
    )

    # Create run directory
    run_dir = tmp_path / 'test_run'
    run_dir.mkdir()

    trainer = MDPTrainer(
        env=env,
        agent=agent,
        log_dir=run_dir,
        log_interval=10,
        save_interval=10
    )

    # Train
    metrics_df = trainer.train(n_episodes=50)

    # Save metrics
    metrics_path = run_dir / 'metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)

    # Save final checkpoint (in case last episode isn't divisible by save_interval)
    final_checkpoint_path = run_dir / 'final_checkpoint.pt'
    agent.save(str(final_checkpoint_path))

    return {
        'run_dir': run_dir,
        'config_path': config_path,
        'config': config,
        'n_episodes': 50,
        'save_interval': 10,
    }


class TestBuildMDPMetaValueDataset:
    """Test suite for MDP meta-value dataset builder."""

    def test_build_single_run_dataset(self, tiny_training_run, tmp_path):
        """Test building dataset from a single training run."""
        run_info = tiny_training_run
        output_path = tmp_path / 'dataset.pt'

        # Build dataset using the module
        from src.analysis.build_mdp_meta_value_dataset import (
            load_config,
            create_environment,
            build_dataset_from_run
        )

        config = load_config(str(run_info['config_path']))
        env = create_environment(config)

        samples = build_dataset_from_run(
            run_dir=run_info['run_dir'],
            config=config,
            env=env,
            snapshot_interval=10,
            n_eval_episodes=10  # Small for speed
        )

        # Check number of samples
        # Snapshots at episodes: 10, 20, 30, 40, 50
        expected_snapshots = [10, 20, 30, 40, 50]
        assert len(samples) == len(expected_snapshots), \
            f"Expected {len(expected_snapshots)} samples, got {len(samples)}"

        # Check each sample has required fields
        for i, sample in enumerate(samples):
            assert 'policy_id' in sample
            assert 'run_id' in sample
            assert 'snapshot_episode' in sample
            assert 'target_return' in sample
            assert 'target_std' in sample
            assert 'n_eval_episodes' in sample
            assert 'policy_params' in sample
            assert 'param_norm' in sample
            assert 'env_name' in sample
            assert 'success_rate' in sample

            # Check snapshot episode is correct
            assert sample['snapshot_episode'] == expected_snapshots[i]

            # Check types
            assert isinstance(sample['target_return'], float)
            assert isinstance(sample['target_std'], float)
            assert isinstance(sample['policy_params'], np.ndarray)
            assert sample['policy_params'].ndim == 1  # Flattened
            assert sample['n_eval_episodes'] == 10

    def test_dataset_target_quality_chain_mdp(self, tiny_training_run):
        """Test that Chain MDP dataset has high-quality targets."""
        run_info = tiny_training_run

        from src.analysis.build_mdp_meta_value_dataset import (
            load_config,
            create_environment,
            build_dataset_from_run
        )

        config = load_config(str(run_info['config_path']))
        env = create_environment(config)

        samples = build_dataset_from_run(
            run_dir=run_info['run_dir'],
            config=config,
            env=env,
            snapshot_interval=10,
            n_eval_episodes=20  # More episodes for better estimates
        )

        # Chain MDP optimal return (for 5-state chain with γ=0.99)
        # Optimal: always right, 4 steps to goal, return = γ^3 ≈ 0.970
        optimal_return = 0.99 ** 3

        # After sufficient training, returns should be close to optimal
        # Check last 2 snapshots (episodes 40, 50)
        for sample in samples[-2:]:
            assert sample['target_return'] >= 0.85, \
                f"Target return {sample['target_return']} too low (expected ≥0.85)"

            # Success rate should be very high
            assert sample['success_rate'] >= 0.80, \
                f"Success rate {sample['success_rate']} too low (expected ≥0.80)"

            # Standard deviation should be relatively low
            # (greedy policy should be deterministic or near-deterministic)
            assert sample['target_std'] <= 0.3, \
                f"Target std {sample['target_std']} too high (expected ≤0.3)"

    def test_metadata_and_shapes(self, tiny_training_run, tmp_path):
        """Test dataset metadata and array shapes."""
        run_info = tiny_training_run
        output_path = tmp_path / 'dataset.pt'

        from src.analysis.build_mdp_meta_value_dataset import (
            load_config,
            create_environment,
            build_dataset_from_run
        )

        config = load_config(str(run_info['config_path']))
        env = create_environment(config)

        samples = build_dataset_from_run(
            run_dir=run_info['run_dir'],
            config=config,
            env=env,
            snapshot_interval=10,
            n_eval_episodes=10
        )

        # Check all samples have same parameter dimensionality
        param_dims = [s['policy_params'].shape[0] for s in samples]
        assert len(set(param_dims)) == 1, "Inconsistent parameter dimensions across samples"

        # Check metadata fields
        for sample in samples:
            assert sample['env_name'] == 'chain_mdp'
            assert sample['discount_factor'] == 0.99
            assert 'env_config' in sample
            assert isinstance(sample['env_config'], dict)

            # Check training context fields
            assert 'training_return_ema' in sample
            assert 'training_episodes_seen' in sample
            assert 'policy_entropy' in sample

            # Check optional fields
            assert 'eval_episode_lengths' in sample
            assert isinstance(sample['eval_episode_lengths'], list)
            assert len(sample['eval_episode_lengths']) == 10

    def test_temporal_ordering(self, tiny_training_run):
        """Test that samples are in temporal order."""
        run_info = tiny_training_run

        from src.analysis.build_mdp_meta_value_dataset import (
            load_config,
            create_environment,
            build_dataset_from_run
        )

        config = load_config(str(run_info['config_path']))
        env = create_environment(config)

        samples = build_dataset_from_run(
            run_dir=run_info['run_dir'],
            config=config,
            env=env,
            snapshot_interval=10,
            n_eval_episodes=10
        )

        # Check snapshots are in increasing order
        snapshot_episodes = [s['snapshot_episode'] for s in samples]
        assert snapshot_episodes == sorted(snapshot_episodes), \
            "Samples not in temporal order"

    def test_full_dataset_structure(self, tiny_training_run, tmp_path):
        """Test complete dataset structure with metadata."""
        run_info = tiny_training_run
        output_path = tmp_path / 'full_dataset.pt'

        from src.analysis.build_mdp_meta_value_dataset import (
            load_config,
            create_environment,
            build_dataset_from_run
        )
        from datetime import datetime

        config = load_config(str(run_info['config_path']))
        env = create_environment(config)

        samples = build_dataset_from_run(
            run_dir=run_info['run_dir'],
            config=config,
            env=env,
            snapshot_interval=10,
            n_eval_episodes=10
        )

        # Create full dataset structure (as builder would)
        dataset = {
            'samples': samples,
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'n_samples': len(samples),
                'snapshot_interval': 10,
                'eval_episodes_per_snapshot': 10,
                'source_runs': [run_info['run_dir'].name],
                'config_path': str(run_info['config_path']),
                'environment': 'chain_mdp',
            }
        }

        # Save and reload
        torch.save(dataset, output_path)
        loaded_dataset = torch.load(output_path, weights_only=False)

        # Check structure
        assert 'samples' in loaded_dataset
        assert 'metadata' in loaded_dataset
        assert len(loaded_dataset['samples']) == len(samples)
        assert loaded_dataset['metadata']['n_samples'] == len(samples)
        assert loaded_dataset['metadata']['environment'] == 'chain_mdp'

    def test_invalid_checkpoint_handling(self, tmp_path):
        """Test handling of missing or corrupt checkpoints."""
        from src.analysis.build_mdp_meta_value_dataset import (
            load_config,
            create_environment,
            build_dataset_from_run
        )

        # Create a fake run directory with metrics but no checkpoints
        fake_run_dir = tmp_path / 'fake_run'
        fake_run_dir.mkdir()

        # Create minimal metrics.csv
        import pandas as pd
        metrics_df = pd.DataFrame({
            'episode_returns': [0.5, 0.6, 0.7, 0.8, 0.9],
            'episode_lengths': [10, 10, 10, 10, 10],
            'success_rate': [0.0, 0.0, 0.5, 0.8, 1.0],
        })
        metrics_df.to_csv(fake_run_dir / 'metrics.csv', index=False)

        # Create minimal config
        config = {
            'environment': {
                'env_type': 'chain_mdp',
                'n_states': 5,
                'max_steps': 50,
                'discount_factor': 0.99,
                'seed': 42,
            },
            'agent': {
                'learning_rate': 0.01,
                'hidden_dim': 32,
            }
        }

        config_path = tmp_path / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        env = create_environment(config)

        # Try to build dataset (should handle missing checkpoints gracefully)
        samples = build_dataset_from_run(
            run_dir=fake_run_dir,
            config=config,
            env=env,
            snapshot_interval=10,
            n_eval_episodes=10
        )

        # Should return empty list (no valid checkpoints)
        assert len(samples) == 0, "Expected empty sample list for missing checkpoints"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
