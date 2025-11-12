"""
Main experiment runner.

Usage:
    python -m experiments.run_experiment --config experiments/bandit_self_gradient.yaml
    python -m experiments.run_experiment --config experiments/bandit_quick_test.yaml seed=456
"""

import argparse
import yaml
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import set_seed, create_experiment_dir
from src.envs import BanditEnvironment
from src.agents import SelfGradientBanditAgent
from src.models.bandit_models import BaselineAgent
from src.training import BanditTrainer


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def override_config(config: dict, overrides: list):
    """
    Override config values from command line.

    Args:
        config: Configuration dictionary
        overrides: List of key=value strings
    """
    for override in overrides:
        if '=' not in override:
            continue

        key, value = override.split('=', 1)

        # Parse value
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False

        # Handle nested keys (e.g., agent.learning_rate)
        parts = key.split('.')
        target = config
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]

        target[parts[-1]] = value


def create_environment(config: dict) -> BanditEnvironment:
    """Create bandit environment from config."""
    env_config = config['environment']

    env = BanditEnvironment(
        n_arms=env_config['n_arms'],
        seed=env_config['seed'],
        non_stationary=env_config.get('non_stationary', False),
        drift_rate=env_config.get('drift_rate', 0.1)
    )

    return env


def create_agent(config: dict) -> SelfGradientBanditAgent:
    """Create self-gradient agent from config."""
    agent_config = config['agent']
    env_config = config['environment']

    agent = SelfGradientBanditAgent(
        n_arms=env_config['n_arms'],
        learning_rate=agent_config['learning_rate'],
        meta_learning_rate=agent_config['meta_learning_rate'],
        gradient_clip=agent_config['gradient_clip'],
        buffer_size=agent_config['buffer_size'],
        batch_size=agent_config['batch_size'],
        exploration_episodes=agent_config['exploration_episodes'],
        hidden_dim=agent_config['hidden_dim']
    )

    return agent


def create_baselines(config: dict) -> dict:
    """Create baseline agents from config."""
    env_config = config['environment']
    baseline_configs = config.get('baselines', [])

    baselines = {}

    for baseline_config in baseline_configs:
        name = baseline_config['name']

        if name == 'epsilon_greedy':
            baselines[name] = BaselineAgent(
                n_arms=env_config['n_arms'],
                algorithm='epsilon_greedy',
                epsilon=baseline_config.get('epsilon', 0.1)
            )
        elif name == 'ucb':
            baselines[name] = BaselineAgent(
                n_arms=env_config['n_arms'],
                algorithm='ucb',
                ucb_c=baseline_config.get('ucb_c', 2.0)
            )
        elif name == 'thompson':
            baselines[name] = BaselineAgent(
                n_arms=env_config['n_arms'],
                algorithm='thompson'
            )
        else:
            raise ValueError(f"Unknown baseline: {name}")

    return baselines


def run_bandit_experiment(config: dict, exp_dir: Path):
    """
    Run bandit experiment.

    Args:
        config: Configuration dictionary
        exp_dir: Experiment directory
    """
    # Set random seed
    set_seed(config['environment']['seed'])

    # Create components
    env = create_environment(config)
    agent = create_agent(config)
    baselines = create_baselines(config)

    # Create trainer
    trainer = BanditTrainer(
        env=env,
        agent=agent,
        baselines=baselines,
        log_dir=exp_dir,
        log_interval=config['training']['log_interval']
    )

    # Train
    metrics = trainer.train(n_episodes=config['training']['n_episodes'])

    # Save results
    if config['logging'].get('save_metrics', True):
        metrics_path = exp_dir / 'metrics.csv'
        trainer.save_metrics(metrics_path)

    if config['logging'].get('save_plots', True):
        plot_path = exp_dir / 'results.png'
        trainer.plot_results(save_path=plot_path)

    # Save final checkpoint
    checkpoint_path = exp_dir / 'final_checkpoint.pt'
    agent.save_checkpoint(str(checkpoint_path))

    # Save config
    config_path = exp_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\nExperiment complete. Results saved to: {exp_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Run gradient world model experiments'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        'overrides',
        nargs='*',
        help='Config overrides (key=value)'
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply overrides
    if args.overrides:
        override_config(config, args.overrides)

    # Create experiment directory
    base_dir = Path(config['logging']['log_dir'])
    exp_name = config['experiment']['name']
    exp_dir = create_experiment_dir(base_dir, exp_name)

    print("=" * 60)
    print(f"Starting experiment: {config['experiment']['name']}")
    print(f"Description: {config['experiment']['description']}")
    print(f"Output directory: {exp_dir}")
    print("=" * 60)

    # Run experiment based on type
    exp_type = config['experiment']['type']

    if exp_type == 'bandit':
        run_bandit_experiment(config, exp_dir)
    else:
        raise ValueError(f"Unknown experiment type: {exp_type}")


if __name__ == '__main__':
    main()
