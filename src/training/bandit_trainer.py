"""
Trainer for bandit experiments.
Handles training loop, metrics tracking, and visualization.
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import matplotlib.pyplot as plt

from ..envs.bandits import BanditEnvironment
from ..agents.self_gradient_agent import SelfGradientBanditAgent
from ..models.bandit_models import BaselineAgent
from ..utils import setup_logger


class BanditTrainer:
    """
    Trainer for bandit experiments.

    Runs our self-gradient agent against baseline algorithms
    and tracks comparative performance.
    """

    def __init__(
        self,
        env: BanditEnvironment,
        agent: SelfGradientBanditAgent,
        baselines: Dict[str, BaselineAgent],
        log_dir: Optional[Path] = None,
        log_interval: int = 100
    ):
        """
        Initialize trainer.

        Args:
            env: Bandit environment
            agent: Self-gradient agent
            baselines: Dictionary of baseline agents
            log_dir: Directory for logs
            log_interval: Frequency of logging
        """
        self.env = env
        self.agent = agent
        self.baselines = baselines
        self.log_interval = log_interval

        # Setup logging
        if log_dir is not None:
            self.logger = setup_logger('bandit_trainer', log_dir)
        else:
            self.logger = setup_logger('bandit_trainer', console=True)

        # Metrics storage
        self.metrics = {
            'our_rewards': [],
            'our_regrets': [],
            'baseline_rewards': {name: [] for name in baselines.keys()},
            'baseline_regrets': {name: [] for name in baselines.keys()},
            'gradient_errors': [],
            'meta_value_losses': [],
            'policy_entropy': [],
            # Planning diagnostics
            'planning_enabled': [],
            'planning_weight': [],
            'grad_error_ema': [],
            'meta_value_current': [],
            'meta_value_correlation': [],
            'planning_scores': {f'arm_{i}': [] for i in range(agent.n_arms)},
            'planning_best_arm': [],
            'planning_chosen_arm': [],
            'planning_score_gap': []
        }

    def train(self, n_episodes: int) -> Dict:
        """
        Run training for specified number of episodes.

        Args:
            n_episodes: Number of episodes to run

        Returns:
            Dictionary of metrics
        """
        self.logger.info(f"Starting training for {n_episodes} episodes")
        self.logger.info(f"Environment: {self.env.n_arms} arms")
        self.logger.info(f"Baselines: {list(self.baselines.keys())}")

        start_time = time.time()

        for episode in range(n_episodes):
            # Our agent's turn
            action = self.agent.act(exploration_mode='auto')
            reward = self.env.pull(action)
            regret = self.env.get_regret(action)

            # Update agent
            self.agent.update(action, reward)

            # Store metrics
            self.metrics['our_rewards'].append(reward)
            self.metrics['our_regrets'].append(regret)

            # Store gradient errors if available
            if self.agent.gradient_errors:
                self.metrics['gradient_errors'].append(
                    self.agent.gradient_errors[-1]
                )

            # Store meta-value losses if available
            if self.agent.meta_value_losses:
                self.metrics['meta_value_losses'].append(
                    self.agent.meta_value_losses[-1]
                )

            # Compute policy entropy
            probs = self.agent.get_policy_probs()
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            self.metrics['policy_entropy'].append(entropy)

            # Store planning diagnostics
            planning_diag = self.agent.get_planning_diagnostics()
            self.metrics['planning_enabled'].append(planning_diag['planning_enabled'])
            self.metrics['planning_weight'].append(planning_diag['planning_weight'])
            self.metrics['grad_error_ema'].append(planning_diag['grad_error_ema'])
            self.metrics['meta_value_current'].append(planning_diag['meta_value_current'])
            self.metrics['meta_value_correlation'].append(planning_diag['meta_value_correlation'])
            for i in range(self.agent.n_arms):
                self.metrics['planning_scores'][f'arm_{i}'].append(
                    planning_diag[f'planning_score_arm_{i}']
                )
            self.metrics['planning_best_arm'].append(planning_diag['planning_best_arm'])
            self.metrics['planning_chosen_arm'].append(planning_diag['planning_chosen_arm'])
            self.metrics['planning_score_gap'].append(planning_diag['planning_score_gap'])

            # Baseline agents
            for name, baseline in self.baselines.items():
                action = baseline.act()
                reward = self.env.pull(action)
                regret = self.env.get_regret(action)
                baseline.update(action, reward)

                self.metrics['baseline_rewards'][name].append(reward)
                self.metrics['baseline_regrets'][name].append(regret)

            # Logging
            if (episode + 1) % self.log_interval == 0:
                self._log_progress(episode + 1, n_episodes, start_time)

        # Final summary
        self._log_final_summary(n_episodes)

        return self.metrics

    def _log_progress(self, episode: int, total: int, start_time: float):
        """Log training progress."""
        elapsed = time.time() - start_time

        # Compute statistics over last 100 episodes
        window = min(100, episode)

        our_avg_reward = np.mean(self.metrics['our_rewards'][-window:])
        our_avg_regret = np.mean(self.metrics['our_regrets'][-window:])

        self.logger.info(f"\nEpisode {episode}/{total} ({elapsed:.1f}s)")
        self.logger.info(f"  Self-Gradient Agent:")
        self.logger.info(f"    Avg reward: {our_avg_reward:.3f}")
        self.logger.info(f"    Avg regret: {our_avg_regret:.3f}")

        if self.metrics['gradient_errors']:
            avg_grad_error = np.mean(self.metrics['gradient_errors'][-window:])
            self.logger.info(f"    Gradient error: {avg_grad_error:.6f}")

        # Baseline statistics
        for name in self.baselines.keys():
            avg_reward = np.mean(
                self.metrics['baseline_rewards'][name][-window:]
            )
            avg_regret = np.mean(
                self.metrics['baseline_regrets'][name][-window:]
            )
            self.logger.info(f"  {name}:")
            self.logger.info(f"    Avg reward: {avg_reward:.3f}")
            self.logger.info(f"    Avg regret: {avg_regret:.3f}")

        # Show top arms
        policy = self.agent.get_policy_probs()
        top_arms = np.argsort(policy)[-3:][::-1]
        self.logger.info(
            f"  Top 3 arms: {top_arms.tolist()} with probs "
            f"{policy[top_arms].tolist()}"
        )

    def _log_final_summary(self, n_episodes: int):
        """Log final training summary."""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("TRAINING COMPLETE")
        self.logger.info("=" * 50)

        final_window = min(500, n_episodes)

        # Our agent
        final_reward = np.mean(self.metrics['our_rewards'][-final_window:])
        final_std = np.std(self.metrics['our_rewards'][-final_window:])
        cum_regret = np.sum(self.metrics['our_regrets'])

        self.logger.info(f"\nFinal {final_window} episodes:")
        self.logger.info(
            f"  Self-Gradient Agent: {final_reward:.3f} +/- {final_std:.3f}"
        )
        self.logger.info(f"  Cumulative regret: {cum_regret:.1f}")

        # Baselines
        for name in self.baselines.keys():
            rewards = self.metrics['baseline_rewards'][name][-final_window:]
            regret = np.sum(self.metrics['baseline_regrets'][name])

            self.logger.info(
                f"  {name}: {np.mean(rewards):.3f} +/- {np.std(rewards):.3f}"
            )
            self.logger.info(f"  Cumulative regret: {regret:.1f}")

        # Environment statistics
        env_stats = self.env.get_statistics()
        self.logger.info(f"\nEnvironment statistics:")
        self.logger.info(f"  Optimal arm: {env_stats['optimal_action']}")
        self.logger.info(f"  Optimal mean: {env_stats['optimal_mean']:.3f}")

    def save_metrics(self, path: Path):
        """
        Save metrics to CSV file.

        Args:
            path: Path to save metrics
        """
        # Convert to DataFrame
        df_data = {
            'episode': list(range(len(self.metrics['our_rewards']))),
            'our_reward': self.metrics['our_rewards'],
            'our_regret': self.metrics['our_regrets'],
            'policy_entropy': self.metrics['policy_entropy']
        }

        # Add baseline metrics
        for name in self.baselines.keys():
            df_data[f'{name}_reward'] = self.metrics['baseline_rewards'][name]
            df_data[f'{name}_regret'] = self.metrics['baseline_regrets'][name]

        # Add gradient errors (may be shorter)
        if self.metrics['gradient_errors']:
            grad_errors = self.metrics['gradient_errors']
            # Pad with NaN
            while len(grad_errors) < len(self.metrics['our_rewards']):
                grad_errors.insert(0, np.nan)
            df_data['gradient_error'] = grad_errors

        # Add planning diagnostics
        df_data['planning_enabled'] = self.metrics['planning_enabled']
        df_data['planning_weight'] = self.metrics['planning_weight']
        df_data['grad_error_ema'] = self.metrics['grad_error_ema']
        df_data['meta_value_current'] = self.metrics['meta_value_current']
        df_data['meta_value_correlation'] = self.metrics['meta_value_correlation']

        # Add planning scores per arm
        for arm_name, scores in self.metrics['planning_scores'].items():
            df_data[f'planning_score_{arm_name}'] = scores

        df_data['planning_best_arm'] = self.metrics['planning_best_arm']
        df_data['planning_chosen_arm'] = self.metrics['planning_chosen_arm']
        df_data['planning_score_gap'] = self.metrics['planning_score_gap']

        df = pd.DataFrame(df_data)
        df.to_csv(path, index=False)

        self.logger.info(f"Metrics saved to {path}")

    def plot_results(self, save_path: Optional[Path] = None):
        """
        Create visualization of results.

        Args:
            save_path: Path to save figure (if None, displays)
        """
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # 1. Cumulative rewards
        ax = axes[0, 0]
        ax.plot(
            np.cumsum(self.metrics['our_rewards']),
            label='Self-Gradient',
            linewidth=2
        )
        for name in self.baselines.keys():
            ax.plot(
                np.cumsum(self.metrics['baseline_rewards'][name]),
                label=name,
                alpha=0.7
            )
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Reward')
        ax.set_title('Cumulative Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Cumulative regret
        ax = axes[0, 1]
        ax.plot(
            np.cumsum(self.metrics['our_regrets']),
            label='Self-Gradient',
            linewidth=2
        )
        for name in self.baselines.keys():
            ax.plot(
                np.cumsum(self.metrics['baseline_regrets'][name]),
                label=name,
                alpha=0.7
            )
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Regret')
        ax.set_title('Cumulative Regret (lower is better)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Rolling average rewards
        ax = axes[0, 2]
        window = 100

        def rolling_mean(x, w):
            if len(x) < w:
                return x
            return np.convolve(x, np.ones(w)/w, 'valid')

        ax.plot(
            rolling_mean(self.metrics['our_rewards'], window),
            label='Self-Gradient',
            linewidth=2
        )
        for name in self.baselines.keys():
            ax.plot(
                rolling_mean(self.metrics['baseline_rewards'][name], window),
                label=name,
                alpha=0.7
            )
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Reward')
        ax.set_title(f'{window}-Episode Rolling Average')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Gradient prediction error
        ax = axes[1, 0]
        if self.metrics['gradient_errors']:
            ax.plot(self.metrics['gradient_errors'])
            ax.set_xlabel('Training Step')
            ax.set_ylabel('MSE')
            ax.set_title('Gradient Prediction Error')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)

        # 5. Policy entropy
        ax = axes[1, 1]
        ax.plot(self.metrics['policy_entropy'])
        ax.set_xlabel('Episode')
        ax.set_ylabel('Entropy (nats)')
        ax.set_title('Policy Entropy')
        ax.grid(True, alpha=0.3)

        # 6. Final performance comparison
        ax = axes[1, 2]
        final_window = 500

        agents = ['Self-Gradient'] + list(self.baselines.keys())
        final_rewards = [np.mean(self.metrics['our_rewards'][-final_window:])]
        final_stds = [np.std(self.metrics['our_rewards'][-final_window:])]

        for name in self.baselines.keys():
            final_rewards.append(
                np.mean(self.metrics['baseline_rewards'][name][-final_window:])
            )
            final_stds.append(
                np.std(self.metrics['baseline_rewards'][name][-final_window:])
            )

        x_pos = np.arange(len(agents))
        ax.bar(x_pos, final_rewards, yerr=final_stds, capsize=5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(agents, rotation=45, ha='right')
        ax.set_ylabel('Average Reward')
        ax.set_title(f'Final {final_window} Episodes')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Figure saved to {save_path}")
        else:
            plt.show()

        plt.close()
