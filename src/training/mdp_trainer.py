"""Trainer for MDP experiments.

Handles episodic training loop, metrics tracking, and visualization for MDP environments.
Similar to BanditTrainer but adapted for multi-step episodes.
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Any
import matplotlib.pyplot as plt

from ..envs.mdp_envs import MDPEnvironment
from ..agents.reinforce_agent import REINFORCEAgent
from ..utils import setup_logger


class MDPTrainer:
    """Trainer for MDP experiments with episodic training.

    Runs episodes, collects trajectories, and tracks performance metrics.
    """

    def __init__(
        self,
        env: MDPEnvironment,
        agent: REINFORCEAgent,
        log_dir: Optional[Path] = None,
        log_interval: int = 10,
    ):
        """Initialize MDP trainer.

        Args:
            env: MDP environment
            agent: REINFORCE agent
            log_dir: Directory for logs
            log_interval: Frequency of logging (in episodes)
        """
        self.env = env
        self.agent = agent
        self.log_interval = log_interval

        # Setup logging
        if log_dir is not None:
            self.logger = setup_logger('mdp_trainer', log_dir)
        else:
            self.logger = setup_logger('mdp_trainer', console=True)

        # Metrics storage
        self.metrics = {
            'episode_returns': [],
            'episode_lengths': [],
            'policy_losses': [],
            'entropies': [],
            'baseline_values': [],
            'success_rate': [],  # Did episode reach goal?
        }

        # Running statistics
        self.total_steps = 0
        self.total_episodes = 0

    def run_episode(self, greedy: bool = False) -> Dict[str, Any]:
        """Run one episode and collect trajectory.

        Args:
            greedy: If True, use greedy policy (default: False)

        Returns:
            episode_info: Dictionary with episode metrics
        """
        state = self.env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0

        while not done:
            # Select action
            action = self.agent.act(state, greedy=greedy)

            # Take step
            next_state, reward, done, info = self.env.step(action)

            # Store transition
            if not greedy:
                self.agent.store_transition(state, action, reward)

            # Update counters
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1

            # Move to next state
            state = next_state

        # Episode info
        episode_info = {
            'return': episode_reward,
            'length': episode_length,
            'success': info.get('at_goal', False),
        }

        return episode_info

    def train(self, n_episodes: int) -> pd.DataFrame:
        """Run training for specified number of episodes.

        Args:
            n_episodes: Number of episodes to run

        Returns:
            metrics_df: DataFrame with episode-level metrics
        """
        self.logger.info(f"Starting training for {n_episodes} episodes")
        self.logger.info(f"Environment: {self.env.__class__.__name__}")
        self.logger.info(f"State dim: {self.env.get_state_dim()}, Action dim: {self.env.get_action_dim()}")

        start_time = time.time()

        for episode in range(n_episodes):
            # Run episode
            episode_info = self.run_episode(greedy=False)

            # Update agent
            update_metrics = self.agent.update()

            # Store metrics
            self.metrics['episode_returns'].append(episode_info['return'])
            self.metrics['episode_lengths'].append(episode_info['length'])
            self.metrics['policy_losses'].append(update_metrics.get('policy_loss', 0.0))
            self.metrics['entropies'].append(update_metrics.get('entropy', 0.0))
            self.metrics['baseline_values'].append(update_metrics.get('baseline_value', 0.0))
            self.metrics['success_rate'].append(1.0 if episode_info['success'] else 0.0)

            self.total_episodes += 1

            # Logging
            if (episode + 1) % self.log_interval == 0:
                # Compute running statistics (last log_interval episodes)
                recent_returns = self.metrics['episode_returns'][-self.log_interval:]
                recent_lengths = self.metrics['episode_lengths'][-self.log_interval:]
                recent_success = self.metrics['success_rate'][-self.log_interval:]

                mean_return = np.mean(recent_returns)
                mean_length = np.mean(recent_lengths)
                success_rate = np.mean(recent_success)

                elapsed = time.time() - start_time
                eps_per_sec = (episode + 1) / elapsed

                self.logger.info(
                    f"Episode {episode + 1}/{n_episodes} | "
                    f"Return: {mean_return:.3f} | "
                    f"Length: {mean_length:.1f} | "
                    f"Success: {success_rate:.2%} | "
                    f"Loss: {self.metrics['policy_losses'][-1]:.4f} | "
                    f"Speed: {eps_per_sec:.1f} eps/s"
                )

        elapsed = time.time() - start_time
        self.logger.info(f"Training completed in {elapsed:.1f}s ({self.total_episodes / elapsed:.1f} eps/s)")

        # Convert to DataFrame
        metrics_df = pd.DataFrame(self.metrics)
        return metrics_df

    def evaluate(self, n_episodes: int = 100) -> Dict[str, float]:
        """Evaluate agent performance with greedy policy.

        Args:
            n_episodes: Number of evaluation episodes

        Returns:
            eval_metrics: Dictionary with evaluation metrics
        """
        self.logger.info(f"Evaluating for {n_episodes} episodes (greedy policy)")

        returns = []
        lengths = []
        successes = []

        for episode in range(n_episodes):
            episode_info = self.run_episode(greedy=True)
            returns.append(episode_info['return'])
            lengths.append(episode_info['length'])
            successes.append(1.0 if episode_info['success'] else 0.0)

        eval_metrics = {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'success_rate': np.mean(successes),
        }

        self.logger.info(
            f"Evaluation: Return {eval_metrics['mean_return']:.3f} ± {eval_metrics['std_return']:.3f}, "
            f"Length {eval_metrics['mean_length']:.1f} ± {eval_metrics['std_length']:.1f}, "
            f"Success {eval_metrics['success_rate']:.2%}"
        )

        return eval_metrics

    def plot_training(self, output_path: Optional[Path] = None, window: int = 10):
        """Plot training curves.

        Args:
            output_path: Path to save plot (if None, show plot)
            window: Window size for moving average
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Convert metrics to numpy arrays
        returns = np.array(self.metrics['episode_returns'])
        lengths = np.array(self.metrics['episode_lengths'])
        losses = np.array(self.metrics['policy_losses'])
        success = np.array(self.metrics['success_rate'])

        # Compute moving averages
        def moving_average(x, w):
            if len(x) < w:
                return x
            return np.convolve(x, np.ones(w) / w, mode='valid')

        returns_ma = moving_average(returns, window)
        lengths_ma = moving_average(lengths, window)
        success_ma = moving_average(success, window)

        # Plot 1: Episode returns
        axes[0, 0].plot(returns, alpha=0.3, label='Raw')
        if len(returns_ma) > 0:
            axes[0, 0].plot(range(window - 1, len(returns)), returns_ma, label=f'{window}-ep MA')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Episode Return')
        axes[0, 0].set_title('Training Returns')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Episode lengths
        axes[0, 1].plot(lengths, alpha=0.3, label='Raw')
        if len(lengths_ma) > 0:
            axes[0, 1].plot(range(window - 1, len(lengths)), lengths_ma, label=f'{window}-ep MA')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Episode Length')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Policy loss
        axes[1, 0].plot(losses)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Policy Loss')
        axes[1, 0].set_title('Policy Gradient Loss')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Success rate
        axes[1, 1].plot(success, alpha=0.3, label='Raw')
        if len(success_ma) > 0:
            axes[1, 1].plot(range(window - 1, len(success)), success_ma, label=f'{window}-ep MA')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].set_title('Goal Achievement Rate')
        axes[1, 1].set_ylim([-0.05, 1.05])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path is not None:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved training plot to {output_path}")
        else:
            plt.show()

        plt.close(fig)
