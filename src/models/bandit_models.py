"""
Models and baseline agents for bandit experiments.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


class SoftmaxBanditPolicy:
    """
    Softmax policy for bandits parameterized by theta.

    Policy: pi(a|theta) = exp(theta_a) / sum_i exp(theta_i)
    """

    def __init__(self, n_arms: int):
        """
        Initialize softmax policy.

        Args:
            n_arms: Number of arms
        """
        self.n_arms = n_arms
        self.theta = torch.zeros(n_arms, requires_grad=False)

    def get_policy(self, theta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute softmax policy from parameters.

        Args:
            theta: Parameters (if None, uses self.theta)

        Returns:
            Action probabilities
        """
        if theta is None:
            theta = self.theta
        return F.softmax(theta, dim=-1)

    def sample_action(self, theta: Optional[torch.Tensor] = None) -> int:
        """
        Sample action from policy.

        Args:
            theta: Parameters (if None, uses self.theta)

        Returns:
            Sampled action
        """
        probs = self.get_policy(theta)
        action = torch.multinomial(probs, 1).item()
        return action

    def get_parameters(self) -> torch.Tensor:
        """Get current parameters."""
        return self.theta.clone()

    def set_parameters(self, theta: torch.Tensor):
        """Set parameters."""
        self.theta = theta.clone()

    def compute_gradient(self, action: int, reward: float) -> torch.Tensor:
        """
        Compute REINFORCE gradient.

        Gradient: ∇L = -r * (e_a - pi(·|theta))

        Args:
            action: Action taken
            reward: Reward received

        Returns:
            Gradient vector
        """
        policy = self.get_policy()

        # One-hot encoding of action
        one_hot = F.one_hot(torch.tensor(action), self.n_arms).float()

        # REINFORCE gradient
        gradient = -reward * (one_hot - policy)

        return gradient


class BaselineAgent:
    """
    Standard bandit algorithms for comparison:
    - Epsilon-greedy
    - UCB (Upper Confidence Bound)
    - Thompson Sampling
    """

    def __init__(
        self,
        n_arms: int,
        algorithm: str = 'epsilon_greedy',
        epsilon: float = 0.1,
        ucb_c: float = 2.0
    ):
        """
        Initialize baseline agent.

        Args:
            n_arms: Number of arms
            algorithm: Algorithm to use ('epsilon_greedy', 'ucb', 'thompson')
            epsilon: Exploration rate for epsilon-greedy
            ucb_c: Exploration constant for UCB
        """
        self.n_arms = n_arms
        self.algorithm = algorithm
        self.epsilon = epsilon
        self.ucb_c = ucb_c

        # Track estimates and counts
        self.q_values = np.zeros(n_arms)
        self.action_counts = np.zeros(n_arms)
        self.total_count = 0

        # For Thompson Sampling (Beta distribution)
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def act(self) -> int:
        """
        Select action based on algorithm.

        Returns:
            Action to take
        """
        if self.algorithm == 'epsilon_greedy':
            if np.random.random() < self.epsilon:
                return np.random.randint(self.n_arms)
            else:
                return int(np.argmax(self.q_values))

        elif self.algorithm == 'ucb':
            if self.total_count < self.n_arms:
                # Ensure all arms pulled once
                return self.total_count

            # UCB formula
            ucb_values = self.q_values + self.ucb_c * np.sqrt(
                np.log(self.total_count) / (self.action_counts + 1e-8)
            )
            return int(np.argmax(ucb_values))

        elif self.algorithm == 'thompson':
            # Sample from Beta distributions
            samples = np.random.beta(self.alpha, self.beta)
            return int(np.argmax(samples))

        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def update(self, action: int, reward: float):
        """
        Update estimates based on reward.

        Args:
            action: Action taken
            reward: Reward received
        """
        self.total_count += 1
        self.action_counts[action] += 1

        # Update Q-values (running average)
        n = self.action_counts[action]
        self.q_values[action] += (reward - self.q_values[action]) / n

        # Update Thompson Sampling parameters
        if self.algorithm == 'thompson':
            # Convert reward to [0,1] for Beta distribution
            # Assuming rewards in approximately [-3, 3]
            normalized_reward = (reward + 3) / 6
            normalized_reward = np.clip(normalized_reward, 0, 1)

            if normalized_reward > 0.5:
                self.alpha[action] += 1
            else:
                self.beta[action] += 1

    def reset(self):
        """Reset agent statistics."""
        self.q_values = np.zeros(self.n_arms)
        self.action_counts = np.zeros(self.n_arms)
        self.total_count = 0
        self.alpha = np.ones(self.n_arms)
        self.beta = np.ones(self.n_arms)

    def get_statistics(self) -> dict:
        """Get agent statistics."""
        return {
            'q_values': self.q_values.copy(),
            'action_counts': self.action_counts.copy(),
            'total_count': self.total_count,
            'best_arm': int(np.argmax(self.q_values))
        }
