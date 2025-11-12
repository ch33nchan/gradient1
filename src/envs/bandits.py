"""
Multi-armed bandit environments.
Includes both stationary and non-stationary variants.
"""

import numpy as np
from typing import Optional


class BanditEnvironment:
    """
    Multi-armed bandit with Gaussian rewards.

    Each arm has a true mean reward, and pulling an arm samples
    from a Gaussian distribution centered at that mean.
    """

    def __init__(
        self,
        n_arms: int,
        seed: Optional[int] = None,
        non_stationary: bool = False,
        drift_rate: float = 0.1
    ):
        """
        Initialize bandit environment.

        Args:
            n_arms: Number of arms
            seed: Random seed for reproducibility
            non_stationary: Whether arm means drift over time
            drift_rate: Standard deviation of mean drift (if non-stationary)
        """
        self.n_arms = n_arms
        self.non_stationary = non_stationary
        self.drift_rate = drift_rate

        self.rng = np.random.RandomState(seed)

        # Initialize arm means
        self.arm_means = self._initialize_arm_means()

        # Tracking
        self.total_pulls = 0
        self.arm_pulls = np.zeros(n_arms, dtype=int)

    def _initialize_arm_means(self) -> np.ndarray:
        """
        Initialize arm means with diverse rewards.

        Creates a mix of good, bad, and mediocre arms to make
        the exploration problem non-trivial.
        """
        means = self.rng.randn(self.n_arms) * 0.5

        # Make some arms clearly better
        n_good_arms = max(1, self.n_arms // 5)
        good_arms = self.rng.choice(self.n_arms, n_good_arms, replace=False)
        means[good_arms] += self.rng.uniform(1, 2, size=n_good_arms)

        # Make some arms clearly worse
        n_bad_arms = max(1, self.n_arms // 5)
        bad_arms_candidates = [i for i in range(self.n_arms) if i not in good_arms]
        bad_arms = self.rng.choice(bad_arms_candidates, n_bad_arms, replace=False)
        means[bad_arms] -= self.rng.uniform(1, 2, size=n_bad_arms)

        return means

    def reset_arms(self) -> None:
        """Reset arm means (for non-stationary scenarios)."""
        self.arm_means = self._initialize_arm_means()
        self.total_pulls = 0
        self.arm_pulls = np.zeros(self.n_arms, dtype=int)

    def pull(self, action: int) -> float:
        """
        Pull an arm and receive reward.

        Args:
            action: Index of arm to pull

        Returns:
            Sampled reward
        """
        if not 0 <= action < self.n_arms:
            raise ValueError(f"Invalid action {action}, must be in [0, {self.n_arms})")

        self.total_pulls += 1
        self.arm_pulls[action] += 1

        # Non-stationary: drift means over time
        if self.non_stationary and self.total_pulls % 500 == 0:
            self.arm_means += self.rng.randn(self.n_arms) * self.drift_rate

        # Sample reward from Gaussian
        reward = self.rng.normal(self.arm_means[action], 1.0)

        return reward

    def get_optimal_action(self) -> int:
        """Get the index of the arm with highest true mean."""
        return int(np.argmax(self.arm_means))

    def get_regret(self, action: int) -> float:
        """
        Compute instant regret for an action.

        Regret is the difference between the optimal reward
        and the expected reward of the chosen action.
        """
        optimal_mean = self.arm_means[self.get_optimal_action()]
        return optimal_mean - self.arm_means[action]

    def get_arm_means(self) -> np.ndarray:
        """Get current arm means (for debugging/analysis)."""
        return self.arm_means.copy()

    def get_statistics(self) -> dict:
        """Get environment statistics."""
        return {
            'total_pulls': self.total_pulls,
            'arm_pulls': self.arm_pulls.copy(),
            'arm_means': self.arm_means.copy(),
            'optimal_action': self.get_optimal_action(),
            'optimal_mean': self.arm_means[self.get_optimal_action()]
        }


class ContextualBandit(BanditEnvironment):
    """
    Contextual bandit where optimal action depends on context.

    Extension of standard bandit for future experiments.
    """

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        seed: Optional[int] = None
    ):
        """
        Initialize contextual bandit.

        Args:
            n_arms: Number of arms
            context_dim: Dimension of context vector
            seed: Random seed
        """
        super().__init__(n_arms, seed=seed)
        self.context_dim = context_dim

        # Each arm has a linear reward function of context
        self.arm_weights = self.rng.randn(n_arms, context_dim) * 0.5
        self.current_context = None

    def reset(self) -> np.ndarray:
        """
        Reset and sample new context.

        Returns:
            Context vector
        """
        self.current_context = self.rng.randn(self.context_dim)
        return self.current_context

    def pull(self, action: int) -> float:
        """
        Pull arm given current context.

        Args:
            action: Arm to pull

        Returns:
            Reward conditioned on context
        """
        if self.current_context is None:
            raise RuntimeError("Must call reset() before pulling arms")

        # Compute expected reward as linear function of context
        expected_reward = np.dot(self.arm_weights[action], self.current_context)

        # Add noise
        reward = self.rng.normal(expected_reward, 1.0)

        self.total_pulls += 1
        self.arm_pulls[action] += 1

        return reward

    def get_optimal_action(self) -> int:
        """Get optimal action for current context."""
        if self.current_context is None:
            raise RuntimeError("Must call reset() before getting optimal action")

        expected_rewards = np.dot(self.arm_weights, self.current_context)
        return int(np.argmax(expected_rewards))
