"""MDP environments for testing meta-value learning with multi-step returns.

This module implements discrete MDP environments where:
- States are discrete (integers)
- Actions are discrete (integers)
- Episodes have multiple steps
- Returns are multi-step (sum of discounted rewards)

This provides better conditions for meta-value learning compared to bandits:
- Multi-step returns have lower variance than single rewards
- Value functions are structurally necessary
- Credit assignment over multiple steps
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import numpy as np


class MDPEnvironment(ABC):
    """Abstract base class for discrete MDP environments."""

    @abstractmethod
    def reset(self) -> int:
        """Reset environment to initial state.

        Returns:
            initial_state: Integer state index
        """
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """Take action and observe next state, reward, done flag.

        Args:
            action: Integer action index

        Returns:
            next_state: Integer state index
            reward: Float reward
            done: Boolean episode termination flag
            info: Dictionary with additional information
        """
        pass

    @abstractmethod
    def get_state_dim(self) -> int:
        """Get number of states."""
        pass

    @abstractmethod
    def get_action_dim(self) -> int:
        """Get number of actions."""
        pass

    def get_discount_factor(self) -> float:
        """Get discount factor gamma for returns."""
        return 0.99


class ChainMDP(MDPEnvironment):
    """Linear chain of states with goal at the end.

    States: 0, 1, 2, ..., n_states-1
    Actions: 0 (left), 1 (right)

    Agent starts at state 0 (leftmost).
    Goal is at state n_states-1 (rightmost).
    Reward is 0 everywhere except +1 at goal.

    Optimal policy: Always go right.
    Optimal expected return: gamma^(n_states-1) * 1.0

    This is the simplest non-trivial MDP for testing:
    - Clear optimal policy (constant action)
    - Multi-step credit assignment (n_states steps to reward)
    - No spatial complexity (1D)
    - Easy to analyze and debug
    """

    def __init__(
        self,
        n_states: int = 10,
        max_steps: int = 100,
        discount_factor: float = 0.99,
        seed: Optional[int] = None,
    ):
        """Initialize Chain MDP.

        Args:
            n_states: Number of states in the chain (default: 10)
            max_steps: Maximum steps per episode (default: 100)
            discount_factor: Discount factor gamma (default: 0.99)
            seed: Random seed for reproducibility
        """
        self.n_states = n_states
        self.n_actions = 2  # 0=left, 1=right
        self.max_steps = max_steps
        self.discount_factor = discount_factor

        # Environment state
        self.current_state = 0
        self.step_count = 0

        # Positions
        self.initial_state = 0
        self.goal_state = n_states - 1

        # Random number generator
        self.rng = np.random.RandomState(seed)

    def reset(self) -> int:
        """Reset to initial state (leftmost)."""
        self.current_state = self.initial_state
        self.step_count = 0
        return self.current_state

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """Take action (0=left, 1=right) and transition.

        Args:
            action: 0 for left, 1 for right

        Returns:
            next_state: New state after action
            reward: 0.0 everywhere except 1.0 at goal
            done: True if reached goal or max steps
            info: Dictionary with additional information
        """
        self.step_count += 1

        # Apply action (deterministic transitions)
        if action == 0:  # Left
            next_state = max(0, self.current_state - 1)
        elif action == 1:  # Right
            next_state = min(self.n_states - 1, self.current_state + 1)
        else:
            raise ValueError(f"Invalid action {action}. Must be 0 (left) or 1 (right).")

        # Compute reward
        reward = 1.0 if next_state == self.goal_state else 0.0

        # Check termination
        done = (next_state == self.goal_state) or (self.step_count >= self.max_steps)

        # Update state
        self.current_state = next_state

        # Info dict
        info = {
            'step_count': self.step_count,
            'at_goal': next_state == self.goal_state,
        }

        return next_state, reward, done, info

    def get_state_dim(self) -> int:
        """Get number of states."""
        return self.n_states

    def get_action_dim(self) -> int:
        """Get number of actions."""
        return self.n_actions

    def get_discount_factor(self) -> float:
        """Get discount factor."""
        return self.discount_factor

    def get_optimal_return(self) -> float:
        """Get expected return of optimal policy (always right).

        Optimal policy reaches goal in (n_states - 1) steps.
        The reward is received at timestep (n_states - 2) (0-indexed).
        Return = gamma^(n_states - 2) * 1.0

        Example: n_states=10, goal at state 9
        - Start at state 0
        - Step 1 (t=0): 0→1, reward=0, discount=γ^0
        - Step 2 (t=1): 1→2, reward=0, discount=γ^1
        - ...
        - Step 9 (t=8): 8→9, reward=1, discount=γ^8
        - Return = γ^8 * 1.0 = γ^(10-2)
        """
        return self.discount_factor ** (self.n_states - 2)
