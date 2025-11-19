"""Tests for Chain MDP environment."""

import pytest
import numpy as np
from src.envs.mdp_envs import ChainMDP


class TestChainMDP:
    """Test suite for Chain MDP environment."""

    def test_initialization(self):
        """Test environment initialization."""
        env = ChainMDP(n_states=10, seed=42)
        assert env.get_state_dim() == 10
        assert env.get_action_dim() == 2
        assert env.discount_factor == 0.99

    def test_reset(self):
        """Test reset returns initial state."""
        env = ChainMDP(n_states=10, seed=42)
        state = env.reset()
        assert state == 0  # Should start at leftmost state
        assert env.step_count == 0

    def test_step_right(self):
        """Test moving right."""
        env = ChainMDP(n_states=10, seed=42)
        env.reset()

        # Move right from state 0 to state 1
        next_state, reward, done, info = env.step(1)  # 1 = right
        assert next_state == 1
        assert reward == 0.0  # No reward except at goal
        assert not done
        assert info['step_count'] == 1

    def test_step_left(self):
        """Test moving left."""
        env = ChainMDP(n_states=10, seed=42)
        env.reset()

        # Try to move left from state 0 (should stay at 0)
        next_state, reward, done, info = env.step(0)  # 0 = left
        assert next_state == 0  # Can't go below 0
        assert reward == 0.0
        assert not done

        # Move right twice, then left
        env.step(1)  # state 1
        env.step(1)  # state 2
        next_state, reward, done, info = env.step(0)  # left
        assert next_state == 1

    def test_boundary_conditions(self):
        """Test boundary conditions (can't go below 0 or above n_states-1)."""
        env = ChainMDP(n_states=5, seed=42)
        env.reset()

        # Try to go left from initial state
        next_state, _, _, _ = env.step(0)
        assert next_state == 0  # Should stay at 0

        # Move to goal state (state 4)
        for _ in range(4):
            env.step(1)

        # Try to go right from goal state
        next_state, _, _, _ = env.step(1)
        assert next_state == 4  # Should stay at 4

    def test_goal_reward(self):
        """Test that reward is only given at goal."""
        env = ChainMDP(n_states=5, seed=42)
        env.reset()

        # Move to state 3 (one before goal)
        for _ in range(3):
            _, reward, _, _ = env.step(1)
            assert reward == 0.0

        # Move to goal (state 4)
        _, reward, done, info = env.step(1)
        assert reward == 1.0
        assert done
        assert info['at_goal']

    def test_episode_termination_at_goal(self):
        """Test episode terminates when reaching goal."""
        env = ChainMDP(n_states=10, seed=42)
        env.reset()

        # Move to goal
        for i in range(8):
            _, _, done, _ = env.step(1)
            assert not done  # Not done until goal

        # Final step to goal
        _, reward, done, info = env.step(1)
        assert done
        assert reward == 1.0
        assert info['at_goal']

    def test_episode_termination_max_steps(self):
        """Test episode terminates at max steps."""
        env = ChainMDP(n_states=10, max_steps=5, seed=42)
        env.reset()

        # Take 5 steps (any actions)
        for i in range(4):
            _, _, done, _ = env.step(1)
            assert not done

        # 5th step should terminate
        _, _, done, info = env.step(1)
        assert done
        assert info['step_count'] == 5

    def test_optimal_policy(self):
        """Test that always-right policy reaches goal optimally."""
        env = ChainMDP(n_states=10, seed=42)
        state = env.reset()

        total_reward = 0
        discount = 1.0

        for step in range(9):  # Should reach goal in 9 steps
            next_state, reward, done, _ = env.step(1)  # Always right
            total_reward += discount * reward
            discount *= env.discount_factor

            if step < 8:
                assert not done
            else:
                assert done  # Should be done on 9th step

        # Check return matches expected
        expected_return = env.get_optimal_return()
        assert abs(total_reward - expected_return) < 1e-6

    def test_random_policy_eventually_reaches_goal(self):
        """Test that random policy can reach goal (given enough steps)."""
        env = ChainMDP(n_states=5, max_steps=100, seed=42)
        env.reset()

        done = False
        reached_goal = False

        for _ in range(100):
            action = env.rng.choice([0, 1])  # Random action
            _, reward, done, info = env.step(action)
            if info['at_goal']:
                reached_goal = True
                break

        assert reached_goal or done  # Either reached goal or hit max steps

    def test_discount_factor(self):
        """Test discount factor is correctly returned."""
        env = ChainMDP(n_states=10, discount_factor=0.95, seed=42)
        assert env.get_discount_factor() == 0.95

    def test_optimal_return_calculation(self):
        """Test optimal return calculation."""
        # For n_states=10, optimal policy takes 9 steps
        # Reward at timestep 8 (0-indexed), so Return = gamma^8 * 1.0
        env = ChainMDP(n_states=10, discount_factor=0.99, seed=42)
        expected = 0.99 ** 8  # gamma^(n_states - 2)
        assert abs(env.get_optimal_return() - expected) < 1e-6

        # For n_states=5, optimal policy takes 4 steps
        # Reward at timestep 3 (0-indexed), so Return = gamma^3 * 1.0
        env = ChainMDP(n_states=5, discount_factor=0.95, seed=42)
        expected = 0.95 ** 3  # gamma^(n_states - 2)
        assert abs(env.get_optimal_return() - expected) < 1e-6

    def test_invalid_action(self):
        """Test that invalid actions raise ValueError."""
        env = ChainMDP(n_states=10, seed=42)
        env.reset()

        with pytest.raises(ValueError):
            env.step(2)  # Only 0 and 1 are valid

        with pytest.raises(ValueError):
            env.step(-1)

    def test_reproducibility(self):
        """Test that same seed produces same behavior."""
        env1 = ChainMDP(n_states=10, seed=42)
        env2 = ChainMDP(n_states=10, seed=42)

        state1 = env1.reset()
        state2 = env2.reset()
        assert state1 == state2

        # Random actions should be the same
        for _ in range(10):
            action1 = env1.rng.choice([0, 1])
            action2 = env2.rng.choice([0, 1])
            assert action1 == action2

    def test_state_tracking(self):
        """Test that current_state is correctly tracked."""
        env = ChainMDP(n_states=10, seed=42)
        env.reset()

        assert env.current_state == 0

        env.step(1)
        assert env.current_state == 1

        env.step(1)
        assert env.current_state == 2

        env.step(0)
        assert env.current_state == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
