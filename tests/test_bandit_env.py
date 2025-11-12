"""
Unit tests for bandit environment.
"""

import pytest
import numpy as np
from src.envs.bandits import BanditEnvironment


def test_bandit_initialization():
    """Test basic initialization."""
    env = BanditEnvironment(n_arms=5, seed=42)

    assert env.n_arms == 5
    assert len(env.arm_means) == 5
    assert env.total_pulls == 0
    assert len(env.arm_pulls) == 5


def test_bandit_pull():
    """Test pulling arms."""
    env = BanditEnvironment(n_arms=5, seed=42)

    # Pull first arm
    reward = env.pull(0)
    assert isinstance(reward, float)
    assert env.total_pulls == 1
    assert env.arm_pulls[0] == 1

    # Pull again
    reward2 = env.pull(0)
    assert env.total_pulls == 2
    assert env.arm_pulls[0] == 2


def test_bandit_invalid_action():
    """Test invalid action raises error."""
    env = BanditEnvironment(n_arms=5, seed=42)

    with pytest.raises(ValueError):
        env.pull(5)

    with pytest.raises(ValueError):
        env.pull(-1)


def test_bandit_optimal_action():
    """Test optimal action computation."""
    env = BanditEnvironment(n_arms=5, seed=42)

    optimal = env.get_optimal_action()
    assert 0 <= optimal < 5

    # Check it's actually optimal
    optimal_mean = env.arm_means[optimal]
    for i in range(5):
        assert env.arm_means[i] <= optimal_mean


def test_bandit_regret():
    """Test regret computation."""
    env = BanditEnvironment(n_arms=5, seed=42)

    optimal = env.get_optimal_action()

    # Regret for optimal action should be 0
    regret = env.get_regret(optimal)
    assert abs(regret) < 1e-6

    # Regret for other actions should be positive
    for i in range(5):
        if i != optimal:
            regret = env.get_regret(i)
            assert regret >= 0


def test_bandit_reproducibility():
    """Test that same seed gives same results."""
    env1 = BanditEnvironment(n_arms=5, seed=42)
    env2 = BanditEnvironment(n_arms=5, seed=42)

    # Same arm means
    assert np.allclose(env1.arm_means, env2.arm_means)

    # Same rewards
    rewards1 = [env1.pull(0) for _ in range(10)]
    rewards2 = [env2.pull(0) for _ in range(10)]

    assert np.allclose(rewards1, rewards2)


def test_bandit_statistics():
    """Test statistics gathering."""
    env = BanditEnvironment(n_arms=5, seed=42)

    # Pull some arms
    for i in range(5):
        env.pull(i)

    stats = env.get_statistics()

    assert stats['total_pulls'] == 5
    assert sum(stats['arm_pulls']) == 5
    assert len(stats['arm_means']) == 5
    assert 'optimal_action' in stats
    assert 'optimal_mean' in stats
