"""
Unit tests for self-gradient agent.
"""

import pytest
import torch
import numpy as np
from src.agents.self_gradient_agent import SelfGradientBanditAgent


def test_agent_initialization():
    """Test agent initialization."""
    agent = SelfGradientBanditAgent(
        n_arms=5,
        learning_rate=0.01,
        exploration_episodes=10
    )

    assert agent.n_arms == 5
    assert agent.learning_rate == 0.01
    assert agent.episode_count == 0


def test_agent_exploratory_action():
    """Test exploratory action selection."""
    agent = SelfGradientBanditAgent(n_arms=5)

    action = agent.act_exploratory()

    assert 0 <= action < 5
    assert isinstance(action, int)


def test_agent_greedy_action():
    """Test greedy action selection."""
    agent = SelfGradientBanditAgent(n_arms=5)

    action = agent.act_greedy()

    assert 0 <= action < 5
    assert isinstance(action, int)


def test_agent_planned_action():
    """Test planned action selection."""
    agent = SelfGradientBanditAgent(n_arms=5)

    # Need some buffer data for planning
    for i in range(50):
        agent.update(i % 5, np.random.randn())

    action = agent.act_planned()

    assert 0 <= action < 5
    assert isinstance(action, int)


def test_agent_update():
    """Test agent update."""
    agent = SelfGradientBanditAgent(n_arms=5)

    initial_params = agent.get_parameters().clone()

    # Update
    agent.update(action=2, reward=1.5)

    # Parameters should have changed
    updated_params = agent.get_parameters()
    assert not torch.allclose(initial_params, updated_params)

    # Episode count should increase
    assert agent.episode_count == 1


def test_agent_buffer_storage():
    """Test that experiences are stored in buffer."""
    agent = SelfGradientBanditAgent(n_arms=5, buffer_size=100)

    assert len(agent.buffer) == 0

    # Add some experiences
    for i in range(20):
        agent.update(i % 5, float(i))

    assert len(agent.buffer) == 20


def test_agent_gradient_predictor_training():
    """Test that gradient predictor is trained."""
    agent = SelfGradientBanditAgent(
        n_arms=5,
        batch_size=10,
        buffer_size=100
    )

    # Add enough experiences to trigger training
    for i in range(50):
        agent.update(i % 5, np.random.randn())

    # Should have gradient errors recorded
    assert len(agent.gradient_errors) > 0


def test_agent_checkpoint_save_load(tmp_path):
    """Test saving and loading checkpoints."""
    agent = SelfGradientBanditAgent(n_arms=5)

    # Do some updates
    for i in range(20):
        agent.update(i % 5, float(i))

    # Save checkpoint
    checkpoint_path = tmp_path / "checkpoint.pt"
    agent.save_checkpoint(str(checkpoint_path))

    # Create new agent and load
    new_agent = SelfGradientBanditAgent(n_arms=5)
    new_agent.load_checkpoint(str(checkpoint_path))

    # Check parameters match
    assert torch.allclose(
        agent.get_parameters(),
        new_agent.get_parameters()
    )

    # Check episode count matches
    assert agent.episode_count == new_agent.episode_count


def test_agent_auto_mode():
    """Test auto exploration mode."""
    agent = SelfGradientBanditAgent(
        n_arms=5,
        exploration_episodes=10
    )

    # First 10 episodes should explore
    for i in range(10):
        action = agent.act(exploration_mode='auto')
        agent.update(action, 1.0)

    # After exploration_episodes, should plan
    assert agent.episode_count >= agent.exploration_episodes


def test_agent_policy_probs():
    """Test getting policy probabilities."""
    agent = SelfGradientBanditAgent(n_arms=5)

    probs = agent.get_policy_probs()

    assert len(probs) == 5
    assert np.allclose(np.sum(probs), 1.0)
    assert np.all(probs >= 0)
    assert np.all(probs <= 1)
