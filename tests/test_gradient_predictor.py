"""
Unit tests for gradient prediction models.
"""

import pytest
import torch
from src.models.gradient_world_model import (
    GradientPredictor,
    MetaValueNetwork,
    ExperienceBuffer
)


def test_gradient_predictor_initialization():
    """Test gradient predictor initialization."""
    model = GradientPredictor(n_arms=5, hidden_dim=32)

    assert model.n_arms == 5
    assert isinstance(model, torch.nn.Module)


def test_gradient_predictor_forward_single():
    """Test forward pass with single input."""
    model = GradientPredictor(n_arms=5, hidden_dim=32)

    theta = torch.randn(5)
    action = torch.tensor(2)

    output = model(theta, action)

    assert output.shape == (5,)
    assert output.requires_grad


def test_gradient_predictor_forward_batch():
    """Test forward pass with batch input."""
    model = GradientPredictor(n_arms=5, hidden_dim=32)

    theta = torch.randn(10, 5)
    action = torch.randint(0, 5, (10,))

    output = model(theta, action)

    assert output.shape == (10, 5)
    assert output.requires_grad


def test_meta_value_initialization():
    """Test meta-value network initialization."""
    model = MetaValueNetwork(param_dim=10, hidden_dim=16)

    assert isinstance(model, torch.nn.Module)


def test_meta_value_forward_single():
    """Test meta-value forward pass with single input."""
    model = MetaValueNetwork(param_dim=10, hidden_dim=16)

    theta = torch.randn(10)
    value = model(theta)

    assert value.shape == ()
    assert value.requires_grad


def test_meta_value_forward_batch():
    """Test meta-value forward pass with batch input."""
    model = MetaValueNetwork(param_dim=10, hidden_dim=16)

    theta = torch.randn(8, 10)
    value = model(theta)

    assert value.shape == (8,)
    assert value.requires_grad


def test_experience_buffer_initialization():
    """Test experience buffer initialization."""
    buffer = ExperienceBuffer(capacity=100, param_dim=5)

    assert len(buffer) == 0
    assert buffer.capacity == 100
    assert buffer.param_dim == 5


def test_experience_buffer_push():
    """Test adding experiences to buffer."""
    buffer = ExperienceBuffer(capacity=10, param_dim=5)

    theta = torch.randn(5)
    action = 2
    gradient = torch.randn(5)
    reward = 1.5

    buffer.push(theta, action, gradient, reward)

    assert len(buffer) == 1


def test_experience_buffer_sample():
    """Test sampling from buffer."""
    buffer = ExperienceBuffer(capacity=100, param_dim=5)

    # Add some experiences
    for i in range(20):
        theta = torch.randn(5)
        action = i % 5
        gradient = torch.randn(5)
        reward = float(i)

        buffer.push(theta, action, gradient, reward)

    assert len(buffer) == 20

    # Sample batch
    theta_batch, action_batch, gradient_batch, reward_batch = buffer.sample(10)

    assert theta_batch.shape == (10, 5)
    assert action_batch.shape == (10,)
    assert gradient_batch.shape == (10, 5)
    assert reward_batch.shape == (10,)


def test_experience_buffer_overflow():
    """Test buffer overflow handling."""
    buffer = ExperienceBuffer(capacity=5, param_dim=3)

    # Add more than capacity
    for i in range(10):
        theta = torch.randn(3)
        buffer.push(theta, 0, torch.zeros(3), 0.0)

    # Should only keep last 5
    assert len(buffer) == 5


def test_experience_buffer_clear():
    """Test clearing buffer."""
    buffer = ExperienceBuffer(capacity=10, param_dim=5)

    # Add experiences
    for i in range(5):
        buffer.push(torch.randn(5), 0, torch.zeros(5), 0.0)

    assert len(buffer) == 5

    buffer.clear()

    assert len(buffer) == 0
