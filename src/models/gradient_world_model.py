"""
Gradient World Models - Core innovation for predicting parameter gradients.

The GradientWorldModel learns to predict how experiences will change
an agent's parameters, enabling planning over learning dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GradientPredictor(nn.Module):
    """
    Neural network that predicts parameter gradients.

    For bandits: predicts gradient from (parameters, action) pairs.
    For RL: extended to handle full trajectories.
    """

    def __init__(
        self,
        n_arms: int,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        """
        Initialize gradient predictor for bandits.

        Args:
            n_arms: Number of arms (parameter dimension)
            hidden_dim: Hidden layer size
            dropout: Dropout probability
        """
        super().__init__()
        self.n_arms = n_arms

        # Input: [parameters, one-hot action]
        input_dim = n_arms * 2

        # CPU-optimized architecture
        # Note: No BatchNorm to support single-sample inputs
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, n_arms)  # Output: predicted gradient
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization for stable gradient flow."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(
        self,
        theta: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict gradient given parameters and action.

        Args:
            theta: Parameters [batch_size, n_arms] or [n_arms]
            action: Action indices [batch_size] or scalar

        Returns:
            Predicted gradient [batch_size, n_arms] or [n_arms]
        """
        # Handle both batched and single inputs
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
            if not torch.is_tensor(action):
                action = torch.tensor([action])
            else:
                action = action.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Create one-hot encoding for actions
        action_one_hot = F.one_hot(action.long(), num_classes=self.n_arms).float()

        # Concatenate inputs
        inputs = torch.cat([theta, action_one_hot], dim=1)

        # Predict gradient
        predicted_gradient = self.network(inputs)

        if squeeze_output:
            predicted_gradient = predicted_gradient.squeeze(0)

        return predicted_gradient


class MetaValueNetwork(nn.Module):
    """
    Evaluates the quality of parameter vectors.

    Learns what parameter configurations lead to good future performance.
    Used for planning: evaluate different future parameter states.
    """

    def __init__(
        self,
        param_dim: int,
        hidden_dim: int = 32
    ):
        """
        Initialize meta-value network.

        Args:
            param_dim: Dimension of parameter vector
            hidden_dim: Hidden layer size
        """
        super().__init__()

        # Note: No BatchNorm to support single-sample inputs
        self.network = nn.Sequential(
            nn.Linear(param_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Evaluate parameter quality.

        Args:
            theta: Parameters [batch_size, param_dim] or [param_dim]

        Returns:
            Value estimate [batch_size] or scalar
            - Single input (1D): returns 0D tensor (scalar)
            - Batched input (2D): returns 1D tensor (batch_size,)
        """
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        value = self.network(theta)  # Shape: (batch_size, 1)
        value = value.squeeze(-1)     # Shape: (batch_size,)

        if squeeze_output:
            value = value.squeeze(0)  # Shape: () - scalar

        return value


class ExperienceBuffer:
    """
    Efficient storage for gradient learning experiences.
    CPU-optimized with numpy backend.
    """

    def __init__(self, capacity: int, param_dim: int):
        """
        Initialize experience buffer.

        Args:
            capacity: Maximum number of experiences to store
            param_dim: Dimension of parameter vectors
        """
        self.capacity = capacity
        self.param_dim = param_dim

        # Pre-allocated arrays for efficiency
        self.theta_buffer = torch.zeros((capacity, param_dim))
        self.action_buffer = torch.zeros(capacity, dtype=torch.long)
        self.gradient_buffer = torch.zeros((capacity, param_dim))
        self.reward_buffer = torch.zeros(capacity)

        self.position = 0
        self.size = 0

    def push(
        self,
        theta: torch.Tensor,
        action: int,
        gradient: torch.Tensor,
        reward: float
    ):
        """Add experience to buffer."""
        self.theta_buffer[self.position] = theta.detach()
        self.action_buffer[self.position] = action
        self.gradient_buffer[self.position] = gradient.detach()
        self.reward_buffer[self.position] = reward

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """
        Sample batch of experiences.

        Returns:
            Tuple of (theta, actions, gradients, rewards)
        """
        if self.size < batch_size:
            indices = torch.arange(self.size)
        else:
            indices = torch.randperm(self.size)[:batch_size]

        return (
            self.theta_buffer[indices],
            self.action_buffer[indices],
            self.gradient_buffer[indices],
            self.reward_buffer[indices]
        )

    def __len__(self):
        return self.size

    def clear(self):
        """Clear the buffer."""
        self.position = 0
        self.size = 0
