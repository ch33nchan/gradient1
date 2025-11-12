"""
Self-Gradient Agent for Bandits.

Agent that predicts how actions will change its parameters,
then plans to maximize future learning.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional

from ..models.gradient_world_model import (
    GradientPredictor,
    MetaValueNetwork,
    ExperienceBuffer
)
from ..models.bandit_models import SoftmaxBanditPolicy


class SelfGradientBanditAgent:
    """
    Bandit agent that models its own gradient dynamics.

    Core innovation: learns g(theta, a) -> gradient, then uses this
    to choose actions that lead to better future parameters.
    """

    def __init__(
        self,
        n_arms: int,
        learning_rate: float = 0.01,
        meta_learning_rate: float = 0.001,
        gradient_clip: float = 1.0,
        buffer_size: int = 2000,
        batch_size: int = 32,
        exploration_episodes: int = 100,
        hidden_dim: int = 64
    ):
        """
        Initialize self-gradient agent.

        Args:
            n_arms: Number of bandit arms
            learning_rate: Learning rate for policy updates
            meta_learning_rate: Learning rate for gradient predictor
            gradient_clip: Gradient clipping threshold
            buffer_size: Size of experience buffer
            batch_size: Batch size for training
            exploration_episodes: Number of episodes to explore randomly
            hidden_dim: Hidden dimension for networks
        """
        self.n_arms = n_arms
        self.learning_rate = learning_rate
        self.meta_learning_rate = meta_learning_rate
        self.gradient_clip = gradient_clip
        self.batch_size = batch_size
        self.exploration_episodes = exploration_episodes

        # Policy (softmax bandit)
        self.policy = SoftmaxBanditPolicy(n_arms)

        # Gradient predictor
        self.gradient_predictor = GradientPredictor(n_arms, hidden_dim)

        # Meta-value network
        self.meta_value = MetaValueNetwork(n_arms, hidden_dim // 2)

        # Optimizers
        self.gp_optimizer = torch.optim.AdamW(
            self.gradient_predictor.parameters(),
            lr=meta_learning_rate,
            weight_decay=1e-4
        )

        self.mv_optimizer = torch.optim.AdamW(
            self.meta_value.parameters(),
            lr=meta_learning_rate,
            weight_decay=1e-4
        )

        # Experience buffer
        self.buffer = ExperienceBuffer(buffer_size, n_arms)

        # Tracking
        self.episode_count = 0
        self.gradient_errors = []
        self.meta_value_losses = []

    def act_exploratory(self) -> int:
        """Random action for initial exploration."""
        return np.random.randint(self.n_arms)

    def act_greedy(self) -> int:
        """Act greedily according to current policy."""
        return self.policy.sample_action()

    def act_planned(self) -> int:
        """
        Choose action by planning over future parameter updates.

        This is the core innovation:
        1. For each possible action
        2. Predict resulting gradient
        3. Simulate parameter update
        4. Evaluate future parameters
        5. Choose action leading to best future state
        """
        best_action = 0
        best_value = -float('inf')

        current_theta = self.policy.get_parameters()

        # Evaluate each possible action
        for action in range(self.n_arms):
            with torch.no_grad():
                # Predict gradient from this action
                predicted_gradient = self.gradient_predictor(
                    current_theta,
                    torch.tensor(action)
                )

                # Simulate parameter update
                future_theta = current_theta - self.learning_rate * predicted_gradient

                # Evaluate future parameters
                future_value = self.meta_value(future_theta).item()

                # Add exploration bonus (gradient magnitude = learning potential)
                exploration_bonus = 0.1 * predicted_gradient.norm().item()
                total_value = future_value + exploration_bonus

            if total_value > best_value:
                best_value = total_value
                best_action = action

        return best_action

    def act(self, exploration_mode: str = 'auto') -> int:
        """
        Select action based on current strategy.

        Args:
            exploration_mode: 'random', 'greedy', 'planned', or 'auto'

        Returns:
            Action to take
        """
        if exploration_mode == 'auto':
            if self.episode_count < self.exploration_episodes:
                return self.act_exploratory()
            else:
                return self.act_planned()
        elif exploration_mode == 'random':
            return self.act_exploratory()
        elif exploration_mode == 'greedy':
            return self.act_greedy()
        elif exploration_mode == 'planned':
            return self.act_planned()
        else:
            raise ValueError(f"Unknown exploration mode: {exploration_mode}")

    def update(self, action: int, reward: float):
        """
        Update agent after observing reward.

        Args:
            action: Action taken
            reward: Reward received
        """
        # Get current parameters
        current_theta = self.policy.get_parameters()

        # Compute actual gradient
        actual_gradient = self.policy.compute_gradient(action, reward)

        # Store experience
        self.buffer.push(current_theta, action, actual_gradient, reward)

        # Update policy parameters
        new_theta = current_theta - self.learning_rate * actual_gradient

        # Clip parameters for stability
        new_theta = torch.clamp(new_theta, -5, 5)

        self.policy.set_parameters(new_theta)

        # Train networks if enough data
        if len(self.buffer) >= self.batch_size:
            self.train_gradient_predictor()
            self.train_meta_value()

        self.episode_count += 1

    def train_gradient_predictor(self) -> float:
        """
        Train gradient predictor on buffered experiences.

        Returns:
            Training loss
        """
        # Sample batch
        theta_batch, action_batch, gradient_batch, _ = self.buffer.sample(
            self.batch_size
        )

        # Predict gradients
        predicted_gradients = self.gradient_predictor(theta_batch, action_batch)

        # Compute loss (MSE)
        loss = F.mse_loss(predicted_gradients, gradient_batch)

        # Add regularization for gradient magnitudes
        magnitude_penalty = 0.01 * predicted_gradients.norm(dim=1).mean()
        total_loss = loss + magnitude_penalty

        # Optimize
        self.gp_optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.gradient_predictor.parameters(),
            self.gradient_clip
        )

        self.gp_optimizer.step()

        # Track error
        self.gradient_errors.append(loss.item())

        return loss.item()

    def train_meta_value(self) -> float:
        """
        Train meta-value network to recognize good parameters.

        Target is based on:
        - Recent rewards (high is good)
        - Policy entropy (lower is better - more decisive)

        Returns:
            Training loss
        """
        # Sample batch
        theta_batch, _, _, reward_batch = self.buffer.sample(self.batch_size)

        # Compute target values
        with torch.no_grad():
            # Criterion 1: Policy entropy (lower is better)
            policies = F.softmax(theta_batch, dim=1)
            entropies = -(policies * (policies + 1e-8).log()).sum(dim=1)

            # Criterion 2: Reward history (higher is better)
            normalized_rewards = (reward_batch - reward_batch.mean()) / (
                reward_batch.std() + 1e-8
            )

            # Combine criteria
            target_values = normalized_rewards - 0.5 * entropies

        # Predict values
        predicted_values = self.meta_value(theta_batch).squeeze()

        # Compute loss
        loss = F.mse_loss(predicted_values, target_values)

        # Optimize
        self.mv_optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.meta_value.parameters(),
            self.gradient_clip
        )

        self.mv_optimizer.step()

        # Track loss
        self.meta_value_losses.append(loss.item())

        return loss.item()

    def get_policy_probs(self) -> np.ndarray:
        """Get current policy probabilities."""
        return self.policy.get_policy().detach().numpy()

    def get_parameters(self) -> torch.Tensor:
        """Get current policy parameters."""
        return self.policy.get_parameters()

    def save_checkpoint(self, path: str):
        """
        Save agent checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'policy_theta': self.policy.theta,
            'gradient_predictor_state': self.gradient_predictor.state_dict(),
            'meta_value_state': self.meta_value.state_dict(),
            'gp_optimizer_state': self.gp_optimizer.state_dict(),
            'mv_optimizer_state': self.mv_optimizer.state_dict(),
            'episode_count': self.episode_count,
            'gradient_errors': self.gradient_errors,
            'meta_value_losses': self.meta_value_losses
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """
        Load agent checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location='cpu')

        self.policy.theta = checkpoint['policy_theta']
        self.gradient_predictor.load_state_dict(checkpoint['gradient_predictor_state'])
        self.meta_value.load_state_dict(checkpoint['meta_value_state'])
        self.gp_optimizer.load_state_dict(checkpoint['gp_optimizer_state'])
        self.mv_optimizer.load_state_dict(checkpoint['mv_optimizer_state'])
        self.episode_count = checkpoint['episode_count']
        self.gradient_errors = checkpoint['gradient_errors']
        self.meta_value_losses = checkpoint['meta_value_losses']
