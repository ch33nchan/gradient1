"""
Self-Gradient Agent for Bandits.

Agent that predicts how actions will change its parameters,
then plans to maximize future performance.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
import logging

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
        hidden_dim: int = 64,
        enable_planning: bool = True,
        planning_exploration_bonus: float = 0.0,
        gradient_step_scale: float = 1.0
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
            enable_planning: Whether to use gradient-based planning
            planning_exploration_bonus: Coefficient for gradient magnitude bonus
            gradient_step_scale: Scale factor for predicted gradient step
        """
        self.n_arms = n_arms
        self.learning_rate = learning_rate
        self.meta_learning_rate = meta_learning_rate
        self.gradient_clip = gradient_clip
        self.batch_size = batch_size
        self.exploration_episodes = exploration_episodes
        self.enable_planning = enable_planning
        self.planning_exploration_bonus = planning_exploration_bonus
        self.gradient_step_scale = gradient_step_scale

        self.logger = logging.getLogger('self_gradient_agent')

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

    def act_ucb_style(self) -> int:
        """
        UCB-style exploration as baseline when planning is disabled.
        Uses policy parameters as Q-values.
        """
        # Use softmax parameters as estimates
        theta = self.policy.get_parameters().detach().numpy()

        # Add small exploration term
        # For simplicity, use a fixed exploration constant
        action = int(np.argmax(theta))
        return action

    def act_planned(self) -> int:
        """
        Choose action by planning over future parameter updates.

        For each action:
        1. Predict gradient from that action
        2. Simulate parameter update: theta' = theta + scale * predicted_gradient
        3. Evaluate quality of theta' using meta-value network
        4. Choose action with best predicted future value
        """
        best_action = 0
        best_value = -float('inf')

        current_theta = self.policy.get_parameters()

        # Set models to eval mode for inference
        self.gradient_predictor.eval()
        self.meta_value.eval()

        # Evaluate each possible action
        for action in range(self.n_arms):
            with torch.no_grad():
                # Predict gradient from this action
                predicted_gradient = self.gradient_predictor(
                    current_theta,
                    torch.tensor(action)
                )

                # Simulate parameter update
                # Note: gradient descent is theta - lr * grad
                # Scale controls how far we look ahead
                future_theta = current_theta - (
                    self.learning_rate * self.gradient_step_scale * predicted_gradient
                )

                # Evaluate future parameters
                future_value = self.meta_value(future_theta).item()

                # Optional: add exploration bonus based on predicted learning
                if self.planning_exploration_bonus > 0:
                    exploration_bonus = (
                        self.planning_exploration_bonus * predicted_gradient.norm().item()
                    )
                    total_value = future_value + exploration_bonus
                else:
                    total_value = future_value

            if total_value > best_value:
                best_value = total_value
                best_action = action

        # Set models back to train mode
        self.gradient_predictor.train()
        self.meta_value.train()

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
                # Random exploration phase
                return self.act_exploratory()
            elif self.enable_planning:
                # Gradient-based planning
                return self.act_planned()
            else:
                # Fallback to greedy (for ablations)
                return self.act_greedy()
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

        This is the ONLY place where the agent receives the environment reward.
        We verify the reward is used correctly for policy updates.

        Args:
            action: Action taken
            reward: Raw environment reward (NOT negated, NOT transformed)
        """
        # Get current parameters
        current_theta = self.policy.get_parameters()

        # Compute actual gradient using REINFORCE
        # This computes ∇L where L is the loss we minimize
        # For REINFORCE: ∇L = -reward * (one_hot - policy)
        actual_gradient = self.policy.compute_gradient(action, reward)

        # Store experience for gradient predictor training
        self.buffer.push(current_theta, action, actual_gradient, reward)

        # Update policy parameters via gradient descent
        # theta_new = theta_old - learning_rate * gradient
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

        Goal: Learn g(theta, action) -> gradient
        Loss: MSE between predicted and actual gradients

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

        # Optimize
        self.gp_optimizer.zero_grad()
        loss.backward()

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
        Train meta-value network to predict quality of parameter vectors.

        The meta-value should predict: "how good is this policy?"
        We use realized rewards as the training signal.

        Target: Recent average reward for parameters that led to good outcomes.

        Returns:
            Training loss
        """
        # Sample batch
        theta_batch, _, _, reward_batch = self.buffer.sample(self.batch_size)

        # Compute target values
        # Simple approach: normalize rewards to get value targets
        with torch.no_grad():
            # Use rewards as proxy for parameter quality
            # Parameters that led to high rewards are "good"
            target_values = (reward_batch - reward_batch.mean()) / (
                reward_batch.std() + 1e-8
            )

            # Optional: penalize high entropy (less decisive policies)
            # This encourages convergence
            policies = F.softmax(theta_batch, dim=1)
            entropies = -(policies * (policies + 1e-8).log()).sum(dim=1)

            # Lower entropy is better (more confident/converged policy)
            target_values = target_values - 0.3 * entropies

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
            'meta_value_losses': self.meta_value_losses,
            'config': {
                'n_arms': self.n_arms,
                'learning_rate': self.learning_rate,
                'enable_planning': self.enable_planning,
                'planning_exploration_bonus': self.planning_exploration_bonus,
                'gradient_step_scale': self.gradient_step_scale,
            }
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
