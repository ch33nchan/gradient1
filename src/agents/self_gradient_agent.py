"""
Self-Gradient Agent for Bandits with Improved Planning.

Agent that predicts how actions will change its parameters,
then uses reward-based planning to select actions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple
import logging

from ..models.gradient_world_model import (
    GradientPredictor,
    MetaValueNetwork,
    ExperienceBuffer
)
from ..models.bandit_models import SoftmaxBanditPolicy


class SelfGradientBanditAgent:
    """
    Bandit agent with gradient prediction and gated reward-based planning.

    Key features:
    - Learns to predict parameter gradients
    - Gates planning based on gradient model quality
    - Optimizes for predicted future reward (not gradient magnitude)
    - Extensive instrumentation for debugging
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
        # Planning control
        enable_planning: bool = True,
        planning_warmup_episodes: int = 100,
        planning_grad_error_threshold: float = 0.3,
        planning_max_weight: float = 1.0,
        planning_ramp_episodes: int = 100,
        # Planning objective
        planning_objective: str = "reward",  # "reward", "update_norm", "reward_plus_update_norm"
        planning_beta_self_change: float = 0.0,
        gradient_step_scale: float = 1.0
    ):
        """
        Initialize self-gradient agent with gated planning.

        Args:
            n_arms: Number of bandit arms
            learning_rate: Learning rate for policy updates
            meta_learning_rate: Learning rate for gradient predictor
            gradient_clip: Gradient clipping threshold
            buffer_size: Size of experience buffer
            batch_size: Batch size for training
            exploration_episodes: Number of episodes to explore randomly
            hidden_dim: Hidden dimension for networks

            Planning control:
            enable_planning: Whether to use gradient-based planning
            planning_warmup_episodes: Episodes before planning can start
            planning_grad_error_threshold: Max gradient error to allow planning
            planning_max_weight: Maximum planning weight (0-1)
            planning_ramp_episodes: Episodes to ramp planning weight

            Planning objective:
            planning_objective: "reward", "update_norm", or "reward_plus_update_norm"
            planning_beta_self_change: Coefficient for update norm in mixed objective
            gradient_step_scale: Scale factor for predicted gradient step
        """
        self.n_arms = n_arms
        self.learning_rate = learning_rate
        self.meta_learning_rate = meta_learning_rate
        self.gradient_clip = gradient_clip
        self.batch_size = batch_size
        self.exploration_episodes = exploration_episodes

        # Planning configuration
        self.enable_planning = enable_planning
        self.planning_warmup_episodes = planning_warmup_episodes
        self.planning_grad_error_threshold = planning_grad_error_threshold
        self.planning_max_weight = planning_max_weight
        self.planning_ramp_episodes = planning_ramp_episodes
        self.planning_objective = planning_objective
        self.planning_beta_self_change = planning_beta_self_change
        self.gradient_step_scale = gradient_step_scale

        self.logger = logging.getLogger('self_gradient_agent')

        # Policy (softmax bandit)
        self.policy = SoftmaxBanditPolicy(n_arms)

        # Gradient predictor
        self.gradient_predictor = GradientPredictor(n_arms, hidden_dim)

        # Meta-value network (for reward prediction)
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

        # Planning tracking
        self.grad_error_ema = 1.0  # Start high
        self.grad_error_ema_alpha = 0.1
        self.current_planning_weight = 0.0
        self.planning_scores = np.zeros(n_arms)  # Last planning scores
        self.last_chosen_arm = 0
        self.planning_enabled_this_episode = False

    def act_exploratory(self) -> int:
        """Random action for initial exploration."""
        self.planning_enabled_this_episode = False
        return np.random.randint(self.n_arms)

    def act_greedy(self) -> int:
        """Act greedily according to current policy."""
        self.planning_enabled_this_episode = False
        return self.policy.sample_action()

    def _compute_planning_weight(self) -> float:
        """
        Compute current planning weight based on gating logic.

        Returns:
            Weight in [0, 1] for how much planning influences decisions
        """
        if not self.enable_planning:
            return 0.0

        # Gate 1: Warmup period
        if self.episode_count < self.planning_warmup_episodes:
            return 0.0

        # Gate 2: Gradient error threshold
        if self.grad_error_ema > self.planning_grad_error_threshold:
            return 0.0

        # Ramp up gradually
        episodes_past_warmup = self.episode_count - self.planning_warmup_episodes
        ramp_factor = min(1.0, episodes_past_warmup / max(1, self.planning_ramp_episodes))

        return ramp_factor * self.planning_max_weight

    def _predict_future_reward(self, future_theta: torch.Tensor) -> float:
        """
        Predict expected reward under future parameters using meta-value network.

        The meta-value network V(θ) is trained on actual returns, so it provides
        a better estimate of parameter quality than raw logits. This replaces the
        previous broken approach of treating logits as rewards.

        Args:
            future_theta: Predicted future parameters

        Returns:
            Predicted value (quality) of future parameters
        """
        with torch.no_grad():
            # Use meta-value network to predict parameter quality
            # This network is trained on actual returns, not logits
            predicted_value = self.meta_value(future_theta).item()

            return predicted_value

    def _compute_planning_scores(self) -> np.ndarray:
        """
        Compute planning score for each arm based on configured objective.

        Returns:
            Array of planning scores, one per arm
        """
        current_theta = self.policy.get_parameters()
        scores = np.zeros(self.n_arms)

        # Set models to eval mode
        self.gradient_predictor.eval()
        self.meta_value.eval()

        for action in range(self.n_arms):
            with torch.no_grad():
                # Predict gradient from this action
                predicted_gradient = self.gradient_predictor(
                    current_theta,
                    torch.tensor(action)
                )

                # Simulate parameter update
                future_theta = current_theta - (
                    self.learning_rate * self.gradient_step_scale * predicted_gradient
                )

                # Compute score based on objective
                if self.planning_objective == "reward":
                    # Pure reward prediction
                    scores[action] = self._predict_future_reward(future_theta)

                elif self.planning_objective == "update_norm":
                    # Pure gradient magnitude (old broken behavior)
                    scores[action] = predicted_gradient.norm().item()

                elif self.planning_objective == "reward_plus_update_norm":
                    # Hybrid objective
                    reward_score = self._predict_future_reward(future_theta)
                    update_score = predicted_gradient.norm().item()
                    scores[action] = reward_score + self.planning_beta_self_change * update_score

                else:
                    raise ValueError(f"Unknown planning objective: {self.planning_objective}")

        # Set models back to train mode
        self.gradient_predictor.train()
        self.meta_value.train()

        return scores

    def act_planned(self) -> int:
        """
        Choose action using gated planning.

        Combines greedy policy with planning scores based on planning_weight.
        When weight=0, pure greedy. When weight=1, pure planning.

        Returns:
            Chosen action
        """
        # Compute planning weight
        self.current_planning_weight = self._compute_planning_weight()

        # If no planning, fall back to greedy
        if self.current_planning_weight == 0.0:
            self.planning_enabled_this_episode = False
            return self.act_greedy()

        self.planning_enabled_this_episode = True

        # Compute planning scores
        self.planning_scores = self._compute_planning_scores()

        # Get greedy policy probabilities
        greedy_probs = self.policy.get_policy().detach().numpy()

        # Normalize planning scores to probabilities
        # Softmax with temperature
        planning_scores_tensor = torch.tensor(self.planning_scores, dtype=torch.float32)
        planning_probs = F.softmax(planning_scores_tensor / 0.1, dim=0).numpy()

        # Blend: (1 - w) * greedy + w * planning
        blended_probs = (
            (1 - self.current_planning_weight) * greedy_probs +
            self.current_planning_weight * planning_probs
        )

        # Sample from blended distribution
        action = np.random.choice(self.n_arms, p=blended_probs)
        self.last_chosen_arm = action

        return action

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
            else:
                # Potentially use planning (gated)
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

        This is the ONLY place where the agent receives the environment reward.

        Args:
            action: Action taken
            reward: Raw environment reward
        """
        # Get current parameters
        current_theta = self.policy.get_parameters()

        # Compute actual gradient using REINFORCE
        actual_gradient = self.policy.compute_gradient(action, reward)

        # Store experience for gradient predictor training
        self.buffer.push(current_theta, action, actual_gradient, reward)

        # Update policy parameters via gradient descent
        new_theta = current_theta - self.learning_rate * actual_gradient

        # Clip parameters for stability
        new_theta = torch.clamp(new_theta, -5, 5)

        self.policy.set_parameters(new_theta)

        # Train networks if enough data
        if len(self.buffer) >= self.batch_size:
            grad_error = self.train_gradient_predictor()
            self.train_meta_value()

            # Update EMA of gradient error
            self.grad_error_ema = (
                self.grad_error_ema_alpha * grad_error +
                (1 - self.grad_error_ema_alpha) * self.grad_error_ema
            )

        self.episode_count += 1

    def train_gradient_predictor(self) -> float:
        """
        Train gradient predictor on buffered experiences.

        Returns:
            Training loss
        """
        theta_batch, action_batch, gradient_batch, _ = self.buffer.sample(
            self.batch_size
        )

        predicted_gradients = self.gradient_predictor(theta_batch, action_batch)
        loss = F.mse_loss(predicted_gradients, gradient_batch)

        self.gp_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.gradient_predictor.parameters(),
            self.gradient_clip
        )
        self.gp_optimizer.step()

        self.gradient_errors.append(loss.item())
        return loss.item()

    def train_meta_value(self) -> float:
        """
        Train meta-value network to predict parameter quality.

        Returns:
            Training loss
        """
        theta_batch, _, _, reward_batch = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            target_values = (reward_batch - reward_batch.mean()) / (
                reward_batch.std() + 1e-8
            )

            policies = F.softmax(theta_batch, dim=1)
            entropies = -(policies * (policies + 1e-8).log()).sum(dim=1)
            target_values = target_values - 0.3 * entropies

        predicted_values = self.meta_value(theta_batch).squeeze()
        loss = F.mse_loss(predicted_values, target_values)

        self.mv_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.meta_value.parameters(),
            self.gradient_clip
        )
        self.mv_optimizer.step()

        self.meta_value_losses.append(loss.item())
        return loss.item()

    def get_policy_probs(self) -> np.ndarray:
        """Get current policy probabilities."""
        return self.policy.get_policy().detach().numpy()

    def get_parameters(self) -> torch.Tensor:
        """Get current policy parameters."""
        return self.policy.get_parameters()

    def get_planning_diagnostics(self) -> Dict[str, float]:
        """
        Get diagnostics about planning behavior.

        Returns:
            Dictionary with planning metrics
        """
        diagnostics = {
            'planning_enabled': float(self.planning_enabled_this_episode),
            'planning_weight': self.current_planning_weight,
            'grad_error_ema': self.grad_error_ema,
        }

        # Add meta-value of current parameters
        with torch.no_grad():
            current_theta = self.policy.get_parameters()
            self.meta_value.eval()
            current_meta_value = self.meta_value(current_theta).item()
            self.meta_value.train()
            diagnostics['meta_value_current'] = current_meta_value

        # Add planning scores for each arm
        for i in range(self.n_arms):
            diagnostics[f'planning_score_arm_{i}'] = float(self.planning_scores[i])

        # Add diagnostics about chosen vs best
        if self.planning_enabled_this_episode:
            best_arm = int(np.argmax(self.planning_scores))
            diagnostics['planning_best_arm'] = float(best_arm)
            diagnostics['planning_chosen_arm'] = float(self.last_chosen_arm)
            diagnostics['planning_score_chosen'] = float(self.planning_scores[self.last_chosen_arm])
            diagnostics['planning_score_best'] = float(self.planning_scores[best_arm])
            diagnostics['planning_score_gap'] = float(
                self.planning_scores[best_arm] - self.planning_scores[self.last_chosen_arm]
            )
        else:
            diagnostics['planning_best_arm'] = -1.0
            diagnostics['planning_chosen_arm'] = -1.0
            diagnostics['planning_score_chosen'] = 0.0
            diagnostics['planning_score_best'] = 0.0
            diagnostics['planning_score_gap'] = 0.0

        return diagnostics

    def save_checkpoint(self, path: str):
        """Save agent checkpoint."""
        checkpoint = {
            'policy_theta': self.policy.theta,
            'gradient_predictor_state': self.gradient_predictor.state_dict(),
            'meta_value_state': self.meta_value.state_dict(),
            'gp_optimizer_state': self.gp_optimizer.state_dict(),
            'mv_optimizer_state': self.mv_optimizer.state_dict(),
            'episode_count': self.episode_count,
            'gradient_errors': self.gradient_errors,
            'meta_value_losses': self.meta_value_losses,
            'grad_error_ema': self.grad_error_ema,
            'config': {
                'n_arms': self.n_arms,
                'learning_rate': self.learning_rate,
                'enable_planning': self.enable_planning,
                'planning_objective': self.planning_objective,
                'planning_warmup_episodes': self.planning_warmup_episodes,
                'planning_grad_error_threshold': self.planning_grad_error_threshold,
                'planning_max_weight': self.planning_max_weight,
            }
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')

        self.policy.theta = checkpoint['policy_theta']
        self.gradient_predictor.load_state_dict(checkpoint['gradient_predictor_state'])
        self.meta_value.load_state_dict(checkpoint['meta_value_state'])
        self.gp_optimizer.load_state_dict(checkpoint['gp_optimizer_state'])
        self.mv_optimizer.load_state_dict(checkpoint['mv_optimizer_state'])
        self.episode_count = checkpoint['episode_count']
        self.gradient_errors = checkpoint['gradient_errors']
        self.meta_value_losses = checkpoint['meta_value_losses']
        if 'grad_error_ema' in checkpoint:
            self.grad_error_ema = checkpoint['grad_error_ema']
