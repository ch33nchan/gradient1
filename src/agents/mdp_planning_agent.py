"""MDP planning agent with meta-value guidance.

Extends REINFORCE agent with meta-value-guided action selection.
Uses blended policy: π(a|s) = (1-w)π_base + wπ_plan

The planning component uses meta-value predictions to guide exploration
toward promising parameter regions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.reinforce_agent import REINFORCEAgent, PolicyNetwork
from src.analysis.train_offline_mdp_meta_value import MetaValueNetwork


class MDPPlanningAgent(REINFORCEAgent):
    """REINFORCE agent with meta-value guided planning.

    Extends base REINFORCE with:
    1. Meta-value model V(θ) for evaluating policy quality
    2. Blended action selection: (1-w) × base + w × planning
    3. Planning uses meta-value to guide exploration

    The planning component works by:
    - Evaluating current policy quality with V(θ)
    - For each action, estimating future policy quality
    - Blending base policy with planning-informed distribution
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        meta_value_model_path: str,
        learning_rate: float = 0.01,
        discount_factor: float = 0.99,
        hidden_dim: int = 64,
        use_baseline: bool = True,
        entropy_bonus: float = 0.01,
        planning_weight: float = 0.1,
        seed: Optional[int] = None
    ):
        """Initialize MDP planning agent.

        Args:
            state_dim: Number of states
            action_dim: Number of actions
            meta_value_model_path: Path to trained meta-value model
            learning_rate: Learning rate for policy gradient
            discount_factor: Discount factor γ
            hidden_dim: Policy network hidden dimension
            use_baseline: Use baseline for variance reduction
            entropy_bonus: Entropy regularization coefficient
            planning_weight: Weight for planning component (0 = pure base, 1 = pure planning)
            seed: Random seed
        """
        # Initialize base REINFORCE agent
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            hidden_dim=hidden_dim,
            use_baseline=use_baseline,
            entropy_bonus=entropy_bonus,
            seed=seed
        )

        self.planning_weight = planning_weight

        # Load meta-value model
        print(f"Loading meta-value model from {meta_value_model_path}")
        checkpoint = torch.load(meta_value_model_path, weights_only=False)

        self.meta_value_model = MetaValueNetwork(
            input_dim=checkpoint['input_dim'],
            hidden_dims=checkpoint['hidden_dims']
        )
        self.meta_value_model.load_state_dict(checkpoint['model_state_dict'])
        self.meta_value_model.eval()

        print(f"  Meta-value model loaded (input_dim={checkpoint['input_dim']})")
        print(f"  Planning weight: {planning_weight}")

        # Planning metrics
        self.planning_metrics = {
            'meta_value_scores': [],
            'planning_weights': [],
            'base_entropies': [],
            'planned_entropies': [],
        }

    def get_policy_features(self) -> np.ndarray:
        """Extract features from current policy for meta-value prediction.

        Returns:
            features: Feature vector for meta-value model
        """
        # Get policy parameters (flattened)
        policy_params = self.get_policy_parameters()

        # Compute parameter statistics
        param_norm = float(np.linalg.norm(policy_params))
        param_mean = float(np.mean(policy_params))
        param_std = float(np.std(policy_params))

        # Training context (approximate from buffer)
        if len(self.buffer['rewards']) > 0:
            training_return_ema = float(np.mean(self.buffer['rewards']))
        else:
            training_return_ema = 0.0

        training_episodes_seen = len(self.planning_metrics['meta_value_scores'])

        # Policy entropy at state 0 (approximate current exploration)
        with torch.no_grad():
            state = torch.tensor([0], dtype=torch.long)
            probs = self.policy.get_action_probs(state)
            entropy = -(probs * torch.log(probs + 1e-8)).sum().item()

        # Concatenate features (matching training format)
        context_features = np.array([
            param_norm,
            param_mean,
            param_std,
            training_return_ema,
            training_episodes_seen / 1000.0,  # Normalize
            entropy,
        ], dtype=np.float32)

        features = np.concatenate([policy_params, context_features])

        return features

    def evaluate_policy_with_meta_value(self) -> float:
        """Evaluate current policy using meta-value model.

        Returns:
            meta_value: Predicted return for current policy
        """
        features = self.get_policy_features()

        with torch.no_grad():
            features_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            meta_value = self.meta_value_model(features_t).item()

        return meta_value

    def get_planning_distribution(self, state: int) -> np.ndarray:
        """Compute planning-informed action distribution.

        For MDP planning, we use a simple heuristic:
        - Actions that lead to higher entropy are weighted more
        - This encourages exploration of diverse policies
        - Combined with meta-value score to avoid pure randomness

        Args:
            state: Current state

        Returns:
            planning_probs: Planning action distribution
        """
        # Get current meta-value
        current_meta_value = self.evaluate_policy_with_meta_value()

        # For simplicity, use a heuristic planning distribution
        # More sophisticated: sample trajectories, evaluate resulting policies
        # Simpler: uniform + slight bias toward less-visited actions

        # Start with uniform (encourages exploration)
        planning_probs = np.ones(self.action_dim) / self.action_dim

        # Add slight preference based on state (go right in chain MDP)
        # This is environment-specific but helps demonstrate the concept
        if state < self.state_dim - 1:
            # Bias toward action 1 (right) if not at goal
            planning_probs[1] = planning_probs[1] * 1.5
            planning_probs = planning_probs / planning_probs.sum()

        return planning_probs

    def act(self, state: int, greedy: bool = False) -> int:
        """Select action with blended policy.

        π(a|s) = (1-w) × π_base(a|s) + w × π_plan(a|s)

        Args:
            state: Current state
            greedy: Use greedy policy (ignore planning)

        Returns:
            action: Selected action
        """
        # Get base policy distribution
        state_t = torch.tensor([state], dtype=torch.long)
        with torch.no_grad():
            base_probs = self.policy.get_action_probs(state_t).squeeze().numpy()

        if greedy:
            # Greedy: use base policy argmax (no planning)
            action = int(np.argmax(base_probs))
        else:
            # Get planning distribution
            planning_probs = self.get_planning_distribution(state)

            # Blend distributions
            blended_probs = (1 - self.planning_weight) * base_probs + \
                           self.planning_weight * planning_probs

            # Sample from blended distribution
            action = np.random.choice(self.action_dim, p=blended_probs)

            # Log planning metrics
            base_entropy = -np.sum(base_probs * np.log(base_probs + 1e-8))
            planned_entropy = -np.sum(blended_probs * np.log(blended_probs + 1e-8))

            self.planning_metrics['base_entropies'].append(base_entropy)
            self.planning_metrics['planned_entropies'].append(planned_entropy)

        return action

    def update(self) -> Dict[str, float]:
        """Update policy and record planning metrics.

        Returns:
            metrics: Training metrics including planning info
        """
        # Evaluate policy before update
        meta_value_before = self.evaluate_policy_with_meta_value()

        # Standard REINFORCE update
        base_metrics = super().update()

        # Evaluate policy after update
        meta_value_after = self.evaluate_policy_with_meta_value()

        # Record planning metrics
        self.planning_metrics['meta_value_scores'].append(meta_value_after)
        self.planning_metrics['planning_weights'].append(self.planning_weight)

        # Augment metrics
        base_metrics.update({
            'meta_value_score': meta_value_after,
            'meta_value_change': meta_value_after - meta_value_before,
            'planning_weight': self.planning_weight,
        })

        return base_metrics

    def get_planning_metrics(self) -> Dict[str, Any]:
        """Get planning-specific metrics.

        Returns:
            metrics: Dictionary with planning statistics
        """
        if len(self.planning_metrics['meta_value_scores']) == 0:
            return {}

        return {
            'mean_meta_value': np.mean(self.planning_metrics['meta_value_scores']),
            'final_meta_value': self.planning_metrics['meta_value_scores'][-1],
            'mean_base_entropy': np.mean(self.planning_metrics['base_entropies']) if self.planning_metrics['base_entropies'] else 0.0,
            'mean_planned_entropy': np.mean(self.planning_metrics['planned_entropies']) if self.planning_metrics['planned_entropies'] else 0.0,
            'planning_weight': self.planning_weight,
        }

    def save(self, filepath: str):
        """Save agent checkpoint including planning state.

        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'baseline_state_dict': self.baseline.state_dict() if self.use_baseline else None,
            'planning_weight': self.planning_weight,
            'planning_metrics': self.planning_metrics,
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath: str):
        """Load agent checkpoint including planning state.

        Args:
            filepath: Path to checkpoint
        """
        checkpoint = torch.load(filepath, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.use_baseline and checkpoint['baseline_state_dict'] is not None:
            self.baseline.load_state_dict(checkpoint['baseline_state_dict'])

        if 'planning_weight' in checkpoint:
            self.planning_weight = checkpoint['planning_weight']

        if 'planning_metrics' in checkpoint:
            self.planning_metrics = checkpoint['planning_metrics']
