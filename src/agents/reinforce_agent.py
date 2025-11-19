"""REINFORCE agent for MDP environments.

Classic policy gradient algorithm (Williams, 1992) for baseline comparison.
Uses Monte Carlo returns with no bootstrapping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging


class PolicyNetwork(nn.Module):
    """Neural network policy π(a|s; θ) for discrete actions."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        """Initialize policy network.

        Args:
            state_dim: Number of states (for discrete states, this is n_states)
            action_dim: Number of actions
            hidden_dim: Hidden layer dimension (default: 64)
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # For discrete states, use embedding
        self.state_embedding = nn.Embedding(state_dim, hidden_dim)

        # Policy network: embedding -> hidden -> action logits
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass: state -> action logits.

        Args:
            state: State index (long tensor)

        Returns:
            logits: Unnormalized action probabilities (batch_size, action_dim)
        """
        # Embed state
        x = self.state_embedding(state)

        # Hidden layers
        x = F.relu(self.fc1(x))

        # Action logits
        logits = self.fc2(x)

        return logits

    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """Get action probabilities π(a|s).

        Args:
            state: State index

        Returns:
            probs: Action probabilities (batch_size, action_dim)
        """
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        return probs

    def get_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get log probability log π(a|s).

        Args:
            state: State index (batch_size,)
            action: Action index (batch_size,)

        Returns:
            log_prob: Log probability (batch_size,)
        """
        logits = self.forward(state)
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather log prob for taken action
        action_log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)

        return action_log_prob


class REINFORCEAgent:
    """REINFORCE agent with Monte Carlo policy gradient.

    Algorithm:
    1. Roll out episode with current policy
    2. Compute discounted returns G_t for each timestep
    3. Update policy: ∇J(θ) = Σ_t ∇log π(a_t|s_t; θ) G_t
    4. Optional baseline subtraction for variance reduction
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        hidden_dim: int = 64,
        use_baseline: bool = False,
        entropy_bonus: float = 0.0,
        seed: Optional[int] = None,
    ):
        """Initialize REINFORCE agent.

        Args:
            state_dim: Number of states
            action_dim: Number of actions
            learning_rate: Policy learning rate (default: 0.001)
            discount_factor: Discount factor gamma (default: 0.99)
            hidden_dim: Hidden layer dimension (default: 64)
            use_baseline: Whether to subtract baseline (mean return) (default: False)
            entropy_bonus: Entropy bonus coefficient for exploration (default: 0.0)
            seed: Random seed
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.use_baseline = use_baseline
        self.entropy_bonus = entropy_bonus

        # Policy network
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Episode buffer (cleared after each update)
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

        # Statistics
        self.episode_count = 0
        self.total_steps = 0

        # Baseline (running mean of returns)
        self.baseline_value = 0.0
        self.baseline_count = 0

        # Random seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Logger
        self.logger = logging.getLogger(__name__)

    def act(self, state: int, greedy: bool = False) -> int:
        """Select action given state.

        Args:
            state: Current state index
            greedy: If True, take argmax action (default: False, sample from policy)

        Returns:
            action: Selected action index
        """
        state_tensor = torch.tensor([state], dtype=torch.long)

        with torch.no_grad():
            probs = self.policy.get_action_probs(state_tensor)

        if greedy:
            action = probs.argmax(dim=-1).item()
        else:
            # Sample from categorical distribution
            action = torch.multinomial(probs, num_samples=1).item()

        return action

    def store_transition(self, state: int, action: int, reward: float):
        """Store transition in episode buffer.

        Args:
            state: State index
            action: Action index
            reward: Reward received
        """
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        self.total_steps += 1

    def compute_returns(self, rewards: List[float]) -> np.ndarray:
        """Compute discounted returns G_t = Σ_{k=0}^∞ γ^k r_{t+k}.

        Args:
            rewards: List of rewards for episode

        Returns:
            returns: Discounted returns for each timestep (T,)
        """
        T = len(rewards)
        returns = np.zeros(T)

        # Compute returns backwards
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + self.discount_factor * G
            returns[t] = G

        return returns

    def update(self) -> Dict[str, float]:
        """Update policy after episode completes using REINFORCE.

        Returns:
            metrics: Dictionary with update metrics
        """
        if len(self.episode_states) == 0:
            return {}

        # Compute returns
        returns = self.compute_returns(self.episode_rewards)

        # Compute episode return (for baseline)
        episode_return = returns[0]

        # Update baseline (running mean)
        if self.use_baseline:
            self.baseline_count += 1
            alpha = 1.0 / self.baseline_count
            self.baseline_value = (1 - alpha) * self.baseline_value + alpha * episode_return

            # Subtract baseline from returns
            advantages = returns - self.baseline_value
        else:
            advantages = returns

        # Convert to tensors
        states = torch.tensor(self.episode_states, dtype=torch.long)
        actions = torch.tensor(self.episode_actions, dtype=torch.long)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32)

        # Compute log probabilities
        log_probs = self.policy.get_log_prob(states, actions)

        # Policy gradient loss: -Σ log π(a|s) * G_t
        policy_loss = -(log_probs * advantages_tensor).mean()

        # Entropy bonus (encourages exploration)
        if self.entropy_bonus > 0:
            probs = self.policy.get_action_probs(states)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            policy_loss = policy_loss - self.entropy_bonus * entropy
        else:
            entropy = torch.tensor(0.0)

        # Optimize
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Clear episode buffer
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_count += 1

        # Return metrics
        metrics = {
            'policy_loss': policy_loss.item(),
            'episode_return': episode_return,
            'episode_length': len(returns),
            'entropy': entropy.item() if isinstance(entropy, torch.Tensor) else 0.0,
            'baseline_value': self.baseline_value,
        }

        return metrics

    def get_policy_parameters(self) -> np.ndarray:
        """Get flattened policy parameters (for meta-value learning).

        Returns:
            params: Flattened parameter vector
        """
        params = []
        for param in self.policy.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def save(self, filepath: str):
        """Save agent checkpoint.

        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'baseline_value': self.baseline_value,
            'baseline_count': self.baseline_count,
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"Saved checkpoint to {filepath}")

    def load(self, filepath: str):
        """Load agent checkpoint.

        Args:
            filepath: Path to checkpoint
        """
        checkpoint = torch.load(filepath, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_count = checkpoint['episode_count']
        self.total_steps = checkpoint['total_steps']
        self.baseline_value = checkpoint['baseline_value']
        self.baseline_count = checkpoint['baseline_count']
        self.logger.info(f"Loaded checkpoint from {filepath}")
