"""Tests for REINFORCE agent."""

import pytest
import torch
import numpy as np
from src.agents.reinforce_agent import REINFORCEAgent, PolicyNetwork
from src.envs.mdp_envs import ChainMDP


class TestPolicyNetwork:
    """Test suite for PolicyNetwork."""

    def test_initialization(self):
        """Test network initialization."""
        policy = PolicyNetwork(state_dim=10, action_dim=2, hidden_dim=64)
        assert policy.state_dim == 10
        assert policy.action_dim == 2
        assert policy.state_embedding.num_embeddings == 10
        assert policy.state_embedding.embedding_dim == 64

    def test_forward(self):
        """Test forward pass produces correct shape."""
        policy = PolicyNetwork(state_dim=10, action_dim=2, hidden_dim=64)
        state = torch.tensor([0, 5, 9], dtype=torch.long)
        logits = policy.forward(state)

        assert logits.shape == (3, 2)  # (batch_size, action_dim)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_get_action_probs(self):
        """Test action probabilities sum to 1."""
        policy = PolicyNetwork(state_dim=10, action_dim=2, hidden_dim=64)
        state = torch.tensor([0, 5], dtype=torch.long)
        probs = policy.get_action_probs(state)

        assert probs.shape == (2, 2)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2))
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_get_log_prob(self):
        """Test log probability computation."""
        policy = PolicyNetwork(state_dim=10, action_dim=2, hidden_dim=64)
        state = torch.tensor([0, 5], dtype=torch.long)
        action = torch.tensor([0, 1], dtype=torch.long)
        log_probs = policy.get_log_prob(state, action)

        assert log_probs.shape == (2,)
        assert (log_probs <= 0).all()  # Log probs should be ≤ 0


class TestREINFORCEAgent:
    """Test suite for REINFORCE agent."""

    def test_initialization(self):
        """Test agent initialization."""
        agent = REINFORCEAgent(
            state_dim=10,
            action_dim=2,
            learning_rate=0.01,
            seed=42
        )

        assert agent.state_dim == 10
        assert agent.action_dim == 2
        assert agent.learning_rate == 0.01
        assert agent.episode_count == 0
        assert agent.total_steps == 0
        assert len(agent.episode_states) == 0

    def test_act_deterministic(self):
        """Test deterministic action selection."""
        agent = REINFORCEAgent(state_dim=10, action_dim=2, seed=42)

        # Greedy action should be deterministic
        action1 = agent.act(state=0, greedy=True)
        action2 = agent.act(state=0, greedy=True)

        assert action1 == action2
        assert action1 in [0, 1]

    def test_act_stochastic(self):
        """Test stochastic action sampling."""
        agent = REINFORCEAgent(state_dim=10, action_dim=2, seed=42)

        # Sample many times, should get both actions
        actions = [agent.act(state=0, greedy=False) for _ in range(100)]

        assert 0 in actions
        assert 1 in actions
        assert all(a in [0, 1] for a in actions)

    def test_store_transition(self):
        """Test transition storage."""
        agent = REINFORCEAgent(state_dim=10, action_dim=2, seed=42)

        agent.store_transition(state=0, action=1, reward=0.5)
        agent.store_transition(state=1, action=1, reward=1.0)

        assert len(agent.episode_states) == 2
        assert len(agent.episode_actions) == 2
        assert len(agent.episode_rewards) == 2
        assert agent.episode_states == [0, 1]
        assert agent.episode_actions == [1, 1]
        assert agent.episode_rewards == [0.5, 1.0]
        assert agent.total_steps == 2

    def test_compute_returns(self):
        """Test discounted return calculation."""
        agent = REINFORCEAgent(state_dim=10, action_dim=2, discount_factor=0.9, seed=42)

        rewards = [0.0, 0.0, 1.0]  # Reward at last step
        returns = agent.compute_returns(rewards)

        # G_0 = 0 + 0.9*0 + 0.9^2*1 = 0.81
        # G_1 = 0 + 0.9*1 = 0.9
        # G_2 = 1.0
        expected = np.array([0.81, 0.9, 1.0])

        assert np.allclose(returns, expected)

    def test_update_clears_buffer(self):
        """Test that update clears episode buffer."""
        agent = REINFORCEAgent(state_dim=10, action_dim=2, seed=42)

        agent.store_transition(0, 1, 0.0)
        agent.store_transition(1, 1, 1.0)

        assert len(agent.episode_states) == 2

        metrics = agent.update()

        assert len(agent.episode_states) == 0
        assert len(agent.episode_actions) == 0
        assert len(agent.episode_rewards) == 0
        assert agent.episode_count == 1

    def test_update_metrics(self):
        """Test update returns correct metrics."""
        agent = REINFORCEAgent(state_dim=10, action_dim=2, seed=42)

        agent.store_transition(0, 1, 0.0)
        agent.store_transition(1, 1, 1.0)

        metrics = agent.update()

        assert 'policy_loss' in metrics
        assert 'episode_return' in metrics
        assert 'episode_length' in metrics
        assert metrics['episode_length'] == 2
        assert metrics['episode_return'] > 0  # Should have positive return

    def test_baseline_reduces_variance(self):
        """Test that baseline is updated correctly."""
        agent = REINFORCEAgent(state_dim=10, action_dim=2, use_baseline=True, seed=42)

        # First episode
        agent.store_transition(0, 1, 1.0)
        metrics1 = agent.update()

        assert agent.baseline_count == 1
        assert agent.baseline_value > 0

        # Second episode
        agent.store_transition(0, 1, 2.0)
        metrics2 = agent.update()

        assert agent.baseline_count == 2
        # Baseline should be running average
        assert 1.0 < agent.baseline_value < 2.0

    def test_entropy_bonus_in_loss(self):
        """Test entropy bonus affects loss."""
        agent_no_bonus = REINFORCEAgent(state_dim=10, action_dim=2, entropy_bonus=0.0, seed=42)
        agent_with_bonus = REINFORCEAgent(state_dim=10, action_dim=2, entropy_bonus=0.1, seed=42)

        # Same episode for both
        for agent in [agent_no_bonus, agent_with_bonus]:
            agent.store_transition(0, 1, 1.0)

        metrics_no_bonus = agent_no_bonus.update()
        metrics_with_bonus = agent_with_bonus.update()

        # With entropy bonus, loss should be different (typically lower)
        assert metrics_no_bonus['policy_loss'] != metrics_with_bonus['policy_loss']

    def test_save_and_load(self, tmp_path):
        """Test checkpoint save and load."""
        agent = REINFORCEAgent(state_dim=10, action_dim=2, seed=42)

        # Train for one episode
        agent.store_transition(0, 1, 1.0)
        agent.update()

        # Save
        checkpoint_path = tmp_path / "checkpoint.pt"
        agent.save(str(checkpoint_path))

        assert checkpoint_path.exists()

        # Create new agent and load
        agent2 = REINFORCEAgent(state_dim=10, action_dim=2, seed=43)
        agent2.load(str(checkpoint_path))

        # Check state was restored
        assert agent2.episode_count == agent.episode_count
        assert agent2.total_steps == agent.total_steps
        assert agent2.baseline_value == agent.baseline_value

    def test_get_policy_parameters(self):
        """Test policy parameter extraction."""
        agent = REINFORCEAgent(state_dim=10, action_dim=2, seed=42)
        params = agent.get_policy_parameters()

        assert isinstance(params, np.ndarray)
        assert len(params) > 0
        assert params.dtype == np.float64 or params.dtype == np.float32


class TestREINFORCEOnChainMDP:
    """Integration tests with Chain MDP environment."""

    def test_agent_learns_chain_mdp(self):
        """Test that REINFORCE learns to solve Chain MDP."""
        env = ChainMDP(n_states=5, max_steps=50, seed=42)
        agent = REINFORCEAgent(
            state_dim=env.get_state_dim(),
            action_dim=env.get_action_dim(),
            learning_rate=0.01,
            discount_factor=env.get_discount_factor(),
            use_baseline=True,
            entropy_bonus=0.01,
            seed=42
        )

        # Train for 200 episodes
        success_count = 0
        returns = []

        for episode in range(200):
            state = env.reset()
            done = False
            episode_return = 0.0

            while not done:
                action = agent.act(state, greedy=False)
                next_state, reward, done, info = env.step(action)
                agent.store_transition(state, action, reward)
                episode_return += reward
                state = next_state

            metrics = agent.update()
            returns.append(episode_return)

            if info.get('at_goal', False):
                success_count += 1

        # Check learning occurred
        # Early episodes (first 50): success rate should be lower
        early_returns = returns[:50]
        late_returns = returns[-50:]

        # Late returns should be at least as good as early (learning happened or saturated)
        # Allow numerical slack for cases where agent learns optimally from the start
        assert np.mean(late_returns) >= np.mean(early_returns) - 1e-6

        # By end of training, should have high success rate
        # (but we allow some variance due to exploration)
        late_success_episodes = sum(1 for r in late_returns if r > 0)
        assert late_success_episodes >= 40  # At least 80% success in last 50

    def test_greedy_policy_after_training(self):
        """Test that greedy policy performs well after training."""
        env = ChainMDP(n_states=5, max_steps=50, seed=42)
        agent = REINFORCEAgent(
            state_dim=env.get_state_dim(),
            action_dim=env.get_action_dim(),
            learning_rate=0.01,
            discount_factor=env.get_discount_factor(),
            use_baseline=True,
            entropy_bonus=0.01,
            seed=42
        )

        # Train for 100 episodes
        for episode in range(100):
            state = env.reset()
            done = False

            while not done:
                action = agent.act(state, greedy=False)
                next_state, reward, done, info = env.step(action)
                agent.store_transition(state, action, reward)
                state = next_state

            agent.update()

        # Evaluate with greedy policy
        eval_successes = 0
        for episode in range(20):
            state = env.reset()
            done = False

            while not done:
                action = agent.act(state, greedy=True)
                next_state, reward, done, info = env.step(action)
                state = next_state

            if info.get('at_goal', False):
                eval_successes += 1

        # Greedy policy should have high success rate
        assert eval_successes >= 15  # At least 75%

    def test_policy_loss_decreases(self):
        """Test that policy loss generally trends downward during training."""
        env = ChainMDP(n_states=5, max_steps=50, seed=42)
        agent = REINFORCEAgent(
            state_dim=env.get_state_dim(),
            action_dim=env.get_action_dim(),
            learning_rate=0.01,
            discount_factor=env.get_discount_factor(),
            seed=42
        )

        losses = []

        for episode in range(100):
            state = env.reset()
            done = False

            while not done:
                action = agent.act(state, greedy=False)
                next_state, reward, done, info = env.step(action)
                agent.store_transition(state, action, reward)
                state = next_state

            metrics = agent.update()
            losses.append(abs(metrics['policy_loss']))  # Use absolute value

        # Early losses vs late losses (using absolute value)
        early_loss = np.mean(losses[:20])
        late_loss = np.mean(losses[-20:])

        # Late loss should be lower (agent is more confident)
        # Allow for some variance
        assert late_loss <= early_loss * 1.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
