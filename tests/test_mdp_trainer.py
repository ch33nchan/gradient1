"""Tests for MDP trainer."""

import pytest
import pandas as pd
from pathlib import Path
from src.training.mdp_trainer import MDPTrainer
from src.agents.reinforce_agent import REINFORCEAgent
from src.envs.mdp_envs import ChainMDP


class TestMDPTrainer:
    """Test suite for MDP trainer."""

    def test_initialization(self):
        """Test trainer initialization."""
        env = ChainMDP(n_states=10, seed=42)
        agent = REINFORCEAgent(
            state_dim=env.get_state_dim(),
            action_dim=env.get_action_dim(),
            seed=42
        )

        trainer = MDPTrainer(env=env, agent=agent, log_interval=10)

        assert trainer.env == env
        assert trainer.agent == agent
        assert trainer.log_interval == 10
        assert trainer.total_steps == 0
        assert trainer.total_episodes == 0

    def test_run_episode(self):
        """Test single episode execution."""
        env = ChainMDP(n_states=10, seed=42)
        agent = REINFORCEAgent(
            state_dim=env.get_state_dim(),
            action_dim=env.get_action_dim(),
            seed=42
        )
        trainer = MDPTrainer(env=env, agent=agent)

        episode_info = trainer.run_episode(greedy=False)

        assert 'return' in episode_info
        assert 'length' in episode_info
        assert 'success' in episode_info
        assert episode_info['length'] > 0
        assert isinstance(episode_info['success'], bool)

    def test_train_basic(self):
        """Test basic training loop runs without errors."""
        env = ChainMDP(n_states=10, seed=42)
        agent = REINFORCEAgent(
            state_dim=env.get_state_dim(),
            action_dim=env.get_action_dim(),
            learning_rate=0.01,
            seed=42
        )
        trainer = MDPTrainer(env=env, agent=agent, log_interval=5)

        # Train for small number of episodes
        metrics_df = trainer.train(n_episodes=20)

        assert isinstance(metrics_df, pd.DataFrame)
        assert len(metrics_df) == 20
        assert 'episode_returns' in metrics_df.columns
        assert 'episode_lengths' in metrics_df.columns
        assert 'policy_losses' in metrics_df.columns
        assert 'success_rate' in metrics_df.columns

    def test_metrics_tracking(self):
        """Test that metrics are correctly tracked."""
        env = ChainMDP(n_states=10, seed=42)
        agent = REINFORCEAgent(
            state_dim=env.get_state_dim(),
            action_dim=env.get_action_dim(),
            seed=42
        )
        trainer = MDPTrainer(env=env, agent=agent)

        metrics_df = trainer.train(n_episodes=10)

        # Check all expected metrics present
        assert 'episode_returns' in metrics_df.columns
        assert 'episode_lengths' in metrics_df.columns
        assert 'policy_losses' in metrics_df.columns
        assert 'entropies' in metrics_df.columns
        assert 'baseline_values' in metrics_df.columns
        assert 'success_rate' in metrics_df.columns

        # Check no NaN values
        assert not metrics_df.isnull().any().any()

        # Check success rate is binary
        assert all(x in [0.0, 1.0] for x in metrics_df['success_rate'])

    def test_evaluate(self):
        """Test evaluation mode."""
        env = ChainMDP(n_states=10, seed=42)
        agent = REINFORCEAgent(
            state_dim=env.get_state_dim(),
            action_dim=env.get_action_dim(),
            learning_rate=0.01,
            entropy_bonus=0.01,
            seed=42
        )
        trainer = MDPTrainer(env=env, agent=agent)

        # Train first
        trainer.train(n_episodes=50)

        # Evaluate
        eval_metrics = trainer.evaluate(n_episodes=10)

        assert 'mean_return' in eval_metrics
        assert 'std_return' in eval_metrics
        assert 'mean_length' in eval_metrics
        assert 'std_length' in eval_metrics
        assert 'success_rate' in eval_metrics

        assert 0 <= eval_metrics['success_rate'] <= 1.0
        assert eval_metrics['mean_length'] > 0
        assert eval_metrics['std_return'] >= 0
        assert eval_metrics['std_length'] >= 0

    def test_plot_training(self, tmp_path):
        """Test training plot generation."""
        env = ChainMDP(n_states=10, seed=42)
        agent = REINFORCEAgent(
            state_dim=env.get_state_dim(),
            action_dim=env.get_action_dim(),
            seed=42
        )
        trainer = MDPTrainer(env=env, agent=agent)

        # Train
        trainer.train(n_episodes=20)

        # Generate plot
        plot_path = tmp_path / "training.png"
        trainer.plot_training(output_path=plot_path, window=5)

        assert plot_path.exists()

    def test_total_steps_increments(self):
        """Test that total_steps counter increments correctly."""
        env = ChainMDP(n_states=5, max_steps=20, seed=42)
        agent = REINFORCEAgent(
            state_dim=env.get_state_dim(),
            action_dim=env.get_action_dim(),
            seed=42
        )
        trainer = MDPTrainer(env=env, agent=agent)

        assert trainer.total_steps == 0

        trainer.train(n_episodes=5)

        # Total steps should be positive (sum of episode lengths)
        assert trainer.total_steps > 0

        # Total steps should equal sum of episode lengths
        episode_lengths = trainer.metrics['episode_lengths']
        assert trainer.total_steps == sum(episode_lengths)

    def test_agent_updates_during_training(self):
        """Test that agent is actually updated during training."""
        env = ChainMDP(n_states=10, seed=42)
        agent = REINFORCEAgent(
            state_dim=env.get_state_dim(),
            action_dim=env.get_action_dim(),
            seed=42
        )
        trainer = MDPTrainer(env=env, agent=agent)

        # Get initial parameters
        params_before = agent.get_policy_parameters().copy()

        # Train
        trainer.train(n_episodes=10)

        # Get final parameters
        params_after = agent.get_policy_parameters()

        # Parameters should have changed
        assert not (params_before == params_after).all()

    def test_greedy_evaluation_no_buffer_storage(self):
        """Test that greedy evaluation doesn't affect agent buffer."""
        env = ChainMDP(n_states=10, seed=42)
        agent = REINFORCEAgent(
            state_dim=env.get_state_dim(),
            action_dim=env.get_action_dim(),
            seed=42
        )
        trainer = MDPTrainer(env=env, agent=agent)

        # Run greedy episode
        episode_info = trainer.run_episode(greedy=True)

        # Agent buffer should be empty (greedy doesn't store)
        assert len(agent.episode_states) == 0
        assert len(agent.episode_actions) == 0
        assert len(agent.episode_rewards) == 0

    def test_training_convergence_chain_mdp(self):
        """Test that training actually improves performance on Chain MDP."""
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
        trainer = MDPTrainer(env=env, agent=agent, log_interval=20)

        # Train
        metrics_df = trainer.train(n_episodes=100)

        # Early vs late performance
        early_returns = metrics_df['episode_returns'][:20].mean()
        late_returns = metrics_df['episode_returns'][-20:].mean()

        # Performance should improve
        assert late_returns > early_returns

        # Success rate should be high by the end
        late_success = metrics_df['success_rate'][-20:].mean()
        assert late_success > 0.7  # At least 70% success

    def test_checkpoint_saved_correctly(self, tmp_path):
        """Test that checkpoint can be saved after training."""
        env = ChainMDP(n_states=10, seed=42)
        agent = REINFORCEAgent(
            state_dim=env.get_state_dim(),
            action_dim=env.get_action_dim(),
            seed=42
        )
        trainer = MDPTrainer(env=env, agent=agent)

        # Train
        trainer.train(n_episodes=10)

        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        agent.save(str(checkpoint_path))

        assert checkpoint_path.exists()

        # Load checkpoint into new agent
        agent2 = REINFORCEAgent(
            state_dim=env.get_state_dim(),
            action_dim=env.get_action_dim(),
            seed=43
        )
        agent2.load(str(checkpoint_path))

        # Episode count should match
        assert agent2.episode_count == agent.episode_count

    def test_metrics_csv_format(self, tmp_path):
        """Test that metrics can be saved to CSV correctly."""
        env = ChainMDP(n_states=10, seed=42)
        agent = REINFORCEAgent(
            state_dim=env.get_state_dim(),
            action_dim=env.get_action_dim(),
            seed=42
        )
        trainer = MDPTrainer(env=env, agent=agent)

        # Train
        metrics_df = trainer.train(n_episodes=20)

        # Save to CSV
        csv_path = tmp_path / "metrics.csv"
        metrics_df.to_csv(csv_path, index=False)

        assert csv_path.exists()

        # Load and verify
        loaded_df = pd.read_csv(csv_path)
        assert len(loaded_df) == 20
        assert list(loaded_df.columns) == list(metrics_df.columns)
        assert loaded_df['episode_returns'].tolist() == metrics_df['episode_returns'].tolist()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
