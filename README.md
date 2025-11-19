# Self-Gradient World Models

A research project exploring agents that explicitly model and predict their own learning dynamics. Agents learn to predict how experiences will change their parameters, then use these predictions to actively seek beneficial learning experiences.

## Core Thesis

"An agent that understands how it learns can choose experiences that accelerate its own learning"

## Project Structure

```
gradient1/
├── src/                          # Core library code
│   ├── models/                   # Neural network models
│   │   ├── gradient_world_model.py   # Gradient predictor and meta-value
│   │   └── bandit_models.py          # Bandit-specific models
│   ├── agents/                   # Agent implementations
│   │   └── self_gradient_agent.py    # Self-gradient bandit agent
│   ├── training/                 # Training utilities
│   │   └── bandit_trainer.py         # Bandit experiment trainer
│   ├── envs/                     # Environments
│   │   └── bandits.py                # Multi-armed bandit
│   └── utils/                    # Utilities
│       ├── logging_utils.py          # Logging setup
│       ├── seed_utils.py             # Reproducibility
│       └── param_utils.py            # Parameter manipulation
├── experiments/                  # Experiment configurations and scripts
│   ├── run_experiment.py             # Main experiment runner
│   ├── bandit_self_gradient.yaml     # Main bandit config
│   ├── bandit_quick_test.yaml        # Quick test config
│   └── bandit_nonstationary.yaml     # Non-stationary config
├── tests/                        # Unit tests
│   ├── test_bandit_env.py
│   ├── test_gradient_predictor.py
│   └── test_self_gradient_agent.py
├── logs/                         # Experiment logs (created at runtime)
└── notes/                        # Research notes and reports
```

## Setup

### Requirements

- Python 3.10 or 3.11
- CPU-only environment (no CUDA required)

### Installation

#### Option 1: Automatic Setup (Recommended)

Use the provided setup script:

```bash
cd /path/to/gradient1
./setup.sh
```

This script will:
- Create a virtual environment
- Install PyTorch (CPU-only)
- Install all dependencies

Then activate the environment:
```bash
source .venv/bin/activate
```

#### Option 2: Manual Setup

1. Create and activate virtual environment:
```bash
cd /path/to/gradient1
python3 -m venv .venv
source .venv/bin/activate
```

2. Upgrade pip:
```bash
pip install --upgrade pip
```

3. Install PyTorch (CPU-only):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

4. Install other dependencies:
```bash
pip install numpy matplotlib pandas tqdm pyyaml pytest pytest-cov gym python-dotenv
```

Alternatively, you can use requirements.txt (but install PyTorch separately first):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Running Experiments

### Quick Start

Run a quick test experiment (reduced parameters for fast iteration):
```bash
python -m experiments.run_experiment --config experiments/bandit_quick_test.yaml
```

### Main Experiments

Run the main self-gradient bandit experiment:
```bash
python -m experiments.run_experiment --config experiments/bandit_self_gradient.yaml
```

Run non-stationary bandit experiment:
```bash
python -m experiments.run_experiment --config experiments/bandit_nonstationary.yaml
```

### Overriding Configuration

You can override any configuration parameter from the command line:

```bash
# Change seed
python -m experiments.run_experiment --config experiments/bandit_quick_test.yaml seed=456

# Change number of episodes
python -m experiments.run_experiment --config experiments/bandit_quick_test.yaml training.n_episodes=2000

# Change learning rate
python -m experiments.run_experiment --config experiments/bandit_self_gradient.yaml agent.learning_rate=0.02
```

### Output

Each experiment run creates a timestamped directory in `logs/` containing:

```
logs/bandit_self_gradient/run_2025-11-13_12-30-00/
├── config.yaml              # Saved configuration
├── train.log                # Training log
├── metrics.csv              # Raw metrics data
├── results.png              # Visualization plots
└── final_checkpoint.pt      # Model checkpoint
```

## Experiment Results

After running an experiment, you will find:

1. **Metrics CSV**: Contains episode-by-episode data:
   - Our agent's rewards and regrets
   - Baseline agent rewards and regrets
   - Gradient prediction errors
   - Policy entropy

2. **Visualization**: Six-panel figure showing:
   - Cumulative rewards over time
   - Cumulative regret over time
   - Rolling average rewards
   - Gradient prediction error
   - Policy entropy evolution
   - Final performance comparison

3. **Logs**: Plain text logs with training progress and statistics

## Running Tests

Run all unit tests:
```bash
pytest tests/
```

Run specific test file:
```bash
pytest tests/test_bandit_env.py
```

Run with coverage:
```bash
pytest --cov=src tests/
```

## Key Concepts

### Self-Gradient Agent

The core innovation is an agent that:
1. **Learns a gradient predictor**: `g(θ, a) → ∇θL` that predicts how taking action `a` with parameters `θ` will update the parameters
2. **Plans over learning dynamics**: Evaluates each possible action by:
   - Predicting the gradient it would produce
   - Simulating the parameter update
   - Evaluating the resulting parameters
3. **Chooses learning-optimal actions**: Selects actions that lead to better future parameter states

### Components

- **GradientPredictor**: Neural network that predicts parameter gradients from (parameters, action) pairs
- **MetaValueNetwork**: Evaluates the quality of parameter vectors (learns what makes good parameters)
- **SoftmaxBanditPolicy**: Softmax policy parameterized by θ
- **ExperienceBuffer**: Stores (θ, action, gradient, reward) tuples for training

### Baseline Comparisons

The agent is compared against standard bandit algorithms:
- **Epsilon-greedy**: Explores randomly with probability ε
- **UCB (Upper Confidence Bound)**: Optimistic exploration based on uncertainty
- **Thompson Sampling**: Bayesian approach with Beta distributions

## Bandit Phase: Negative Result (CLOSED)

**Status**: The bandit meta-value planning experiments have concluded with a **negative result**. This work is considered **CLOSED**.

### Summary

After comprehensive investigation, we found that **meta-value-based planning is not viable for bandit tasks** with online single-episode training:

- **Root cause**: Meta-value network learns inverted preferences (assigns lowest score to optimal arm)
- **Performance impact**: Planning makes performance **6.2x worse** than no planning, **107x worse** than UCB
- **Sample complexity**: Single-episode rewards have **17.3x higher variance** than needed (385 episodes required for reliable estimates)
- **Offline validation**: Meta-value architecture works perfectly offline (0.98 correlation with 300-episode averages)
- **Online failure**: Online correlation only 0.018-0.070 (98% degradation)

**Key insight**: "Confident wrong decisions are worse than uncertain random exploration."

### Documentation

Complete analysis and findings available in:

- **Main reports**:
  - [`analysis/planning_failure_diagnosis_report.md`](analysis/planning_failure_diagnosis_report.md) - Complete diagnostic report
  - [`analysis/meta_value_summary.md`](analysis/meta_value_summary.md) - Performance summary

- **Figures and data**:
  - [`analysis/meta_value_vs_planning_overview.png`](analysis/meta_value_vs_planning_overview.png) - Comparison plots
  - [`analysis/meta_value_vs_planning_overview.csv`](analysis/meta_value_vs_planning_overview.csv) - Statistics table
  - [`analysis/planning_diagnosis.png`](analysis/planning_diagnosis.png) - Diagnostic visualizations
  - [`analysis/meta_value_noise_ablation.png`](analysis/meta_value_noise_ablation.png) - Noise ablation study
  - [`analysis/sample_efficiency_analysis.png`](analysis/sample_efficiency_analysis.png) - Variance analysis

- **Regenerate all analysis**:
  ```bash
  python analysis/run_bandit_story.py
  ```

### Final Bandit Configs

The following configs represent final experiments (no further tuning):

- [`experiments/meta_value_improved.yaml`](experiments/meta_value_improved.yaml) - Improved meta-value training without planning
- [`experiments/planning_meta_value_improved.yaml`](experiments/planning_meta_value_improved.yaml) - Planning with improved meta-value (negative result)
- [`experiments/planning_test_instrumented.yaml`](experiments/planning_test_instrumented.yaml) - Short instrumented test run
- [`experiments/ablation_*.yaml`](experiments/) - Planning objective ablations

### Recommendations

**For bandits**: Use direct methods (UCB, Thompson Sampling, ε-greedy)

**Do NOT use**: Meta-value planning (adds complexity without benefit)

### Next Steps

Meta-value learning may succeed in **multi-step environments (MDPs)** where:
- Multi-step returns provide less noisy targets
- Value functions are structurally necessary
- Credit assignment is non-trivial

See [`mdp_experiments/README.md`](mdp_experiments/README.md) for MDP project specifications.

## MDP Baseline: Chain MDP (REINFORCE)

**Status**: Baseline established and verified. All MDP experiments must meet or exceed this baseline.

### Chain MDP Environment

A simple linear chain of states for testing multi-step learning:
- **States**: 10 (indexed 0-9)
- **Actions**: 2 (0=left, 1=right)
- **Start state**: 0 (leftmost)
- **Goal state**: 9 (rightmost)
- **Rewards**: 0.0 everywhere except 1.0 at goal
- **Discount factor**: γ = 0.99
- **Optimal policy**: Always go right
- **Optimal return**: γ^8 ≈ 0.923 (reward received at timestep 8)

### REINFORCE Baseline Agent

Classic policy gradient algorithm (Williams, 1992):
- Policy network: State embedding → hidden layer → action logits
- Monte Carlo returns (no bootstrapping)
- Baseline subtraction for variance reduction
- Entropy bonus (0.01) for exploration
- Learning rate: 0.01

### Baseline Performance

**Training (500 episodes):**
- Mean return: **1.000** (maximum possible)
- Episode length: **9.0 steps** (optimal)
- Success rate: **100%**
- Training time: ~1.3s (394.7 eps/s)

**Evaluation (100 episodes, greedy policy):**
- Mean return: **1.000 ± 0.000**
- Episode length: **9.0 ± 0.0**
- Success rate: **100%**

The agent reliably learns the optimal policy (always go right) within 100 episodes.

### Running the Baseline

```bash
python -m experiments.run_experiment --config experiments/chain_mdp_baseline.yaml
```

Results saved to: `logs/chain_mdp_baseline/run_YYYY-MM-DD_HH-MM-SS/`

### Files

- Config: [`experiments/chain_mdp_baseline.yaml`](experiments/chain_mdp_baseline.yaml)
- Environment: [`src/envs/mdp_envs.py`](src/envs/mdp_envs.py) (ChainMDP class)
- Agent: [`src/agents/reinforce_agent.py`](src/agents/reinforce_agent.py)
- Trainer: [`src/training/mdp_trainer.py`](src/training/mdp_trainer.py)
- Tests: [`tests/test_chain_mdp.py`](tests/test_chain_mdp.py)
- Notes: [`mdp_experiments/CHAIN_MDP_NOTES.md`](mdp_experiments/CHAIN_MDP_NOTES.md)

### Requirements for Future MDP Experiments

Any meta-value or planning experiments on Chain MDP must:
1. Match baseline performance (100% success, ~9 steps, return ~1.0)
2. If planning helps, show clear improvement over baseline
3. If planning hurts, diagnose why (as we did for bandits)

This baseline ensures meta-value learning is tested in controlled conditions where:
- Optimal policy is known and simple
- Multi-step returns provide better signal than single rewards
- Success is easily measured (reached goal or not)

## MDP Meta-Value Planning

**Status**: Infrastructure complete. Ready for experimentation.

### Overview

Unlike bandits (where meta-value planning failed due to high single-episode variance), MDPs can use **multi-episode average returns** as meta-value targets. This reduces variance by ~17x, making meta-value learning tractable.

### Architecture

The MDP meta-value system consists of three components:

#### 1. Dataset Builder

Extract policy snapshots from training runs and evaluate with K_eval episodes:

```bash
python -m src.analysis.build_mdp_meta_value_dataset \
    --run-dir logs/chain_mdp_baseline/run_2025-11-19_06-18-51 \
    --config experiments/chain_mdp_baseline.yaml \
    --snapshot-interval 20 \
    --n-eval-episodes 200 \
    --output analysis/mdp_meta_value_dataset.pt
```

**Output**: Dataset with policy parameters → multi-episode returns

#### 2. Meta-Value Trainer

Train MLP regressor V(θ) to predict policy returns from parameters:

```bash
python -m src.analysis.train_offline_mdp_meta_value \
    --dataset-path analysis/mdp_meta_value_dataset.pt \
    --output-dir analysis/meta_value_model \
    --hidden-dims 256 128 64 \
    --epochs 200
```

**Target quality**: Pearson correlation ≥ 0.95 on validation set

#### 3. Planning Agent

REINFORCE agent that blends base policy with meta-value-guided exploration:

```bash
python -m experiments.run_experiment --config experiments/chain_mdp_planning.yaml
```

**Blended policy**: π(a|s) = (1-w)π_base + wπ_plan

### Evaluation

Compare baseline vs planning performance:

```bash
python analysis/mdp_planning_comparison.py \
    --baseline-dir logs/chain_mdp_baseline/run_XXX \
    --planning-dir logs/chain_mdp_planning/run_YYY \
    --output-dir analysis/planning_comparison
```

**Metrics**:
- Final return improvement
- Learning speed (episodes to 80% success)
- Meta-value prediction quality

### Files

**Scripts**:
- [`src/analysis/build_mdp_meta_value_dataset.py`](src/analysis/build_mdp_meta_value_dataset.py) - Dataset builder
- [`src/analysis/train_offline_mdp_meta_value.py`](src/analysis/train_offline_mdp_meta_value.py) - Meta-value trainer
- [`src/analysis/evaluate_mdp_meta_value.py`](src/analysis/evaluate_mdp_meta_value.py) - Model evaluation
- [`src/agents/mdp_planning_agent.py`](src/agents/mdp_planning_agent.py) - Planning agent
- [`analysis/mdp_planning_comparison.py`](analysis/mdp_planning_comparison.py) - Comparison tool

**Configs**:
- [`experiments/chain_mdp_planning.yaml`](experiments/chain_mdp_planning.yaml) - Planning experiment

**Documentation**:
- [`mdp_experiments/README.md`](mdp_experiments/README.md) - Project overview
- [`analysis/mdp_meta_value_spec.md`](analysis/mdp_meta_value_spec.md) - Dataset specification
- [`mdp_experiments/CHAIN_MDP_NOTES.md`](mdp_experiments/CHAIN_MDP_NOTES.md) - Environment notes

**Tests**:
- [`tests/test_build_mdp_meta_value_dataset.py`](tests/test_build_mdp_meta_value_dataset.py) - Dataset builder tests

### Key Differences from Bandits

| Aspect | Bandits | MDPs |
|--------|---------|------|
| **Target** | Single-episode reward | Multi-episode average return |
| **Variance** | High (σ=1.0) | Low (σ/√K_eval) |
| **Sample complexity** | 385 episodes | ~22 episodes (17.3x better) |
| **Offline correlation** | 0.98 | Expected ≥0.95 |
| **Online correlation** | 0.018-0.070 (failed) | To be measured |

### Usage Example

Full workflow from baseline to planning:

```bash
# 1. Train baseline (with periodic checkpoints)
python -m experiments.run_experiment --config experiments/chain_mdp_baseline.yaml

# 2. Build meta-value dataset
python -m src.analysis.build_mdp_meta_value_dataset \
    --run-dir logs/chain_mdp_baseline/run_XXX \
    --config experiments/chain_mdp_baseline.yaml \
    --snapshot-interval 20 \
    --n-eval-episodes 200 \
    --output analysis/mdp_meta_value_dataset.pt

# 3. Train meta-value model
python -m src.analysis.train_offline_mdp_meta_value \
    --dataset-path analysis/mdp_meta_value_dataset.pt \
    --output-dir analysis/meta_value_model

# 4. Evaluate meta-value model
python -m src.analysis.evaluate_mdp_meta_value \
    --dataset-path analysis/mdp_meta_value_dataset.pt \
    --model-path analysis/meta_value_model/model.pt \
    --output-dir analysis/meta_value_eval

# 5. Train planning agent
python -m experiments.run_experiment --config experiments/chain_mdp_planning.yaml

# 6. Compare performance
python analysis/mdp_planning_comparison.py \
    --baseline-dir logs/chain_mdp_baseline/run_XXX \
    --planning-dir logs/chain_mdp_planning/run_YYY \
    --output-dir analysis/planning_comparison
```

## Experiment Guidelines

Following strict research standards:

### Do's
- All experiments must use configuration files (YAML)
- Set random seeds for reproducibility
- Log all metrics to CSV files
- Save experiment configurations with results
- Use proper logging (no print statements, no decorative output)
- Write clean, modular, reusable code

### Don'ts
- No dummy scripts (test_code.py, try_stuff.py, etc.)
- No fake experiments or sanity plots without real training runs
- No undocumented hyperparameter changes
- No emojis or decorative output in logs
- No hardcoded parameters (use configs)

## Creating New Experiments

1. Create a new config file in `experiments/`:
```yaml
experiment:
  name: "my_experiment"
  type: "bandit"
  description: "Description of experiment"

environment:
  n_arms: 10
  seed: 42

agent:
  learning_rate: 0.01
  # ... other parameters

training:
  n_episodes: 5000
```

2. Run the experiment:
```bash
python -m experiments.run_experiment --config experiments/my_experiment.yaml
```

3. Document results in `notes/`:
```bash
echo "# Experiment: my_experiment" > notes/2025-11-13_my_experiment.md
echo "Config: experiments/my_experiment.yaml" >> notes/2025-11-13_my_experiment.md
echo "Results: logs/my_experiment/run_2025-11-13_XX-XX-XX/" >> notes/2025-11-13_my_experiment.md
```

## CPU Optimization

The codebase is optimized for CPU-only training:

- Smaller model architectures (hidden_dim=64 default)
- Smaller batch sizes (16-32)
- Efficient numpy-backed buffers
- GRU instead of LSTM for temporal processing
- Batch normalization for stability
- Gradient clipping for stability

To adjust CPU thread count, modify `src/utils/seed_utils.py`:
```python
torch.set_num_threads(4)  # Adjust based on your CPU
```

## Reproducibility

All experiments are fully reproducible:

1. Random seeds are set for Python, NumPy, and PyTorch
2. Seeds are saved in experiment configs
3. All configurations are logged
4. Deterministic algorithms are used where possible

To reproduce an experiment:
```bash
python -m experiments.run_experiment --config logs/my_experiment/run_XXX/config.yaml
```

## Extending the Project

### Adding New Environments

1. Implement environment in `src/envs/`
2. Add to `src/envs/__init__.py`
3. Create config in `experiments/`
4. Update `run_experiment.py` if needed

### Adding New Agents

1. Implement agent in `src/agents/`
2. Add to `src/agents/__init__.py`
3. Update config schema
4. Update `run_experiment.py` to handle new agent type

### Adding New Models

1. Implement model in `src/models/`
2. Add to `src/models/__init__.py`
3. Use in agent implementations

## Troubleshooting

### Import Errors

If you see import errors, ensure you're running from the project root:
```bash
cd /path/to/gradient1
python -m experiments.run_experiment --config ...
```

### Memory Issues

If running out of memory on CPU:
- Reduce `batch_size` in config
- Reduce `buffer_size` in config
- Reduce `hidden_dim` in config
- Reduce `n_episodes` for testing

### Slow Training

Training is CPU-bound. To speed up:
- Use smaller models (reduce `hidden_dim`)
- Use fewer episodes (`n_episodes`)
- Reduce batch size (`batch_size`)
- Use the quick test config for development

## Citation

If you use this code in your research, please cite:

```
@misc{gradient-world-models,
  title={Self-Gradient World Models},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/gradient1}}
}
```

## License

This project is for research purposes.

## Contact

For questions or issues, please open an issue on GitHub.
