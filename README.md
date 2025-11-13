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
