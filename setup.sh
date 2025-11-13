#!/bin/bash
# Setup script for Self-Gradient World Models project
# CPU-only installation

set -e

echo "Setting up Self-Gradient World Models project..."
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch CPU-only version
echo "Installing PyTorch (CPU-only)..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
echo "Installing other dependencies..."
pip install numpy matplotlib pandas tqdm pyyaml pytest pytest-cov gym python-dotenv

echo ""
echo "Installation complete!"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run a quick test experiment:"
echo "  python -m experiments.run_experiment --config experiments/bandit_quick_test.yaml"
echo ""
echo "To run tests:"
echo "  pytest tests/"
echo ""
