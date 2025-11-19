"""Train offline meta-value network V(θ) from MDP dataset.

This script:
1. Loads MDP meta-value dataset (.pt file)
2. Performs temporal 70/30 train/val split
3. Trains MLP regressor to predict policy returns from parameters
4. Logs training metrics (loss, Pearson correlation)
5. Saves model, metrics, and predictions

Usage:
    python -m src.analysis.train_offline_mdp_meta_value \
        --dataset-path analysis/mdp_meta_value_dataset.pt \
        --output-dir analysis/meta_value_model \
        --hidden-dims 256 128 64 \
        --learning-rate 0.001 \
        --epochs 200 \
        --batch-size 32
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class MetaValueNetwork(nn.Module):
    """MLP regressor for predicting policy returns from parameters.

    Architecture:
        Input: policy_params + context features
        Hidden: Configurable MLP layers with ReLU
        Output: Single scalar (predicted return)
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64]):
        """Initialize meta-value network.

        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        # Output layer (single scalar prediction)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features (batch_size, input_dim)

        Returns:
            predictions: Predicted returns (batch_size, 1)
        """
        return self.network(x)


def load_dataset(dataset_path: str) -> Dict[str, Any]:
    """Load MDP meta-value dataset from .pt file."""
    print(f"Loading dataset from {dataset_path}")
    dataset = torch.load(dataset_path, weights_only=False)

    print(f"Dataset metadata:")
    for key, value in dataset['metadata'].items():
        print(f"  {key}: {value}")

    return dataset


def extract_features_and_targets(samples: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix and target vector from samples.

    Args:
        samples: List of dataset samples

    Returns:
        X: Feature matrix (n_samples, feature_dim)
        y: Target vector (n_samples,)
    """
    X_list = []
    y_list = []

    for sample in samples:
        # Core features: policy parameters
        policy_params = sample['policy_params']

        # Additional context features
        context_features = np.array([
            sample['param_norm'],
            sample['param_mean'],
            sample['param_std'],
            sample['training_return_ema'],
            sample['training_episodes_seen'] / 1000.0,  # Normalize episode count
            sample['policy_entropy'],
        ])

        # Concatenate all features
        features = np.concatenate([policy_params, context_features])
        X_list.append(features)

        # Target: average return over evaluation episodes
        y_list.append(sample['target_return'])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    return X, y


def temporal_split(samples: List[Dict[str, Any]], train_ratio: float = 0.7) -> Tuple[List[Dict], List[Dict]]:
    """Split dataset temporally (earlier episodes = train, later = val).

    Args:
        samples: List of dataset samples (should be temporally ordered)
        train_ratio: Fraction for training (default: 0.7)

    Returns:
        train_samples: Training samples
        val_samples: Validation samples
    """
    # Ensure temporal ordering
    samples_sorted = sorted(samples, key=lambda s: (s['run_id'], s['snapshot_episode']))

    n_train = int(len(samples_sorted) * train_ratio)
    train_samples = samples_sorted[:n_train]
    val_samples = samples_sorted[n_train:]

    print(f"\nTemporal split:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val: {len(val_samples)} samples")

    return train_samples, val_samples


def compute_pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    return np.corrcoef(y_true, y_pred)[0, 1]


def train_epoch(model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module,
                X_train: torch.Tensor, y_train: torch.Tensor, batch_size: int) -> float:
    """Train for one epoch.

    Args:
        model: Meta-value network
        optimizer: Optimizer
        criterion: Loss function (MSE)
        X_train: Training features
        y_train: Training targets
        batch_size: Batch size

    Returns:
        avg_loss: Average loss over epoch
    """
    model.train()

    n_samples = X_train.shape[0]
    indices = torch.randperm(n_samples)

    total_loss = 0.0
    n_batches = 0

    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]

        # Forward pass
        optimizer.zero_grad()
        predictions = model(X_batch).squeeze()
        loss = criterion(predictions, y_batch)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model: nn.Module, X_val: torch.Tensor, y_val: torch.Tensor) -> Dict[str, float]:
    """Evaluate model on validation set.

    Args:
        model: Meta-value network
        X_val: Validation features
        y_val: Validation targets

    Returns:
        metrics: Dictionary with loss and correlation
    """
    model.eval()

    with torch.no_grad():
        predictions = model(X_val).squeeze()
        loss = nn.MSELoss()(predictions, y_val).item()

        # Convert to numpy for correlation
        y_val_np = y_val.cpu().numpy()
        predictions_np = predictions.cpu().numpy()

        pearson = compute_pearson_correlation(y_val_np, predictions_np)
        rmse = np.sqrt(loss)
        mae = np.mean(np.abs(y_val_np - predictions_np))

    return {
        'loss': loss,
        'rmse': rmse,
        'mae': mae,
        'pearson': pearson
    }


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dims: List[int],
    learning_rate: float,
    epochs: int,
    batch_size: int,
    device: str = 'cpu'
) -> Tuple[MetaValueNetwork, List[Dict[str, float]]]:
    """Train meta-value network.

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        hidden_dims: Hidden layer dimensions
        learning_rate: Learning rate
        epochs: Number of training epochs
        batch_size: Batch size
        device: Device to train on

    Returns:
        model: Trained model
        history: Training history (metrics per epoch)
    """
    input_dim = X_train.shape[1]

    # Create model
    model = MetaValueNetwork(input_dim=input_dim, hidden_dims=hidden_dims)
    model = model.to(device)

    print(f"\nModel architecture:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dims: {hidden_dims}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training loop
    history = []
    best_val_pearson = -1.0
    best_model_state = None

    print(f"\nTraining for {epochs} epochs...")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'Val Pearson':>12}")
    print("-" * 48)

    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, optimizer, criterion, X_train_t, y_train_t, batch_size)

        # Evaluate
        val_metrics = evaluate(model, X_val_t, y_val_t)

        # Log
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_rmse': val_metrics['rmse'],
            'val_mae': val_metrics['mae'],
            'val_pearson': val_metrics['pearson']
        })

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"{epoch+1:6d} {train_loss:12.6f} {val_metrics['loss']:12.6f} {val_metrics['pearson']:12.4f}")

        # Save best model
        if val_metrics['pearson'] > best_val_pearson:
            best_val_pearson = val_metrics['pearson']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print("-" * 48)
    print(f"Best validation Pearson: {best_val_pearson:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def save_outputs(
    model: MetaValueNetwork,
    history: List[Dict[str, float]],
    samples: List[Dict[str, Any]],
    X_all: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace
):
    """Save model, metrics, and predictions.

    Args:
        model: Trained model
        history: Training history
        samples: All dataset samples
        X_all: All features
        output_dir: Output directory
        args: Command-line arguments
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save model
    model_path = output_dir / 'model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': X_all.shape[1],
        'hidden_dims': args.hidden_dims,
        'training_args': vars(args),
        'timestamp': datetime.now().isoformat(),
    }, model_path)
    print(f"\nSaved model: {model_path}")

    # 2. Save metrics
    metrics_path = output_dir / 'metrics.json'
    final_metrics = history[-1]
    with open(metrics_path, 'w') as f:
        json.dump({
            'final_metrics': final_metrics,
            'best_val_pearson': max(h['val_pearson'] for h in history),
            'training_args': vars(args),
            'n_epochs': len(history),
        }, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    # 3. Save full training history
    history_path = output_dir / 'training_history.csv'
    history_df = pd.DataFrame(history)
    history_df.to_csv(history_path, index=False)
    print(f"Saved history: {history_path}")

    # 4. Save predictions
    model.eval()
    with torch.no_grad():
        X_all_t = torch.tensor(X_all, dtype=torch.float32)
        predictions = model(X_all_t).squeeze().numpy()

    predictions_df = pd.DataFrame({
        'policy_id': [s['policy_id'] for s in samples],
        'run_id': [s['run_id'] for s in samples],
        'snapshot_episode': [s['snapshot_episode'] for s in samples],
        'true_return': [s['target_return'] for s in samples],
        'predicted_return': predictions,
        'error': predictions - np.array([s['target_return'] for s in samples]),
    })

    predictions_path = output_dir / 'predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Saved predictions: {predictions_path}")

    # Print summary statistics
    print(f"\nFinal validation metrics:")
    print(f"  Loss (MSE): {final_metrics['val_loss']:.6f}")
    print(f"  RMSE: {final_metrics['val_rmse']:.6f}")
    print(f"  MAE: {final_metrics['val_mae']:.6f}")
    print(f"  Pearson correlation: {final_metrics['val_pearson']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Train offline meta-value network from MDP dataset'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        required=True,
        help='Path to MDP meta-value dataset (.pt file)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save model and outputs'
    )
    parser.add_argument(
        '--hidden-dims',
        type=int,
        nargs='+',
        default=[256, 128, 64],
        help='Hidden layer dimensions (default: 256 128 64)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='Number of training epochs (default: 200)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Fraction of data for training (default: 0.7)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to train on (default: cpu)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset
    dataset = load_dataset(args.dataset_path)
    samples = dataset['samples']

    print(f"\nDataset info:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Environment: {dataset['metadata']['environment']}")

    # Temporal split
    train_samples, val_samples = temporal_split(samples, train_ratio=args.train_ratio)

    # Extract features and targets
    X_train, y_train = extract_features_and_targets(train_samples)
    X_val, y_val = extract_features_and_targets(val_samples)
    X_all, y_all = extract_features_and_targets(samples)

    print(f"\nFeature dimensions:")
    print(f"  Input features: {X_train.shape[1]}")
    print(f"  Train samples: {X_train.shape[0]}")
    print(f"  Val samples: {X_val.shape[0]}")

    print(f"\nTarget statistics:")
    print(f"  Train: mean={y_train.mean():.3f}, std={y_train.std():.3f}, range=[{y_train.min():.3f}, {y_train.max():.3f}]")
    print(f"  Val: mean={y_val.mean():.3f}, std={y_val.std():.3f}, range=[{y_val.min():.3f}, {y_val.max():.3f}]")

    # Train model
    model, history = train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        hidden_dims=args.hidden_dims,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )

    # Save outputs
    output_dir = Path(args.output_dir)
    save_outputs(model, history, samples, X_all, output_dir, args)

    print(f"\n{'=' * 80}")
    print(f"Training complete!")
    print(f"Model saved to: {output_dir}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
