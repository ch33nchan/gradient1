"""Evaluate trained meta-value network V(θ).

This script:
1. Loads trained meta-value model
2. Loads test/validation dataset
3. Computes comprehensive evaluation metrics:
   - Pearson correlation (linear relationship)
   - Spearman correlation (rank order)
   - RMSE, MAE (prediction error)
   - Calibration analysis (predicted vs actual in buckets)
4. Generates evaluation report

Usage:
    python -m src.analysis.evaluate_mdp_meta_value \
        --dataset-path analysis/mdp_meta_value_dataset.pt \
        --model-path analysis/meta_value_model/model.pt \
        --output-dir analysis/meta_value_eval
"""

import argparse
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
from scipy.stats import spearmanr
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.json_utils import convert_to_json_serializable
from src.analysis.train_offline_mdp_meta_value import (
    MetaValueNetwork,
    extract_features_and_targets,
    temporal_split
)


def load_model(model_path: str, device: str = 'cpu') -> MetaValueNetwork:
    """Load trained meta-value model."""
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, weights_only=False)

    model = MetaValueNetwork(
        input_dim=checkpoint['input_dim'],
        hidden_dims=checkpoint['hidden_dims']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"  Input dim: {checkpoint['input_dim']}")
    print(f"  Hidden dims: {checkpoint['hidden_dims']}")
    print(f"  Trained at: {checkpoint.get('timestamp', 'N/A')}")

    return model


def compute_predictions(model: MetaValueNetwork, X: np.ndarray, device: str = 'cpu') -> np.ndarray:
    """Compute predictions for all samples."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        predictions = model(X_t).squeeze().cpu().numpy()
    return predictions


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics.

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        metrics: Dictionary with all evaluation metrics
    """
    # Prediction errors
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2

    # Basic metrics
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(squared_errors))
    mse = np.mean(squared_errors)

    # Correlation metrics
    pearson = np.corrcoef(y_true, y_pred)[0, 1]
    spearman, _ = spearmanr(y_true, y_pred)

    # R² score
    ss_res = np.sum(squared_errors)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Relative metrics
    mean_abs_error_relative = mae / np.mean(np.abs(y_true)) if np.mean(np.abs(y_true)) > 0 else float('inf')

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mse': mse,
        'pearson': pearson,
        'spearman': spearman,
        'r2': r2,
        'mean_abs_error_relative': mean_abs_error_relative,
        'max_abs_error': np.max(abs_errors),
        'mean_true': np.mean(y_true),
        'std_true': np.std(y_true),
        'mean_pred': np.mean(y_pred),
        'std_pred': np.std(y_pred),
    }

    return metrics


def analyze_calibration(y_true: np.ndarray, y_pred: np.ndarray, n_buckets: int = 5) -> pd.DataFrame:
    """Analyze calibration by bucketing predictions.

    Args:
        y_true: True target values
        y_pred: Predicted values
        n_buckets: Number of buckets

    Returns:
        calibration_df: DataFrame with bucket statistics
    """
    # Sort by predictions and create buckets
    sorted_indices = np.argsort(y_pred)
    bucket_size = len(y_pred) // n_buckets

    calibration_data = []

    for i in range(n_buckets):
        start_idx = i * bucket_size
        end_idx = (i + 1) * bucket_size if i < n_buckets - 1 else len(y_pred)

        bucket_indices = sorted_indices[start_idx:end_idx]

        bucket_data = {
            'bucket': i + 1,
            'n_samples': len(bucket_indices),
            'pred_min': np.min(y_pred[bucket_indices]),
            'pred_max': np.max(y_pred[bucket_indices]),
            'pred_mean': np.mean(y_pred[bucket_indices]),
            'true_mean': np.mean(y_true[bucket_indices]),
            'mae': np.mean(np.abs(y_pred[bucket_indices] - y_true[bucket_indices])),
            'bias': np.mean(y_pred[bucket_indices] - y_true[bucket_indices]),
        }

        calibration_data.append(bucket_data)

    return pd.DataFrame(calibration_data)


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str = "Meta-Value Predictions"
):
    """Plot predicted vs true values.

    Args:
        y_true: True target values
        y_pred: Predicted values
        output_path: Path to save plot
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')

    # Compute metrics for annotation
    pearson = np.corrcoef(y_true, y_pred)[0, 1]
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

    axes[0].set_xlabel('True Return')
    axes[0].set_ylabel('Predicted Return')
    axes[0].set_title(f'{title}\nPearson: {pearson:.3f}, RMSE: {rmse:.4f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Residuals
    residuals = y_pred - y_true
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('Predicted Return')
    axes[1].set_ylabel('Residual (Pred - True)')
    axes[1].set_title('Residual Plot')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close(fig)


def plot_calibration(calibration_df: pd.DataFrame, output_path: Path):
    """Plot calibration analysis.

    Args:
        calibration_df: Calibration DataFrame from analyze_calibration
        output_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Predicted vs True (bucket means)
    axes[0].plot(calibration_df['pred_mean'], calibration_df['true_mean'], 'o-', markersize=8)

    min_val = min(calibration_df['pred_mean'].min(), calibration_df['true_mean'].min())
    max_val = max(calibration_df['pred_mean'].max(), calibration_df['true_mean'].max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect calibration')

    axes[0].set_xlabel('Mean Predicted Return (Bucket)')
    axes[0].set_ylabel('Mean True Return (Bucket)')
    axes[0].set_title('Calibration: Predicted vs True by Bucket')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Bucket-wise MAE
    axes[1].bar(calibration_df['bucket'], calibration_df['mae'])
    axes[1].set_xlabel('Bucket')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Prediction Error by Bucket')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved calibration plot: {output_path}")
    plt.close(fig)


def generate_report(
    metrics: Dict[str, float],
    calibration_df: pd.DataFrame,
    output_path: Path
):
    """Generate markdown evaluation report.

    Args:
        metrics: Evaluation metrics
        calibration_df: Calibration DataFrame
        output_path: Path to save report
    """
    report = f"""# Meta-Value Network Evaluation Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Metrics

### Correlation (Goodness of Fit)
- **Pearson correlation**: {metrics['pearson']:.4f}
- **Spearman correlation**: {metrics['spearman']:.4f}
- **R² score**: {metrics['r2']:.4f}

### Prediction Error
- **MAE**: {metrics['mae']:.4f}
- **RMSE**: {metrics['rmse']:.4f}
- **MSE**: {metrics['mse']:.6f}
- **Max absolute error**: {metrics['max_abs_error']:.4f}

### Distribution Statistics
- **True returns**: mean={metrics['mean_true']:.3f}, std={metrics['std_true']:.3f}
- **Predicted returns**: mean={metrics['mean_pred']:.3f}, std={metrics['std_pred']:.3f}

## Calibration Analysis

Calibration measures whether predicted values match true values across different ranges.

| Bucket | N | Pred Range | Pred Mean | True Mean | MAE | Bias |
|--------|---|------------|-----------|-----------|-----|------|
"""

    for _, row in calibration_df.iterrows():
        report += f"| {row['bucket']:.0f} | {row['n_samples']:.0f} | "
        report += f"[{row['pred_min']:.3f}, {row['pred_max']:.3f}] | "
        report += f"{row['pred_mean']:.3f} | {row['true_mean']:.3f} | "
        report += f"{row['mae']:.4f} | {row['bias']:.4f} |\n"

    report += f"""
## Interpretation

### Correlation Quality
"""

    if metrics['pearson'] >= 0.95:
        report += "- ✓ **Excellent**: Pearson ≥ 0.95 (strong linear relationship)\n"
    elif metrics['pearson'] >= 0.80:
        report += "- ~ **Good**: Pearson ≥ 0.80 (moderate-strong relationship)\n"
    else:
        report += "- ✗ **Poor**: Pearson < 0.80 (weak relationship)\n"

    report += f"""
### Prediction Accuracy
- Relative MAE: {metrics['mean_abs_error_relative']:.1%} of mean true value
"""

    if metrics['mean_abs_error_relative'] < 0.10:
        report += "- ✓ Predictions within 10% of true values (excellent)\n"
    elif metrics['mean_abs_error_relative'] < 0.20:
        report += "- ~ Predictions within 20% of true values (acceptable)\n"
    else:
        report += "- ✗ High relative error (>20%)\n"

    report += """
### Usage Recommendations

**For policy selection (ranking)**:
- Use Spearman correlation as primary metric
- Spearman ≥ 0.90: Excellent for ranking policies
- Spearman ≥ 0.80: Good for coarse ranking

**For quantitative predictions**:
- Use Pearson correlation and RMSE
- Pearson ≥ 0.95 + low RMSE: Suitable for quantitative use
- Otherwise: Use only for relative comparisons

## Files Generated
- `eval_metrics.json`: Machine-readable metrics
- `predictions_vs_true.png`: Scatter and residual plots
- `calibration.png`: Calibration analysis plots
- `eval_report.md`: This report
"""

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Saved report: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained meta-value network'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        required=True,
        help='Path to MDP meta-value dataset (.pt file)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model (.pt file)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save evaluation outputs'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['train', 'val', 'all'],
        help='Which split to evaluate (default: val)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Train ratio for temporal split (default: 0.7)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use (default: cpu)'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.model_path, device=args.device)

    # Load dataset
    print(f"\nLoading dataset from {args.dataset_path}")
    dataset = torch.load(args.dataset_path, weights_only=False)
    samples = dataset['samples']

    # Select split
    if args.split == 'all':
        eval_samples = samples
        split_name = 'All'
    else:
        train_samples, val_samples = temporal_split(samples, train_ratio=args.train_ratio)
        if args.split == 'train':
            eval_samples = train_samples
            split_name = 'Train'
        else:
            eval_samples = val_samples
            split_name = 'Validation'

    print(f"\nEvaluating on {split_name} split: {len(eval_samples)} samples")

    # Extract features and targets
    X, y_true = extract_features_and_targets(eval_samples)

    # Compute predictions
    print("Computing predictions...")
    y_pred = compute_predictions(model, X, device=args.device)

    # Compute metrics
    print("\nComputing evaluation metrics...")
    metrics = compute_metrics(y_true, y_pred)

    print(f"\nEvaluation Results ({split_name} split):")
    print(f"{'=' * 60}")
    print(f"Correlation:")
    print(f"  Pearson:  {metrics['pearson']:.4f}")
    print(f"  Spearman: {metrics['spearman']:.4f}")
    print(f"  R²:       {metrics['r2']:.4f}")
    print(f"\nPrediction Error:")
    print(f"  MAE:      {metrics['mae']:.4f}")
    print(f"  RMSE:     {metrics['rmse']:.4f}")
    print(f"  Max err:  {metrics['max_abs_error']:.4f}")
    print(f"{'=' * 60}")

    # Save metrics
    metrics_path = output_dir / 'eval_metrics.json'
    metrics_dict = {
        'split': args.split,
        'n_samples': len(eval_samples),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
    }
    # Convert numpy/torch types to JSON-serializable Python types
    with open(metrics_path, 'w') as f:
        json.dump(convert_to_json_serializable(metrics_dict), f, indent=2)
    print(f"\nSaved metrics: {metrics_path}")

    # Calibration analysis
    print("\nAnalyzing calibration...")
    calibration_df = analyze_calibration(y_true, y_pred, n_buckets=5)
    calibration_path = output_dir / 'calibration.csv'
    calibration_df.to_csv(calibration_path, index=False)
    print(f"Saved calibration: {calibration_path}")

    # Generate plots
    print("\nGenerating plots...")
    plot_predictions(
        y_true, y_pred,
        output_dir / 'predictions_vs_true.png',
        title=f'Meta-Value Predictions ({split_name} Split)'
    )
    plot_calibration(calibration_df, output_dir / 'calibration.png')

    # Generate report
    print("\nGenerating report...")
    generate_report(metrics, calibration_df, output_dir / 'eval_report.md')

    print(f"\n{'=' * 80}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
