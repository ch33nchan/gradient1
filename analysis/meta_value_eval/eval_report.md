# Meta-Value Network Evaluation Report

Generated: 2025-11-20 07:02:33

## Summary Metrics

### Correlation (Goodness of Fit)
- **Pearson correlation**: nan
- **Spearman correlation**: nan
- **R² score**: -inf

### Prediction Error
- **MAE**: 1.0183
- **RMSE**: 1.0183
- **MSE**: 1.036900
- **Max absolute error**: 1.0183

### Distribution Statistics
- **True returns**: mean=0.923, std=0.000
- **Predicted returns**: mean=-0.096, std=0.000

## Calibration Analysis

Calibration measures whether predicted values match true values across different ranges.

| Bucket | N | Pred Range | Pred Mean | True Mean | MAE | Bias |
|--------|---|------------|-----------|-----------|-----|------|
| 0 | 1 | [-0.096, -0.096] | -0.096 | 0.923 | 1.0183 | -1.0183 |

## Interpretation

### Correlation Quality
- ✗ **Poor**: Pearson < 0.80 (weak relationship)

### Prediction Accuracy
- Relative MAE: 110.4% of mean true value
- ✗ High relative error (>20%)

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
