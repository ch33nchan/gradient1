# MDP Planning vs Baseline Comparison

Generated: 2025-11-20 13:17:09

## Experiment Setup

### Baseline
- **Type**: mdp
- **Agent**: REINFORCE
- **Learning rate**: 0.01
- **Hidden dim**: 64
- **Entropy bonus**: 0.01

### Planning
- **Type**: mdp_planning
- **Agent**: REINFORCE + Meta-Value Planning
- **Learning rate**: 0.01
- **Hidden dim**: 64
- **Entropy bonus**: 0.01
- **Planning weight**: 0.1
- **Meta-value model**: analysis/meta_value_model/model.pt

## Results Summary

### Final Performance (last 20 episodes)

| Metric | Baseline | Planning | Difference | Improvement |
|--------|----------|----------|------------|-------------|
| Mean Return | 1.000 ± 0.000 | 1.000 ± 0.000 | +0.000 | +0.0% |
| Success Rate | 100.0% | 100.0% | +0.0% | - |
| Best Return | 1.000 | 1.000 | +0.000 | - |

### Learning Speed

| Metric | Baseline | Planning |
|--------|----------|----------|
| Episodes to 80% success | 10 | 10 |

## Planning Metrics

- **Mean meta-value score**: 0.837
- **Final meta-value score**: 0.815
- **Mean base entropy**: 0.365
- **Mean planned entropy**: 0.420
- **Planning weight**: 0.1

## Interpretation

### Performance Comparison
- ~ **Neutral effect**: Performance difference within ±5%
- ~ Similar success rates (100.0% vs 100.0%)

### Learning Speed
- ~ Similar learning speeds

## Conclusion

**Neutral result**: Planning has minimal impact on performance. This could indicate:
- The baseline is already near-optimal for this task
- Planning weight may need tuning
- Meta-value model may need more training data

## Files Generated
- `comparison.png`: Learning curves comparison
- `summary.md`: This report
