# Research Notes

This directory contains research notes and reports for experiments.

## Format

Each experiment should have a corresponding note file:

```
notes/
  2025-11-13_bandit_self_gradient.md
  2025-11-14_nonstationary_results.md
  ...
```

## Template

Use this template for experiment notes:

```markdown
# Experiment: [Name]

Date: YYYY-MM-DD

## Configuration

- Config file: `experiments/[config_name].yaml`
- Run directory: `logs/[experiment]/run_[timestamp]/`
- Commit hash: [git commit hash]

## Objective

What was this experiment trying to test or demonstrate?

## Setup

- Environment: [description]
- Agent: [description]
- Baselines: [list]
- Key parameters: [list important hyperparameters]

## Results

### Quantitative

- Final reward: X.XX ± Y.YY
- Cumulative regret: XXXX
- Gradient prediction MSE: X.XXXXX
- [other metrics]

### Qualitative

- What patterns emerged?
- How did the agent behave?
- Any unexpected results?

## Observations

- What worked well?
- What didn't work?
- Any bugs or issues discovered?

## Conclusions

What did we learn from this experiment?

## Next Steps

What should be tried next based on these results?

## Changes Made

Document any code changes made during or because of this experiment.
```

## Guidelines

- Document ALL experiments, even failed ones
- Include both quantitative and qualitative observations
- Note any parameter changes or code modifications
- Link to actual run directories for reproducibility
- Be honest about what worked and what didn't
- No fabrication or alteration of results
