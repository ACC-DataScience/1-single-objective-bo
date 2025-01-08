# Single-objective Bayesian Optimization: Alloy Yield Strength

## Overview
This assignment focuses on using Bayesian Optimization (BO) to maximize the yield strength of an alloy by optimizing its composition and processing conditions.

## Background
The yield strength of a metal determines the point at which it begins to plastically deform. Maximizing yield strength is crucial for:
- Aerospace applications
- Automotive industry
- Nautical engineering

A common strengthening mechanism is the formation of vanadium carbide precipitates, which:
- Inhibit atomic plane movement
- Provide thermal and chemical stability
- Form based on specific processing conditions

## Optimization Parameters

| Parameter | Range | Description |
|-----------|--------|-------------|
| Vanadium Content | 1-5 wt% | Weight percentage of vanadium |
| Temperature | 500-1100°C | Aging temperature |
| Time | 0.5-24 hours | Aging duration |
| Process | CR/RX | Cold Rolling (CR) or Recrystallization (RX) |

## Example Usage
```python
t = 12            # hours
temperature = 800 # °C
v_prct = 3       # weight percentage of Vanadium
process = "RX"    # recrystallization
ys = measure_yield_strength(t, temperature, v_prct, process)
print(ys)
```

## Tasks

### Task A: Optimization Setup
Use Honegumi to:
- Generate optimization template
- Configure parameter space
- Set up experiment with 25-trial budget

### Task B: Parameter Optimization
- Find optimal parameters
- Store results in `optimal_params`
- Record best yield strength in `optimal_yield_strength`

### Task C: Feature Importance
- Use `get_feature_importances()`
- Analyze parameter significance
- Store results in `feature_importances`

### Task D: Model Validation
- Perform cross-validation using `cross_validate()`
- Calculate diagnostics with `compute_diagnostic()`
- Record:
  - Correlation coefficient in `corr_coeff`
  - Root mean squared error in `rmse`

### Task E: Stability Analysis
- Analyze parameter perturbations (±3%)
- Generate stability heatmap
- Report minimum performance impact

## Development Setup

### Prerequisites
- Python 3.8+
- Required packages listed in `requirements.txt`

### Installation
```bash
# Environment will be automatically set up in GitHub Codespace
# Manual setup if needed:
pip install -r requirements.txt
```

### Testing
```bash
pytest
```

## Documentation
- [Honegumi Documentation](https://honegumi.readthedocs.io)
- [Honegumi Tutorials](https://honegumi.readthedocs.io/en/latest/tutorials.html)
- [Ax Model Bridge Documentation](https://ax.dev/api/modelbridge.html#ax.modelbridge.cross_validation.compute_diagnostics)

## Notes
- The objective function `measure_yield_strength()` is provided in `utils.py`
- Limited to 25 experimental trials
- Consider practical parameter variations in production settings
