# Discrimination Experiments

Synthetic Data experiments for discriminating multivariate probabilistic forecasts using various scoring rules.

## Usage

```bash
python main.py --config configs/example.yaml
```

## Example Usage

```bash
# Run with default example configuration
python main.py --config configs/example.yaml

# Run with custom seed
python main.py --config configs/example.yaml --seed 42

# Run with specific experiment name
python main.py --config configs/example.yaml --experiment_name my_experiment
```

## Structure

- `main.py` - Main experiment runner
- `metrics/` - Evaluation metrics (energy score, CRPS, variogram score, etc.)
- `data/` - Data generation with copulas and dependence structures
- `configs/` - YAML configuration files
- `ground_truth.py` - Ground truth metric computation
- `visualise.py` - Results visualisation
