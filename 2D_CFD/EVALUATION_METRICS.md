# Evaluation Metrics for ConvCNP Models

This document describes the comprehensive evaluation metrics implemented in `evaluate_model.py` and the persistence baseline in `evaluate_persistence.py`.

## Usage

### Evaluating Trained Models

```bash
python evaluate_model.py \
    --model_dir experiments/convcnp_energy \
    --n_samples 100 \
    --batch_size 8 \
    --device cuda \
    --save_results evaluation_results.json
```

### Evaluating Persistence Baseline

```bash
python evaluate_persistence.py \
    --config experiments/convcnp_energy/config.yaml \
    --n_samples 100 \
    --batch_size 8 \
    --device cuda \
    --save_results persistence_evaluation.json
```

### Comparing Models

```bash
python compare_models.py \
    --results persistence_evaluation.json model_evaluation.json \
    --names "Persistence" "ConvCNP" \
    --save_table comparison.csv \
    --save_plot comparison.png
```

## Implemented Metrics

### 1. Mean Absolute Error (MAE)
- **Definition**: Average absolute difference between predicted mean and ground truth
- **Formula**: `MAE = mean(|pred_mean - target|)`
- **Interpretation**: Lower is better; measures average prediction error

### 2. Mean Squared Error (MSE)
- **Definition**: Average squared difference between predicted mean and ground truth
- **Formula**: `MSE = mean((pred_mean - target)²)`
- **Interpretation**: Lower is better; penalises large errors more than MAE

### 3. Variogram Score
- **Definition**: Measures how well the model captures spatial correlation structure
- **Implementation**: 
  - Computes pairwise squared differences for all spatial locations
  - Weights by inverse distance (closer points have higher weight)
  - Compares variogram of predictions vs ground truth
- **Interpretation**: Lower is better; indicates better spatial structure preservation

### 4. Patched Energy Score
- **Definition**: Energy score computed on spatially pooled data
- **Implementation**:
  - Pools 16x16 grid to 4x4 using average pooling (4x4 patches)
  - Computes energy score on the reduced resolution
  - Helps evaluate coarse-scale prediction quality
- **Formula**: `ES = E[||X - y||] - 0.5 * E[||X - X'||]`
- **Interpretation**: Lower is better; measures distributional accuracy at coarse scale

### 5. Pairwise Energy Score
- **Definition**: Energy score computed on randomly sampled adjacent cell pairs
- **Implementation**:
  - Randomly samples N points on the grid
  - For each point, selects a random adjacent cell (4-connected)
  - Computes 2D energy score for each pair
  - Aggregates over all pairs
- **Interpretation**: Lower is better; evaluates local spatial dependencies

## Output Format

The evaluation script outputs:
1. Console display of all metrics with standard deviations
2. Optional JSON file with detailed results and metadata

Example output:
```json
{
    "metrics": {
        "mae": 0.023456,
        "mae_std": 0.002345,
        "mse": 0.001234,
        "mse_std": 0.000234,
        "variogram_score": 0.045678,
        "variogram_score_std": 0.004567,
        "patched_energy_score": 0.123456,
        "patched_energy_score_std": 0.012345,
        "pairwise_energy_score": 0.234567,
        "pairwise_energy_score_std": 0.023456
    },
    "evaluation_config": {
        "model_dir": "experiments/convcnp_energy",
        "n_samples": 100,
        "batch_size": 8,
        "device": "cuda"
    }
}
```

## Notes

- All metrics are computed on the test set
- Standard deviations are computed across batches
- The variogram score uses inverse distance weighting by default
- Patched energy score uses 4x4 average pooling (16x16 → 4x4)
- Pairwise energy score samples 100 pairs per batch by default

## Persistence Baseline

The persistence model is a simple baseline that predicts the future state will be identical to the current state:
- **Prediction**: `y(t+1) = y(t)`
- **Energy Score**: Since all samples are identical, the second term (sample diversity) is 0
- **Use Case**: Provides a simple baseline to compare against learned models
- **Interpretation**: Models should outperform persistence to demonstrate they've learned meaningful dynamics
