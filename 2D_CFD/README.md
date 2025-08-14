# 2D CFD Next-Step Prediction

Probabilistic machine learning models for next-timestep prediction of 2D compressible Navier-Stokes CFD simulations.

## Quick Start

### Training

```bash
# Train with CRPS loss
python train.py --config configs/config.yaml

# Train with Energy Score loss
python train.py --config configs/energy_config.yaml
```

### Evaluation

```bash
# Evaluate trained model
python evaluate_model.py \
    --model_dir experiments/crps_uni \
    --n_samples 100 \
    --save_results results.json

# Evaluate persistence baseline
python evaluate_persistence.py \
    --config experiments/crps_uni/config.yaml \
    --n_samples 100 \
    --save_results persistence.json

# Compare models
python compare_models.py \
    --results persistence.json results.json \
    --names "Persistence" "ConvCNP" \
    --save_table comparison.csv
```

### Visualisation

```bash
python visualise_predictions.py \
    --checkpoint experiments/crps_uni/best_model.pt \
    --config experiments/crps_uni/config.yaml \
    --sample_idx 0 \
    --split test
```

## Models

- **ConvCNPSampler**: Convolutional Conditional Neural Process
- **MLPSampler**: Multi-layer perceptron with noise injection
- **FGNEncoderSampler**: Feature-based Gaussian Noise encoder

## Data Format

- Input: `(batch_size, input_vars, 16, 16)`
- Output: `(batch_size, n_samples, output_vars, 16, 16)`
- Variables: Vx, Vy, density, pressure
