# Energy Score Experiments

Code repository for MPhil thesis: "Are Variants of the CRPS Sufficient for Generative Modelling?"

## Overview

This repository contains experimental implementations and evaluations of various scoring rules for probabilistic forecasting and generative modelling, with a focus on variants of the Continuous Ranked Probability Score (CRPS) and Energy Score.

## Structure

### 2D_CFD/
Probabilistic machine learning models for next-timestep prediction of 2D compressible Navier-Stokes CFD simulations.

- **Models**: ConvCNP, MLP, and FGN samplers
- **Losses**: CRPS, Energy Score, and projection-based losses
- **Usage**: Training, evaluation, and comparison of probabilistic forecasting models

### Discrimination/
Synthetic data experiments for discriminating multivariate probabilistic forecasts using various scoring rules.

- **Metrics**: Energy Score, CRPS, Variogram Score, and projection-based variants
- **Data**: Configurable dependence structures via copulas and common-shock methods
- **Purpose**: Evaluate metric discrimination power between different forecast distributions

## Quick Start

```bash
# 2D CFD experiments
cd 2D_CFD
python train.py --config configs/config.yaml

# Discrimination experiments  
cd Discrimination
python main.py --config configs/example.yaml
```

## Key Contributions

- Implementation of projection-based CRPS variants
- Evaluation of scoring rule discrimination power
- Probabilistic forecasting for CFD dynamics
- Comparison of univariate vs multivariate scoring approaches

## Requirements

- Python 3.12
- PyTorch
- NumPy
- SciPy
- h5py
- PyYAML
- matplotlib
