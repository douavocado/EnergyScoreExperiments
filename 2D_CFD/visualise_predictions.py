import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
from pathlib import Path

from models import MLPSampler, FGNEncoderSampler
from dataset import CFDDataset
from train import create_model


def load_model(checkpoint_path, config):
    model = create_model(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def visualize_predictions(model, dataset, sample_idx, config, save_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    inputs, targets = dataset[sample_idx]
    inputs = inputs.unsqueeze(0).to(device)
    targets = targets.unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(inputs, n_samples=100)
    
    inputs = inputs.cpu().numpy()[0]
    targets = targets.cpu().numpy()[0]
    predictions = predictions.cpu().numpy()[0]
    
    pred_mean = predictions.mean(axis=0)
    pred_std = predictions.std(axis=0)
    
    output_vars = config['data']['output_vars']
    input_vars = config['data']['input_vars']
    
    n_output_vars = len(output_vars)
    fig, axes = plt.subplots(n_output_vars, 4, figsize=(16, 4*n_output_vars))
    if n_output_vars == 1:
        axes = axes.reshape(1, -1)
    
    for i, var_name in enumerate(output_vars):
        im1 = axes[i, 0].imshow(targets[i], cmap='RdBu_r')
        axes[i, 0].set_title(f'{var_name} - Ground Truth')
        plt.colorbar(im1, ax=axes[i, 0])
        
        im2 = axes[i, 1].imshow(pred_mean[i], cmap='RdBu_r')
        axes[i, 1].set_title(f'{var_name} - Mean Prediction')
        plt.colorbar(im2, ax=axes[i, 1])
        
        im3 = axes[i, 2].imshow(pred_std[i], cmap='viridis')
        axes[i, 2].set_title(f'{var_name} - Uncertainty (Std)')
        plt.colorbar(im3, ax=axes[i, 2])
        
        error = np.abs(pred_mean[i] - targets[i])
        im4 = axes[i, 3].imshow(error, cmap='hot')
        axes[i, 3].set_title(f'{var_name} - Absolute Error')
        plt.colorbar(im4, ax=axes[i, 3])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    print(f"\nPrediction statistics:")
    for i, var_name in enumerate(output_vars):
        mse = np.mean((pred_mean[i] - targets[i])**2)
        mae = np.mean(np.abs(pred_mean[i] - targets[i]))
        print(f"{var_name}:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  Mean uncertainty (std): {pred_std[i].mean():.6f}")


def main():
    parser = argparse.ArgumentParser(description="Visualize model predictions")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Sample index to visualize')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to use')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save visualization')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    model = load_model(args.checkpoint, config)
    
    data_config = config['data']
    dataset = CFDDataset(
        file_path=data_config['file_path'],
        input_vars=data_config['input_vars'],
        output_vars=data_config['output_vars'],
        time_skip=data_config['time_skip'],
        split=args.split,
        train_ratio=data_config['train_ratio'],
        val_ratio=data_config['val_ratio'],
        random_seed=data_config['random_seed']
    )
    
    visualize_predictions(model, dataset, args.sample_idx, config, args.save_path)


if __name__ == "__main__":
    main()
