import torch
import torch.nn.functional as F
import numpy as np
import yaml
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple

from dataset import CFDDataset


def compute_mae_mse(predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    pred_mean = predictions.mean(dim=1)
    
    mae = torch.abs(pred_mean - targets).mean().item()
    mse = ((pred_mean - targets) ** 2).mean().item()
    
    return mae, mse


def compute_variogram_score(predictions: torch.Tensor, targets: torch.Tensor, 
                           max_distance: int = None) -> float:
    batch_size, n_samples, n_vars, height, width = predictions.shape
    
    pred_mean = predictions.mean(dim=1)
    
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, device=predictions.device),
        torch.arange(width, device=predictions.device),
        indexing='ij'
    )
    coords = torch.stack([x_coords, y_coords], dim=-1).float()
    
    coords_flat = coords.view(-1, 2)
    distances = torch.cdist(coords_flat, coords_flat)
    
    weights = 1.0 / (distances + 1e-6)
    weights[torch.eye(height * width, dtype=torch.bool, device=weights.device)] = 0
    
    if max_distance is not None:
        weights[distances > max_distance] = 0
    
    weights = weights / weights.sum()
    
    score = 0
    for b in range(batch_size):
        for v in range(n_vars):
            pred_flat = pred_mean[b, v].flatten()
            target_flat = targets[b, v].flatten()
            
            pred_diff = (pred_flat.unsqueeze(0) - pred_flat.unsqueeze(1)) ** 2
            target_diff = (target_flat.unsqueeze(0) - target_flat.unsqueeze(1)) ** 2
            
            variogram_error = torch.abs(pred_diff - target_diff) * weights
            score += variogram_error.sum().item()
    
    return score / (batch_size * n_vars)


def compute_patched_energy_score(predictions: torch.Tensor, targets: torch.Tensor,
                                patch_size: int = 4) -> float:
    batch_size, n_samples, n_vars, height, width = predictions.shape
    
    predictions_reshaped = predictions.view(batch_size * n_samples, n_vars, height, width)
    pooled_predictions_reshaped = F.avg_pool2d(
        predictions_reshaped,
        kernel_size=patch_size,
        stride=patch_size
    )
    
    _, _, pooled_height, pooled_width = pooled_predictions_reshaped.shape
    pooled_predictions = pooled_predictions_reshaped.view(
        batch_size, n_samples, n_vars, pooled_height, pooled_width
    )
    
    pooled_targets = F.avg_pool2d(
        targets,
        kernel_size=patch_size,
        stride=patch_size
    )
    
    batch_size, n_samples, n_vars, height, width = pooled_predictions.shape
    
    predictions_flat = pooled_predictions.reshape(batch_size, n_samples, -1)
    targets_flat = pooled_targets.reshape(batch_size, -1)
    
    targets_expanded = targets_flat.unsqueeze(1)
    diff_to_target = predictions_flat - targets_expanded
    norm_to_target = torch.norm(diff_to_target, p=2, dim=-1)
    first_term = norm_to_target.mean(dim=1)
    
    second_term = torch.zeros_like(first_term)
    
    energy_score = first_term - 0.5 * second_term
    return energy_score.mean().item()


def compute_pairwise_energy_score(predictions: torch.Tensor, targets: torch.Tensor,
                                 n_pairs: int = 100, seed: int = 42) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    batch_size, n_samples, n_vars, height, width = predictions.shape
    device = predictions.device
    
    offsets = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    all_scores = []
    
    for b in range(batch_size):
        batch_scores = []
        
        for _ in range(n_pairs):
            y = np.random.randint(1, height - 1)
            x = np.random.randint(1, width - 1)
            
            dy, dx = offsets[np.random.randint(len(offsets))]
            y2 = y + dy
            x2 = x + dx
            
            if 0 <= y2 < height and 0 <= x2 < width:
                pred_pair = torch.stack([
                    predictions[b, :, :, y, x],
                    predictions[b, :, :, y2, x2]
                ], dim=-1)
                
                target_pair = torch.stack([
                    targets[b, :, y, x],
                    targets[b, :, y2, x2]
                ], dim=-1)
                
                pred_flat = pred_pair.view(n_samples, -1)
                target_flat = target_pair.flatten()
                
                diff_to_target = pred_flat - target_flat.unsqueeze(0)
                norm_to_target = torch.norm(diff_to_target, p=2, dim=-1)
                first_term = norm_to_target.mean()
                
                second_term = torch.tensor(0.0, device=device)
                
                pair_score = first_term - 0.5 * second_term
                batch_scores.append(pair_score.item())
        
        if batch_scores:
            all_scores.append(np.mean(batch_scores))
    
    return np.mean(all_scores) if all_scores else 0.0


def generate_persistence_predictions(inputs: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
    batch_size, n_vars, height, width = inputs.shape
    
    predictions = inputs.unsqueeze(1).repeat(1, n_samples, 1, 1, 1)
    
    return predictions


def evaluate_persistence(config_path: str, n_eval_samples: int = 100, 
                        batch_size: int = 8, device: str = 'cuda') -> Dict[str, float]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    dataset = CFDDataset(
        file_path=data_config['file_path'],
        input_vars=data_config['input_vars'],
        output_vars=data_config['output_vars'],
        time_skip=data_config['time_skip'],
        split='test',
        train_ratio=data_config['train_ratio'],
        val_ratio=data_config['val_ratio'],
        random_seed=data_config['random_seed']
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    metrics = {
        'mae': [],
        'mse': [],
        'variogram_score': [],
        'patched_energy_score': [],
        'pairwise_energy_score': []
    }
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc="Evaluating Persistence")):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            input_vars = data_config['input_vars']
            output_vars = data_config['output_vars']
            
            if input_vars == output_vars:
                predictions = generate_persistence_predictions(inputs, n_samples=n_eval_samples)
            else:
                output_indices = []
                for out_var in output_vars:
                    if out_var in input_vars:
                        output_indices.append(input_vars.index(out_var))
                    else:
                        raise ValueError(f"Output variable '{out_var}' not found in input variables")
                
                selected_inputs = inputs[:, output_indices, :, :]
                predictions = generate_persistence_predictions(selected_inputs, n_samples=n_eval_samples)
            
            mae, mse = compute_mae_mse(predictions, targets)
            metrics['mae'].append(mae)
            metrics['mse'].append(mse)
            
            variogram = compute_variogram_score(predictions, targets)
            metrics['variogram_score'].append(variogram)
            
            patched_es = compute_patched_energy_score(predictions, targets)
            metrics['patched_energy_score'].append(patched_es)
            
            pairwise_es = compute_pairwise_energy_score(predictions, targets)
            metrics['pairwise_energy_score'].append(pairwise_es)
    
    final_metrics = {
        metric: np.mean(values) for metric, values in metrics.items()
    }
    
    for metric, values in metrics.items():
        final_metrics[f'{metric}_std'] = np.std(values)
    
    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate persistence baseline model")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file (from a trained model)')
    parser.add_argument('--n_samples', type=int, default=100,
                       help='Number of samples to generate for evaluation')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for evaluation')
    parser.add_argument('--save_results', type=str, default=None,
                       help='Path to save evaluation results (JSON format)')
    
    args = parser.parse_args()
    
    print(f"Evaluating persistence baseline using config: {args.config}")
    print(f"Persistence model: future state = current state")
    
    metrics = evaluate_persistence(
        config_path=args.config,
        n_eval_samples=args.n_samples,
        batch_size=args.batch_size,
        device=args.device
    )
    
    print("\n" + "="*50)
    print("PERSISTENCE BASELINE EVALUATION RESULTS")
    print("="*50)
    for metric, value in sorted(metrics.items()):
        if not metric.endswith('_std'):
            std_key = f'{metric}_std'
            if std_key in metrics:
                print(f"{metric:25s}: {value:.6f} ± {metrics[std_key]:.6f}")
            else:
                print(f"{metric:25s}: {value:.6f}")
    
    if args.save_results:
        save_path = Path(args.save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'model_type': 'persistence_baseline',
            'metrics': metrics,
            'evaluation_config': {
                'config_path': str(args.config),
                'n_samples': args.n_samples,
                'batch_size': args.batch_size,
                'device': args.device
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
