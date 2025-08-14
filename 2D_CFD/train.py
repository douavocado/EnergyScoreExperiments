import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import yaml
import argparse
import os
from pathlib import Path
import json
from tqdm import tqdm
import time

from models import MLPSampler, FGNEncoderSampler, SpatialMLPSampler, SpatialFGNEncoderSampler, ConvCNPSampler
from losses import crps_loss, energy_score_loss, opt_proj_loss, optpart_proj_loss
from dataset import CFDDataset
from progressive_sampler import ProgressiveSampleScheduler


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config):
    model_config = config['model']
    data_config = config['data']
    
    n_input_vars = len(data_config['input_vars'])
    n_output_vars = len(data_config['output_vars'])
    
    model_type = model_config['type']
    
    model_params = {k: v for k, v in model_config.items() if k != 'type'}
    
    if model_type == 'MLPSampler':
        model = MLPSampler(
            input_vars=n_input_vars,
            output_vars=n_output_vars,
            **model_params
        )
    elif model_type == 'FGNEncoderSampler':
        model = FGNEncoderSampler(
            input_vars=n_input_vars,
            output_vars=n_output_vars,
            **model_params
        )
    elif model_type == 'SpatialMLPSampler':
        model = SpatialMLPSampler(
            input_vars=n_input_vars,
            output_vars=n_output_vars,
            **model_params
        )
    elif model_type == 'SpatialFGNEncoderSampler':
        model = SpatialFGNEncoderSampler(
            input_vars=n_input_vars,
            output_vars=n_output_vars,
            **model_params
        )
    elif model_type == 'ConvCNPSampler':
        model = ConvCNPSampler(
            input_vars=n_input_vars,
            output_vars=n_output_vars,
            **model_params
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def get_loss_function(loss_name, loss_params=None):
    if loss_params is None:
        loss_params = {}
    
    if loss_name == 'crps':
        return crps_loss
    elif loss_name == 'energy_score':
        return energy_score_loss
    elif loss_name == 'opt_proj':
        def opt_proj_wrapper(predictions, targets):
            return opt_proj_loss(predictions, targets, **loss_params)
        return opt_proj_wrapper
    elif loss_name == 'optpart_proj':
        def optpart_proj_wrapper(predictions, targets):
            return optpart_proj_loss(predictions, targets, **loss_params)
        return optpart_proj_wrapper
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def create_optimizer(model, config):
    train_config = config['training']
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(train_config['learning_rate']),
        weight_decay=float(train_config['weight_decay'])
    )
    
    scheduler = None
    if train_config['scheduler']['type'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_config['num_epochs'] - train_config['scheduler']['warmup_epochs']
        )
    elif train_config['scheduler']['type'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    
    return optimizer, scheduler


def create_progressive_sampler(config):
    train_config = config['training']
    
    if 'progressive_sampling' not in train_config or not train_config['progressive_sampling']['enabled']:
        return None
    
    prog_config = train_config['progressive_sampling']
    
    return ProgressiveSampleScheduler(
        initial_samples=prog_config.get('initial_samples', train_config['n_samples'] // 2),
        max_samples=prog_config.get('max_samples', train_config['n_samples']),
        increase_factor=prog_config.get('increase_factor', 1.5),
        patience=prog_config.get('patience', 10),
        min_delta=prog_config.get('min_delta', 0.001),
        increase_mode=prog_config.get('mode', 'patience')
    )


def train_epoch(model, dataloader, loss_fn, optimizer, n_samples, device):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc="Training")):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        predictions = model(inputs, n_samples=n_samples)
        loss = loss_fn(predictions, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, metrics, n_samples, device):
    model.eval()
    results = {metric: 0.0 for metric in metrics}
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            predictions = model(inputs, n_samples=n_samples)
            
            for metric in metrics:
                if metric == 'crps':
                    score = crps_loss(predictions, targets)
                elif metric == 'energy_score':
                    score = energy_score_loss(predictions, targets)
                elif metric == 'opt_proj':
                    score = opt_proj_loss(predictions, targets)
                elif metric == 'optpart_proj':
                    score = optpart_proj_loss(predictions, targets)
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                results[metric] += score.item()
    
    for metric in results:
        results[metric] /= len(dataloader)
    
    return results


def main(config_path):
    config = load_config(config_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    save_dir = Path(config['logging']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    data_config = config['data']
    train_dataset = CFDDataset(
        file_path=data_config['file_path'],
        input_vars=data_config['input_vars'],
        output_vars=data_config['output_vars'],
        time_skip=data_config['time_skip'],
        split='train',
        train_ratio=data_config['train_ratio'],
        val_ratio=data_config['val_ratio'],
        random_seed=data_config['random_seed']
    )
    
    val_dataset = CFDDataset(
        file_path=data_config['file_path'],
        input_vars=data_config['input_vars'],
        output_vars=data_config['output_vars'],
        time_skip=data_config['time_skip'],
        split='val',
        train_ratio=data_config['train_ratio'],
        val_ratio=data_config['val_ratio'],
        random_seed=data_config['random_seed']
    )
    
    test_dataset = CFDDataset(
        file_path=data_config['file_path'],
        input_vars=data_config['input_vars'],
        output_vars=data_config['output_vars'],
        time_skip=data_config['time_skip'],
        split='test',
        train_ratio=data_config['train_ratio'],
        val_ratio=data_config['val_ratio'],
        random_seed=data_config['random_seed']
    )
    
    train_config = config['training']
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    model = create_model(config)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    loss_params = train_config.get('loss_params', {})
    loss_fn = get_loss_function(train_config['loss'], loss_params)
    
    optimizer, scheduler = create_optimizer(model, config)
    
    progressive_sampler = create_progressive_sampler(config)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    sample_history = []
    
    for epoch in range(train_config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{train_config['num_epochs']}")
        
        if progressive_sampler is not None:
            current_n_samples = progressive_sampler.get_current_samples()
        else:
            current_n_samples = train_config['n_samples']
        
        print(f"Using {current_n_samples} samples for training")
        
        train_loss = train_epoch(
            model, train_loader, loss_fn, optimizer,
            current_n_samples, device
        )
        train_losses.append(train_loss)
        print(f"Train loss: {train_loss:.6f}")
        
        eval_frequency = config.get('evaluation', {}).get('eval_every', 5)
        should_evaluate = ((epoch + 1) % eval_frequency == 0) or (epoch + 1 == train_config['num_epochs'])
        
        if should_evaluate:
            val_metrics = evaluate(
                model, val_loader, config['evaluation']['metrics'],
                config['evaluation']['n_samples'], device
            )
            val_loss = val_metrics[train_config['loss']]
            val_losses.append(val_loss)
            
            print(f"Validation metrics:")
            for metric, value in val_metrics.items():
                print(f"  {metric}: {value:.6f}")
        else:
            print("Skipping validation evaluation this epoch")
            val_loss = None
            if len(val_losses) > 0:
                val_loss = val_losses[-1]
        
        if progressive_sampler is not None:
            if should_evaluate:
                sampler_val_loss = val_loss
                val_loss_is_fresh = True
            elif val_losses:
                sampler_val_loss = val_losses[-1]
                val_loss_is_fresh = False
            else:
                sampler_val_loss = None
                val_loss_is_fresh = False
            
            if sampler_val_loss is not None:
                current_samples, samples_increased = progressive_sampler.step(
                    sampler_val_loss, epoch + 1, val_loss_updated=val_loss_is_fresh
                )
                sample_history.append({
                    'epoch': epoch + 1,
                    'samples': current_samples,
                    'samples_increased': samples_increased,
                    'validation_performed': should_evaluate,
                    'val_loss_fresh': val_loss_is_fresh
                })
        
        if scheduler is not None:
            if epoch >= train_config['scheduler']['warmup_epochs']:
                scheduler.step()
        
        if should_evaluate and val_loss is not None:
            if val_loss < best_val_loss - float(train_config['early_stopping']['min_delta']):
                best_val_loss = val_loss
                patience_counter = 0
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': config
                }
                if progressive_sampler is not None:
                    checkpoint_dict['progressive_sampler_state'] = progressive_sampler.state_dict()
                
                torch.save(checkpoint_dict, save_dir / 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= train_config['early_stopping']['patience']:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        if (epoch + 1) % config['logging']['save_every'] == 0:
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }
            if progressive_sampler is not None:
                checkpoint_dict['progressive_sampler_state'] = progressive_sampler.state_dict()
            
            torch.save(checkpoint_dict, save_dir / f'checkpoint_epoch_{epoch + 1}.pt')
        
        val_loss_history = []
        val_epoch_mapping = []
        for i, (epoch_num, val_loss) in enumerate(zip(range(1, len(train_losses) + 1), val_losses)):
            eval_epoch = ((i + 1) * eval_frequency) if (i + 1) * eval_frequency <= len(train_losses) else len(train_losses)
            val_loss_history.append(val_loss)
            val_epoch_mapping.append(eval_epoch)
        
        history = {
            'train_losses': train_losses,
            'train_epochs': list(range(1, len(train_losses) + 1)),
            'val_losses': val_loss_history,
            'val_epochs': val_epoch_mapping,
            'eval_frequency': eval_frequency,
            'sample_history': sample_history
        }
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    checkpoint = torch.load(save_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\nFinal evaluation on test set:")
    test_metrics = evaluate(
        model, test_loader, config['evaluation']['metrics'],
        config['evaluation']['n_samples'], device
    )
    
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    with open(save_dir / 'test_results.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    print(f"\nTraining completed. Results saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 2D CFD prediction model")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    args = parser.parse_args()
    
    main(args.config)
