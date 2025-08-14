import torch
from torch.utils.data import Dataset
import h5py
import numpy as np


class CFDDataset(Dataset):
    def __init__(self, file_path, input_vars, output_vars, time_skip=1, 
                 split='train', train_ratio=0.7, val_ratio=0.15, random_seed=42):
        self.file_path = file_path
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.time_skip = time_skip
        self.split = split
        
        self.var_names = ['Vx', 'Vy', 'density', 'pressure']
        self.input_var_indices = [self.var_names.index(var) for var in input_vars]
        self.output_var_indices = [self.var_names.index(var) for var in output_vars]
        
        with h5py.File(file_path, 'r') as f:
            self.data = []
            for var_name in self.var_names:
                self.data.append(f[var_name][:])
            self.data = np.stack(self.data, axis=2)
            
            self.n_samples, self.n_timesteps, self.n_vars, self.height, self.width = self.data.shape
            
        self.valid_time_indices = list(range(self.n_timesteps - time_skip))
        
        all_pairs = []
        for sample_idx in range(self.n_samples):
            for time_idx in self.valid_time_indices:
                all_pairs.append((sample_idx, time_idx))
        
        np.random.seed(random_seed)
        np.random.shuffle(all_pairs)
        
        n_total = len(all_pairs)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        if split == 'train':
            self.pairs = all_pairs[:n_train]
        elif split == 'val':
            self.pairs = all_pairs[n_train:n_train + n_val]
        elif split == 'test':
            self.pairs = all_pairs[n_train + n_val:]
        else:
            raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'val', 'test']")
        
        print(f"Created {split} dataset with {len(self.pairs)} samples")
        print(f"Input variables: {input_vars}")
        print(f"Output variables: {output_vars}")
        print(f"Time skip: {time_skip}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        sample_idx, time_idx = self.pairs[idx]
        
        input_data = self.data[sample_idx, time_idx, self.input_var_indices]
        
        output_time_idx = time_idx + self.time_skip
        output_data = self.data[sample_idx, output_time_idx, self.output_var_indices]
        
        input_tensor = torch.from_numpy(input_data).float()
        output_tensor = torch.from_numpy(output_data).float()
        
        return input_tensor, output_tensor
    
    def get_normalization_stats(self):
        input_stats = {}
        for i, var_name in enumerate(self.input_vars):
            var_idx = self.var_names.index(var_name)
            var_data = self.data[:, :, var_idx].flatten()
            input_stats[var_name] = (float(np.mean(var_data)), float(np.std(var_data)))
        
        output_stats = {}
        for i, var_name in enumerate(self.output_vars):
            var_idx = self.var_names.index(var_name)
            var_data = self.data[:, :, var_idx].flatten()
            output_stats[var_name] = (float(np.mean(var_data)), float(np.std(var_data)))
        
        return input_stats, output_stats
