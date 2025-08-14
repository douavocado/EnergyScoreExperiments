import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPSampler(nn.Module):
    def __init__(self, input_vars, output_vars, spatial_size=16, hidden_size=64, 
                 latent_dim=16, n_layers=2, dropout_rate=0.0, activation_function='relu'):
        super(MLPSampler, self).__init__()
        
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.spatial_size = spatial_size
        self.latent_dim = latent_dim
        
        self.input_size = input_vars * spatial_size * spatial_size
        self.output_size = output_vars * spatial_size * spatial_size
        
        self.activation_functions = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'gelu': nn.GELU(),
            'elu': nn.ELU(),
            'softplus': nn.Softplus()
        }
        
        if activation_function not in self.activation_functions:
            raise ValueError(f"Unsupported activation function: {activation_function}. "
                           f"Supported functions: {list(self.activation_functions.keys())}")
        
        self.activation_fn = self.activation_functions[activation_function]
        
        if isinstance(hidden_size, (list, tuple)):
            self.hidden_sizes = list(hidden_size)
            if len(self.hidden_sizes) != n_layers:
                raise ValueError(f"Length of hidden_sizes list ({len(self.hidden_sizes)}) must match "
                               f"n_layers ({n_layers}).")
        else:
            self.hidden_sizes = [hidden_size] * n_layers
        
        layers = []
        layer_sizes = [self.input_size] + self.hidden_sizes
        
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(self.activation_fn)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        
        self.feature_extractor = nn.Sequential(*layers)
        
        self.output_layer = nn.Linear(self.hidden_sizes[-1] + latent_dim, self.output_size)
        
    def forward(self, x, n_samples=10):
        batch_size = x.shape[0]
        device = x.device
        
        x_flat = x.view(batch_size, -1)
        
        features = self.feature_extractor(x_flat)
        
        noise = torch.randn(batch_size, n_samples, self.latent_dim, device=device)
        
        features_expanded = features.unsqueeze(1).expand(-1, n_samples, -1)
        
        combined = torch.cat([features_expanded, noise], dim=-1)
        
        output_flat = self.output_layer(combined)
        
        output = output_flat.view(batch_size, n_samples, self.output_vars, self.spatial_size, self.spatial_size)
        
        return output
