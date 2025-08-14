import torch
import torch.nn as nn


class ConditionalLayerNorm(nn.Module):
    def __init__(self, normalized_shape, conditioning_dim):
        super(ConditionalLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.conditioning_dim = conditioning_dim
        
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        
        self.gamma_proj = nn.Linear(conditioning_dim, normalized_shape)
        self.beta_proj = nn.Linear(conditioning_dim, normalized_shape)
        
        self.eps = 1e-5
    
    def forward(self, x, conditioning):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        gamma_cond = self.gamma + self.gamma_proj(conditioning)
        beta_cond = self.beta + self.beta_proj(conditioning)
        
        while len(gamma_cond.shape) < len(x_norm.shape):
            gamma_cond = gamma_cond.unsqueeze(-2)
            beta_cond = beta_cond.unsqueeze(-2)
        
        return gamma_cond * x_norm + beta_cond


class FGNEncoderSampler(nn.Module):
    def __init__(self, input_vars, output_vars, spatial_size=16, hidden_size=64, 
                 latent_dim=16, n_layers=2, dropout_rate=0.0, activation_function='relu'):
        super(FGNEncoderSampler, self).__init__()
        
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.spatial_size = spatial_size
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        
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
        
        self.noise_encoder = nn.Linear(latent_dim, latent_dim)
        
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList() if dropout_rate > 0 else None
        
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        
        for i in range(len(layer_sizes) - 1):
            input_dim = layer_sizes[i]
            output_dim = layer_sizes[i + 1]
            
            if i == 0:
                self.layer_norms.append(ConditionalLayerNorm(input_dim, latent_dim))
            else:
                self.layer_norms.append(ConditionalLayerNorm(input_dim, latent_dim))
            
            self.layers.append(nn.Linear(input_dim, output_dim))
            
            if dropout_rate > 0 and i < len(layer_sizes) - 2:
                self.dropouts.append(nn.Dropout(dropout_rate))
    
    def _generate_encoded_noise(self, batch_size, n_samples, device):
        noise = torch.randn(batch_size, n_samples, self.latent_dim, device=device)
        
        encoded_noise = self.noise_encoder(noise)
        
        return encoded_noise
    
    def forward(self, x, n_samples=10):
        batch_size = x.shape[0]
        device = x.device
        
        x_flat = x.view(batch_size, -1)
        
        encoded_noise = self._generate_encoded_noise(batch_size, n_samples, device)
        
        current_input = x_flat.unsqueeze(1).expand(-1, n_samples, -1)
        
        for i, (layer_norm, linear_layer) in enumerate(zip(self.layer_norms, self.layers)):
            original_shape = current_input.shape
            current_input_flat = current_input.reshape(-1, current_input.shape[-1])
            encoded_noise_flat = encoded_noise.reshape(-1, encoded_noise.shape[-1])
            
            normed_input = layer_norm(current_input_flat, encoded_noise_flat)
            current_input = normed_input.reshape(original_shape)
            
            current_input = linear_layer(current_input)
            
            if i < len(self.layers) - 1:
                current_input = self.activation_fn(current_input)
            
            if self.dropouts is not None and i < len(self.layers) - 1:
                current_input = self.dropouts[i](current_input)
        
        output = current_input.view(batch_size, n_samples, self.output_vars, self.spatial_size, self.spatial_size)
        
        return output
