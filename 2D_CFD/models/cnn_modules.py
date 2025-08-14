import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    def __init__(self, input_channels, base_channels=32, n_layers=3, 
                 kernel_size=3, activation='relu', use_batch_norm=True):
        super(CNNEncoder, self).__init__()
        
        self.input_channels = input_channels
        self.base_channels = base_channels
        self.n_layers = n_layers
        
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU()
        }
        self.activation = activations.get(activation, nn.ReLU())
        
        layers = []
        in_channels = input_channels
        
        for i in range(n_layers):
            out_channels = base_channels * (2 ** i)
            
            layers.append(nn.Conv2d(
                in_channels, out_channels, 
                kernel_size=kernel_size, 
                stride=1,
                padding=kernel_size // 2
            ))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            
            layers.append(self.activation)
            
            in_channels = out_channels
        
        self.encoder = nn.Sequential(*layers)
        self.output_channels = in_channels
        
    def forward(self, x):
        return self.encoder(x)


class CNNDecoder(nn.Module):
    def __init__(self, input_channels, output_channels, base_channels=32, 
                 n_layers=3, kernel_size=3, activation='relu', 
                 use_batch_norm=True, final_activation=None):
        super(CNNDecoder, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_layers = n_layers
        
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        self.activation = activations.get(activation, nn.ReLU())
        self.final_activation = activations.get(final_activation, nn.Identity())
        
        layers = []
        
        channel_sizes = []
        for i in range(n_layers - 1, -1, -1):
            channel_sizes.append(base_channels * (2 ** i))
        
        in_channels = input_channels
        
        for i, out_channels in enumerate(channel_sizes):
            layers.append(nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ))
            
            if use_batch_norm and i < len(channel_sizes) - 1:
                layers.append(nn.BatchNorm2d(out_channels))
            
            if i < len(channel_sizes) - 1:
                layers.append(self.activation)
            
            in_channels = out_channels
        
        layers.append(nn.Conv2d(
            in_channels, output_channels,
            kernel_size=1,
            stride=1,
            padding=0
        ))
        
        layers.append(self.final_activation)
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.decoder(x)


class SpatialBridge(nn.Module):
    def __init__(self, spatial_channels, spatial_size, flat_dim, 
                 use_positional_encoding=True):
        super(SpatialBridge, self).__init__()
        
        self.spatial_channels = spatial_channels
        self.spatial_size = spatial_size
        self.flat_dim = flat_dim
        self.use_positional_encoding = use_positional_encoding
        
        if use_positional_encoding:
            self.register_buffer('pos_encoding', self._create_positional_encoding())
            self.pos_encoding_channels = 16
        else:
            self.pos_encoding_channels = 0
        
        total_channels = spatial_channels + self.pos_encoding_channels
        self.spatial_flat_dim = total_channels * spatial_size * spatial_size
        
        self.projection = nn.Linear(self.spatial_flat_dim, flat_dim)
        
        self.reverse_projection = nn.Linear(flat_dim, self.spatial_flat_dim)
    
    def _create_positional_encoding(self):
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, self.spatial_size),
            torch.linspace(-1, 1, self.spatial_size),
            indexing='ij'
        )
        
        pos_features = []
        for i in range(4):
            freq = 2 ** i
            pos_features.append(torch.sin(freq * torch.pi * x))
            pos_features.append(torch.cos(freq * torch.pi * x))
            pos_features.append(torch.sin(freq * torch.pi * y))
            pos_features.append(torch.cos(freq * torch.pi * y))
        
        pos_encoding = torch.stack(pos_features, dim=0)
        return pos_encoding
    
    def encode(self, spatial_features):
        batch_size = spatial_features.shape[0]
        
        if self.use_positional_encoding:
            pos_enc = self.pos_encoding.unsqueeze(0).expand(batch_size, -1, -1, -1)
            spatial_features = torch.cat([spatial_features, pos_enc], dim=1)
        
        flat = spatial_features.view(batch_size, -1)
        
        return self.projection(flat)
    
    def decode(self, flat_features, target_channels):
        batch_size, n_samples, _ = flat_features.shape
        
        spatial_flat = self.reverse_projection(flat_features)
        
        total_channels = self.spatial_channels + self.pos_encoding_channels
        
        spatial = spatial_flat.view(
            batch_size, n_samples, total_channels, self.spatial_size, self.spatial_size
        )
        
        spatial = spatial[:, :, :target_channels]
        
        return spatial


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, activation='relu', use_batch_norm=True):
        super(ResidualBlock, self).__init__()
        
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU()
        }
        activation_fn = activations.get(activation, nn.ReLU())
        
        layers = []
        
        layers.append(nn.Conv2d(channels, channels, kernel_size, 
                               padding=kernel_size//2))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(channels))
        layers.append(activation_fn)
        
        layers.append(nn.Conv2d(channels, channels, kernel_size, 
                               padding=kernel_size//2))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(channels))
            
        self.block = nn.Sequential(*layers)
        self.activation = activation_fn
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.activation(out)
