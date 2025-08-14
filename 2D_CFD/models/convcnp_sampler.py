import torch
import torch.nn as nn
import torch.nn.functional as F


class SetConv(nn.Module):
    def __init__(self, length_scale: float):
        super().__init__()
        self.log_length_scale = nn.Parameter(
            torch.log(torch.tensor(length_scale, dtype=torch.float32)), 
            requires_grad=True
        )

    def rbf(self, dists: torch.Tensor) -> torch.Tensor:
        length_scale = torch.exp(self.log_length_scale)
        return torch.exp(-dists.pow(2) / (2 * length_scale ** 2))

    def forward(self, x_query: torch.Tensor, x_context: torch.Tensor, y_context: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(x_query, x_context)
        weights = self.rbf(dists)

        density = weights.sum(dim=-1, keepdim=True)

        weights_norm = weights / (density + 1e-8)

        y_query = torch.bmm(weights_norm, y_context)

        return torch.cat([y_query, density], dim=-1)


class GridInterpolator(nn.Module):
    def __init__(self, length_scale: float):
        super().__init__()
        self.log_length_scale = nn.Parameter(
            torch.log(torch.tensor(length_scale, dtype=torch.float32)), 
            requires_grad=True
        )

    def rbf(self, dists: torch.Tensor) -> torch.Tensor:
        length_scale = torch.exp(self.log_length_scale)
        return torch.exp(-dists.pow(2) / (2 * length_scale ** 2))

    def forward(self, x_query: torch.Tensor, x_grid: torch.Tensor, y_grid: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(x_query, x_grid)
        weights = self.rbf(dists)
        weights_norm = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        y_query = torch.bmm(weights_norm, y_grid)
        return y_query


class ResConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class ConvCNPSampler(nn.Module):
    def __init__(self, 
                 input_vars: int,
                 output_vars: int,
                 spatial_size: int = 16,
                 hidden_size: int = 64,
                 latent_dim: int = 16,
                 n_layers: int = 4,
                 dropout_rate: float = 0.0,
                 activation_function: str = 'relu',
                 grid_size: int = 32,
                 length_scale: float = 0.1,
                 use_batch_norm: bool = True):
        super().__init__()
        
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.spatial_size = spatial_size
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        if isinstance(hidden_size, (list, tuple)):
            self.hidden_sizes = list(hidden_size)
            if len(self.hidden_sizes) != n_layers:
                self.hidden_sizes = [self.hidden_sizes[0]] * n_layers
        else:
            self.hidden_sizes = [hidden_size] * n_layers
        
        x_grid = torch.linspace(0, 1, grid_size)
        y_grid = torch.linspace(0, 1, grid_size)
        grid_y, grid_x = torch.meshgrid(y_grid, x_grid, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)
        self.register_buffer('grid', grid)
        
        x_coords = torch.linspace(0, 1, spatial_size)
        y_coords = torch.linspace(0, 1, spatial_size)
        coords_y, coords_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coords = torch.stack([coords_x, coords_y], dim=-1).view(-1, 2)
        self.register_buffer('spatial_coords', coords)
        
        self.encoder_set_conv = SetConv(length_scale=length_scale)
        
        cnn_layers = []
        in_channels = input_vars + 1
        for i in range(n_layers):
            out_channels = self.hidden_sizes[i]
            cnn_layers.append(ResConvBlock(in_channels, out_channels))
            in_channels = out_channels
        self.cnn = nn.Sequential(*cnn_layers)
        
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        self.decoder_interpolator = GridInterpolator(length_scale=length_scale)
        
        final_hidden_size = self.hidden_sizes[-1]
        decoder_layers = []
        
        decoder_hidden = hidden_size if isinstance(hidden_size, int) else hidden_size[0]
        
        decoder_layers.append(nn.Linear(final_hidden_size + latent_dim, decoder_hidden))
        decoder_layers.append(nn.ReLU())
        
        current_size = decoder_hidden
        for _ in range(max(1, n_layers // 2)):
            decoder_layers.append(nn.Linear(current_size, decoder_hidden))
            decoder_layers.append(nn.ReLU())
            current_size = decoder_hidden
        
        decoder_layers.append(nn.Linear(current_size, output_vars))
        
        self.decoder_mlp = nn.Sequential(*decoder_layers)
        
    def forward(self, x, n_samples=1):
        batch_size = x.shape[0]
        device = x.device
        
        x_flat = x.view(batch_size, self.input_vars, -1).permute(0, 2, 1)
        
        input_coords = self.spatial_coords.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        grid_coords = self.grid.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        
        grid_rep = self.encoder_set_conv(grid_coords, input_coords, x_flat)
        
        grid_rep_2d = grid_rep.permute(0, 2, 1).view(
            batch_size, -1, self.grid_size, self.grid_size
        )
        cnn_features = self.cnn(grid_rep_2d)
        cnn_features = self.dropout(cnn_features)
        
        cnn_features_flat = cnn_features.view(
            batch_size, cnn_features.shape[1], -1
        ).permute(0, 2, 1)
        
        target_features = self.decoder_interpolator(
            input_coords, grid_coords, cnn_features_flat
        )
        
        samples_list = []
        for _ in range(n_samples):
            latent = torch.randn(
                batch_size, self.spatial_size * self.spatial_size, self.latent_dim
            ).to(device)
            
            decoder_input = torch.cat([target_features, latent], dim=-1)
            
            decoder_input_flat = decoder_input.view(-1, decoder_input.shape[-1])
            
            output_flat = self.decoder_mlp(decoder_input_flat)
            
            output = output_flat.view(
                batch_size, self.spatial_size, self.spatial_size, self.output_vars
            ).permute(0, 3, 1, 2)
            
            samples_list.append(output)
        
        samples = torch.stack(samples_list, dim=1)
        
        return samples


if __name__ == '__main__':
    batch_size = 4
    input_vars = 3
    output_vars = 1
    spatial_size = 16
    n_samples = 10
    
    model = ConvCNPSampler(
        input_vars=input_vars,
        output_vars=output_vars,
        spatial_size=spatial_size,
        hidden_size=64,
        latent_dim=16,
        n_layers=4,
        grid_size=32,
        length_scale=0.1
    )
    
    x = torch.randn(batch_size, input_vars, spatial_size, spatial_size)
    
    samples = model(x, n_samples=n_samples)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {samples.shape}")
    print(f"Expected shape: ({batch_size}, {n_samples}, {output_vars}, {spatial_size}, {spatial_size})")
