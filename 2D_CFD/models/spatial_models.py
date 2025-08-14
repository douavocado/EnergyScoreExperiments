import torch
import torch.nn as nn
from .mlp_sampler import MLPSampler
from .fgn_encoder_sampler import FGNEncoderSampler
from .cnn_modules import CNNEncoder, CNNDecoder, SpatialBridge


class SpatialMLPSampler(nn.Module):
    def __init__(self, input_vars, output_vars, spatial_size=16,
                 cnn_base_channels=32, cnn_n_layers=3, cnn_kernel_size=3,
                 cnn_activation='relu', use_batch_norm=True,
                 use_positional_encoding=True,
                 hidden_size=64, latent_dim=16, n_layers=2, 
                 dropout_rate=0.0, activation_function='relu'):
        super(SpatialMLPSampler, self).__init__()
        
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.spatial_size = spatial_size
        
        self.input_cnn_encoder = CNNEncoder(
            input_channels=input_vars,
            base_channels=cnn_base_channels,
            n_layers=cnn_n_layers,
            kernel_size=cnn_kernel_size,
            activation=cnn_activation,
            use_batch_norm=use_batch_norm
        )
        
        self.output_cnn_encoder = CNNEncoder(
            input_channels=output_vars,
            base_channels=cnn_base_channels,
            n_layers=cnn_n_layers,
            kernel_size=cnn_kernel_size,
            activation=cnn_activation,
            use_batch_norm=use_batch_norm
        )
        
        cnn_output_channels = cnn_base_channels * (2 ** (cnn_n_layers - 1))
        
        self.cnn_decoder = CNNDecoder(
            input_channels=cnn_output_channels,
            output_channels=output_vars,
            base_channels=cnn_base_channels,
            n_layers=cnn_n_layers,
            kernel_size=cnn_kernel_size,
            activation=cnn_activation,
            use_batch_norm=use_batch_norm
        )
        
        mlp_input_dim = input_vars * spatial_size * spatial_size
        self.spatial_bridge = SpatialBridge(
            spatial_channels=cnn_output_channels,
            spatial_size=spatial_size,
            flat_dim=mlp_input_dim,
            use_positional_encoding=use_positional_encoding
        )
        
        self.mlp_sampler = MLPSampler(
            input_vars=input_vars,
            output_vars=output_vars,
            spatial_size=spatial_size,
            hidden_size=hidden_size,
            latent_dim=latent_dim,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            activation_function=activation_function
        )
        
    def forward(self, x, n_samples=10):
        batch_size = x.shape[0]
        
        spatial_features = self.input_cnn_encoder(x)
        
        flat_features = self.spatial_bridge.encode(spatial_features)
        
        flat_input = flat_features.view(batch_size, self.input_vars, 
                                       self.spatial_size, self.spatial_size)
        
        mlp_output = self.mlp_sampler(flat_input, n_samples)
        
        B, N, C, H, W = mlp_output.shape
        mlp_flat = mlp_output.view(B * N, C, H, W)
        
        spatial_encoded = self.output_cnn_encoder(mlp_flat)
        
        decoded = self.cnn_decoder(spatial_encoded)
        
        output = decoded.view(B, N, self.output_vars, H, W)
        
        return output


class SpatialFGNEncoderSampler(nn.Module):
    def __init__(self, input_vars, output_vars, spatial_size=16,
                 cnn_base_channels=32, cnn_n_layers=3, cnn_kernel_size=3,
                 cnn_activation='relu', use_batch_norm=True,
                 use_positional_encoding=True,
                 hidden_size=64, latent_dim=16, n_layers=2,
                 dropout_rate=0.0, activation_function='relu'):
        super(SpatialFGNEncoderSampler, self).__init__()
        
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.spatial_size = spatial_size
        
        self.input_cnn_encoder = CNNEncoder(
            input_channels=input_vars,
            base_channels=cnn_base_channels,
            n_layers=cnn_n_layers,
            kernel_size=cnn_kernel_size,
            activation=cnn_activation,
            use_batch_norm=use_batch_norm
        )
        
        self.output_cnn_encoder = CNNEncoder(
            input_channels=output_vars,
            base_channels=cnn_base_channels,
            n_layers=cnn_n_layers,
            kernel_size=cnn_kernel_size,
            activation=cnn_activation,
            use_batch_norm=use_batch_norm
        )
        
        cnn_output_channels = cnn_base_channels * (2 ** (cnn_n_layers - 1))
        
        self.cnn_decoder = CNNDecoder(
            input_channels=cnn_output_channels,
            output_channels=output_vars,
            base_channels=cnn_base_channels,
            n_layers=cnn_n_layers,
            kernel_size=cnn_kernel_size,
            activation=cnn_activation,
            use_batch_norm=use_batch_norm
        )
        
        fgn_input_dim = input_vars * spatial_size * spatial_size
        self.spatial_bridge = SpatialBridge(
            spatial_channels=cnn_output_channels,
            spatial_size=spatial_size,
            flat_dim=fgn_input_dim,
            use_positional_encoding=use_positional_encoding
        )
        
        self.fgn_sampler = FGNEncoderSampler(
            input_vars=input_vars,
            output_vars=output_vars,
            spatial_size=spatial_size,
            hidden_size=hidden_size,
            latent_dim=latent_dim,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            activation_function=activation_function
        )
        
    def forward(self, x, n_samples=10):
        batch_size = x.shape[0]
        
        spatial_features = self.input_cnn_encoder(x)
        
        flat_features = self.spatial_bridge.encode(spatial_features)
        
        flat_input = flat_features.view(batch_size, self.input_vars, 
                                       self.spatial_size, self.spatial_size)
        
        fgn_output = self.fgn_sampler(flat_input, n_samples)
        
        B, N, C, H, W = fgn_output.shape
        fgn_flat = fgn_output.view(B * N, C, H, W)
        
        spatial_encoded = self.output_cnn_encoder(fgn_flat)
        
        decoded = self.cnn_decoder(spatial_encoded)
        
        output = decoded.view(B, N, self.output_vars, H, W)
        
        return output


class UNetStyleDecoder(nn.Module):
    def __init__(self, skip_channels, latent_channels, output_channels,
                 base_channels=32, n_layers=3, kernel_size=3,
                 activation='relu', use_batch_norm=True):
        super(UNetStyleDecoder, self).__init__()
        
        self.n_layers = n_layers
        
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU()
        }
        self.activation = activations.get(activation, nn.ReLU())
        
        self.decode_blocks = nn.ModuleList()
        self.skip_projections = nn.ModuleList()
        
        in_channels = latent_channels
        
        for i in range(n_layers - 1, -1, -1):
            out_channels = base_channels * (2 ** i)
            skip_ch = base_channels * (2 ** i)
            
            self.skip_projections.append(
                nn.Conv2d(skip_ch, out_channels, kernel_size=1)
            )
            
            block = []
            block.append(nn.Conv2d(
                in_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ))
            
            if use_batch_norm:
                block.append(nn.BatchNorm2d(out_channels))
            block.append(self.activation)
            
            self.decode_blocks.append(nn.Sequential(*block))
            in_channels = out_channels
        
        self.final_conv = nn.Conv2d(in_channels, output_channels, kernel_size=1)
        
    def forward(self, latent_features, skip_features):
        x = latent_features
        
        for i, (decode_block, skip_proj) in enumerate(
            zip(self.decode_blocks, self.skip_projections)
        ):
            skip = skip_proj(skip_features[i])
            
            x = torch.cat([x, skip], dim=1)
            x = decode_block(x)
        
        output = self.final_conv(x)
        
        return output
