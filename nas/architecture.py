import torch
import torch.nn as nn
from nas.layers import LayerConfig

def build_layer(cfg: LayerConfig, in_channels: int) -> nn.Module:
    """
    Factory function to build a PyTorch layer from a LayerConfig.
    """
    l_type = cfg.layer_type
    
    if l_type == 'Conv2D':
        padding = 'same' if cfg.padding == 'same' else 0
        layer = nn.Conv2d(
            in_channels, 
            cfg.out_channels, 
            kernel_size=cfg.kernel_size, 
            stride=cfg.stride or 1, 
            padding=padding
        )
        if cfg.activation in ('relu', 'relu6'):
            return nn.Sequential(layer, nn.ReLU6())
        return layer
        
    elif l_type == 'DepthwiseSepConv':
        padding = 'same' if cfg.padding == 'same' else 0
        # Depthwise + Pointwise + BatchNorm + ReLU6
        return nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, 
                kernel_size=cfg.kernel_size, 
                stride=cfg.stride or 1, 
                padding=padding, 
                groups=in_channels, 
                bias=False
            ),
            nn.Conv2d(in_channels, cfg.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(cfg.out_channels),
            nn.ReLU6()
        )
        
    elif l_type == 'Dense':
        return nn.Linear(in_channels, cfg.units)
        
    elif l_type == 'GlobalAvgPool':
        return nn.AdaptiveAvgPool2d(1)
        
    elif l_type == 'MaxPool2D':
        return nn.MaxPool2d(kernel_size=cfg.kernel_size, stride=cfg.stride)
        
    elif l_type == 'Flatten':
        return nn.Flatten()
        
    elif l_type == 'ReLU' or l_type == 'ReLU6':
        return nn.ReLU6()
        
    elif l_type == 'BatchNorm':
        return nn.BatchNorm2d(in_channels)
        
    elif l_type == 'Dropout':
        return nn.Dropout(p=0.2)
        
    else:
        raise ValueError(f"Unknown layer type: {l_type}")

class Architecture(nn.Module):
    """
    Dynamic PyTorch model built from a list of LayerConfigs.
    """
    def __init__(self, layer_configs: list, num_classes: int, in_channels: int = 1):
        super().__init__()
        
        self.layers = nn.ModuleList()
        curr_ch = in_channels
        
        for cfg in layer_configs:
            layer = build_layer(cfg, curr_ch)
            self.layers.append(layer)
            
            # Track the channel dimension for the next layer or head
            if cfg.out_channels:
                curr_ch = cfg.out_channels
            elif cfg.units:
                curr_ch = cfg.units
            # Note: GAP and MaxPool don't change curr_ch in 2D application
            
        # The head must always exist
        # We use AdaptiveAvgPool2d(1) + Flatten to handle any spatial size
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(curr_ch, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.head(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
