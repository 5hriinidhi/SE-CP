from dataclasses import dataclass
from typing import Optional, List

@dataclass
class LayerConfig:
    layer_type: str
    out_channels: Optional[int] = None
    kernel_size: Optional[int] = None
    stride: Optional[int] = None
    padding: Optional[str] = None
    activation: Optional[str] = None
    units: Optional[int] = None
    multiplier: Optional[float] = None
    input_shape: Optional[List[int]] = None

    def param_count(self, in_channels: int) -> int:
        if self.layer_type == 'Conv2D':
            if self.kernel_size is None or self.out_channels is None:
                return 0
            count = (self.kernel_size ** 2) * in_channels * self.out_channels
            return max(0, count)
        
        elif self.layer_type == 'DepthwiseSepConv':
            if self.kernel_size is None or self.out_channels is None:
                return 0
            count = ((self.kernel_size ** 2) * in_channels) + (in_channels * self.out_channels)
            return max(0, count)
            
        elif self.layer_type == 'Dense':
            if self.units is None:
                return 0
            count = in_channels * self.units
            return max(0, count)
            
        return 0

LAYER_OPTIONS = [
    LayerConfig(layer_type='Conv2D', out_channels=8, kernel_size=3, stride=1, padding='same', activation='relu'),
    LayerConfig(layer_type='Conv2D', out_channels=16, kernel_size=3, stride=1, padding='same', activation='relu'),
    LayerConfig(layer_type='Conv2D', out_channels=32, kernel_size=3, stride=1, padding='same', activation='relu'),
    LayerConfig(layer_type='DepthwiseSepConv', out_channels=16, kernel_size=3, stride=1, padding='same', activation='relu'),
    LayerConfig(layer_type='DepthwiseSepConv', out_channels=32, kernel_size=3, stride=1, padding='same', activation='relu'),
    LayerConfig(layer_type='DepthwiseSepConv', out_channels=64, kernel_size=3, stride=1, padding='same', activation='relu'),
    LayerConfig(layer_type='GlobalAvgPool'),
    LayerConfig(layer_type='MaxPool2D', kernel_size=2, stride=2)
]
