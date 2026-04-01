import itertools
import random
from nas.layers import LAYER_OPTIONS, LayerConfig
from nas.hardware_config import HardwareConfig

class SearchSpace:
    """
    SearchSpace builds and prunes the pool of candidate architectures.
    It generates all 2, 3, and 4-layer combinations from LAYER_OPTIONS
    and applies LLM-guided filtering.
    """
    def __init__(self, hw: HardwareConfig, hints: list[dict]):
        self.hw = hw
        
        # Generate all 2, 3, 4-layer combinations
        all_combos = []
        for r in [2, 3, 4]:
            # Convert tuples from product to lists as requested
            all_combos.extend([list(c) for c in itertools.product(LAYER_OPTIONS, repeat=r)])
        
        self.original_size = len(all_combos)
        print(f"Original search space size: {self.original_size}")
        
        # Apply LLM hints to prune the space
        self.candidates = self._apply_hints(all_combos, hints)
        
        self.pruned_size = len(self.candidates)
        print(f"Pruned search space size: {self.pruned_size}")

    def _apply_hints(self, candidates: list, hints: list[dict]) -> list:
        # Define filter functions for various hint keys
        def filter_prefer_ds(layers):
            return all(l.layer_type != 'Conv2D' for l in layers)

        def filter_small_kernels(layers):
            return all(l.kernel_size is None or l.kernel_size <= 3 for l in layers)

        def filter_avoid_large_channels(layers):
            return all(l.out_channels is None or l.out_channels <= 32 for l in layers)

        def filter_max_3_conv(layers):
            conv_count = sum(1 for l in layers if l.layer_type in ('Conv2D', 'DepthwiseSepConv'))
            return conv_count <= 3

        def filter_max_2_conv(layers):
            conv_count = sum(1 for l in layers if l.layer_type in ('Conv2D', 'DepthwiseSepConv'))
            return conv_count <= 2

        hint_filters = {
            'prefer_depthwise_separable': filter_prefer_ds,
            'use_small_kernels': filter_small_kernels,
            'avoid_large_channels': filter_avoid_large_channels,
            'max_3_conv_blocks': filter_max_3_conv,
            'max_2_conv_blocks': filter_max_2_conv
        }

        pruned = []
        for cand in candidates:
            keep = True
            for h in hints:
                hint_key = h.get('hint')
                priority = h.get('priority', 3)
                
                if hint_key in hint_filters:
                    if not hint_filters[hint_key](cand):
                        if priority == 1:
                            # Hard filter
                            keep = False
                            break
                        elif priority in (2, 3):
                            # Soft filter: 50% drop
                            if random.random() < 0.5:
                                keep = False
                                break
            if keep:
                pruned.append(cand)
        return pruned

    def sample(self) -> list[LayerConfig]:
        if not self.candidates:
            return []
        return random.choice(self.candidates)

    @property
    def size(self) -> int:
        return len(self.candidates)
