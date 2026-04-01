from nas.hardware_config import HardwareConfig
from nas.layers import LayerConfig

# Module-level LUT as requested
OP_LUT = {
    'Conv2D': {'ms_per_mflop': 0.042, 'bytes_per_param': 4.0},
    'DepthwiseSepConv': {'ms_per_mflop': 0.018, 'bytes_per_param': 4.0},
    'Dense': {'ms_per_mflop': 0.035, 'bytes_per_param': 4.0},
    'GlobalAvgPool': {'ms_per_mflop': 0.001, 'bytes_per_param': 0.0},
    'BatchNorm': {'ms_per_mflop': 0.001, 'bytes_per_param': 4.0},
    'Dropout': {'ms_per_mflop': 0.0, 'bytes_per_param': 0.0},
    'MaxPool2D': {'ms_per_mflop': 0.001, 'bytes_per_param': 0.0},
    'ReLU': {'ms_per_mflop': 0.0, 'bytes_per_param': 0.0},
    'Flatten': {'ms_per_mflop': 0.0, 'bytes_per_param': 0.0},
}

INT8_COMPRESSION = 0.25

class LatencySimulator:
    """
    LatencySimulator estimates inference metrics (latency, memory, flash)
    for a given architecture on a specific hardware target.
    """
    def __init__(self, hw: HardwareConfig):
        self.hw = hw

    def estimate(self, layers: list[LayerConfig], input_shape=(1, 40, 40)) -> dict:
        curr_c, curr_h, curr_w = input_shape
        
        total_latency = 0.0
        total_params = 0
        peak_ram_kb = 0.0
        per_layer_breakdown = []
        
        # Initial tensor size in KB
        prev_tensor_kb = (curr_c * curr_h * curr_w * 4) / 1024.0
        
        for i, config in enumerate(layers):
            layer_type = config.layer_type
            lut = OP_LUT.get(layer_type, {'ms_per_mflop': 0.001, 'bytes_per_param': 0.0})
            
            # Param count
            params = config.param_count(curr_c)
            total_params += params
            
            # Calculate output shape and FLOPs
            k = config.kernel_size or 1
            s = config.stride or 1
            out_c = config.out_channels or curr_c
            if layer_type == 'Dense':
                out_c = config.units or out_c
            
            if layer_type in ['Conv2D', 'DepthwiseSepConv', 'MaxPool2D']:
                out_h, out_w = curr_h // s, curr_w // s
            elif layer_type == 'GlobalAvgPool':
                out_h, out_w = 1, 1
            elif layer_type == 'Flatten':
                out_h, out_w = 1, 1
                out_c = curr_c * curr_h * curr_w
            else:
                out_h, out_w = curr_h, curr_w
            
            # FLOPs calculation
            flops = 0
            if layer_type == 'Conv2D':
                flops = k * k * curr_c * out_c * out_h * out_w
            elif layer_type == 'DepthwiseSepConv':
                flops = (k * k * curr_c + curr_c * out_c) * out_h * out_w
            elif layer_type == 'Dense':
                flops = curr_c * out_c # in_features * units
                
            # Latency (ms)
            mflops = flops / 1e6
            layer_latency = (mflops * lut['ms_per_mflop']) * (240.0 / self.hw.mhz)
            total_latency += layer_latency
            
            # Layer Model Size (KB) - using uncompressed params for breakdown
            layer_size_kb = (params * lut['bytes_per_param']) / 1024.0
            
            # RAM tracking: peak RAM is max(input + output) at any layer
            curr_tensor_kb = (out_c * out_h * out_w * 4) / 1024.0
            peak_ram_kb = max(peak_ram_kb, prev_tensor_kb + curr_tensor_kb)
            
            per_layer_breakdown.append({
                "layer_index": i,
                "type": layer_type,
                "latency_ms": round(layer_latency, 4),
                "size_kb": round(layer_size_kb * INT8_COMPRESSION, 2)
            })
            
            # Update current state for next layer
            curr_c, curr_h, curr_w = out_c, out_h, out_w
            prev_tensor_kb = curr_tensor_kb

        # Quantized model size
        estimated_model_size_kb = (total_params * 4.0 * INT8_COMPRESSION) / 1024.0
        
        # Constraints check
        violations = []
        if estimated_model_size_kb > self.hw.max_model_weights_kb:
            violations.append(f"Model size {estimated_model_size_kb:.1f}KB > max {self.hw.max_model_weights_kb:.1f}KB")
        if peak_ram_kb > self.hw.max_peak_ram_kb:
            violations.append(f"Peak RAM {peak_ram_kb:.1f}KB > max {self.hw.max_peak_ram_kb:.1f}KB")
            
        return {
            "estimated_latency_ms": round(total_latency, 3),
            "estimated_model_size_kb": round(estimated_model_size_kb, 2),
            "estimated_peak_ram_kb": round(peak_ram_kb, 2),
            "feasibility_check_passed": len(violations) == 0,
            "constraint_violations": violations,
            "per_layer_breakdown": per_layer_breakdown
        }
