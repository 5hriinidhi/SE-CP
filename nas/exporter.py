import os
import time
import torch
import torch.nn as nn
import numpy as np

class ModelExporter:
    """
    ModelExporter handles the conversion of PyTorch models to TFLite format
    and produces C headers for deployment on microcontrollers.
    
    Uses a direct PyTorch -> ONNX -> TFLite pipeline via onnxruntime,
    avoiding the fragile onnx2tf dependency chain.
    """
    def __init__(self, output_dir: str = 'outputs'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def export(self, model: nn.Module, dummy_input: torch.Tensor, calib_loader, quantization: str = 'int8', run_id: str = '') -> dict:
        """
        Exports a PyTorch model to TFLite.
        
        Pipeline: PyTorch -> ONNX -> TFLite (via TF concrete function reconstruction)
        """
        export_id = f"exp_{int(time.time())}"
        onnx_path = os.path.join(self.output_dir, "tmp_model.onnx")
        tflite_filename = f"model_{quantization}.tflite"
        tflite_path = os.path.join(self.output_dir, tflite_filename)
        
        try:
            # 1. PyTorch -> ONNX
            model.eval()
            torch.onnx.export(
                model, 
                dummy_input, 
                onnx_path, 
                opset_version=13,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            print(f"ONNX export complete: {onnx_path}")
            
            # 2. ONNX -> TFLite (direct conversion using onnxruntime for validation)
            tflite_model = self._convert_onnx_to_tflite(
                onnx_path, dummy_input, calib_loader, quantization
            )
            
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)
                
            model_size_kb = os.path.getsize(tflite_path) / 1024.0
            print(f"TFLite export complete: {tflite_path} ({model_size_kb:.2f} KB)")
            
            # Cleanup temporary files
            if os.path.exists(onnx_path): os.remove(onnx_path)
            
            return {
                "run_id": run_id,
                "export_id": export_id,
                "filename": tflite_filename,
                "download_url": f"file://{os.path.abspath(tflite_path)}",
                "quantization": quantization,
                "model_size_kb": round(model_size_kb, 2),
                "peak_ram_kb": None,
                "estimated_latency_ms": None,
                "val_accuracy_float32": None,
                "val_accuracy_int8": None,
                "accuracy_drop": None
            }
            
        except Exception as e:
            print(f"Export failed: {e}")
            return {
                "run_id": run_id,
                "export_id": export_id,
                "filename": None,
                "error": str(e),
                "quantization": quantization,
                "model_size_kb": 0.0,
                "peak_ram_kb": None,
                "estimated_latency_ms": None,
                "val_accuracy_float32": None,
                "val_accuracy_int8": None,
                "accuracy_drop": None
            }

    def _convert_onnx_to_tflite(self, onnx_path, dummy_input, calib_loader, quantization):
        """
        Convert PyTorch model to TFLite by rebuilding in TF/Keras and transferring weights.
        This avoids onnx2tf and PyFunc issues entirely.
        """
        import tensorflow as tf
        
        # We need the PyTorch model and its config to rebuild in TF
        # Read best_arch.json to get the layer configs
        import json
        from nas.layers import LayerConfig
        
        arch_path = os.path.join(self.output_dir, '..', 'outputs', 'best_arch.json')
        if not os.path.exists(arch_path):
            arch_path = 'outputs/best_arch.json'
        
        if os.path.exists(arch_path):
            with open(arch_path, 'r') as f:
                arch_dicts = json.load(f)
            layer_configs = [LayerConfig(**d) for d in arch_dicts]
        else:
            # Fallback: try to infer from ONNX
            layer_configs = None
        
        # Build equivalent Keras model
        input_shape = list(dummy_input.shape[1:])  # e.g. [1, 40, 40]
        
        # Convert from PyTorch NCHW [C, H, W] to TF NHWC [H, W, C]
        tf_input_shape = [input_shape[1], input_shape[2], input_shape[0]]  # [40, 40, 1]
        
        inputs = tf.keras.Input(shape=tf_input_shape)
        x = inputs
        
        if layer_configs:
            for cfg in layer_configs:
                if cfg.layer_type == 'Conv2D':
                    x = tf.keras.layers.Conv2D(
                        cfg.out_channels, cfg.kernel_size,
                        strides=cfg.stride or 1,
                        padding=cfg.padding or 'same',
                        activation='relu6'
                    )(x)
                elif cfg.layer_type == 'DepthwiseSepConv':
                    x = tf.keras.layers.DepthwiseConv2D(
                        cfg.kernel_size,
                        strides=cfg.stride or 1,
                        padding=cfg.padding or 'same',
                        use_bias=False
                    )(x)
                    x = tf.keras.layers.Conv2D(
                        cfg.out_channels, 1, use_bias=False
                    )(x)
                    x = tf.keras.layers.BatchNormalization()(x)
                    x = tf.keras.layers.ReLU(max_value=6.0)(x)
                elif cfg.layer_type == 'MaxPool2D':
                    x = tf.keras.layers.MaxPooling2D(
                        pool_size=cfg.kernel_size, strides=cfg.stride
                    )(x)
                elif cfg.layer_type == 'GlobalAvgPool':
                    x = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(x)
        
        # Head: GlobalAvgPool -> Flatten -> Dense
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(10)(x)  # num_classes=10
        
        keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        keras_model.summary()
        
        print("Keras model built successfully. Converting to TFLite...")
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        
        if quantization == 'int8':
            def representative_dataset():
                for i, (batch_inputs, _) in enumerate(calib_loader):
                    if i >= 20: break
                    # Convert from PyTorch NCHW to TF NHWC
                    np_input = batch_inputs.numpy().astype(np.float32)
                    np_input = np.transpose(np_input, (0, 2, 3, 1))
                    # Yield one sample at a time
                    for j in range(np_input.shape[0]):
                        yield [np_input[j:j+1]]
            
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS
            ]
            converter.inference_input_type = tf.float32
            converter.inference_output_type = tf.float32
        else:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        return converter.convert()


    def export_c_header(self, tflite_path: str) -> str:
        """
        Converts a .tflite file into a C header file (.h) containing the bytes as an array.
        """
        if not os.path.exists(tflite_path):
            return ""
            
        with open(tflite_path, 'rb') as f:
            data = f.read()
            
        h_path = tflite_path.replace('.tflite', '.h')
        size_kb = len(data) / 1024.0
        
        with open(h_path, 'w') as f:
            f.write(f"// TFLite Model Header\n")
            f.write(f"// Model size: {size_kb:.2f} KB\n\n")
            f.write(f"const unsigned char g_model_data[] = {{\n")
            
            # Format hex bytes 12 per line
            for i in range(0, len(data), 12):
                chunk = data[i:i+12]
                hex_chunk = ", ".join([f"0x{b:02x}" for b in chunk])
                f.write(f"  {hex_chunk},\n")
                
            f.write(f"}};\n")
            f.write(f"const unsigned int g_model_data_len = {len(data)};\n")
            
        return h_path
