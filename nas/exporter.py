import os
import time
import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
import subprocess

class ModelExporter:
    """
    ModelExporter handles the conversion of PyTorch models to TFLite format
    and produces C headers for deployment on microcontrollers.
    """
    def __init__(self, output_dir: str = 'outputs'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def export(self, model: nn.Module, dummy_input: torch.Tensor, calib_loader, quantization: str = 'int8', run_id: str = '') -> dict:
        """
        Exports a PyTorch model to TFLite via ONNX.
        """
        export_id = f"exp_{int(time.time())}"
        onnx_path = os.path.join(self.output_dir, "tmp_model.onnx")
        tflite_filename = f"model_{quantization}.tflite"
        tflite_path = os.path.join(self.output_dir, tflite_filename)
        saved_model_dir = os.path.join(self.output_dir, "tmp_saved_model")
        
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
            
            # 2. ONNX -> TensorFlow SavedModel (using onnx2tf for better compatibility)
            # We use onnx2tf to generate a SavedModel which TFLiteConverter can then optimize
            subprocess.run([
                "onnx2tf", 
                "-i", onnx_path, 
                "-o", saved_model_dir,
                "--non_verbose"
            ], check=True)
            
            # 3. TensorFlow SavedModel -> TFLite with optional quantization
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
            
            if quantization == 'int8':
                def representative_dataset():
                    # Use a few batches from calib_loader for representative dataset
                    for i, (inputs, _) in enumerate(calib_loader):
                        if i >= 20: break # Limit 20 batches for speed
                        # Ensure input is float32 and matched shape
                        yield [inputs.numpy().astype(np.float32)]
                
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = representative_dataset
                # Ensure fully integer quantization for MCU
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
                
            tflite_model = converter.convert()
            
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)
                
            model_size_kb = os.path.getsize(tflite_path) / 1024.0
            
            # Cleanup temporary files
            if os.path.exists(onnx_path): os.remove(onnx_path)
            
            return {
                "run_id": run_id,
                "export_id": export_id,
                "filename": tflite_filename,
                "download_url": f"file://{os.path.abspath(tflite_path)}",
                "quantization": quantization,
                "model_size_kb": round(model_size_kb, 2),
                "peak_ram_kb": None, # Complex to estimate outside of profiling
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
