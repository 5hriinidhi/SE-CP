# ESP32 Inference Firmware

This Arduino sketch runs TFLite Micro inference on your ESP32 using the model exported by TinyML AutoNAS.

## Setup Instructions

### 1. Open in Arduino IDE
- Open **Arduino IDE** (v2.x recommended)
- Go to **File → Open** and select `esp32_inference.ino`
- The IDE will automatically load the sketch folder including `best_model_data.h`

### 2. Install Required Libraries
- Go to **Tools → Manage Libraries**
- Search for **TensorFlowLite_ESP32** and install it
- If unavailable, download from [TFLite Micro Arduino](https://github.com/tensorflow/tflite-micro-arduino-examples)

### 3. Select Board
- Go to **Tools → Board → ESP32 Arduino**
- Select **ESP32 Dev Module**
- Set **Upload Speed** to `921600`
- Set **Flash Size** to `4MB (32Mb)`

### 4. Configure Tensor Arena Size
At the top of `esp32_inference.ino`, adjust `TENSOR_ARENA_SIZE` based on your model's export output:

```cpp
#define TENSOR_ARENA_SIZE  (64 * 1024)  // Set to peak_ram_kb * 1024
```

You can find `peak_ram_kb` in the export result JSON or from the CLI:
```bash
python run_nas.py export --checkpoint /tmp/best_candidate.pth
```

### 5. Upload & Monitor
- Click **Upload** (→ button)
- Open **Serial Monitor** at **115200 baud**
- You should see:
  ```
  Model loaded. Arena: 65536 bytes
  Class: 3 | Conf: 0.87 | Latency: 14200 us
  ```

## Customization

| Define | Description |
|---|---|
| `TENSOR_ARENA_SIZE` | Memory pool for TFLite tensors. Must be ≥ `peak_ram_kb * 1024` from export. |
| `NUM_CLASSES` | Number of output classes. Must match training config. |

## Replacing Dummy Input
In `loop()`, replace the dummy data fill with your real sensor input (e.g., microphone FFT for audio classification).
