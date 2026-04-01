# TinyML AutoNAS

**Automatically find the best neural network for your microcontroller.**

TinyML AutoNAS is a tool that designs tiny, optimized neural networks that can run on cheap microcontrollers like ESP32. Instead of manually guessing which neural network architecture will fit on your chip, this tool automatically searches through thousands of possibilities, tests them, and gives you the best one — ready to flash onto your device.

**How it works in plain English:**
1. You tell it which chip you're using (e.g., ESP32 with 4MB flash, 520KB RAM)
2. An AI (Claude/GPT) suggests what kind of layers will work best for your task
3. The tool generates thousands of candidate architectures and instantly eliminates ones that are too big or too slow
4. It trains the remaining candidates and picks the winner
5. It exports the best model as a `.tflite` file + C header ready to flash

---

## 📁 Project Structure

```
tinyml-autonas/
├── nas/                    # Core NAS engine
│   ├── hardware_config.py  # Chip specs (flash, RAM, clock)
│   ├── llm_advisor.py      # AI-powered architecture hints
│   ├── search_space.py     # Candidate generation + pruning
│   ├── simulator.py        # Fast latency/memory estimation
│   ├── architecture.py     # PyTorch model builder
│   ├── trainer.py          # Training loop
│   ├── controller.py       # Orchestrates the full pipeline
│   └── exporter.py         # PyTorch → TFLite → C header
├── api/                    # REST API server
│   ├── server.py           # FastAPI app
│   ├── models.py           # Pydantic schemas
│   └── routes/             # Hardware, Search, LLM, Simulator endpoints
├── config/                 # Editable YAML configs
│   ├── hardware.yaml       # Your chip's specs
│   └── search.yaml         # Search parameters
├── scripts/                # Utilities
│   ├── check_install.py    # Verify environment
│   └── benchmark_llm_speedup.py  # Measure LLM pruning speedup
├── firmware/               # ESP32 Arduino sketch
│   └── esp32_inference/
│       ├── esp32_inference.ino
│       └── best_model_data.h
├── tests/                  # Unit tests
├── run_nas.py              # Main CLI (search, simulate, export)
├── run_trial.py            # NNI trial script
├── nni_config.yaml         # NNI experiment config
└── requirements.txt        # Python dependencies
```

---

## 🚀 Step-by-Step Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/5hriinidhi/SE-CP.git
cd SE-CP/tinyml-autonas
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

For full export pipeline support, also install:

```bash
pip install onnx onnx2tf onnxruntime pydantic
```

### Step 4: (Optional) Set API Key

If you want AI-guided search space pruning, set your Anthropic API key:

```bash
# Windows PowerShell
$env:ANTHROPIC_API_KEY = "sk-ant-api03-your-key-here"

# macOS/Linux
export ANTHROPIC_API_KEY="sk-ant-api03-your-key-here"
```

> **Without the API key, everything still works.** The search just won't use LLM hints — it searches blindly through all candidates instead of a pruned subset.

### Step 5: Verify Installation

```bash
python scripts/check_install.py
```

---

## 🏃 How to Run

### Run a Quick Simulation

Test if an architecture fits your hardware — no training needed:

```bash
python run_nas.py simulate --arch "DSConv16,DSConv32" --hw config/hardware.yaml
```

This instantly tells you the estimated latency, model size, and RAM usage.

### Run the Full NAS Search

```bash
python run_nas.py search --config config/search.yaml
```

This will:
1. Query the LLM for architecture hints (or skip if no API key)
2. Generate and prune the search space
3. Loop through candidates: simulate → train → score
4. Save the best model to `outputs/`

### Export the Best Model

```bash
python run_nas.py export --checkpoint /tmp/best_candidate.pth
```

This converts the best PyTorch model to `.tflite` format and generates a C header file.

### Run the API Server

```bash
uvicorn api.server:app --port 8000
```

Then open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger UI.

**Available API Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/hardware` | Register a hardware profile |
| GET | `/api/v1/hardware` | List all hardware profiles |
| POST | `/api/v1/search` | Start a NAS search (runs in background) |
| GET | `/api/v1/search/{run_id}` | Check search progress |
| POST | `/api/v1/llm/hints` | Get LLM architecture hints |
| POST | `/api/v1/simulator/estimate` | Simulate an architecture |

### Run Unit Tests

```bash
python -m pytest tests/ -v
```

### Run the LLM Speedup Benchmark

```bash
python scripts/benchmark_llm_speedup.py --domain audio_classification
```

This compares NAS with vs. without LLM hints and prints a comparison table.

---

## 🫀 Use Case: AFib Detection on ESP32-S3

This walkthrough shows how to use AutoNAS to build a neural network that detects Atrial Fibrillation (AFib) from ECG signals on a wearable device.

### Step 1: Update Hardware Config

Replace the contents of `config/hardware.yaml` with your ESP32-S3 specs:

```yaml
chip_id: ESP32-S3-WROOM-1
flash_kb: 8192
sram_kb: 512
mhz: 240
has_fpu: true
supports_simd: true
```

### Step 2: Update Search Config

Replace the contents of `config/search.yaml`:

```yaml
hardware_id: ESP32-S3-WROOM-1
domain: time_series
task_description: Binary ECG classification Normal vs AFib, 30s segments, wearable device
dataset_path: ./data/ecg_afib
num_classes: 2
trial_budget: 60
max_latency_ms: 20
max_model_size_kb: 80
target_accuracy: 0.990
tuner_strategy: TPE
llm_model: gpt-4o
quantization: int8
epochs: 30
```

### Step 3: Prepare Your ECG Dataset

Place your ECG data in `./data/ecg_afib/` with this structure:

```
data/ecg_afib/
├── train/
│   ├── normal/       # Normal ECG samples (.npy or .wav)
│   └── afib/         # AFib ECG samples
├── val/
│   ├── normal/
│   └── afib/
└── calibration/      # Small subset for INT8 quantization
    ├── normal/
    └── afib/
```

> **No dataset?** The system will fall back to dummy data for demo purposes.

### Step 4: Run the Search

```bash
python run_nas.py search --config config/search.yaml
```

The tool will:
- Ask the LLM: *"What architecture works best for binary ECG classification on ESP32-S3?"*
- The LLM might suggest: prefer 1D convolutions, avoid LSTMs, use small kernels
- Search through 60 candidates with those constraints
- Train each feasible candidate for 30 epochs
- Stop early if accuracy ≥ 99.0%

### Step 5: Export the Best Model

```bash
python run_nas.py export --checkpoint /tmp/best_candidate.pth
```

This creates:
- `outputs/model_int8.tflite` — the quantized model for ESP32
- `outputs/model_int8.h` — C header with the model as a byte array

### Step 6: Copy to Firmware

```bash
copy outputs\model_int8.h firmware\esp32_inference\best_model_data.h
```

### Step 7: Update Firmware Constants

Open `firmware/esp32_inference/esp32_inference.ino` and update:

```cpp
#define TENSOR_ARENA_SIZE  (64 * 1024)  // Set to peak_ram_kb * 1024 from export output
#define NUM_CLASSES        2            // 2 for Normal vs AFib
```

---

## 🔌 Hardware Setup: Connecting & Flashing the ESP32

### What You Need

- **ESP32 Dev Board** (ESP32-WROOM-32, ESP32-S3, or similar)
- **USB Cable** (Micro-USB or USB-C depending on your board)
- **Arduino IDE** (v2.x recommended)

### Step 1: Install Arduino IDE

Download from [arduino.cc/en/software](https://www.arduino.cc/en/software)

### Step 2: Add ESP32 Board Support

1. Open Arduino IDE
2. Go to **File → Preferences**
3. In "Additional Board Manager URLs", add:
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
4. Go to **Tools → Board → Boards Manager**
5. Search **"ESP32"** and install **"esp32 by Espressif Systems"**

### Step 3: Install TFLite Micro Library

1. Go to **Tools → Manage Libraries**
2. Search **"TensorFlowLite_ESP32"** and install it
3. If not found, download from [tflite-micro-arduino-examples](https://github.com/tensorflow/tflite-micro-arduino-examples)

### Step 4: Open the Sketch

1. Go to **File → Open**
2. Navigate to `firmware/esp32_inference/esp32_inference.ino`

### Step 5: Select Board & Port

1. **Tools → Board → ESP32 Arduino → ESP32 Dev Module** (or ESP32-S3 Dev Module for S3)
2. **Tools → Port** → Select the COM port where your ESP32 is connected
3. Set **Upload Speed** to `921600`
4. Set **Flash Size** to `4MB` (or `8MB` for S3)

### Step 6: Upload

1. Click the **Upload** button (→ arrow)
2. When you see `Connecting...`, hold the **BOOT** button on your ESP32 until upload starts
3. Wait for "Done uploading"

### Step 7: Monitor Output

1. Open **Tools → Serial Monitor**
2. Set baud rate to **115200**
3. You should see:

```
============================
 TinyML AutoNAS — ESP32
============================
Model loaded. Arena: 65536 bytes
Ready for inference.

Class: 0 | Conf: 0.94 | Latency: 14200 us
Class: 1 | Conf: 0.87 | Latency: 14350 us
```

- **Class 0** = Normal sinus rhythm
- **Class 1** = Atrial Fibrillation detected

### Connecting ECG Sensor (for real inference)

To replace the dummy data with real ECG input, connect an ECG sensor module (e.g., AD8232) to your ESP32:

| AD8232 Pin | ESP32 Pin |
|------------|-----------|
| GND | GND |
| 3.3V | 3V3 |
| OUTPUT | GPIO36 (ADC) |
| LO+ | GPIO34 |
| LO- | GPIO35 |

Then modify the `loop()` function in `esp32_inference.ino` to read from the ADC instead of using dummy data.

---

## 📋 Requirements Summary

| Requirement | Needed For | Required? |
|---|---|---|
| Python 3.9+ | Everything | ✅ Yes |
| `pip install -r requirements.txt` | Core functionality | ✅ Yes |
| `ANTHROPIC_API_KEY` | LLM-guided pruning | ❌ Optional |
| `onnx`, `onnx2tf` | Model export | ❌ Optional |
| Arduino IDE + ESP32 board | Hardware deployment | ❌ Optional |
| ECG sensor (AD8232) | Real AFib detection | ❌ Optional |

---

## SHRINIDHI SE PROJECT DONEEEEEE LESSGOOOOO WOOOOHOOOOO
