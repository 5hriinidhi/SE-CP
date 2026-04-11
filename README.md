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

> **Note:** Out of the box, `run_nas.py` is configured to instantly download a subset of the **MNIST Handwritten Digits** dataset and train the AI to classify pictures of numbers 0-9.

### Export the Best Model

```bash
python run_nas.py export --checkpoint /tmp/best_candidate.pth
```

This converts the best PyTorch model to `.tflite` format and generates a C header file (`outputs/model_int8.h`) for the microcontroller.

### Test the Model Locally

Before flashing to actual hardware, you can test the model's accuracy on your PC using real handwritten digit images from the MNIST dataset:

```bash
python scripts/local_inference.py
```

This script will pick 5 random images, run them through your optimized architecture, and print the True Class vs Predicted Class side by side!

### 🎯 Presentation Demo: Offline Digit Recognition on ESP32

To truly impress an audience, you want to show the ESP32 recognizing a real handwritten digit **entirely offline** without sending any image data to the cloud (which saves battery, reduces latency, and protects privacy).

1. Before flashing, generate the real C-array test images from the MNIST dataset:
   ```bash
   python scripts/generate_mnist_c_array.py
   ```
   *This downloads 3 real handwritten digits (a '3', '7', and '0') and converts them into C-code inside `firmware/esp32_inference/mnist_samples.h`.*

2. Run another quick python script to export those exact pixels to `.png` pictures so you can flash them on screen during the demo:
   ```bash
   python scripts/export_demo_images.py
   ```
   *You'll find these in the `outputs/demo_images/` folder.*

3. Follow the standard flashing instructions (down in the **Hardware Setup** section) to upload the firmware to your ESP32.
4. Open the **Arduino Serial Monitor (115200 baud)** on the right half of your screen. Open the exported PNG pictures on the left half.
5. You will see the ESP32 automatically read the raw pixels entirely offline and print out the correct classification!

**Expected Serial Output during your Demo:**
```text
============================
 TinyML AutoNAS — ESP32 Demo
============================
Model loaded. Arena: 65536 bytes
Offline digit recognition initialized.

----------------------------------------
Loading test image #1 (Expected Digit: 3)...
Predicted Digit: 3  (Confidence: 98.40%)
Inference time: 14.20 ms
Result: CORRECT! ✅
```

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

---

## 📋 Requirements Summary

| Requirement | Needed For | Required? |
|---|---|---|
| Python 3.9+ | Everything | ✅ Yes |
| `pip install -r requirements.txt` | Core functionality | ✅ Yes |
| `ANTHROPIC_API_KEY` | LLM-guided pruning | ❌ Optional |
| `onnx`, `onnx2tf` | Model export | ❌ Optional |
| Arduino IDE + ESP32 board | Hardware deployment | ❌ Optional |

---

## SHRINIDHI SE PROJECT DONEEEEEE LESSGOOOOO WOOOOHOOOOO
