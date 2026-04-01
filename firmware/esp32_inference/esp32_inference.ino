/*
 * TinyML AutoNAS — ESP32 Inference Firmware
 * Runs the exported TFLite INT8 model on ESP32 using TFLite Micro.
 *
 * Adjust TENSOR_ARENA_SIZE based on your model's peak_ram_kb from the export step.
 * Adjust NUM_CLASSES to match your training configuration.
 */

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "best_model_data.h"

// ── User-configurable defines ────────────────────────────
#define TENSOR_ARENA_SIZE  (64 * 1024)   // bytes — set to peak_ram_kb * 1024 from export
#define NUM_CLASSES        10             // must match num_classes used during training

// ── Globals (statically allocated — no malloc in loop) ───
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }

  Serial.println("============================");
  Serial.println(" TinyML AutoNAS — ESP32");
  Serial.println("============================");

  // Load the TFLite model from the C header array
  model = tflite::GetModel(g_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("Model schema mismatch: expected %d, got %d\n",
                  TFLITE_SCHEMA_VERSION, model->version());
    while (1) { delay(1000); }
  }

  // Register all ops (can be replaced with specific ops for smaller binary)
  static tflite::AllOpsResolver resolver;

  // Build the interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
  interpreter = &static_interpreter;

  // Allocate tensors
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed!");
    while (1) { delay(1000); }
  }

  // Get input/output tensor pointers
  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  Serial.printf("Model loaded. Arena: %d bytes\n", TENSOR_ARENA_SIZE);
  Serial.printf("Input shape:  [%d", input_tensor->dims->data[0]);
  for (int i = 1; i < input_tensor->dims->size; i++) {
    Serial.printf(", %d", input_tensor->dims->data[i]);
  }
  Serial.println("]");
  Serial.printf("Output shape: [%d, %d]\n",
                output_tensor->dims->data[0],
                output_tensor->dims->data[1]);
  Serial.println("Ready for inference.\n");
}

void loop() {
  // Fill input tensor with dummy data (replace with real sensor input)
  int input_size = input_tensor->bytes / sizeof(float);
  float* input_data = input_tensor->data.f;
  for (int i = 0; i < input_size; i++) {
    input_data[i] = (float)(i % 256) / 255.0f;
  }

  // Run inference
  unsigned long t0 = micros();
  TfLiteStatus invoke_status = interpreter->Invoke();
  unsigned long elapsed_us = micros() - t0;

  if (invoke_status != kTfLiteOk) {
    Serial.println("ERROR: Invoke() failed!");
    delay(500);
    return;
  }

  // Read output and find argmax
  float* output_data = output_tensor->data.f;
  int best_class = 0;
  float best_conf = output_data[0];
  for (int i = 1; i < NUM_CLASSES; i++) {
    if (output_data[i] > best_conf) {
      best_conf = output_data[i];
      best_class = i;
    }
  }

  Serial.printf("Class: %d | Conf: %.2f | Latency: %lu us\n",
                best_class, best_conf, elapsed_us);

  delay(500);
}
