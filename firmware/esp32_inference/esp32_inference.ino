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

#include "mnist_samples.h"

// ── User-configurable defines ────────────────────────────
#define TENSOR_ARENA_SIZE  (64 * 1024)   // bytes — set to peak_ram_kb * 1024 from export
#define NUM_CLASSES        10             // must match num_classes used during training

// ── Globals (statically allocated — no malloc in loop) ───
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

// Track which test image we are on
int current_test_idx = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }

  Serial.println("============================");
  Serial.println(" TinyML AutoNAS — ESP32 Demo");
  Serial.println("============================");

  // Load the TFLite model from the C header array
  model = tflite::GetModel(g_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("Model schema mismatch: expected %d, got %d\n",
                  TFLITE_SCHEMA_VERSION, model->version());
    while (1) { delay(1000); }
  }

  // Register all ops
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
  Serial.printf("Offline digit recognition initialized.\n\n");
  delay(2000);
}

void loop() {
  Serial.printf("----------------------------------------\n");
  Serial.printf("Loading test image #%d (Expected Digit: %d)...\n", 
                current_test_idx + 1, test_labels[current_test_idx]);

  // Copy real MNIST test image into the input tensor
  int input_size = input_tensor->bytes / sizeof(float);
  float* input_data = input_tensor->data.f;
  
  for (int i = 0; i < input_size; i++) {
    input_data[i] = test_samples[current_test_idx][i];
  }

  // Run inference locally on the ESP32!
  unsigned long t0 = micros();
  TfLiteStatus invoke_status = interpreter->Invoke();
  unsigned long elapsed_us = micros() - t0;

  if (invoke_status != kTfLiteOk) {
    Serial.println("ERROR: Invoke() failed!");
    delay(500);
    return;
  }

  // Read output and find argmax (predicted digit)
  float* output_data = output_tensor->data.f;
  int best_class = 0;
  float best_conf = output_data[0];
  for (int i = 1; i < NUM_CLASSES; i++) {
    if (output_data[i] > best_conf) {
      best_conf = output_data[i];
      best_class = i;
    }
  }

  Serial.printf("Predicted Digit: %d  (Confidence: %.2f%%)\n",
                best_class, best_conf * 100.0);
  Serial.printf("Inference time: %.2f ms\n", elapsed_us / 1000.0);
  
  if (best_class == test_labels[current_test_idx]) {
    Serial.println("Result: CORRECT! ✅");
  } else {
    Serial.println("Result: INCORRECT ❌");
  }

  // Move to next image, pause so presentation audience can read
  current_test_idx = (current_test_idx + 1) % NUM_TEST_SAMPLES;
  delay(4000);
}
