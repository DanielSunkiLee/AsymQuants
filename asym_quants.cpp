/** 
 *This is for Asymmetric Quantization(Zero-point Quantization)
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <omp.h>
constexpr int TOTAL_WEIGHTS = 256;
constexpr int GROUP_SIZE = 32;
constexpr int NUM_GROUPS = TOTAL_WEIGHTS / GROUP_SIZE;

//Struct to hold quantized group
struct QuantGroup{
  float scale; // scale factor for this group
  int8_t z;
  int8_t values[GROUP_SIZE]; // quantized values
};

// Quantize one group using asymmetric quntization
void quantize_group(const float* input, QuantGroup& qg) {
  float max_val = input[0];
  float min_val = input[0];

  //Find min,max_val
  for (int i = 0; i < GROUP_SIZE; i++) {
    max_val = std::max(max_val, input[i]);
    min_val = std::min(min_val, input[i]);
  }

  // Avoid division by zero
  if (max_val == 0.0f) {
    qg.scale = 1.0f;
    for (int i = 0; i < GROUP_SIZE; i++) {
      qg.values[i] = 0;
    }
    return;
  }
  
  // value s. scale maps float -> int8 [-127,127].
  // The value must be a positive integer.
  qg.scale = (max_val-min_val)/255.0f;
  // value z. the value 0 mapped.
  qg.z = static_cast<int8_t>(std::round(-min_val/qg.scale)-128.0f);


  // Quantize
  for (int i = 0; i < GROUP_SIZE; i++) {
    float scaled = (input[i] / qg.scale)+static_cast<float>(qg.z);
    int q = static_cast<int>(std::round(scaled));

    // Clamp to int8 range
    if (q > 127) q = 127;
    if (q < -127) q = -128;

    qg.values[i] = static_cast<int8_t>(q);
  }
}

// Dequantize one group
void dequantize_group(const QuantGroup& qg, float* output) {
  for (int i = 0; i < GROUP_SIZE; i++) {
    output[i] = (static_cast<int>(qg.values[i])-static_cast<float>(qg.z))*qg.scale;
  }
}


// Full tensor quantization
void quantize(const std::vector<float>& input, std::vector<QuantGroup>& output) {
  assert(input.size() == TOTAL_WEIGHTS);
  output.resize(NUM_GROUPS);

  #pragma omp parallel for
  for (int g = 0; g < NUM_GROUPS; g++) {
    quantize_group(&input[g * GROUP_SIZE],output[g]);
  }
}

//Full tensor dequantization
void dequantize(const std::vector<QuantGroup>& input, std::vector<float>& output) {
  output.resize(TOTAL_WEIGHTS);

  #pragma omp parallel for
  for (int g = 0; g < NUM_GROUPS; g++) {
    dequantize_group(input[g], &output[g * GROUP_SIZE]);
  }
}

int main() {
  std::vector<float> weights(TOTAL_WEIGHTS);

  // Initialize with some data
  for (int i = 0; i < TOTAL_WEIGHTS; i++) {
    weights[i] = std::sin(i*0.1f) * 10.0f;
  }

  std::vector<QuantGroup> quantized;
  std::vector<float> reconstructed;

  quantize(weights, quantized);
  dequantize(quantized, reconstructed);

  // Compute recontruction error
  float mse = 0.0f;
  for (int i = 0; i < TOTAL_WEIGHTS; i++) {
    float diff = weights[i] - reconstructed[i];
    mse += diff * diff;
  }
  mse /= TOTAL_WEIGHTS;

  std::cout << mse << std::endl;
  
  for (int i = 0; i < 10; i++) {
    std::cout << "origin: " << weights[i]
    << " retrieval: " << reconstructed[i] << std::endl;
  }
  return 0;

}