#include "TinySparse.h"

TinySparse::TinySparse() {
  // Constructor: Ready for future hardware-specific initializations (e.g., DMA setup)
}

uint32_t TinySparse::evaluateDense(const uint8_t* inputVector, uint16_t length) {
  uint32_t accumulator = 0;
  
  // Dense Execution: The CPU is forced to compute every single element.
  for (uint16_t i = 0; i < length; i++) {
    // Simulating a MAC (Multiply-Accumulate) operation with a static weight
    accumulator += inputVector[i] * 2; 
  }
  
  return accumulator;
}

uint32_t TinySparse::evaluateSparse(const uint8_t* inputVector, uint16_t length) {
  uint32_t accumulator = 0;
  
  // Sparse Execution: The CPU evaluates the condition and skips the MAC for zeros.
  // On an ESP32-S3 or UNO, bypassing the multiplication instruction saves significant cycles.
  for (uint16_t i = 0; i < length; i++) {
    if (inputVector[i] != 0) {
      accumulator += inputVector[i] * 2;
    }
  }
  
  return accumulator;
}