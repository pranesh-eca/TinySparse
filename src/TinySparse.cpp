#include "TinySparse.h"

TinySparse::TinySparse() {}

const char* TinySparse::getEnvironment() {
  return TARGET_ENV;
}

uint32_t TinySparse::evaluateDense(const uint8_t* inputVector, uint16_t length) {
  uint32_t accumulator = 0;
  for (uint16_t i = 0; i < length; i++) {
    accumulator += inputVector[i] * 2; 
  }
  return accumulator;
}

uint32_t TinySparse::evaluateSparse(const uint8_t* inputVector, uint16_t length) {
  uint32_t accumulator = 0;
  for (uint16_t i = 0; i < length; i++) {
    if (inputVector[i] != 0) {
      accumulator += inputVector[i] * 2;
    }
  }
  return accumulator;
}

uint32_t TinySparse::evaluateSparseIndexed(const uint8_t* values, const uint8_t* indices, uint16_t nonZeroCount) {
  uint32_t accumulator = 0;
  for (uint16_t i = 0; i < nonZeroCount; i++) {
    accumulator += values[i] * 2;
  }
  return accumulator;
}