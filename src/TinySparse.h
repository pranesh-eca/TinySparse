#ifndef TINY_SPARSE_H
#define TINY_SPARSE_H

#include <Arduino.h>

// ---------------------------------------------------------
// HARDWARE ABSTRACTION LAYER (HAL)
// ---------------------------------------------------------
#if defined(ESP32)
    #define TARGET_ENV "Microcontroller (ESP32)"
#elif defined(USE_CUDA) || defined(__NVCC__)
    #define TARGET_ENV "Desktop GPU (CUDA)"
#elif defined(__AVX2__) || defined(__x86_64__)
    #define TARGET_ENV "Desktop CPU (x86_64 SIMD)"
#else
    #define TARGET_ENV "Generic C++"
#endif

class TinySparse {
  public:
    TinySparse();
    const char* getEnvironment();
    
    uint32_t evaluateDense(const uint8_t* inputVector, uint16_t length);
    uint32_t evaluateSparse(const uint8_t* inputVector, uint16_t length);
    uint32_t evaluateSparseIndexed(const uint8_t* values, const uint8_t* indices, uint16_t nonZeroCount);
};

#endif