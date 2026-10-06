#ifndef TINY_SPARSE_H
#define TINY_SPARSE_H

#include <Arduino.h>

class TinySparse {
  public:
    // Constructor to initialize the library
    TinySparse();

    // Standard dense matrix evaluation (The baseline for comparison)
    uint32_t evaluateDense(const uint8_t* inputVector, uint16_t length);

    // Optimized zero-skipping evaluation (Your core thesis algorithm)
    uint32_t evaluateSparse(const uint8_t* inputVector, uint16_t length);
};

#endif