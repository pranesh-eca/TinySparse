#ifndef TINY_SPARSE_H
#define TINY_SPARSE_H

#include <Arduino.h>

class TinySparse {
  public:
    TinySparse();
    uint32_t evaluateDense(const uint8_t* inputVector, uint16_t length);
    uint32_t evaluateSparse(const uint8_t* inputVector, uint16_t length);
    uint32_t evaluateSparseIndexed(const uint8_t* values, const uint8_t* indices, uint16_t nonZeroCount);
};

#endif