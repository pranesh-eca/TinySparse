#ifndef TINY_SPARSE_H
#define TINY_SPARSE_H

#include <stdint.h>

class TinySparse {
private:
    constexpr static float SPARSITY_SPEEDUP_THRESHOLD = 0.45f;

public:
    TinySparse() {}

    const char* getEnvironment() {
#if defined(ESP32)
        return "Microcontroller (ESP32)";
#elif defined(__ARM_ARCH)
        return "Microcontroller (ARM Cortex)";
#else
        return "Host / x86";
#endif
    }

    // ---------------------------------------------------------
    // 1D INFERENCE (COO FORMAT)
    // ---------------------------------------------------------
    
    int32_t evaluateDense(const uint8_t* input, const int8_t* weights, uint16_t totalElements) {
        int32_t accumulator = 0;
        for (uint16_t i = 0; i < totalElements; i++) {
            accumulator += (int32_t)input[i] * (int32_t)weights[i];
        }
        return accumulator;
    }

    template <typename IndexType>
    int32_t evaluateSparseIndexed(const uint8_t* input, const int8_t* values, const IndexType* indices, uint16_t nonZeroCount) {
        int32_t accumulator = 0;
        for (uint16_t i = 0; i < nonZeroCount; i++) {
            accumulator += (int32_t)input[indices[i]] * (int32_t)values[i];
        }
        return accumulator;
    }

    template <typename IndexType>
    int32_t infer1D(const uint8_t* input, const int8_t* sparseValues, const IndexType* sparseIndices, 
                    uint16_t nonZeroCount, uint16_t totalElements, const int8_t* fallbackDenseWeights = nullptr) {
        
        float sparsity = 1.0f - ((float)nonZeroCount / (float)totalElements);

        if (sparsity >= SPARSITY_SPEEDUP_THRESHOLD || fallbackDenseWeights == nullptr) {
            return evaluateSparseIndexed(input, sparseValues, sparseIndices, nonZeroCount);
        } else {
            return evaluateDense(input, fallbackDenseWeights, totalElements);
        }
    }

    // ---------------------------------------------------------
    // 2D INFERENCE (CSR FORMAT)
    // ---------------------------------------------------------

    template <typename IndexType>
    void evaluateSparseCSR(const uint8_t* input, const int8_t* values, const IndexType* colIndices, 
                           const IndexType* rowPtr, uint16_t numRows, int32_t* output) {
        for (uint16_t i = 0; i < numRows; i++) {
            int32_t accumulator = 0;
            IndexType rowStart = rowPtr[i];
            IndexType rowEnd = rowPtr[i + 1];
            
            for (IndexType j = rowStart; j < rowEnd; j++) {
                accumulator += (int32_t)input[colIndices[j]] * (int32_t)values[j];
            }
            output[i] = accumulator;
        }
    }

    template <typename IndexType>
    void infer2D(const uint8_t* input, const int8_t* sparseValues, const IndexType* colIndices, 
                 const IndexType* rowPtr, uint16_t nonZeroCount, uint16_t numRows, uint16_t numCols, 
                 int32_t* output, const int8_t* fallbackDenseWeights = nullptr) {
        
        uint32_t totalElements = numRows * numCols;
        float sparsity = 1.0f - ((float)nonZeroCount / (float)totalElements);

        if (sparsity >= SPARSITY_SPEEDUP_THRESHOLD || fallbackDenseWeights == nullptr) {
            evaluateSparseCSR(input, sparseValues, colIndices, rowPtr, numRows, output);
        } else {
            for (uint16_t i = 0; i < numRows; i++) {
                int32_t accumulator = 0;
                for (uint16_t j = 0; j < numCols; j++) {
                    accumulator += (int32_t)input[j] * (int32_t)fallbackDenseWeights[i * numCols + j];
                }
                output[i] = accumulator;
            }
        }
    }
};

#endif
