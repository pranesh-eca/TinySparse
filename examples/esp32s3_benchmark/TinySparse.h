#ifndef TINY_SPARSE_H
#define TINY_SPARSE_H

#include <stdint.h>
#include <stddef.h>

class TinySparse {
public:
    /**
     * @brief Computes a single 1D Sparse Dot Product with Bias and ReLU activation.
     *        Neuron Output = ReLU( SUM(input[indices[k]] * values[k]) + bias )
     *
     * @tparam IndexType uint8_t (<= 256 inputs) or uint16_t (> 256 inputs)
     * @param input Dense 1D sensor buffer stored in SRAM (INT8)
     * @param values Non-zero weights array stored in Flash (INT8)
     * @param indices Array of column indices for non-zero weights in Flash
     * @param nnz Number of non-zero weights in this specific neuron
     * @param bias 32-bit pre-scaled bias for this neuron
     * @return int8_t Activated output clamped to [0, 127]
     */
    template <typename IndexType>
    static inline int8_t sparseNeuron1D(
        const int8_t*    __restrict input,
        const int8_t*    __restrict values,
        const IndexType* __restrict indices,
        uint16_t         nnz,
        int32_t          bias
    ) {
        int32_t acc = bias;
        uint16_t k = 0;

        // Unroll by 4 to utilize 32-bit ALU execution pipelines
        for (; k + 3 < nnz; k += 4) {
            acc += (int32_t)input[indices[k]]     * (int32_t)values[k];
            acc += (int32_t)input[indices[k + 1]] * (int32_t)values[k + 1];
            acc += (int32_t)input[indices[k + 2]] * (int32_t)values[k + 2];
            acc += (int32_t)input[indices[k + 3]] * (int32_t)values[k + 3];
        }

        // Handle remaining non-zero weights
        for (; k < nnz; ++k) {
            acc += (int32_t)input[indices[k]] * (int32_t)values[k];
        }

        // ReLU Activation and symmetric integer scaling back to INT8 range
        // Division by 128 (arithmetic right-shift by 7) rescales INT8 x INT8 product
        int32_t scaled = acc >> 7;
        if (scaled < 0) scaled = 0;
        if (scaled > 127) scaled = 127;

        return (int8_t)scaled;
    }

    /**
     * @brief Executes a 1D Sparse Layer across multiple neurons.
     * 
     * @tparam IndexType uint8_t or uint16_t
     * @param input Input vector in SRAM (length = inputDim)
     * @param values Contiguous array of all non-zero weights in Flash
     * @param indices Contiguous array of column indices in Flash
     * @param offsets Index boundaries demarcating where each neuron's weights start/end
     * @param biases Pre-scaled 32-bit biases in Flash
     * @param numNeurons Number of output neurons
     * @param output Output vector buffer allocated in SRAM (length = numNeurons)
     */
    template <typename IndexType>
    static inline void forwardSparseLayer1D(
        const int8_t*    __restrict input,
        const int8_t*    __restrict values,
        const IndexType* __restrict indices,
        const uint16_t*  __restrict offsets,
        const int32_t*   __restrict biases,
        uint16_t         numNeurons,
        int8_t*          __restrict output
    ) {
        for (uint16_t n = 0; n < numNeurons; ++n) {
            uint16_t start = offsets[n];
            uint16_t end   = offsets[n + 1];
            uint16_t nnz   = end - start;

            output[n] = sparseNeuron1D<IndexType>(
                input,
                &values[start],
                &indices[start],
                nnz,
                biases[n]
            );
        }
    }

    /**
     * @brief Executes a Dense 1D Layer (used for fc3 to bypass the index tax).
     * 
     * @param input Input vector in SRAM (length = inputDim)
     * @param weights Dense weights matrix in Flash (numNeurons * inputDim)
     * @param biases 32-bit biases in Flash
     * @param inputDim Number of input connections
     * @param numNeurons Number of output classes
     * @param output Logits buffer in SRAM (length = numNeurons)
     */
    static inline void forwardDenseLayer1D(
        const int8_t*  __restrict input,
        const int8_t*  __restrict weights,
        const int32_t* __restrict biases,
        uint16_t       inputDim,
        uint16_t       numNeurons,
        int32_t*       __restrict output
    ) {
        for (uint16_t n = 0; n < numNeurons; ++n) {
            int32_t acc = biases[n];
            const int8_t* w_row = &weights[n * inputDim];

            uint16_t i = 0;
            for (; i + 3 < inputDim; i += 4) {
                acc += (int32_t)input[i]     * (int32_t)w_row[i];
                acc += (int32_t)input[i + 1] * (int32_t)w_row[i + 1];
                acc += (int32_t)input[i + 2] * (int32_t)w_row[i + 2];
                acc += (int32_t)input[i + 3] * (int32_t)w_row[i + 3];
            }
            for (; i < inputDim; ++i) {
                acc += (int32_t)input[i] * (int32_t)w_row[i];
            }

            output[n] = acc;
        }
    }

    /**
     * @brief Argmax helper to determine the predicted class label.
     */
    static inline uint8_t argmax(const int32_t* logits, uint16_t count) {
        uint8_t bestClass = 0;
        int32_t maxVal = logits[0];
        for (uint16_t i = 1; i < count; ++i) {
            if (logits[i] > maxVal) {
                maxVal = logits[i];
                bestClass = (uint8_t)i;
            }
        }
        return bestClass;
    }
};

#endif // TINY_SPARSE_H