/**
 * @file TinySparse.h
 * @brief Bare-Metal CSR Sparse & Dense INT8 Inference Engine
 * 
 * Designed for resource-constrained microcontrollers (ESP32-S3, ARM Cortex-M, RISC-V).
 * Features:
 *  - Zero dynamic memory allocation (no malloc/new).
 *  - Unstructured compressed sparse row (CSR) execution for hidden layers.
 *  - Symmetric INT8 weights and activations with 32-bit accumulators.
 *  - Configurable per-layer fixed-point bit-shift requantization.
 *  - Dense INT8 baseline kernels for direct hardware cycle profiling.
 */

#ifndef TINYSPARSE_H
#define TINYSPARSE_H

#include <stdint.h>
#include <stddef.h>

namespace TinySparse {

/**
 * @brief Executes a Sparse Fully-Connected Layer (CSR format) with ReLU and Fixed-Point Shift.
 * 
 * Mathematical Contract:
 *   acc[row] = bias[row] + sum(values[idx] * input[col_indices[idx]])
 *   output[row] = clamp(ReLU(acc[row]) >> bit_shift, 0, 127)
 *
 * @param input        Pointer to INT8 input feature vector.
 * @param values       Pointer to non-zero INT8 weight values.
 * @param col_indices  Pointer to column index for each non-zero weight (uint8_t).
 * @param row_offsets  Pointer to row boundary offsets in values/col_indices arrays (uint16_t).
 * @param bias         Pointer to pre-scaled INT32 bias vector.
 * @param output       Pointer to INT8 output activation buffer.
 * @param out_features Number of output neurons (rows).
 * @param bit_shift    Calibrated right-shift factor for INT32 -> INT8 requantization (default 7).
 */
inline void sparse_fc_relu(
    const int8_t*   __restrict input,
    const int8_t*   __restrict values,
    const uint8_t*  __restrict col_indices,
    const uint16_t* __restrict row_offsets,
    const int32_t*  __restrict bias,
    int8_t*         __restrict output,
    uint16_t        out_features,
    uint8_t         bit_shift = 7
) {
    for (uint16_t row = 0; row < out_features; ++row) {
        int32_t acc = bias[row];
        uint16_t start = row_offsets[row];
        uint16_t end   = row_offsets[row + 1];

        for (uint16_t idx = start; idx < end; ++idx) {
            acc += static_cast<int32_t>(values[idx]) * static_cast<int32_t>(input[col_indices[idx]]);
        }

        // ReLU activation in INT32 domain followed by calibrated fixed-point scaling
        if (acc <= 0) {
            output[row] = 0;
        } else {
            acc = acc >> bit_shift;
            if (acc > 127) {
                acc = 127;
            }
            output[row] = static_cast<int8_t>(acc);
        }
    }
}

/**
 * @brief Executes a Standard Dense Fully-Connected Layer with ReLU and Fixed-Point Shift.
 * 
 * Used for direct cycle-accurate baseline benchmarking against sparse_fc_relu on silicon.
 *
 * @param input        Pointer to INT8 input feature vector.
 * @param weights      Pointer to contiguous row-major INT8 weight matrix (out_features * in_features).
 * @param bias         Pointer to pre-scaled INT32 bias vector.
 * @param output       Pointer to INT8 output activation buffer.
 * @param in_features  Number of input features (columns).
 * @param out_features Number of output neurons (rows).
 * @param bit_shift    Calibrated right-shift factor for INT32 -> INT8 requantization (default 7).
 */
inline void dense_fc_relu(
    const int8_t*  __restrict input,
    const int8_t*  __restrict weights,
    const int32_t* __restrict bias,
    int8_t*        __restrict output,
    uint16_t       in_features,
    uint16_t       out_features,
    uint8_t        bit_shift = 7
) {
    for (uint16_t row = 0; row < out_features; ++row) {
        int32_t acc = bias[row];
        const int8_t* row_weights = weights + (row * in_features);

        for (uint16_t col = 0; col < in_features; ++col) {
            acc += static_cast<int32_t>(row_weights[col]) * static_cast<int32_t>(input[col]);
        }

        if (acc <= 0) {
            output[row] = 0;
        } else {
            acc = acc >> bit_shift;
            if (acc > 127) {
                acc = 127;
            }
            output[row] = static_cast<int8_t>(acc);
        }
    }
}

/**
 * @brief Executes a Dense Fully-Connected Output Layer returning raw INT32 logits.
 * 
 * Preserves full 32-bit accumulator precision at the final classification head
 * so no quantization rounding occurs prior to argmax selection.
 *
 * @param input         Pointer to INT8 input activation vector.
 * @param weights       Pointer to row-major INT8 weight matrix.
 * @param bias          Pointer to INT32 bias vector.
 * @param output_logits Pointer to INT32 output logits buffer.
 * @param in_features   Number of input features.
 * @param out_features  Number of output classes.
 */
inline void dense_fc_logits(
    const int8_t*  __restrict input,
    const int8_t*  __restrict weights,
    const int32_t* __restrict bias,
    int32_t*       __restrict output_logits,
    uint16_t       in_features,
    uint16_t       out_features
) {
    for (uint16_t row = 0; row < out_features; ++row) {
        int32_t acc = bias[row];
        const int8_t* row_weights = weights + (row * in_features);

        for (uint16_t col = 0; col < in_features; ++col) {
            acc += static_cast<int32_t>(row_weights[col]) * static_cast<int32_t>(input[col]);
        }
        output_logits[row] = acc;
    }
}

/** Sparse final layer with the same INT32 logit contract as dense_fc_logits. */
inline void sparse_fc_logits(
    const int8_t* __restrict input,
    const int8_t* __restrict values,
    const uint8_t* __restrict col_indices,
    const uint16_t* __restrict row_offsets,
    const int32_t* __restrict bias,
    int32_t* __restrict output_logits,
    uint16_t out_features
) {
    for (uint16_t row = 0; row < out_features; ++row) {
        int32_t acc = bias[row];
        for (uint16_t idx = row_offsets[row]; idx < row_offsets[row + 1]; ++idx) {
            acc += static_cast<int32_t>(values[idx]) * static_cast<int32_t>(input[col_indices[idx]]);
        }
        output_logits[row] = acc;
    }
}

/**
 * @brief Returns the class index with the highest INT32 logit score.
 */
inline uint16_t argmax(const int32_t* logits, uint16_t num_classes) {
    uint16_t best_idx = 0;
    int32_t max_val  = logits[0];
    for (uint16_t i = 1; i < num_classes; ++i) {
        if (logits[i] > max_val) {
            max_val  = logits[i];
            best_idx = i;
        }
    }
    return best_idx;
}

} // namespace TinySparse

#endif // TINYSPARSE_H
