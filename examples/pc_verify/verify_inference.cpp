#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "TinySparse.h"
#include "model_weights.h"

// Static Working Memory Allocation (192 Bytes Total)
static int8_t  input_buf[128];            // 128 Bytes
static int8_t  l1_out_buf[FC1_OUT_DIM];   //  32 Bytes
static int8_t  l2_out_buf[FC2_OUT_DIM];   //  16 Bytes
static int32_t logits_buf[FC3_OUT_DIM];   //  16 Bytes

int main() {
    printf("======================================================================\n");
    printf(" TinySparse Host C++ Bit-Exact Numerical Verification Suite\n");
    printf("======================================================================\n");

    // Copy golden input into working input buffer
    for (int i = 0; i < 128; ++i) {
        input_buf[i] = golden_input[i];
    }

    // 1. Execute Layer 1 (Sparse COO)
    TinySparse::sparse_fc_relu(
        input_buf, fc1_values, fc1_col_indices, fc1_row_offsets,
        fc1_bias, l1_out_buf, FC1_OUT_DIM, FC1_SHIFT
    );

    // Verify Layer 1 Bit-Exactness
    int l1_errors = 0;
    for (int i = 0; i < FC1_OUT_DIM; ++i) {
        if (l1_out_buf[i] != golden_l1_out[i]) {
            printf("[FAIL] L1 Index %d: C++=%d vs Python=%d\n", i, l1_out_buf[i], golden_l1_out[i]);
            l1_errors++;
        }
    }
    printf("[*] Layer 1 (128->32, Shift >> %d) Bit-Exact Check : %s (%d/%d matched)\n",
           FC1_SHIFT, (l1_errors == 0) ? "PASSED" : "FAILED", FC1_OUT_DIM - l1_errors, FC1_OUT_DIM);

    // 2. Execute Layer 2 (Sparse COO)
    TinySparse::sparse_fc_relu(
        l1_out_buf, fc2_values, fc2_col_indices, fc2_row_offsets,
        fc2_bias, l2_out_buf, FC2_OUT_DIM, FC2_SHIFT
    );

    // Verify Layer 2 Bit-Exactness
    int l2_errors = 0;
    for (int i = 0; i < FC2_OUT_DIM; ++i) {
        if (l2_out_buf[i] != golden_l2_out[i]) {
            printf("[FAIL] L2 Index %d: C++=%d vs Python=%d\n", i, l2_out_buf[i], golden_l2_out[i]);
            l2_errors++;
        }
    }
    printf("[*] Layer 2 (32->16,  Shift >> %d) Bit-Exact Check : %s (%d/%d matched)\n",
           FC2_SHIFT, (l2_errors == 0) ? "PASSED" : "FAILED", FC2_OUT_DIM - l2_errors, FC2_OUT_DIM);

    // 3. Execute Layer 3 (Dense Output Head)
    TinySparse::dense_fc_logits(
        l2_out_buf, fc3_weights_dense, fc3_bias,
        logits_buf, FC3_IN_DIM, FC3_OUT_DIM
    );

    // Verify Layer 3 Logits Bit-Exactness
    int l3_errors = 0;
    for (int i = 0; i < FC3_OUT_DIM; ++i) {
        if (logits_buf[i] != golden_logits[i]) {
            printf("[FAIL] Logit %d: C++=%d vs Python=%d\n", i, logits_buf[i], golden_logits[i]);
            l3_errors++;
        }
    }
    printf("[*] Layer 3 (16->4,   INT32 Logits) Bit-Exact Check: %s (%d/%d matched)\n",
           (l3_errors == 0) ? "PASSED" : "FAILED", FC3_OUT_DIM - l3_errors, FC3_OUT_DIM);

    uint8_t pred = TinySparse::argmax(logits_buf, FC3_OUT_DIM);
    printf("----------------------------------------------------------------------\n");
    printf("  Output Logits        : [%d, %d, %d, %d]\n",
           logits_buf[0], logits_buf[1], logits_buf[2], logits_buf[3]);
    printf("  Predicted Class      : %u (Expected Golden Class: %u)\n", pred, GOLDEN_EXPECTED_CLASS);
    printf("  Activation Scratchpad: %zu Bytes (L1: 32B, L2: 16B, Logits: 16B)\n",
           sizeof(l1_out_buf) + sizeof(l2_out_buf) + sizeof(logits_buf));
    printf("  Total Working Buffers: %zu Bytes (including 128B input window)\n",
           sizeof(input_buf) + sizeof(l1_out_buf) + sizeof(l2_out_buf) + sizeof(logits_buf));
    printf("======================================================================\n");

    if (l1_errors > 0 || l2_errors > 0 || l3_errors > 0 || pred != GOLDEN_EXPECTED_CLASS) {
        return 1;
    }
    return 0;
}