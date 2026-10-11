#include <stdio.h>
#include <stdint.h>
#include "TinySparse.h"
#include "model_weights.h"

int main() {
    printf("=== Step 6: 1D Sensor Sparse Inference Test ===\n");

    // Real normalized 1D vibration window (Class 1: Inner Race Fault sample)
    // Scaled into signed INT8 range [-128, 127]
    int8_t sensorWindow[FC1_INPUT_DIM] = {
         -4,   12,   28,  -19,  -35,   10,   45,  -12,
         18,  -50,  -82,   34,   91,  -15,  -77,   42,
         22,  -61,  -19,   55,   31,  -48,  -22,   67,
         14,  -33,  -45,   29,   70,  -12,  -66,   38,
         -8,   21,   39,  -24,  -41,   15,   52,  -18,
         25,  -58,  -79,   31,   88,  -12,  -74,   45,
         19,  -64,  -16,   58,   28,  -45,  -19,   64,
         11,  -30,  -42,   26,   67,  -15,  -63,   35,
         -2,   15,   32,  -17,  -38,   12,   48,  -15,
         21,  -53,  -75,   28,   85,  -10,  -71,   41,
         16,  -60,  -13,   54,   25,  -42,  -16,   61,
          8,  -27,  -39,   23,   64,  -18,  -60,   32,
         -6,   18,   35,  -21,  -36,   14,   50,  -14,
         23,  -55,  -77,   30,   87,  -11,  -73,   43,
         17,  -62,  -15,   56,   27,  -44,  -18,   63,
         10,  -29,  -41,   25,   66,  -16,  -62,   34
    };

    // Intermediate activation buffers allocated on the stack (zero dynamic heap)
    int8_t  fc1_output[FC1_NUM_NEURONS]; // 32 bytes
    int8_t  fc2_output[FC2_NUM_NEURONS]; // 16 bytes
    int32_t fc3_output[FC3_NUM_NEURONS]; // 16 bytes (4 x 4-byte logits)

    // 1. Forward Layer 1: 1D Sparse COO (128 -> 32)
    TinySparse::forwardSparseLayer1D<uint8_t>(
        sensorWindow,
        fc1_values,
        fc1_indices,
        fc1_offsets,
        fc1_biases,
        FC1_NUM_NEURONS,
        fc1_output
    );

    // 2. Forward Layer 2: 1D Sparse COO (32 -> 16)
    TinySparse::forwardSparseLayer1D<uint8_t>(
        fc1_output,
        fc2_values,
        fc2_indices,
        fc2_offsets,
        fc2_biases,
        FC2_NUM_NEURONS,
        fc2_output
    );

    // 3. Forward Layer 3: Dense Fallback (16 -> 4 classes)
    TinySparse::forwardDenseLayer1D(
        fc2_output,
        fc3_weights,
        fc3_biases,
        FC3_INPUT_DIM,
        FC3_NUM_NEURONS,
        fc3_output
    );

    // 4. Output Classification
    uint8_t predicted = TinySparse::argmax(fc3_output, FC3_NUM_NEURONS);
    const char* classNames[] = {"Normal", "Inner Race Fault", "Ball Fault", "Outer Race Fault"};

    printf("\n--- Inference Results ---\n");
    for (int i = 0; i < FC3_NUM_NEURONS; ++i) {
        printf("Class %d [%s]: Logit = %d\n", i, classNames[i], fc3_output[i]);
    }
    printf("\nPredicted Diagnosis: Class %d (%s)\n", predicted, classNames[predicted]);
    printf("Total SRAM Used for Activations: %zu Bytes\n", sizeof(fc1_output) + sizeof(fc2_output) + sizeof(fc3_output));

    return 0;
}