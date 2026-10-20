#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "TinySparse.h"

int main() {
    printf("==================================================\n");
    printf(" Unit Test: TinySparse Edge-Case Kernels\n");
    printf("==================================================\n");

    // Test 1: Saturation Clamping (Values > 127 must clamp to 127)
    int8_t input1[2] = {100, 100};
    int8_t w1[2] = {10, 10}; // 1000 + 1000 = 2000
    int32_t b1[1] = {0};
    int8_t out1[1];
    TinySparse::dense_fc_relu(input1, w1, b1, out1, 2, 1, 0); // shift=0
    assert(out1[0] == 127);
    printf("[*] Saturation Clamping: PASSED\n");

    // Test 2: Negative ReLU Clamping (Values < 0 must clamp to 0)
    int8_t input2[2] = {10, 10};
    int8_t w2[2] = {-5, -5}; // -50 - 50 = -100
    int32_t b2[1] = {0};
    int8_t out2[1];
    TinySparse::dense_fc_relu(input2, w2, b2, out2, 2, 1, 0);
    assert(out2[0] == 0);
    printf("[*] Negative ReLU Clamping: PASSED\n");

    // Test 3: Sparse COO Zero-Row Handling
    int8_t input3[4] = {10, 20, 30, 40};
    int8_t val3[2] = {1, 1};
    uint8_t col3[2] = {0, 3}; // Selects 10 and 40
    uint16_t off3[3] = {0, 0, 2}; // Row 0 is empty (0 to 0), Row 1 has 2 elements (0 to 2)
    int32_t b3[2] = {5, 0};
    int8_t out3[2];
    TinySparse::sparse_fc_relu(input3, val3, col3, off3, b3, out3, 2, 0);
    assert(out3[0] == 5);  // Empty row + bias
    assert(out3[1] == 50); // (10*1) + (40*1) + 0
    printf("[*] Sparse COO Zero-Row Logic: PASSED\n");

    printf("==================================================\n");
    return 0;
}