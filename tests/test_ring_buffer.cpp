#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "RingBuffer1D.h"

int main() {
    printf("======================================================================\n");
    printf(" Unit Test: RingBuffer1D Boundary & Unwrapping Verification\n");
    printf("======================================================================\n");

    const size_t W = 128;
    const size_t STRIDE = 64;
    RingBuffer1D<W> ring;
    int8_t out_window[W];

    // Test 1: Buffer underfill behavior
    for (int8_t i = 0; i < 50; ++i) {
        ring.push(i);
    }
    assert(!ring.is_ready(STRIDE));
    printf("[*] Test 1: Underfill ready-check passed (count = %zu / %zu)\n", ring.size(), W);

    // Test 2: Fill to exactly W elements
    for (int8_t i = 50; i < 127; ++i) {
        ring.push(i);
    }
    ring.push(127);
    assert(ring.is_ready(STRIDE));
    printf("[*] Test 2: Full-capacity threshold passed (count = %zu)\n", ring.size());

    // Extract first full window
    ring.extract_window(out_window);
    for (size_t i = 0; i < W; ++i) {
        assert(out_window[i] == (int8_t)i);
    }
    printf("[*] Test 3: Linear unwrap fidelity passed for first window\n");

    // Stride counter should reset after extraction
    assert(!ring.is_ready(STRIDE));

    // Test 4: Push next STRIDE elements to trigger wraparound
    for (size_t i = 0; i < STRIDE; ++i) {
        ring.push((int8_t)(100 + i));
    }
    assert(ring.is_ready(STRIDE));

    ring.extract_window(out_window);
    // Oldest STRIDE elements should be discarded; newer elements appended at tail
    for (size_t i = 0; i < (W - STRIDE); ++i) {
        assert(out_window[i] == (int8_t)(STRIDE + i));
    }
    for (size_t i = 0; i < STRIDE; ++i) {
        assert(out_window[(W - STRIDE) + i] == (int8_t)(100 + i));
    }
    printf("[*] Test 4: Wraparound and stride alignment passed (two-segment copy)\n");

    printf("----------------------------------------------------------------------\n");
    printf(" All RingBuffer1D unit tests PASSED cleanly.\n");
    printf("======================================================================\n");
    return 0;
}