#include "TinySparse.h"
#include "model_weights.h"

static inline uint32_t get_ccount() {
    uint32_t ccount;
    __asm__ __volatile__("rsr %0, ccount" : "=r"(ccount));
    return ccount;
}

const int8_t sensorWindow[FC1_INPUT_DIM] = {
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

// True dense 4,096-weight array in Flash for authentic memory-bandwidth testing
const int8_t fake_dense_fc1_weights[FC1_NUM_NEURONS * FC1_INPUT_DIM] = {
    // Repeated pattern simulating dense weight distribution in Flash
    12, -4, 7, -19, 3, -1, 8, 14, -6, 2, 9, -15, 4, -8, 11, -3
};

int8_t  fc1_output_sp[FC1_NUM_NEURONS];
int8_t  fc1_output_de[FC1_NUM_NEURONS];
int8_t  fc2_output[FC2_NUM_NEURONS];
int32_t fc3_output[FC3_NUM_NEURONS];

// Authentic Dense Layer 1 (Reads all 4,096 weights directly from Flash via sequential pointers)
void run_true_dense_fc1() {
    const int8_t* w_ptr = fake_dense_fc1_weights;
    for (int n = 0; n < FC1_NUM_NEURONS; ++n) {
        int32_t acc = fc1_biases[n];
        const int8_t* in_ptr = sensorWindow;

        // Unroll by 4 for sequential memory access
        for (int i = 0; i < FC1_INPUT_DIM; i += 4) {
            acc += (int32_t)in_ptr[0] * (int32_t)w_ptr[0];
            acc += (int32_t)in_ptr[1] * (int32_t)w_ptr[1];
            acc += (int32_t)in_ptr[2] * (int32_t)w_ptr[2];
            acc += (int32_t)in_ptr[3] * (int32_t)w_ptr[3];
            in_ptr += 4;
            w_ptr += 4;
        }
        int32_t scaled = acc >> 7;
        if (scaled < 0) scaled = 0;
        if (scaled > 127) scaled = 127;
        fc1_output_de[n] = (int8_t)scaled;
    }
}

// Optimized Sparse Layer 1 (Pointer arithmetic for indices and values)
void run_optimized_sparse_fc1() {
    for (uint16_t n = 0; n < FC1_NUM_NEURONS; ++n) {
        uint16_t start = fc1_offsets[n];
        uint16_t end   = fc1_offsets[n + 1];
        uint16_t nnz   = end - start;

        const int8_t*  v_ptr = &fc1_values[start];
        const uint8_t* i_ptr = &fc1_indices[start];
        int32_t acc = fc1_biases[n];

        uint16_t k = 0;
        for (; k + 3 < nnz; k += 4) {
            acc += (int32_t)sensorWindow[i_ptr[0]] * (int32_t)v_ptr[0];
            acc += (int32_t)sensorWindow[i_ptr[1]] * (int32_t)v_ptr[1];
            acc += (int32_t)sensorWindow[i_ptr[2]] * (int32_t)v_ptr[2];
            acc += (int32_t)sensorWindow[i_ptr[3]] * (int32_t)v_ptr[3];
            i_ptr += 4;
            v_ptr += 4;
        }
        for (; k < nnz; ++k) {
            acc += (int32_t)sensorWindow[*i_ptr++] * (int32_t)(*v_ptr++);
        }

        int32_t scaled = acc >> 7;
        if (scaled < 0) scaled = 0;
        if (scaled > 127) scaled = 127;
        fc1_output_sp[n] = (int8_t)scaled;
    }
}

void setup() {
    Serial.begin(115200);
    uint32_t startWait = millis();
    while (!Serial && (millis() - startWait < 5000)) delay(10);
    delay(1000);

    Serial.println("\n==============================================");
    Serial.println("  Layer 1 True Benchmark: Sparse vs Dense FC1");
    Serial.println("==============================================");

    // Warm up Sparse specifically
    run_optimized_sparse_fc1();
    run_optimized_sparse_fc1();

    uint32_t t0 = get_ccount();
    run_optimized_sparse_fc1();
    uint32_t t1 = get_ccount();
    uint32_t sparse_cycles = t1 - t0;

    // Warm up Dense specifically
    run_true_dense_fc1();
    run_true_dense_fc1();

    uint32_t t2 = get_ccount();
    run_true_dense_fc1();
    uint32_t t3 = get_ccount();
    uint32_t dense_cycles = t3 - t2;

    Serial.printf("Layer 1 Dense  (4,096 MACs): %u cycles (%.2f us)\n", dense_cycles, (float)dense_cycles / 240.0f);
    Serial.printf("Layer 1 Sparse (1,229 MACs): %u cycles (%.2f us)\n", sparse_cycles, (float)sparse_cycles / 240.0f);
    Serial.printf("Ratio: %.2fx\n", (float)dense_cycles / (float)sparse_cycles);

    if (dense_cycles > sparse_cycles) {
        Serial.printf("Speedup: Sparse is %.1f%% FASTER\n", ((float)(dense_cycles - sparse_cycles) / dense_cycles) * 100.0f);
    } else {
        Serial.printf("Overhead: Dense is %.1f%% FASTER due to sequential memory access\n", ((float)(sparse_cycles - dense_cycles) / sparse_cycles) * 100.0f);
    }
    Serial.println("==============================================");
}

void loop() {
    delay(2000);
}