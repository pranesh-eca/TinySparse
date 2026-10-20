#include <Arduino.h>
#include "TinySparse.h"
#include "sweep_weights.h"

#define WINDOW_SIZE 128
#define FC1_OUT     32
#define FC2_OUT     16
#define FC3_OUT     4
#define NUM_RUNS    1000

// Static Working Memory (192 Bytes Total)
static int8_t  input_buf[WINDOW_SIZE];
static int8_t  l1_out[FC1_OUT];
static int8_t  l2_out[FC2_OUT];
static int32_t logits[FC3_OUT];

static inline uint32_t get_ccount() {
    uint32_t ccount;
    __asm__ __volatile__("rsr %0, ccount" : "=r"(ccount));
    return ccount;
}

struct SweepResult {
    int      sparsity_pct;
    uint32_t l1_cycles;
    uint32_t l2_cycles;
    uint32_t l3_cycles;
    uint32_t total_cycles;
    uint32_t flash_bytes;
    float    int8_acc;
    uint8_t  pred_class;
};

// Macro accepts both lowercase array prefix and uppercase constant prefix
#define PROFILE_SPARSE_MODEL(P_LOW, P_UP, SP_PCT, RESULT_STRUCT) \
    do { \
        uint32_t sum_l1 = 0, sum_l2 = 0, sum_l3 = 0, sum_tot = 0; \
        uint8_t pred = 0; \
        for (int r = 0; r < NUM_RUNS; ++r) { \
            uint32_t t0 = get_ccount(); \
            TinySparse::sparse_fc_relu(input_buf, P_LOW##_fc1_val, P_LOW##_fc1_idx, P_LOW##_fc1_off, P_LOW##_fc1_b, l1_out, FC1_OUT, P_UP##_S1); \
            uint32_t t1 = get_ccount(); \
            TinySparse::sparse_fc_relu(l1_out, P_LOW##_fc2_val, P_LOW##_fc2_idx, P_LOW##_fc2_off, P_LOW##_fc2_b, l2_out, FC2_OUT, P_UP##_S2); \
            uint32_t t2 = get_ccount(); \
            TinySparse::dense_fc_logits(l2_out, P_LOW##_fc3_dense, P_LOW##_fc3_b, logits, FC2_OUT, FC3_OUT); \
            pred = TinySparse::argmax(logits, FC3_OUT); \
            uint32_t t3 = get_ccount(); \
            sum_l1  += (t1 - t0); \
            sum_l2  += (t2 - t1); \
            sum_l3  += (t3 - t2); \
            sum_tot += (t3 - t0); \
        } \
        RESULT_STRUCT.sparsity_pct = SP_PCT; \
        RESULT_STRUCT.l1_cycles    = sum_l1 / NUM_RUNS; \
        RESULT_STRUCT.l2_cycles    = sum_l2 / NUM_RUNS; \
        RESULT_STRUCT.l3_cycles    = sum_l3 / NUM_RUNS; \
        RESULT_STRUCT.total_cycles = sum_tot / NUM_RUNS; \
        RESULT_STRUCT.flash_bytes  = P_UP##_FLASH_BYTES; \
        RESULT_STRUCT.int8_acc     = P_UP##_INT8_ACC; \
        RESULT_STRUCT.pred_class   = pred; \
    } while (0)

void setup() {
    Serial.begin(115200);
    uint32_t startWait = millis();
    while (!Serial && (millis() - startWait < 5000)) delay(10);
    delay(1000);

    for (int i = 0; i < WINDOW_SIZE; ++i) {
        input_buf[i] = sweep_test_input[i];
    }

    Serial.println("\n==========================================================================================");
    Serial.println("  TinySparse Phase 3: Empirical Multi-Sparsity Crossover Sweep (ESP32-S3 @ 240 MHz)");
    Serial.println("==========================================================================================");

    // 1. Profile 0% Dense Baseline
    SweepResult res_dense;
    {
        uint32_t sum_l1 = 0, sum_l2 = 0, sum_l3 = 0, sum_tot = 0;
        uint8_t pred = 0;
        for (int r = 0; r < NUM_RUNS; ++r) {
            uint32_t t0 = get_ccount();
            TinySparse::dense_fc_relu(input_buf, sp0_fc1_dense, sp0_fc1_b, l1_out, WINDOW_SIZE, FC1_OUT, SP0_S1);
            uint32_t t1 = get_ccount();
            TinySparse::dense_fc_relu(l1_out, sp0_fc2_dense, sp0_fc2_b, l2_out, FC1_OUT, FC2_OUT, SP0_S2);
            uint32_t t2 = get_ccount();
            TinySparse::dense_fc_logits(l2_out, sp0_fc3_dense, sp0_fc3_b, logits, FC2_OUT, FC3_OUT);
            pred = TinySparse::argmax(logits, FC3_OUT);
            uint32_t t3 = get_ccount();
            sum_l1  += (t1 - t0);
            sum_l2  += (t2 - t1);
            sum_l3  += (t3 - t2);
            sum_tot += (t3 - t0);
        }
        res_dense = {0, sum_l1 / NUM_RUNS, sum_l2 / NUM_RUNS, sum_l3 / NUM_RUNS, sum_tot / NUM_RUNS, SP0_FLASH_BYTES, SP0_INT8_ACC, pred};
    }

    // 2. Profile Sparse Models (0% COO overhead check, 50%, 70%, 80%, 90%)
    SweepResult res_sp0_coo, res_sp50, res_sp70, res_sp80, res_sp90;
    PROFILE_SPARSE_MODEL(sp0,  SP0,   0, res_sp0_coo);
    PROFILE_SPARSE_MODEL(sp50, SP50, 50, res_sp50);
    PROFILE_SPARSE_MODEL(sp70, SP70, 70, res_sp70);
    PROFILE_SPARSE_MODEL(sp80, SP80, 80, res_sp80);
    PROFILE_SPARSE_MODEL(sp90, SP90, 90, res_sp90);

    SweepResult all_results[6] = {res_dense, res_sp0_coo, res_sp50, res_sp70, res_sp80, res_sp90};
    const char* labels[6] = {
        "0% (Dense Kernel)",
        "0% (COO Kernel)  ",
        "50% (COO Sparse) ",
        "70% (COO Sparse) ",
        "80% (COO Sparse) ",
        "90% (COO Sparse) "
    };

    Serial.println("Model Config      | L1 Cyc | L2 Cyc | L3 Cyc | Total Cyc | Latency (us) | Speedup | Flash (B) | INT8 Acc");
    Serial.println("--------------------------------------------------------------------------------------------------------");
    for (int i = 0; i < 6; ++i) {
        float latency_us = (float)all_results[i].total_cycles / 240.0f;
        float speedup    = (float)res_dense.total_cycles / (float)all_results[i].total_cycles;
        Serial.printf("%s | %6u | %6u | %6u | %9u | %9.2f us |  %5.2fx | %7u B |  %6.2f%%\n",
                      labels[i],
                      all_results[i].l1_cycles,
                      all_results[i].l2_cycles,
                      all_results[i].l3_cycles,
                      all_results[i].total_cycles,
                      latency_us,
                      speedup,
                      all_results[i].flash_bytes,
                      all_results[i].int8_acc);
    }
    Serial.println("==========================================================================================");
}

void loop() {
    delay(5000);
}