#include <Arduino.h>
#include "TinySparse.h"
#include "RingBuffer1D.h"
#include "sweep_weights.h"

#define WINDOW_SIZE   128
#define STRIDE_STEP   128   // Step 1 full window to verify exact multi-class CWRU stream

#define FC1_OUT        32
#define FC2_OUT        16
#define FC3_OUT         4

// Static zero-allocation memory buffers (192B working inference memory)
static RingBuffer1D<WINDOW_SIZE> ringBuffer;
static int8_t  inputWindow[WINDOW_SIZE];
static int8_t  l1_act[FC1_OUT];
static int8_t  l2_act[FC2_OUT];
static int32_t logits[FC3_OUT];

const char* CLASS_NAMES[4] = {
    "Normal Condition",
    "Inner Race Fault",
    "Ball Fault",
    "Outer Race Fault"
};

static inline uint32_t get_ccount() {
    uint32_t ccount;
    __asm__ __volatile__("rsr %0, ccount" : "=r"(ccount));
    return ccount;
}

void setup() {
    Serial.begin(115200);
    uint32_t startWait = millis();
    while (!Serial && (millis() - startWait < 5000)) delay(10);
    delay(1000);

    Serial.println("\n==================================================");
    Serial.println("  TinySparse Continuous Streaming Edge Pipeline   ");
    Serial.println("==================================================");
    Serial.printf("Window Size   : %d samples\n", WINDOW_SIZE);
    Serial.printf("Inference Hop : %d samples (%.1f%% overlap)\n", 
                  STRIDE_STEP, (1.0f - ((float)STRIDE_STEP / WINDOW_SIZE)) * 100.0f);
    Serial.printf("Stream Source : %d Real Held-Out CWRU Test Windows\n", STREAM_NUM_WINDOWS);
    Serial.println("Target Core   : ESP32-S3 (Xtensa LX7 @ 240 MHz)\n");
}

void loop() {
    static uint32_t global_sample_counter = 0;
    static uint32_t inference_event_counter = 0;
    const uint32_t total_stream_samples = STREAM_NUM_WINDOWS * WINDOW_SIZE;

    // Ingest real CWRU test stream sample-by-sample in batches of 16
    for (int b = 0; b < 16; ++b) {
        uint32_t stream_idx = global_sample_counter % total_stream_samples;
        int8_t sample = sweep_stream_input[stream_idx];
        global_sample_counter++;
        ringBuffer.push(sample);

        if (ringBuffer.is_ready(STRIDE_STEP)) {
            uint32_t window_idx = inference_event_counter % STREAM_NUM_WINDOWS;
            uint8_t expected_class = sweep_stream_expected[window_idx];
            inference_event_counter++;

            // 1. Unwrap circular ring buffer
            uint32_t t_unwrap_start = get_ccount();
            ringBuffer.extract_window(inputWindow);
            uint32_t t_unwrap_end = get_ccount();
            uint32_t unwrap_cycles = t_unwrap_end - t_unwrap_start;

            // 2. Execute TinySparse forward pass using the 70% Sparse model
            uint32_t t_infer_start = get_ccount();
            TinySparse::sparse_fc_relu(
                inputWindow, sp70_fc1_val, sp70_fc1_idx, sp70_fc1_off,
                sp70_fc1_b, l1_act, FC1_OUT, SP70_S1
            );
            TinySparse::sparse_fc_relu(
                l1_act, sp70_fc2_val, sp70_fc2_idx, sp70_fc2_off,
                sp70_fc2_b, l2_act, FC2_OUT, SP70_S2
            );
            TinySparse::dense_fc_logits(
                l2_act, sp70_fc3_dense, sp70_fc3_b,
                logits, FC2_OUT, FC3_OUT
            );
            uint8_t predicted_class = TinySparse::argmax(logits, FC3_OUT);
            uint32_t t_infer_end = get_ccount();
            uint32_t infer_cycles = t_infer_end - t_infer_start;

            const char* status = (predicted_class == expected_class) ? "MATCH" : "MISMATCH";

            Serial.printf("[Event #%03u | Sample %05u] Pred: %u (Exp: %u) [%s] [%-16s] | Infer: %5u cyc (%.2f us) | Unwrap: %2u cyc\n",
                          inference_event_counter,
                          global_sample_counter,
                          predicted_class,
                          expected_class,
                          status,
                          CLASS_NAMES[predicted_class],
                          infer_cycles,
                          (float)infer_cycles / 240.0f,
                          unwrap_cycles);
        }
    }

    delay(10);
}