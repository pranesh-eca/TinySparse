#include <Arduino.h>
#include "TinySparse.h"
#include "RingBuffer1D.h"
#include "model_weights.h"

#define WINDOW_SIZE   128
#define STRIDE_STEP    64   // Hop stride: trigger inference every 64 new samples (50% overlap)

// Static zero-allocation memory buffers (192B working inference memory)
static RingBuffer1D<WINDOW_SIZE> ringBuffer;
static int8_t  inputWindow[WINDOW_SIZE];
static int8_t  l1_act[FC1_OUT_DIM];
static int8_t  l2_act[FC2_OUT_DIM];
static int32_t logits[FC3_OUT_DIM];

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

// Simulates continuous sensor acquisition stream transitioning from Healthy to Damaged
int8_t read_simulated_sensor(uint32_t sample_idx) {
    float t = (float)sample_idx * 0.05f;
    float wave = 0.0f;

    if (sample_idx < 400) {
        // Stage 1: Healthy bearing vibration baseline
        wave = 15.0f * sin(t) + ((rand() % 10) - 5);
    } else {
        // Stage 2: Bearing impact shock ring
        wave = 65.0f * sin(t * 3.7f) + 30.0f * sin(t * 8.1f) + ((rand() % 20) - 10);
    }

    if (wave > 127.0f)  wave = 127.0f;
    if (wave < -128.0f) wave = -128.0f;
    return (int8_t)wave;
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
    Serial.printf("Inference Hop : %d samples (50.0%% overlap)\n", STRIDE_STEP);
    Serial.println("Target Core   : ESP32-S3 (Xtensa LX7 @ 240 MHz)\n");
}

void loop() {
    static uint32_t global_sample_counter = 0;
    static uint32_t inference_event_counter = 0;

    // Ingest stream in batches
    for (int b = 0; b < 16; ++b) {
        int8_t sample = read_simulated_sensor(global_sample_counter++);
        ringBuffer.push(sample);

        if (ringBuffer.is_ready(STRIDE_STEP)) {
            inference_event_counter++;

            // 1. Unwrap circular ring buffer
            uint32_t t_unwrap_start = get_ccount();
            ringBuffer.extract_window(inputWindow);
            uint32_t t_unwrap_end = get_ccount();
            uint32_t unwrap_cycles = t_unwrap_end - t_unwrap_start;

            // 2. Execute TinySparse forward pass
            uint32_t t_infer_start = get_ccount();
            TinySparse::sparse_fc_relu(
                inputWindow, fc1_values, fc1_col_indices, fc1_row_offsets,
                fc1_bias, l1_act, FC1_OUT_DIM, FC1_SHIFT
            );
            TinySparse::sparse_fc_relu(
                l1_act, fc2_values, fc2_col_indices, fc2_row_offsets,
                fc2_bias, l2_act, FC2_OUT_DIM, FC2_SHIFT
            );
            TinySparse::dense_fc_logits(
                l2_act, fc3_weights_dense, fc3_bias,
                logits, FC3_IN_DIM, FC3_OUT_DIM
            );
            uint8_t predicted_class = TinySparse::argmax(logits, FC3_OUT_DIM);
            uint32_t t_infer_end = get_ccount();
            uint32_t infer_cycles = t_infer_end - t_infer_start;

            Serial.printf("[Event #%03u | Sample %05u] Class: %u [%-17s] | Infer: %5u cyc (%.2f us) | Unwrap: %2u cyc\n",
                          inference_event_counter,
                          global_sample_counter,
                          predicted_class,
                          CLASS_NAMES[predicted_class],
                          infer_cycles,
                          (float)infer_cycles / 240.0f,
                          unwrap_cycles);
        }
    }

    delay(5);
}