#include <Arduino.h>
#include "TinySparse.h"

TinySparse sparseEngine;

#define INPUT_SIZE 64
uint8_t liveDataBuffer[INPUT_SIZE];

// Your validated 17-weight sparse model
const int8_t sparseValues[17] = { 68, 57, 34, 58, 35, 21, 25, 64, 76, 42, -50, 38, 75, 36, 22, 17, 25 }; 
const uint16_t sparseIndices[17] = { 0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18 };

/**
 * HARDWARE AGNOSTIC INGESTION
 * Replace the contents of this function with your actual real-time read.
 * e.g., I2C/SPI accelerometer burst read, analogRead(), or camera DMA transfer.
 */
void fetchLiveSensorData(uint8_t* buffer, uint16_t size) {
  for(uint16_t i = 0; i < size; i++) {
    // Simulating live, fluctuating sensor data 
    // Replace with: buffer[i] = readHardware();
    buffer[i] = random(0, 100); 
  }
}

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println("\n--- TinySparse Real-Time Pipeline ---");
  Serial.print("Target Environment: ");
  Serial.println(sparseEngine.getEnvironment());
  Serial.println("-------------------------------------");
}

void loop() {
  // 1. Data Acquisition
  uint32_t fetchStart = micros();
  fetchLiveSensorData(liveDataBuffer, INPUT_SIZE);
  uint32_t fetchTime = micros() - fetchStart;

  // 2. Execution (Dynamic Routing)
  uint32_t inferStart = micros();
  int32_t result = sparseEngine.infer1D(liveDataBuffer, sparseValues, sparseIndices, 17, INPUT_SIZE);
  uint32_t inferTime = micros() - inferStart;

  // 3. Telemetry
  Serial.print("[Pipeline] Fetch: "); Serial.print(fetchTime); Serial.print(" us | ");
  Serial.print("Inference: "); Serial.print(inferTime); Serial.print(" us | ");
  Serial.print("Output: "); Serial.println(result);
  
  // Maintain a steady 10Hz sampling rate loop
  delay(100); 
}
