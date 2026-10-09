#include <Arduino.h>
#include "TinySparse.h"

TinySparse sparseEngine;

// Simulated 64-bin FFT sensor reading (Hardware Agnostic)
const uint8_t testVector[64] = {
  10, 0, 12, 0, 0, 0, 0, 0,
  0, 25, 0, 0, 45, 8, 0, 0,
  0, 0, 0, 0, 0, 0, 14, 0,
  0, 0, 0, 42, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 15, 0,
  0, 0, 33, 0, 0, 0, 0, 0,
  0, 5, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 3, 0, 0, 0
};

// 17-value sparse weights
const int8_t sparseValues[17] = { 68, 57, 34, 58, 35, 21, 25, 64, 76, 42, -50, 38, 75, 36, 22, 17, 25 }; 
const uint8_t sparseIndices[17] = { 0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18 };

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println("\n--- TinySparse Generalized Engine Benchmark ---");
  Serial.print("Target Environment: ");
  Serial.println(sparseEngine.getEnvironment());
  Serial.println("----------------------------------------------");

  // 1. Direct Production Call via Dynamic Routing
  uint32_t startProduction = micros();
  int32_t resultProduction = 0;
  for (int i = 0; i < 1000; i++) {
    resultProduction = sparseEngine.infer1D(testVector, sparseValues, sparseIndices, 17, 64);
  }
  uint32_t timeProduction = micros() - startProduction;

  // 2. Direct Indexed Execution
  uint32_t startIndexed = micros();
  int32_t resultIndexed = 0;
  for (int i = 0; i < 1000; i++) {
    resultIndexed = sparseEngine.evaluateSparseIndexed(testVector, sparseValues, sparseIndices, 17);
  }
  uint32_t timeIndexed = micros() - startIndexed;

  Serial.println("\n[ Benchmark Results (1000 iterations) ]");
  Serial.print("infer1D (Production Routed): "); Serial.print(timeProduction); Serial.println(" us");
  Serial.print("Direct Indexed COO:          "); Serial.print(timeIndexed); Serial.println(" us");
  Serial.print("Inference Result:            "); Serial.println(resultProduction);
}

void loop() {
  delay(1000);
}
