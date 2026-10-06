#include <Arduino.h>
#include <TinySparse.h>

TinySparse sparseEngine;

const uint8_t testVector[64] = {
  0, 0, 12, 0, 0, 0, 0, 0,
  0, 25, 0, 0, 0, 8, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 42, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 15, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 5, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 3, 0, 0, 0
};

const uint8_t sparseValues[7]  = {12, 25, 8, 42, 15, 5, 3};
const uint8_t sparseIndices[7] = {2, 9, 13, 27, 38, 49, 60};

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); } 
  
  Serial.println("\n--- TinySparse Indexed Architecture Benchmark ---");

  uint32_t startDense = micros();
  uint32_t resultDense = 0;
  for (int i = 0; i < 1000; i++) {
    resultDense = sparseEngine.evaluateDense(testVector, 64);
  }
  uint32_t timeDense = micros() - startDense;

  uint32_t startSparse = micros();
  uint32_t resultSparse = 0;
  for (int i = 0; i < 1000; i++) {
    resultSparse = sparseEngine.evaluateSparse(testVector, 64);
  }
  uint32_t timeSparse = micros() - startSparse;

  uint32_t startIndexed = micros();
  uint32_t resultIndexed = 0;
  for (int i = 0; i < 1000; i++) {
    resultIndexed = sparseEngine.evaluateSparseIndexed(sparseValues, sparseIndices, 7);
  }
  uint32_t timeIndexed = micros() - startIndexed;

  Serial.println("\n[ Execution Latency ]");
  Serial.print("Dense Math Time:    "); Serial.print(timeDense); Serial.println(" us");
  Serial.print("Branch Sparse Time: "); Serial.print(timeSparse); Serial.println(" us");
  Serial.print("Indexed Sparse:     "); Serial.print(timeIndexed); Serial.println(" us");

  Serial.print("\nHardware Speedup (vs Dense): ");
  Serial.print((float)timeDense / timeIndexed);
  Serial.println("x faster");

  if (resultDense == resultSparse && resultDense == resultIndexed) {
    Serial.println("Integrity Check:    PASSED");
  } else {
    Serial.println("Integrity Check:    FAILED");
  }
}

void loop() {
  delay(1000);
}