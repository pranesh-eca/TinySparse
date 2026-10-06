#include <Arduino.h>
#include <TinySparse.h>

TinySparse sparseEngine;

// A synthetic 64-element array representing an 8x8 image block with ~85% sparsity
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

void setup() {
  Serial.begin(115200);
  // Wait for the serial monitor to open before printing
  while (!Serial) { delay(10); } 
  
  Serial.println("\n--- TinySparse Hardware Benchmark ---");
  Serial.println("Running 1,000 iterations for precision...");

  // 1. Benchmark Dense Execution
  uint32_t startDense = micros();
  uint32_t resultDense = 0;
  for (int i = 0; i < 1000; i++) {
    resultDense = sparseEngine.evaluateDense(testVector, sizeof(testVector));
  }
  uint32_t timeDense = micros() - startDense;

  // 2. Benchmark Sparse Execution
  uint32_t startSparse = micros();
  uint32_t resultSparse = 0;
  for (int i = 0; i < 1000; i++) {
    resultSparse = sparseEngine.evaluateSparse(testVector, sizeof(testVector));
  }
  uint32_t timeSparse = micros() - startSparse;

  // 3. Print Results to Serial Monitor
  Serial.println("\n[ Execution Latency ]");
  Serial.print("Dense Math Time:  ");
  Serial.print(timeDense);
  Serial.println(" us");
  
  Serial.print("Sparse Math Time: ");
  Serial.print(timeSparse);
  Serial.println(" us");

  Serial.print("\nHardware Speedup: ");
  Serial.print((float)timeDense / timeSparse);
  Serial.println("x faster");

  // 4. Verify Mathematical Integrity
  if (resultDense == resultSparse) {
    Serial.println("Integrity Check:  PASSED (Outputs perfectly match)");
  } else {
    Serial.println("Integrity Check:  FAILED (Outputs differ!)");
  }
}

void loop() {
  delay(1000);
}