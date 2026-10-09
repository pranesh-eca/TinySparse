#include <Arduino.h>
#include "TinySparse.h"

TinySparse sparseEngine;

void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("--- ADXL345 Live Inference Engine ---");
  Serial.print("Environment: ");
  Serial.println(sparseEngine.getEnvironment());
  
  // Hardware initialization pending I2C/SPI selection
}

void loop() {
  // Sensor reading and infer1D() execution will go here
  delay(100);
}
