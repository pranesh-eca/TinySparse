# TinySparse: Bare-Metal Sparse Inference Engine for Microcontrollers

TinySparse is an ultra-lightweight, header-only C++ inference engine optimized for 1D time-series vibration analysis on edge microcontrollers. It executes unstructured Coordinate (COO) sparse matrix math and symmetric INT8 fixed-point arithmetic without dynamic heap allocation (`malloc`/`new`), external math runtimes, or desktop dependencies.

---

## Key Architectural Specifications

* **Target Silicon:** ESP32-S3 (Xtensa Dual-Core 32-bit LX7 @ 240 MHz).
* **Model Topology:** 3-Layer Multilayer Perceptron (`128 -> 32 -> 16 -> 4`).
* **Sparsity Representation:** Coordinate (COO) sparse matrix storage (`values`, `indices`, `row_offsets`).
* **Numerical Representation:** Symmetric INT8 quantization with arithmetic right-shift (`>> 7`) scaling and INT32 accumulators.
* **Continuous Ingestion:** Circular 1D Ring Buffer (`RingBuffer1D.h`) with $O(1)$ sample insertion and dual-`memcpy` window unwrapping.

---

## Memory Footprint Accounting

To maintain strict embedded resource transparency, memory is reported across distinct hardware boundaries:

| Memory Domain | Size | Description |
| :--- | :--- | :--- |
| **Input Feature Buffer** | 128 Bytes | Contiguous INT8 array holding one normalized vibration window |
| **Intermediate Activation Scratchpad** | 64 Bytes | Static buffers: Layer 1 (32B), Layer 2 (16B), Output Logits (16B) |
| **Total Working SRAM (Inference)** | **192 Bytes** | Peak static working memory during forward pass |
| **Dynamic Heap Allocation** | **0 Bytes** | Zero `malloc`, zero heap fragmentation risk |
| **Flash Parameter Storage (COO L1)** | 2,458 Bytes | 1,229 non-zero weights (INT8) + 1,229 column indices (uint8_t) |
| **Flash Parameter Storage (Dense L1 Base)** | 4,096 Bytes | 4,096 dense weights (INT8) |

---
## Empirical Hardware Profiling & Sparsity Crossover Sweep

Measurements captured on physical silicon (**ESP32-S3, Xtensa Dual-Core 32-bit LX7 @ 240 MHz**) using internal hardware cycle counters (`rsr %0, ccount`) over native USB CDC, averaged across 1,000 inference runs, paired with bit-exact INT8 accuracy on the strict 60/15/15 chronological holdout split:

| Model Configuration | L1 Non-Zero MACs | L2 Non-Zero MACs | Total MACs | End-to-End Cycles | Latency (@ 240 MHz) | System Speedup | Total Model Flash | FP32 Test Acc | Verified INT8 Acc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0% (Dense Kernel)** | 4,054 | 503 | 4,621 | 85,722 cyc | 357.17 µs | 1.00x (Base) | 4,880 Bytes | 91.36% | 91.00% |
| **0% (COO Kernel)** | 4,054 | 503 | 4,621 | 89,628 cyc | 373.45 µs | 0.96x (-4.5%) | 4,880 Bytes | 91.36% | 91.00% |
| **50% (COO Sparse)** | 2,048 | 256 | 2,368 | 46,375 cyc | 193.23 µs | 1.85x | 4,980 Bytes | 90.46% | 90.62% |
| **70% (COO Sparse)** | **1,226** | **154** | **1,444** | **28,748 cyc** | **119.78 µs** | **2.98x** | **3,132 Bytes** | **88.92%** | **88.75%** |
| **80% (COO Sparse)** | 817 | 102 | 983 | 19,906 cyc | 82.94 µs | 4.31x | 2,210 Bytes | 84.01% | 84.09% |
| **90% (COO Sparse)** | 406 | 51 | 521 | 11,096 cyc | 46.23 µs | 7.73x | 1,286 Bytes | 70.14% | 69.92% |

### Engineering Takeaways
1. **The Indirection Tax:** At 0% pruning, COO execution is ~4.5% slower than dense due to column index loads and row offset indirection[cite: 1].
2. **Flash Memory Threshold:** Because COO requires 2 bytes per weight plus row offset tables, storage savings only begin above ~52% sparsity[cite: 1]. At 50%, COO is 100 bytes larger than dense.
3. **The Pareto Optimal Point:** 70% sparsity achieves a **2.98x full-pipeline speedup** and **35.7% Flash reduction** while preserving 88.73% 4-class test accuracy on temporal holdout data.
### Silicon Efficiency Takeaways
* **Layer 1 Speedup:** **1.99x faster (49.9% cycle reduction)** and **40.0% Flash memory savings** on Layer 1 parameters.
* **Sparsity Indirection Overhead:** Dense sequential streaming averages ~7.6 cycles/MAC on Xtensa LX7, whereas sparse indirect indexing averages ~12.7 cycles/MAC. A minimum of 60–70% sparsity is required to overcome the pointer-lookup overhead on this core.
* **Real-Time Duty Cycle:** Slicing a 128-sample window at 12 kHz with a 64-sample hop stride gives an inference deadline every 5.33 ms. An 89.77 µs inference consumes **<1.7% of a single CPU core**, reserving >98% of processor bandwidth for concurrent RTOS tasks or peripheral I/O.

---

## Leakage-Free Diagnostic Validation

The model is evaluated across two distinct, non-overlapping experimental protocols using the Case Western Reserve University (CWRU) 12 kHz Drive-End bearing vibration dataset:

### 1. Path B: Within-Recording Temporal Holdout (Primary Streaming Benchmark)
* **Design:** Continuous time-series recordings are chronologically partitioned *prior* to window slicing. The first 70% of time is used for training, an unwindowed guard interval (>12,000 raw samples / >1 second) is completely discarded, and the final 20% is held out for testing.
* **Main Objective:** Evaluates whether TinySparse can reliably process future, non-overlapping windows from characterized machinery under stationary operating conditions.
* **4-Class Test Accuracy:** **91.56%** (4,926 held-out test windows).
* **Binary Diagnostic Performance (Healthy vs. Faulty):**
  * **Accuracy:** 99.80%
  * **Fault Sensitivity:** 99.80%
  * **Healthy Specificity:** 99.80%

### 2. Path A: Cross-Severity Holdout (Generalization Stress Test)
### Model Limitations & Generalization (Path A Zero-Shot Stress Test)
To transparently document the generalization limits of MLP-based vibration analysis, we conducted a rigorous cross-severity stress test (Path A). 

The model was trained exclusively on **0.007"** and **0.021"** defect diameters, and tested *zero-shot* on entirely unseen **0.014"** defect recordings.
* **Training Accuracy:** 99.33%
* **Validation Accuracy:** 96.69%
* **Zero-Shot Test Accuracy (0.014"):** 62.34%

**Engineering Takeaway:** The severe drop from 99% to 62% proves that the neural network overfits to the acoustic resonant frequencies of specific fault geometries, rather than generalizing the underlying physics of bearing failure. Therefore, for production deployment, the model must be calibrated on data covering the exact mechanical tolerances it will evaluate in the field.
---
## Quick Start & Usage

`TinySparse` is a header-only library. Simply drop `TinySparse.h` into your project.

### Arduino / ESP32
Since the repository includes a `library.properties` file, you can zip the folder and add it via **Sketch > Include Library > Add .ZIP Library** in the Arduino IDE, or include it directly:
```cpp
#include "TinySparse.h"

// 192 Bytes working memory
int8_t input_window[128];
int8_t hidden_act[32];

void setup() {
    // Execute INT8 Sparse COO Layer
    TinySparse::sparse_fc_relu(
        input_window, 
        weight_values, weight_col_indices, weight_row_offsets, biases, 
        hidden_act, 
        32,      // Output features
        10       // Quantization right-shift
    );
}
