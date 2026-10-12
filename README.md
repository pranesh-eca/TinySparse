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

## Empirical Hardware Profiling (ESP32-S3 @ 240 MHz)

Execution cycles measured directly on silicon using internal processor cycle counters (`rsr %0, ccount`) over native USB CDC:

| Execution Stage | Architecture | Compute Workload | CPU Cycles | Latency (@ 240 MHz) |
| :--- | :--- | :--- | :--- | :--- |
| **Layer 1 (128 -> 32)** | **Sparse COO (70% Sparsity)** | **1,229 MACs** | **15,591** | **64.96 µs** |
| *Layer 1 Baseline* | *Dense Kernel (0% Sparsity)* | *4,096 MACs* | *31,091* | *129.55 µs* |
| **Layer 2 (32 -> 16)** | Sparse COO (70% Sparsity) | 154 MACs | 2,987 | 12.45 µs |
| **Layer 3 (16 -> 4)** | Dense Kernel (Output Logits) | 64 MACs | 932 | 3.88 µs |
| **Total Engine Latency** | **Hybrid TinySparse Pipeline** | **1,447 MACs** | **21,545** | **89.77 µs** |

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
* **Design:** The network is trained solely on mild (0.007") and severe (0.021") defect diameters, plus healthy baselines. The intermediate defect severity (0.014") is completely withheld and evaluated zero-shot.
* **Main Objective:** Evaluates whether raw time-domain waveforms transfer across defect geometries without frequency-domain transformations.
* **4-Class Test Accuracy:** **61.92%** (Geometric phase shifts across defect sizes create ambiguity between Inner and Outer race fault modes).
* **Binary Diagnostic Performance (Healthy vs. Faulty):**
  * **Accuracy:** 97.25%
  * **Fault Sensitivity:** 96.69%
  * **Healthy Specificity:** 99.34%

---

## Repository Structure

```text
├── include/
│   ├── TinySparse.h              # Header-only bare-metal C++ inference core
│   └── RingBuffer1D.h            # O(1) circular ring buffer for continuous streams
├── scripts/
│   ├── prepare_cwru_dataset.py   # Path A cross-severity dataset generator
│   ├── train_baseline.py         # Path A training and evaluation pipeline
│   ├── prepare_cwru_dataset_temporal.py  # Path B chronological holdout generator
│   ├── train_temporal_baseline.py        # Path B training and evaluation pipeline
│   ├── prune_and_quantize.py     # L1 unstructured pruning and INT8 export
│   └── tinysparse_compiler.py    # Auto-generation of model_weights.h
├── models/                       # Processed datasets and PyTorch weight checkpoints
└── examples/
    ├── esp32s3_benchmark/        # Hardware cycle benchmark sketch for ESP32-S3
    └── esp32s3_streaming/        # Continuous ring-buffer streaming ingestion sketch
