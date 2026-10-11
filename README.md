# TinySparse

TinySparse is an ultra-lightweight, bare-metal C++ inference engine designed for edge microcontrollers. It accelerates 1D sensor time-series classification using structured INT8 quantization and unstructured Coordinate (COO) sparse execution.

Tested and validated on **ESP32-S3 (Xtensa Dual-Core LX7 @ 240 MHz)** using Case Western Reserve University (CWRU) bearing vibration data.

---

## Key Highlights

- **Zero Dynamic Allocation:** 100% static SRAM usage—no `malloc` or heap fragmentation.
- **Hardware Validated:** 1.99x speedup on Layer 1 over dense baseline on bare-metal silicon.
- **Ultra-Low Memory Footprint:** Uses only **64 bytes** of working SRAM and **2.4 KB** of Flash memory.
- **End-to-End Pipeline:** Complete PyTorch training, L1 unstructured pruning, INT8 quantization, and C++ header generation.

---

## ESP32-S3 Hardware Profile (240 MHz)

### Latency Breakdown

| Execution Stage | Kernel Type | Workload | Latency (Cycles) | Latency (µs) |
| :--- | :--- | :--- | :--- | :--- |
| **Layer 1 (128 → 32)** | Sparse COO | 1,229 MACs (70% Sparse) | 15,591 | 64.96 µs |
| **Layer 2 (32 → 16)**  | Sparse COO | 154 MACs (70% Sparse)   | 2,987  | 12.45 µs |
| **Layer 3 (16 → 4)**   | Dense Fallback | 64 MACs             | 932    | 3.88 µs  |
| **Total Inference**    | **End-to-End** | **1,447 MACs**      | **21,545** | **89.77 µs** |

### Dense vs. Sparse Comparison (Layer 1)

| Metric | Dense Baseline | TinySparse (70% Pruned) | Gain |
| :--- | :--- | :--- | :--- |
| **MAC Count** | 4,096 | 1,229 | **-70.0%** |
| **Flash Usage** | 4,096 Bytes | 2,458 Bytes | **-40.0%** |
| **Execution Cycles** | 31,091 | 15,591 | **-49.9% (1.99x speedup)** |
| **Classification Accuracy** | 99.13% | 97.70% | -1.43% |
| **Peak SRAM Allocation** | 64 Bytes | 64 Bytes | Identical |

---

## Repository Structure

- `include/`: Header-only C++ inference kernels (`TinySparse.h`).
- `scripts/`: Data preparation, PyTorch training, pruning, quantization, and C++ header generator.
- `models/`: Saved model weights (`.pth`), quantized numpy arrays (`.npz`), and compiled headers (`.h`).
- `examples/`:
  - `esp32s3_benchmark/`: Arduino benchmark sketch reading hardware cycle counters (`ccount`).
  - `pc_verify/`: Host verification test to validate cross-platform bit-exactness.

---

## Quickstart

### 1. Host Verification (C++)
```bash
g++ -O3 -Iinclude -Imodels examples/pc_verify/verify_inference.cpp -o verify_inference
./verify_inference
