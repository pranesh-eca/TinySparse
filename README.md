# TinySparse

TinySparse is a small, header-only C++17 library for INT8 fully connected inference on microcontrollers. Its current example classifies 128-sample vibration windows with a 128 -> 32 -> 16 -> 4 network. Hidden layers use row-offset sparse storage; the output layer is dense. The inference kernels allocate no heap memory. Training, pruning, calibration, and model export happen offline in Python.

The project has been tested on an ESP32-S3 and built with host GCC. The C++ kernels are intended to be portable, but other MCU families have not been validated. ESP32-S3 has a single-precision FPU; TinySparse's inference path uses integer arithmetic and does not require that FPU or a dedicated neural accelerator.

## Current scope

- The original CWRU header has fixed dimensions and quantization shifts. The new model descriptor supports varying fully connected layer widths and per-layer dense/CSR storage; it does not yet support convolutional or recurrent layers.
- The row-offset format is compressed sparse row (CSR): INT8 values, 8-bit input-column indices, 16-bit row offsets, and INT32 biases.
- Skipping zero weights reduces inference work. TinySparse does not compress or discard raw sensor samples.
- The Arduino streaming example replays eight stored held-out windows through the ring buffer. It does not acquire a live sensor stream. Its inference hop is 128 samples; the dataset windows use stride 64.

## Build and verify on a host

~~~sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
~~~

The four C++ tests cover ring-buffer behavior, a generated Python reference vector at every network layer, kernel edge cases, and the reusable model interface. A separate Python test exports a mixed dense/CSR model, compiles it, and checks its logits. The golden-vector tests check selected inputs. They do not establish correctness for every possible input. CI runs all five tests.

To reproduce the model experiments, install the packages in `requirements.txt` and use the scripts in `scripts/`. The CWRU .mat files are not bundled. If you change a checkpoint locally, regenerate its header and verify the reference outputs before using it as a released model. Dataset and training provenance remain planned improvements.

## Memory accounting

The CWRU inference example uses 128 bytes for a contiguous input window, 32 and 16 bytes for hidden activations, and 16 bytes for four INT32 logits: **192 bytes of explicit inference arrays**. This excludes the streaming ring buffer, stack, framework, serial telemetry, and other application state. No `malloc` or `new` appears in the inference kernels; the 192-byte figure is not whole-application peak RAM.

Parameter bytes below count compiled weight, index, row-offset, and bias arrays. They are not the size of the whole firmware image. The 0% CSR array count is 9,486 bytes, while the dense representation is 4,880 bytes. The 50% CSR model is also larger than dense. The point at which sparse storage becomes smaller depends on actual nonzero counts and layer dimensions.

The Arduino compiler's program-storage and global-RAM totals, plus stack usage, have not yet been recorded here.

## Reported ESP32-S3 benchmark

These are the previously recorded figures from the ESP32-S3 crossover sweep at 240 MHz, averaged over 1,000 runs. The cycle measurement covers the three inference layers and argmax; it excludes sensor acquisition, preprocessing, ring-buffer copying, and serial output. INT8 accuracy was calculated offline on the within-recording temporal test set. No new hardware measurements were made for this README revision.

| Model | MACs | Cycles | Inference latency | Speedup vs dense | Parameter arrays | INT8 4-class accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0% dense | 4,621 | 85,722 | 357.17 us | 1.00x | 4,880 B | 91.00% |
| 0% CSR | 4,621 | 89,628 | 373.45 us | 0.96x | 9,486 B | 91.00% |
| 50% CSR | 2,368 | 46,375 | 193.23 us | 1.85x | 4,980 B | 90.62% |
| 70% CSR | 1,444 | 28,748 | 119.78 us | 2.98x | 3,132 B | 88.75% |
| 80% CSR | 983 | 19,906 | 82.94 us | 4.31x | 2,210 B | 84.09% |
| 90% CSR | 521 | 11,096 | 46.23 us | 7.73x | 1,286 B | 69.92% |

The 70% point trades 2.25 percentage points of INT8 accuracy against the 0% dense model for lower latency and parameter storage. The right choice depends on an application's accuracy, storage, and timing requirements. The 0% CSR case shows that sparse indexing has a cost when few weights are zero.

The crossover sketch now checks each model's predicted class against the class exported by Python for its test input. This is a class check, not a hardware bit-exact check of intermediate activations and logits. It must be rerun on hardware after code changes.

## Dataset evaluation

The CWRU demonstration is an application benchmark for the library. These results describe this model and these data splits; they are not general accuracy guarantees for the engine.

**Path B: within-recording temporal holdout.** Each raw recording is split chronologically into 60% training, 5% discarded guard, 15% validation, 5% discarded guard, and 15% test before windowing (128 samples, stride 64). This produced 14,808 training, 3,690 validation, and 3,690 test windows. The recorded dense FP32 4-class test accuracy was 91.36%. The sparse 70% INT8 model from the sweep achieved 88.75% on the same protocol. The guard bands prevent windows from crossing partitions, but segments of the same recording can remain statistically correlated; this experiment does not test unseen bearings.

**Path A: cross-severity stress test.** The separate model was trained on 0.007-inch and 0.021-inch faults and evaluated on held-out 0.014-inch fault recordings. Its reported 4-class test accuracy was 62.34%. This is evidence of limited transfer for this model and protocol; it does not identify the physical cause of the errors or establish a general limit for time-domain models.

A bearing or specimen holdout, if the source metadata permits it, remains future work for stronger claims about transfer to new machinery.

## Using the kernels

Arduino IDE can install the repository as a ZIP library using `library.properties`. The examples copy headers locally so each sketch opens as a standalone Arduino project. For a new model, offline quantization and header export are currently required.

~~~cpp
#include "TinySparse.h"

// For one sparse hidden layer, the caller owns input and output buffers.
TinySparse::sparse_fc_relu(
    input, values, column_indices, row_offsets, biases,
    output, output_neurons, calibrated_shift
);
~~~

For an already-quantized model in the `fc1_w`, `fc1_b`, `fc1_shift`, `fc2_w`, ... `.npz` format, export a model descriptor and arrays with:

~~~sh
python scripts/export_adaptive_model.py --input models/pruned_quantized_model.npz --output include/my_model.h
~~~

The exporter selects the smaller dense or CSR representation for each layer. It does not prune, train, or quantize weights, and choosing smaller arrays does not guarantee faster execution. It validates supported dimensions, types, shifts, and a conservative INT32 overflow bound. The generated header contains a `TinySparseExport::model` descriptor and optional golden input/logits when those are present in the checkpoint. Checkpoint changes require a fresh export.

For models with different dimensions, include `TinySparseModel.h` and the exported header. Call `validate_model(model)` once before inference; it checks dimensions, CSR indices and offsets, and the overflow bound. Allocate `required_workspace_bytes(model)` bytes yourself, then call `run_model(model, input, workspace, workspace_bytes, logits)`. The workspace uses two alternating INT8 buffers sized for the widest hidden layer. The final layer returns INT32 logits. Model arrays and buffers must remain alive and must not overlap. See `tests/test_model.cpp` for a complete mixed dense/CSR example. The older `tinysparse_compiler.py` remains available for the original fixed CWRU example.

The exact arithmetic and input assumptions are documented in `docs/numerical-contract.md`.

Use `examples/pc_verify/` as the exact arithmetic example, `examples/esp32s3_crossover_sweep/` as the hardware benchmark, and `examples/esp32s3_streaming/` as the replay demonstration. The input must use the same sample ordering and INT8 preprocessing as the exported model.

## Next library milestone

Tie each exported header to its checkpoint and measure the generated model on hardware. Automatic storage selection currently minimizes parameter-array bytes; hardware speed-based selection requires measurements on the target board. Broader layer support can follow after those checks.

The source is provided under the terms in `LICENSE.txt`; the CWRU dataset has its own attribution and usage terms.
