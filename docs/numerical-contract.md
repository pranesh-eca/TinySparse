# TinySparse numerical contract

This document describes the current INT8 fully connected kernels and the
quantized CWRU example. It is the contract to use when exporting another model.

## Inputs and parameters

- Input and weight elements are signed INT8. Kernels accept the full INT8 input
  range, -128 through 127. The CWRU data script normalizes each window by its
  maximum absolute sample, rounds with NumPy, then clips to -127 through 127.
  A live sensor application must implement equivalent preprocessing to use that
  trained model.
- Weights are stored as signed INT8. The existing Python quantizer maps each
  layer's largest absolute FP32 weight to 127, uses NumPy rounding, and clips
  to -127 through 127. Previously pruned exact zeros remain zero.
- Each row has an INT32 bias. The existing Python quantizer rounds the FP32 bias
  after multiplying by the layer's weight scale and its input activation scale.
- Dense weights are row-major. CSR weights use values and column indices for
  each row, with `offsets[row]` inclusive and `offsets[row + 1]` exclusive.
  Column indices are 8-bit, so a CSR layer can have at most 256 input features.
  The total number of nonzeros must fit in a 16-bit row offset.

## Layer computation

For output row `r`, both storage formats calculate:

~~~text
acc = bias[r] + sum(weight[r, c] * input[c])
~~~

Each product and addition is computed in signed INT32. The model validator
conservatively checks `abs(bias[r]) + 128 * sum(abs(weight[r, c])) <= INT32_MAX`
for every row. Models that fail this check are rejected by the exporter. Call
`validate_model()` before running manually constructed model descriptions.

For each hidden layer, TinySparse computes:

~~~text
output[r] = min(127, max(0, acc) >> shift)
~~~

The right shift occurs after ReLU, so its input is nonnegative. It therefore
discards low bits toward zero. `shift` must be between 0 and 30. The existing
CWRU exporter calibrates shifts from validation-set accumulators; its 70%
sparse example uses 10 and 7 for the two hidden layers.

The final layer does not shift, clamp, or apply ReLU. It returns signed INT32
logits. `argmax()` returns the first class with the largest logit, including
when logits tie.

## Equivalence and limits

Dense and CSR forms of the *same* quantized layer have the same arithmetic and
output values if their parameters and row order match. The automatic exporter
selects the representation with fewer parameter-array bytes; it does not change
weights or improve a model's classification accuracy. Comparing separate
pruned models at different sparsity levels also measures model changes.

The old `tinysparse_compiler.py` and the new `export_adaptive_model.py` take
already-quantized checkpoints. The latter accepts consecutive `fcN_w` and
`fcN_b` arrays, plus `fcN_shift` for every hidden layer. It supports fully
connected layers with ReLU hidden activations and an INT32-logit final layer.
Other layer types, asymmetric quantization, and arbitrary rescaling factors
are outside the current format.
