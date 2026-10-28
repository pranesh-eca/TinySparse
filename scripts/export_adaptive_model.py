"""Export an INT8 .npz network as a TinySparse Model with per-layer storage.

The input contains fc1_w/fc1_b, fc2_w/fc2_b, ... in execution order.
Every hidden layer also has fcN_shift. This tool does not prune or quantize.
"""

import argparse
import re
from pathlib import Path

import numpy as np


def array_text(name, values, c_type):
    flat = np.asarray(values).reshape(-1)
    # Standard C++ does not permit zero-length arrays.
    items = [int(x) for x in flat] or [0]
    lines = [f"static const {c_type} {name}[{len(items)}] = {{"]
    for start in range(0, len(items), 16):
        lines.append("    " + ", ".join(str(x) for x in items[start:start + 16]) + ",")
    lines.append("};")
    return "\n".join(lines)


def export_model(source: Path, destination: Path, namespace: str = "TinySparseExport"):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", namespace):
        raise ValueError("namespace must be a C++ identifier")
    with np.load(source, allow_pickle=False) as data:
        names = set(data.files)
        layer_numbers = sorted(int(m.group(1)) for key in names
                               if (m := re.fullmatch(r"fc([1-9][0-9]*)_w", key)))
        if not layer_numbers or layer_numbers != list(range(1, len(layer_numbers) + 1)):
            raise ValueError("expected consecutive fc1_w, fc2_w, ... layers")
        blocks = ["#pragma once", "#include \"TinySparseModel.h\"", "",
                  f"namespace {namespace} {{", ""]
        descriptors = []
        previous_width = None
        summary = []
        for number in layer_numbers:
            prefix = f"fc{number}"
            if f"{prefix}_b" not in names:
                raise ValueError(f"missing {prefix}_b")
            weight = data[f"{prefix}_w"]
            bias = data[f"{prefix}_b"]
            if weight.dtype != np.int8 or weight.ndim != 2 or bias.dtype != np.int32 or bias.shape != (weight.shape[0],):
                raise ValueError(f"{prefix} needs 2D INT8 weights and matching 1D INT32 biases")
            outputs, inputs = weight.shape
            if not (0 < inputs <= 65535 and 0 < outputs <= 65535):
                raise ValueError(f"{prefix} dimensions exceed the uint16_t interface")
            if previous_width is not None and inputs != previous_width:
                raise ValueError(f"{prefix} input width does not match the previous layer")
            previous_width = outputs
            final = number == layer_numbers[-1]
            shift_key = f"{prefix}_shift"
            if final:
                shift = 0
            else:
                if shift_key not in names:
                    raise ValueError(f"missing {shift_key}")
                shift_array = data[shift_key]
                if (shift_array.shape != () or not np.issubdtype(shift_array.dtype, np.integer)
                        or not (0 <= int(shift_array) <= 30)):
                    raise ValueError(f"{shift_key} must be an integer from 0 to 30")
                shift = int(shift_array)
            bound = np.abs(bias.astype(np.int64)) + 128 * np.abs(weight.astype(np.int64)).sum(axis=1)
            if np.any(bound > np.iinfo(np.int32).max):
                raise ValueError(f"{prefix} may overflow INT32 for an INT8 input")

            nonzeros = int(np.count_nonzero(weight))
            offsets = [0]
            values, columns = [], []
            for row in weight:
                for col in np.flatnonzero(row):
                    columns.append(int(col))
                    values.append(int(row[col]))
                offsets.append(len(values))
            dense_bytes = weight.size
            csr_bytes = max(1, nonzeros) * 2 + 2 * (outputs + 1)
            use_csr = inputs <= 256 and nonzeros <= 65535 and csr_bytes < dense_bytes

            blocks.append(array_text(f"{prefix}_bias", bias, "int32_t"))
            if use_csr:
                blocks.append(array_text(f"{prefix}_weights", values, "int8_t"))
                blocks.append(array_text(f"{prefix}_columns", columns, "uint8_t"))
                blocks.append(array_text(f"{prefix}_offsets", offsets, "uint16_t"))
                descriptors.append(
                    f"    {{TinySparse::Storage::Csr, {inputs}, {outputs}, {prefix}_weights, "
                    f"{prefix}_columns, {prefix}_offsets, {prefix}_bias, {nonzeros}, {shift}}},")
                parameter_bytes = csr_bytes + 4 * outputs
            else:
                blocks.append(array_text(f"{prefix}_weights", weight, "int8_t"))
                descriptors.append(
                    f"    {{TinySparse::Storage::Dense, {inputs}, {outputs}, {prefix}_weights, "
                    f"nullptr, nullptr, {prefix}_bias, 0, {shift}}},")
                parameter_bytes = dense_bytes + 4 * outputs
            blocks.append("")
            summary.append((prefix, "CSR" if use_csr else "dense", parameter_bytes))

        blocks.extend(["static const TinySparse::Layer layers[] = {", *descriptors, "};",
                       f"static const TinySparse::Model model = {{layers, {len(descriptors)}}};", ""])
        if "ref_input" in names and "ref_logits" in names:
            reference_input = data["ref_input"]
            reference_logits = data["ref_logits"]
            if (reference_input.dtype == np.int8 and reference_input.shape == (data["fc1_w"].shape[1],)
                    and reference_logits.dtype == np.int32 and reference_logits.shape == (previous_width,)):
                blocks.append(array_text("golden_input", reference_input, "int8_t"))
                blocks.append(array_text("golden_logits", reference_logits, "int32_t"))
        blocks.extend(["", f"}} // namespace {namespace}", ""])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(blocks), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="quantized model .npz")
    parser.add_argument("--output", type=Path, required=True, help="generated C++ header")
    parser.add_argument("--namespace", default="TinySparseExport")
    args = parser.parse_args()
    for name, storage, size in export_model(args.input, args.output, args.namespace):
        print(f"{name}: {storage}, {size} parameter bytes")
