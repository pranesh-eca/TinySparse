import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

WINDOW_SIZE = 128
NUM_CLASSES = 4
SPARSITY_LEVELS = [0.0, 0.50, 0.70, 0.80, 0.90]
FINETUNE_EPOCHS = 15
BATCH_SIZE = 64
DEVICE = torch.device("cpu")

class TinySparseMLP(nn.Module):
    def __init__(self):
        super(TinySparseMLP, self).__init__()
        self.fc1 = nn.Linear(WINDOW_SIZE, 32, bias=True)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16, bias=True)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, NUM_CLASSES, bias=True)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

def calibrate_shift(acc_values):
    max_val = np.percentile(np.maximum(0, acc_values), 99.9)
    if max_val <= 127:
        return 0
    return max(0, int(np.ceil(np.log2(max_val / 127.0))))

def quantize_layer(weight_fp32, bias_fp32, input_scale_factor=1.0):
    w_max = np.max(np.abs(weight_fp32))
    w_scale = 127.0 / w_max if w_max > 1e-8 else 1.0
    w_q = np.clip(np.round(weight_fp32 * w_scale), -127, 127).astype(np.int8)
    w_q[weight_fp32 == 0.0] = 0
    b_scale = w_scale * input_scale_factor
    b_q = np.round(bias_fp32 * b_scale).astype(np.int32)
    return w_q, b_q, w_scale

def bit_exact_int8_forward(X_int8, w1_q, b1_q, s1, w2_q, b2_q, s2, w3_q, b3_q):
    X_i32 = X_int8.astype(np.int32)
    acc1 = np.matmul(X_i32, w1_q.astype(np.int32).T) + b1_q.astype(np.int32)
    out1 = np.clip(np.maximum(0, acc1) >> s1, 0, 127).astype(np.int8)

    acc2 = np.matmul(out1.astype(np.int32), w2_q.astype(np.int32).T) + b2_q.astype(np.int32)
    out2 = np.clip(np.maximum(0, acc2) >> s2, 0, 127).astype(np.int8)

    logits = np.matmul(out2.astype(np.int32), w3_q.astype(np.int32).T) + b3_q.astype(np.int32)
    return logits

def dense_to_coo(weight_matrix):
    values, col_indices, row_offsets = [], [], [0]
    for row in weight_matrix:
        nz = np.nonzero(row)[0]
        for col in nz:
            values.append(int(row[col]))
            col_indices.append(int(col))
        row_offsets.append(len(values))
    if len(values) == 0:
        values.append(0)
        col_indices.append(0)
    return (
        np.array(values, dtype=np.int8),
        np.array(col_indices, dtype=np.uint8),
        np.array(row_offsets, dtype=np.uint16)
    )

def format_c_array(arr, c_type, name, per_line=16):
    lines = [f"const {c_type} {name}[{len(arr)}] = {{"]
    for i in range(0, len(arr), per_line):
        chunk = ", ".join(str(x) for x in arr[i:i + per_line])
        lines.append(f"    {chunk},")
    lines.append("};")
    return "\n".join(lines)

def main():
    print("=" * 80)
    print(" Phase 3: Automated Multi-Sparsity Sweep on Strict 60/15/15 Split")
    print("=" * 80)

    data = np.load(os.path.join("models", "cwru_temporal_dataset.npz"))
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val     = data["X_val"], data["y_val"]
    X_test, y_test   = data["X_test"], data["y_test"]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train.astype(np.float32)), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val.astype(np.float32)), torch.from_numpy(y_val)),
        batch_size=BATCH_SIZE, shuffle=False
    )

    baseline_path = os.path.join("models", "dense_baseline_temporal.pth")
    header_blocks = [
        "#ifndef SWEEP_WEIGHTS_H",
        "#define SWEEP_WEIGHTS_H",
        "#include <stdint.h>",
        ""
    ]

    results_table = []

    for sp in SPARSITY_LEVELS:
        sp_pct = int(sp * 100)
        torch.manual_seed(42)
        np.random.seed(42)

        model = TinySparseMLP().to(DEVICE)
        model.load_state_dict(torch.load(baseline_path, map_location=DEVICE))

        if sp > 0.0:
            prune.l1_unstructured(model.fc1, name="weight", amount=sp)
            prune.l1_unstructured(model.fc2, name="weight", amount=sp)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=5e-4)
            best_val_acc = 0.0
            best_state = None

            for _ in range(FINETUNE_EPOCHS):
                model.train()
                for inputs, targets in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(inputs), targets)
                    loss.backward()
                    optimizer.step()

                model.eval()
                val_correct, val_total = 0, 0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        _, pred = model(inputs).max(1)
                        val_total += targets.size(0)
                        val_correct += pred.eq(targets).sum().item()
                val_acc = 100.0 * val_correct / val_total
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

            model.load_state_dict(best_state)
            prune.remove(model.fc1, "weight")
            prune.remove(model.fc2, "weight")

        model.eval()
        with torch.no_grad():
            fp32_logits = model(torch.from_numpy(X_test.astype(np.float32)))
            fp32_acc = 100.0 * np.mean(fp32_logits.argmax(dim=1).numpy() == y_test)

        w1_fp, b1_fp = model.fc1.weight.detach().numpy(), model.fc1.bias.detach().numpy()
        w2_fp, b2_fp = model.fc2.weight.detach().numpy(), model.fc2.bias.detach().numpy()
        w3_fp, b3_fp = model.fc3.weight.detach().numpy(), model.fc3.bias.detach().numpy()

        # Fixed-point quantization & shift calibration on X_val
        X_val_i32 = X_val.astype(np.int32)
        w1_q, b1_q, w1_s = quantize_layer(w1_fp, b1_fp, 1.0)
        acc1_val = np.matmul(X_val_i32, w1_q.astype(np.int32).T) + b1_q
        s1 = calibrate_shift(acc1_val)
        out1_val = np.clip(np.maximum(0, acc1_val) >> s1, 0, 127).astype(np.int8)
        act1_s = w1_s / (2 ** s1)

        w2_q, b2_q, w2_s = quantize_layer(w2_fp, b2_fp, act1_s)
        acc2_val = np.matmul(out1_val.astype(np.int32), w2_q.astype(np.int32).T) + b2_q
        s2 = calibrate_shift(acc2_val)
        act2_s = (act1_s * w2_s) / (2 ** s2)

        w3_q, b3_q, _ = quantize_layer(w3_fp, b3_fp, act2_s)

        # Evaluate INT8 on untouched X_test
        int8_logits = bit_exact_int8_forward(X_test, w1_q, b1_q, s1, w2_q, b2_q, s2, w3_q, b3_q)
        int8_preds  = np.argmax(int8_logits, axis=1)
        int8_acc    = 100.0 * np.mean(int8_preds == y_test)
        sample0_pred = int(int8_preds[0])

        fc1_val, fc1_idx, fc1_off = dense_to_coo(w1_q)
        fc2_val, fc2_idx, fc2_off = dense_to_coo(w2_q)
        fc3_flat = w3_q.ravel()

        l1_macs = int(np.count_nonzero(w1_q))
        l2_macs = int(np.count_nonzero(w2_q))
        l3_macs = int(w3_q.size)
        total_macs = l1_macs + l2_macs + l3_macs

        if sp == 0.0:
            flash_bytes = (w1_q.size + len(b1_q)*4) + (w2_q.size + len(b2_q)*4) + (w3_q.size + len(b3_q)*4)
        else:
            l1_coo_b = l1_macs * 2 + len(fc1_off) * 2 + len(b1_q) * 4
            l2_coo_b = l2_macs * 2 + len(fc2_off) * 2 + len(b2_q) * 4
            l3_b     = w3_q.size + len(b3_q) * 4
            flash_bytes = l1_coo_b + l2_coo_b + l3_b

        results_table.append((sp_pct, l1_macs, l2_macs, total_macs, flash_bytes, fp32_acc, int8_acc, s1, s2))

        prefix = f"sp{sp_pct}"
        header_blocks.append(f"// --- Sparsity Target: {sp_pct}% (INT8 Acc: {int8_acc:.2f}%, Flash: {flash_bytes} B) ---")
        header_blocks.append(f"#define {prefix.upper()}_S1 {s1}")
        header_blocks.append(f"#define {prefix.upper()}_S2 {s2}")
        header_blocks.append(f"#define {prefix.upper()}_FLASH_BYTES {flash_bytes}")
        header_blocks.append(f"#define {prefix.upper()}_INT8_ACC {int8_acc:.2f}f")
        header_blocks.append(f"#define {prefix.upper()}_EXPECTED_CLASS {sample0_pred}")
        if sp == 0.0:
            header_blocks.append(format_c_array(w1_q.ravel(), "int8_t", f"{prefix}_fc1_dense"))
            header_blocks.append(format_c_array(w2_q.ravel(), "int8_t", f"{prefix}_fc2_dense"))
        header_blocks.append(format_c_array(fc1_val, "int8_t",   f"{prefix}_fc1_val"))
        header_blocks.append(format_c_array(fc1_idx, "uint8_t",  f"{prefix}_fc1_idx"))
        header_blocks.append(format_c_array(fc1_off, "uint16_t", f"{prefix}_fc1_off"))
        header_blocks.append(format_c_array(b1_q,    "int32_t",  f"{prefix}_fc1_b"))
        header_blocks.append(format_c_array(fc2_val, "int8_t",   f"{prefix}_fc2_val"))
        header_blocks.append(format_c_array(fc2_idx, "uint8_t",  f"{prefix}_fc2_idx"))
        header_blocks.append(format_c_array(fc2_off, "uint16_t", f"{prefix}_fc2_off"))
        header_blocks.append(format_c_array(b2_q,    "int32_t",  f"{prefix}_fc2_b"))
        header_blocks.append(format_c_array(fc3_flat,"int8_t",   f"{prefix}_fc3_dense"))
        header_blocks.append(format_c_array(b3_q,    "int32_t",  f"{prefix}_fc3_b"))
        header_blocks.append("")

        if sp == 0.70:
            # Export a 10-window stream across all 4 classes from X_test for streaming verification
            indices = [0, 5, 1200, 1205, 2100, 2105, 3000, 3005]
            stream_windows = X_test[indices].ravel()
            stream_expected = int8_preds[indices]

    header_blocks.append(f"#define GOLDEN_EXPECTED_CLASS { int(y_test[0]) }")
    header_blocks.append(format_c_array(X_test[0], "int8_t", "sweep_test_input"))
    header_blocks.append(f"#define STREAM_NUM_WINDOWS {len(indices)}")
    header_blocks.append(format_c_array(stream_windows, "int8_t", "sweep_stream_input"))
    header_blocks.append(format_c_array(stream_expected, "uint8_t", "sweep_stream_expected"))
    header_blocks.append("#endif // SWEEP_WEIGHTS_H\n")

    out_header = os.path.join("include", "sweep_weights.h")
    with open(out_header, "w") as f:
        f.write("\n".join(header_blocks))

    print("\n" + "-" * 80)
    print(f"{'Sparsity':<10} | {'L1 MACs':<8} | {'L2 MACs':<8} | {'Total MACs':<10} | {'Flash (B)':<10} | {'FP32 Acc':<9} | {'INT8 Acc':<9}")
    print("-" * 80)
    for r in results_table:
        print(f"{r[0]:>7}%   | {r[1]:>8} | {r[2]:>8} | {r[3]:>10} | {r[4]:>9} B | {r[5]:>8.2f}% | {r[6]:>8.2f}%")
    print("-" * 80)
    print(f"Exported all 5 quantized models + streaming test vectors to: {out_header}")
    print("=" * 80)

if __name__ == "__main__":
    main()