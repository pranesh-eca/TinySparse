import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

WINDOW_SIZE = 128
NUM_CLASSES = 4
SPARSITY_TARGET = 0.70
FINETUNE_EPOCHS = 20
BATCH_SIZE = 64
DEVICE = torch.device("cpu")
CLASS_NAMES = ["Normal", "Inner Race", "Ball", "Outer Race"]

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
    """Finds the optimal bit-shift S so 99.9% of positive activations fit in [0, 127]."""
    max_val = np.percentile(np.maximum(0, acc_values), 99.9)
    if max_val <= 127:
        return 0
    shift = int(np.ceil(np.log2(max_val / 127.0)))
    return max(0, shift)

def bit_exact_int8_forward(X_int8, w1_q, b1_q, s1, w2_q, b2_q, s2, w3_q, b3_q):
    """
    Instruction-for-instruction mirror of the C++ TinySparse integer kernel.
    Uses int32 accumulators, exact bit-shifts, and int8 clamping.
    """
    X_i32 = X_int8.astype(np.int32)
    
    # Layer 1: INT8 MAC -> INT32 Acc -> ReLU -> Shift -> Clamp INT8
    acc1 = np.matmul(X_i32, w1_q.astype(np.int32).T) + b1_q.astype(np.int32)
    act1 = np.maximum(0, acc1) >> s1
    out1_int8 = np.clip(act1, 0, 127).astype(np.int8)

    # Layer 2: INT8 MAC -> INT32 Acc -> ReLU -> Shift -> Clamp INT8
    acc2 = np.matmul(out1_int8.astype(np.int32), w2_q.astype(np.int32).T) + b2_q.astype(np.int32)
    act2 = np.maximum(0, acc2) >> s2
    out2_int8 = np.clip(act2, 0, 127).astype(np.int8)

    # Layer 3 (Dense Output): INT8 MAC -> INT32 Logits
    logits_int32 = np.matmul(out2_int8.astype(np.int32), w3_q.astype(np.int32).T) + b3_q.astype(np.int32)
    return out1_int8, out2_int8, logits_int32

def quantize_layer(weight_fp32, bias_fp32, input_scale_factor=1.0):
    """
    Quantizes weights symmetrically to [-127, 127] and scales bias to match
    the accumulator scale: Scale_acc = Scale_w * input_scale_factor.
    """
    w_max = np.max(np.abs(weight_fp32))
    w_scale = 127.0 / w_max if w_max > 1e-8 else 1.0
    w_q = np.clip(np.round(weight_fp32 * w_scale), -127, 127).astype(np.int8)
    
    # Enforce exact zero where pruned
    w_q[weight_fp32 == 0.0] = 0

    b_scale = w_scale * input_scale_factor
    b_q = np.round(bias_fp32 * b_scale).astype(np.int32)
    return w_q, b_q, w_scale, b_scale

def main():
    print("=" * 70)
    print(f" Phase 1.3: L1 Unstructured Pruning ({int(SPARSITY_TARGET*100)}%) & Bit-Exact INT8 Verification")
    print("=" * 70)

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

    model = TinySparseMLP().to(DEVICE)
    baseline_path = os.path.join("models", "dense_baseline_temporal.pth")
    model.load_state_dict(torch.load(baseline_path, map_location=DEVICE))

    # Apply L1 unstructured pruning to FC1 and FC2
    prune.l1_unstructured(model.fc1, name="weight", amount=SPARSITY_TARGET)
    prune.l1_unstructured(model.fc2, name="weight", amount=SPARSITY_TARGET)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    best_val_acc = 0.0
    best_state = None

    print("\nFine-Tuning Pruned Model (Validating strictly on X_val, X_test untouched)...")
    for epoch in range(1, FINETUNE_EPOCHS + 1):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_acc = 100.0 * val_correct / val_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{FINETUNE_EPOCHS:02d}] Val Acc (Pruned FP32): {val_acc:.2f}%")

    model.load_state_dict(best_state)
    prune.remove(model.fc1, "weight")
    prune.remove(model.fc2, "weight")
    model.eval()

    # Evaluate Pruned FP32 on Held-Out Test Set
    with torch.no_grad():
        test_logits_fp32 = model(torch.from_numpy(X_test.astype(np.float32)))
        pred_fp32 = test_logits_fp32.argmax(dim=1).numpy()
    fp32_test_acc = 100.0 * np.mean(pred_fp32 == y_test)

    # Extract FP32 numpy arrays
    w1_fp = model.fc1.weight.detach().numpy()
    b1_fp = model.fc1.bias.detach().numpy()
    w2_fp = model.fc2.weight.detach().numpy()
    b2_fp = model.fc2.bias.detach().numpy()
    w3_fp = model.fc3.weight.detach().numpy()
    b3_fp = model.fc3.bias.detach().numpy()

    # Step-by-Step Fixed-Point Calibration on Validation Set (X_val)
    X_val_i32 = X_val.astype(np.int32)

    # Layer 1 Quantization & Shift Calibration
    w1_q, b1_q, w1_scale, _ = quantize_layer(w1_fp, b1_fp, input_scale_factor=1.0)
    acc1_val = np.matmul(X_val_i32, w1_q.astype(np.int32).T) + b1_q
    s1 = calibrate_shift(acc1_val)
    out1_val = np.clip(np.maximum(0, acc1_val) >> s1, 0, 127).astype(np.int8)
    act1_scale = w1_scale / (2 ** s1)

    # Layer 2 Quantization & Shift Calibration
    w2_q, b2_q, w2_scale, _ = quantize_layer(w2_fp, b2_fp, input_scale_factor=act1_scale)
    acc2_val = np.matmul(out1_val.astype(np.int32), w2_q.astype(np.int32).T) + b2_q
    s2 = calibrate_shift(acc2_val)
    act2_scale = (act1_scale * w2_scale) / (2 ** s2)

    # Layer 3 Quantization
    w3_q, b3_q, w3_scale, _ = quantize_layer(w3_fp, b3_fp, input_scale_factor=act2_scale)

    # Execute Bit-Exact INT8 Inference on Untouched Held-Out Test Set (X_test)
    _, _, test_logits_int32 = bit_exact_int8_forward(
        X_test, w1_q, b1_q, s1, w2_q, b2_q, s2, w3_q, b3_q
    )
    pred_int8 = np.argmax(test_logits_int32, axis=1)
    int8_test_acc = 100.0 * np.mean(pred_int8 == y_test)

    # Calculate Sparsity Metrics
    l1_nz = np.count_nonzero(w1_q)
    l2_nz = np.count_nonzero(w2_q)
    l1_sparsity = 100.0 * (1.0 - l1_nz / w1_q.size)
    l2_sparsity = 100.0 * (1.0 - l2_nz / w2_q.size)

    print("\n" + "=" * 70)
    print(" Final Held-Out Test Set Results (Path B Chronological Split)")
    print("=" * 70)
    print(f"  Layer 1 Sparsity           : {l1_sparsity:.2f}% ({l1_nz} / {w1_q.size} non-zero)")
    print(f"  Layer 2 Sparsity           : {l2_sparsity:.2f}% ({l2_nz} / {w2_q.size} non-zero)")
    print(f"  Calibrated Bit-Shifts      : Layer 1 >> {s1} | Layer 2 >> {s2}")
    print(f"  Pruned FP32 Test Accuracy  : {fp32_test_acc:.2f}%")
    print(f"  Verified INT8 Test Accuracy: {int8_test_acc:.2f}% (Delta: {int8_test_acc - fp32_test_acc:+.2f}%)")
    print("-" * 70)
    print("\nINT8 Classification Report:")
    print(classification_report(y_test, pred_int8, target_names=CLASS_NAMES, digits=4))
    print("INT8 Confusion Matrix:")
    print(confusion_matrix(y_test, pred_int8))
    print("=" * 70)

    # Save Quantized Artifact + Golden Reference Test Vectors for C++ Verification
    sample_idx = 0
    ref_l1, ref_l2, ref_logits = bit_exact_int8_forward(
        X_test[sample_idx:sample_idx+1], w1_q, b1_q, s1, w2_q, b2_q, s2, w3_q, b3_q
    )

    out_npz = os.path.join("models", "pruned_quantized_model.npz")
    np.savez_compressed(
        out_npz,
        fc1_w=w1_q, fc1_b=b1_q, fc1_shift=np.int32(s1),
        fc2_w=w2_q, fc2_b=b2_q, fc2_shift=np.int32(s2),
        fc3_w=w3_q, fc3_b=b3_q,
        ref_input=X_test[sample_idx],
        ref_l1=ref_l1[0],
        ref_l2=ref_l2[0],
        ref_logits=ref_logits[0],
        ref_label=np.int32(y_test[sample_idx])
    )
    print(f"\nSaved verified INT8 model & golden test vectors to: {out_npz}")

if __name__ == "__main__":
    main()