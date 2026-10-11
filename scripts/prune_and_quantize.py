import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

torch.manual_seed(42)
np.random.seed(42)

# Same architecture as baseline
class TinySensorNet(nn.Module):
    def __init__(self):
        super(TinySensorNet, self).__init__()
        self.fc1 = nn.Linear(128, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 4)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return (correct / total) * 100.0

def main():
    print("=== Step 3: Accuracy-Preserving Pruning & INT8 Quantization ===")
    
    # 1. Load Data
    data = np.load("cwru_1d_dataset.npz")
    X, y = data["X"], data["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=32, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
        batch_size=64, shuffle=False
    )

    # 2. Load the trained dense baseline
    model = TinySensorNet()
    model.load_state_dict(torch.load("dense_baseline_model.pth"))
    baseline_acc = evaluate(model, test_loader)
    print(f"Loaded Dense Baseline Accuracy: {baseline_acc:.2f}%\n")

    # 3. Apply Magnitude Pruning (Target: 70% sparsity on hidden layers)
    # Layer 1: 128 -> 32 (4096 weights - our primary candidate for acceleration)
    # Layer 2: 32 -> 16 (512 weights)
    # Output layer (fc3) has only 64 weights; we keep it dense to protect class boundaries.
    print("Applying 70% L1-norm magnitude pruning on fc1 and fc2...")
    prune.l1_unstructured(model.fc1, name='weight', amount=0.70)
    prune.l1_unstructured(model.fc2, name='weight', amount=0.70)

    acc_after_raw_prune = evaluate(model, test_loader)
    print(f"Accuracy immediately after raw pruning (before recovery): {acc_after_raw_prune:.2f}%")

    # 4. Fine-Tuning to Recover Accuracy
    print("\nFine-tuning for 5 epochs to restore model accuracy...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(1, 6):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        acc = evaluate(model, test_loader)
        print(f"  Fine-tune Epoch [{epoch}/5] -> Test Accuracy: {acc:.2f}%")

    # Make pruning permanent (removes PyTorch internal masks)
    prune.remove(model.fc1, 'weight')
    prune.remove(model.fc2, 'weight')

    final_pruned_acc = evaluate(model, test_loader)
    print(f"\nFinal Pruned FP32 Accuracy: {final_pruned_acc:.2f}%")

    # Self-Audit: Verify accuracy did not drop beyond acceptable margin
    if final_pruned_acc < (baseline_acc - 2.0):
        print("[!] AUDIT FAILED: Accuracy dropped by more than 2%. Aborting export.")
        return
    else:
        print("[v] AUDIT PASSED: Accuracy preserved within acceptable threshold.")

    # 5. INT8 Quantization (Symmetric per-tensor scaling)
    print("\n--- Symmetric INT8 Quantization & Sparsity Audit ---")
    
    quantized_weights = {}
    for name, param in [("fc1", model.fc1.weight), ("fc2", model.fc2.weight), ("fc3", model.fc3.weight)]:
        w = param.detach().numpy()
        total_elements = w.size
        
        # Max-abs symmetric scaling to signed 8-bit range [-127, 127]
        max_val = np.max(np.abs(w))
        scale = 127.0 / (max_val + 1e-8)
        w_int8 = np.round(w * scale).astype(np.int8)

        # Count actual zeros
        nnz = np.count_nonzero(w_int8)
        sparsity = (1.0 - (nnz / total_elements)) * 100.0
        
        dense_bytes = total_elements * 1  # 1 byte per int8
        sparse_bytes = nnz * 2            # 1 byte value + 1 byte uint8 index

        print(f"Layer [{name}]: Shape {w_int8.shape} | Sparsity: {sparsity:.1f}%")
        print(f"  Total Weights: {total_elements} | Non-Zero (NNZ): {nnz}")
        print(f"  Storage Cost: Dense = {dense_bytes} Bytes | Sparse = {sparse_bytes} Bytes")
        
        if sparse_bytes < dense_bytes:
            savings = ((dense_bytes - sparse_bytes) / dense_bytes) * 100.0
            print(f"  -> Sparse is efficient: Saves {savings:.1f}% Flash storage & skips {total_elements - nnz} MACs.")
        else:
            print(f"  -> Dense is more efficient: Sparsity too low to beat Index Tax. Keep dense.")

        quantized_weights[name] = w_int8

    # Save quantized parameters for the C++ compiler
    np.savez("pruned_quantized_model.npz", 
             fc1=quantized_weights["fc1"], 
             fc2=quantized_weights["fc2"], 
             fc3=quantized_weights["fc3"],
             fc1_bias=model.fc1.bias.detach().numpy(),
             fc2_bias=model.fc2.bias.detach().numpy(),
             fc3_bias=model.fc3.bias.detach().numpy())
    
    print("\nQuantized parameters saved to: pruned_quantized_model.npz")

if __name__ == "__main__":
    main()