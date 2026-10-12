import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# Hyperparameters
WINDOW_SIZE = 128
NUM_CLASSES = 4
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def main():
    print("=" * 70)
    print(" Path B Training: Within-Recording Temporal Holdout Baseline")
    print(f" Compute Device: {DEVICE}")
    print("=" * 70)

    data_path = os.path.join("models", "cwru_temporal_dataset.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing {data_path}. Run prepare_cwru_dataset_temporal.py first.")

    data = np.load(data_path)
    X_train, y_train = data["X_train"].astype(np.float32), data["y_train"]
    X_val, y_val = data["X_val"].astype(np.float32), data["y_val"]
    X_test, y_test = data["X_test"].astype(np.float32), data["y_test"]

    print(f"Training Samples   : {len(X_train)}")
    print(f"Validation Samples : {len(X_val)}")
    print(f"Held-Out Test      : {len(X_test)} (Future chronological segments)")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=BATCH_SIZE, shuffle=False
    )

    model = TinySparseMLP().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    best_val_acc = 0.0
    best_weights_path = os.path.join("models", "dense_baseline_temporal.pth")

    print("\nExecuting Training Loop...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100.0 * correct / total
        train_loss = running_loss / total

        # Validation phase (strictly tuned on X_val, test untouched)
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_acc = 100.0 * val_correct / val_total
        val_loss = val_loss / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_weights_path)
            saved_tag = "[Checkpoint Saved]"
        else:
            saved_tag = ""

        if epoch % 5 == 0 or epoch == 1 or saved_tag != "":
            print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:6.2f}% {saved_tag}")

    print("\n" + "=" * 70)
    print(" Final Evaluation on Path B Held-Out Test Set (Chronological Holdout)")
    print("=" * 70)

    model.load_state_dict(torch.load(best_weights_path))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            y_true.extend(targets.numpy())
            y_pred.extend(predicted.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    test_acc = 100.0 * np.mean(y_true == y_pred)

    print(f"\nFinal Measured 4-Class Test Accuracy: {test_acc:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # Derived Binary Metrics (Healthy vs Fault)
    true_fault = (y_true > 0)
    pred_fault = (y_pred > 0)
    binary_acc = 100.0 * np.mean(true_fault == pred_fault)
    sensitivity = 100.0 * np.sum((true_fault == 1) & (pred_fault == 1)) / np.sum(true_fault == 1)
    specificity = 100.0 * np.sum((true_fault == 0) & (pred_fault == 0)) / np.sum(true_fault == 0)

    print("\n" + "-" * 70)
    print(" Derived Binary Diagnostic Performance (Healthy vs Faulty):")
    print(f"  Binary Accuracy       : {binary_acc:.2f}%")
    print(f"  Fault Sensitivity     : {sensitivity:.2f}%")
    print(f"  Healthy Specificity   : {specificity:.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()