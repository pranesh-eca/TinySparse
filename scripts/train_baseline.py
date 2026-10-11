import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# 1. Self-Check: Verify reproducibility
# -----------------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

# -----------------------------------------------------------------------------
# 2. Define the MCU-Friendly 1D Neural Network
# -----------------------------------------------------------------------------
class TinySensorNet(nn.Module):
    def __init__(self):
        super(TinySensorNet, self).__init__()
        # Layer 1: 128 input sensor points -> 32 features
        self.fc1 = nn.Linear(128, 32)
        self.relu1 = nn.ReLU()
        
        # Layer 2: 32 features -> 16 features
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        
        # Layer 3: 16 features -> 4 class logits
        self.fc3 = nn.Linear(16, 4)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    print("=== Step 2: Training Dense Baseline Model ===")
    
    # Load the authentic CWRU dataset created in Step 1
    data = np.load("cwru_1d_dataset.npz")
    X = data["X"]  # Shape: (13271, 128)
    y = data["y"]  # Shape: (13271,)

    # Split into 80% Training and 20% Testing sets
    # stratify=y ensures balanced class distribution in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Convert to PyTorch tensors
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                  torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), 
                                 torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Instantiate model, loss function, and Adam optimizer
    model = TinySensorNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train for 15 epochs
    epochs = 15
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = total_loss / total
        epoch_acc = (correct / total) * 100.0

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")

    # -------------------------------------------------------------------------
    # 3. Evaluate Dense Baseline Accuracy on Unseen Test Data
    # -------------------------------------------------------------------------
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    baseline_accuracy = (test_correct / test_total) * 100.0
    print("\n-------------------------------------------")
    print(f"Dense Baseline Test Accuracy: {baseline_accuracy:.2f}%")
    print("-------------------------------------------")

    # Save trained baseline model weights
    torch.save(model.state_dict(), "dense_baseline_model.pth")
    print("Baseline weights saved to: dense_baseline_model.pth")

if __name__ == "__main__":
    main()