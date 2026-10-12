import os
import numpy as np
import scipy.io as sio

# Configuration
WINDOW_SIZE = 128
STRIDE = 64
BASE_PATH = "."  # Run from repository root

CLASSES = {
    0: "Normal",
    1: "Inner Race Fault",
    2: "Ball Fault",
    3: "Outer Race Fault"
}

def extract_drive_end_vibration(mat_path):
    """Extracts Drive-End (DE) accelerometer time series from CWRU .mat file."""
    data = sio.loadmat(mat_path)
    for key in data.keys():
        if "DE_time" in key:
            return data[key].ravel()
    raise KeyError(f"No DE_time channel found in {mat_path}")

def slice_windows(signal, window_size=WINDOW_SIZE, stride=STRIDE):
    """Slices a 1D signal into windows with a defined stride."""
    num_windows = (len(signal) - window_size) // stride + 1
    windows = np.empty((num_windows, window_size), dtype=np.float32)
    for i in range(num_windows):
        start = i * stride
        windows[i] = signal[start : start + window_size]
    return windows

def normalize_to_int8(windows):
    """
    Min-Max scale each window to [-1.0, 1.0], then symmetrically
    quantize to INT8 [-127, 127] for embedded MCU input.
    """
    normalized = np.empty_like(windows, dtype=np.int8)
    for i, w in enumerate(windows):
        max_abs = np.max(np.abs(w))
        if max_abs > 1e-6:
            scaled = (w / max_abs) * 127.0
        else:
            scaled = np.zeros_like(w)
        normalized[i] = np.clip(np.round(scaled), -127, 127).astype(np.int8)
    return normalized

def build_dataset():
    print("=" * 60)
    print(" CWRU Zero-Leakage Dataset Generator (Cross-Severity Split)")
    print("=" * 60)

    # 1. Process Normal Baseline (Single file with temporal partition & guard band)
    normal_raw = extract_drive_end_vibration(os.path.join(BASE_PATH, "normal", "Normal_1.mat"))
    N = len(normal_raw)
    
    # 70% Train, 10% Guard Band (Discarded), 20% Test
    train_end = int(0.70 * N)
    test_start = int(0.80 * N)
    
    normal_train = normal_raw[:train_end]
    normal_test = normal_raw[test_start:]
    
    # 2. Define File Groups (Hold out 0.014" entirely for testing)
    train_files = [
        # (filepath, label, description)
        (os.path.join(BASE_PATH, "0.007", "IR007_1.mat"), 1, "Inner Race 0.007"),
        (os.path.join(BASE_PATH, "0.007", "B007_1.mat"),   2, "Ball 0.007"),
        (os.path.join(BASE_PATH, "0.007", "OR007@6_1.mat"), 3, "Outer Race 0.007"),
        (os.path.join(BASE_PATH, "0.021", "IR021_1.mat"), 1, "Inner Race 0.021"),
        (os.path.join(BASE_PATH, "0.021", "B021_1.mat"),   2, "Ball 0.021"),
        (os.path.join(BASE_PATH, "0.021", "OR021@6_1.mat"), 3, "Outer Race 0.021"),
    ]

    test_files = [
        (os.path.join(BASE_PATH, "0.014", "IR014_1.mat"), 1, "Inner Race 0.014"),
        (os.path.join(BASE_PATH, "0.014", "B014_1.mat"),   2, "Ball 0.014"),
        (os.path.join(BASE_PATH, "0.014", "OR014@6_1.mat"), 3, "Outer Race 0.014"),
    ]

    # 3. Assemble Training Windows
    X_train_list, y_train_list = [], []

    # Add Normal train
    norm_tr_win = normalize_to_int8(slice_windows(normal_train))
    X_train_list.append(norm_tr_win)
    y_train_list.append(np.zeros(len(norm_tr_win), dtype=np.int64))
    print(f"[*] Train: Normal_1 (0-70%) -> {len(norm_tr_win)} windows")

    # Add Fault train
    for path, label, desc in train_files:
        raw_sig = extract_drive_end_vibration(path)
        win = normalize_to_int8(slice_windows(raw_sig))
        X_train_list.append(win)
        y_train_list.append(np.full(len(win), label, dtype=np.int64))
        print(f"[*] Train: {desc} -> {len(win)} windows")

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    # Shuffle training set
    np.random.seed(42)
    train_perm = np.random.permutation(len(X_train))
    X_train = X_train[train_perm]
    y_train = y_train[train_perm]

    # Split 15% of the train pool for validation (early stopping / tuning)
    val_split_idx = int(len(X_train) * 0.85)
    X_tr, y_tr = X_train[:val_split_idx], y_train[:val_split_idx]
    X_val, y_val = X_train[val_split_idx:], y_train[val_split_idx:]

    # 4. Assemble Held-Out Test Windows (0% overlap, 100% unseen files)
    X_test_list, y_test_list = [], []

    norm_te_win = normalize_to_int8(slice_windows(normal_test))
    X_test_list.append(norm_te_win)
    y_test_list.append(np.zeros(len(norm_te_win), dtype=np.int64))
    print(f"[!] Test (Held-out): Normal_1 (80-100%) -> {len(norm_te_win)} windows")

    for path, label, desc in test_files:
        raw_sig = extract_drive_end_vibration(path)
        win = normalize_to_int8(slice_windows(raw_sig))
        X_test_list.append(win)
        y_test_list.append(np.full(len(win), label, dtype=np.int64))
        print(f"[!] Test (Held-out): {desc} -> {len(win)} windows")

    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    # 5. Export Processed Artifact
    out_dir = os.path.join(BASE_PATH, "models")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "cwru_1d_dataset.npz")

    np.savez_compressed(
        out_file,
        X_train=X_tr, y_train=y_tr,
        X_val=X_val,   y_val=y_val,
        X_test=X_test, y_test=y_test
    )

    print("-" * 60)
    print(f"Dataset successfully compiled to: {out_file}")
    print(f"  Training Windows   : {X_tr.shape[0]}")
    print(f"  Validation Windows : {X_val.shape[0]}")
    print(f"  Held-Out Test      : {X_test.shape[0]}")
    print(f"  Window Shape       : {X_tr.shape[1]} samples (INT8)")
    print("=" * 60)

if __name__ == "__main__":
    build_dataset()