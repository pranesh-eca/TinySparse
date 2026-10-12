import os
import numpy as np
import scipy.io as sio

WINDOW_SIZE = 128
STRIDE = 64
BASE_PATH = "."

CLASSES = {
    0: "Normal",
    1: "Inner Race Fault",
    2: "Ball Fault",
    3: "Outer Race Fault"
}

def extract_de_signal(mat_path):
    data = sio.loadmat(mat_path)
    for k in data.keys():
        if "DE_time" in k:
            return data[k].ravel()
    raise KeyError(f"No DE_time channel found in {mat_path}")

def slice_windows(signal, window_size=WINDOW_SIZE, stride=STRIDE):
    num_windows = (len(signal) - window_size) // stride + 1
    windows = np.empty((num_windows, window_size), dtype=np.float32)
    for i in range(num_windows):
        start = i * stride
        windows[i] = signal[start : start + window_size]
    return windows

def normalize_to_int8(windows):
    normalized = np.empty_like(windows, dtype=np.int8)
    for i, w in enumerate(windows):
        max_abs = np.max(np.abs(w))
        if max_abs > 1e-6:
            scaled = (w / max_abs) * 127.0
        else:
            scaled = np.zeros_like(w)
        normalized[i] = np.clip(np.round(scaled), -127, 127).astype(np.int8)
    return normalized

def build_temporal_dataset():
    print("=" * 70)
    print(" Path B: Within-Recording Temporal Holdout (Zero Boundary Overlap)")
    print("=" * 70)

    # 1. Define files across all fault classes (0.007", 0.014", 0.021" balanced)
    files = [
        # Normal
        (os.path.join(BASE_PATH, "normal", "Normal_1.mat"), 0, "Normal"),
        # 0.007"
        (os.path.join(BASE_PATH, "0.007", "IR007_1.mat"),   1, "Inner Race 0.007"),
        (os.path.join(BASE_PATH, "0.007", "B007_1.mat"),     2, "Ball 0.007"),
        (os.path.join(BASE_PATH, "0.007", "OR007@6_1.mat"), 3, "Outer Race 0.007"),
        # 0.014"
        (os.path.join(BASE_PATH, "0.014", "IR014_1.mat"),   1, "Inner Race 0.014"),
        (os.path.join(BASE_PATH, "0.014", "B014_1.mat"),     2, "Ball 0.014"),
        (os.path.join(BASE_PATH, "0.014", "OR014@6_1.mat"), 3, "Outer Race 0.014"),
        # 0.021"
        (os.path.join(BASE_PATH, "0.021", "IR021_1.mat"),   1, "Inner Race 0.021"),
        (os.path.join(BASE_PATH, "0.021", "B021_1.mat"),     2, "Ball 0.021"),
        (os.path.join(BASE_PATH, "0.021", "OR021@6_1.mat"), 3, "Outer Race 0.021"),
    ]

    X_train_list, y_train_list = [], []
    X_test_list,  y_test_list  = [], []

    for path, label, desc in files:
        raw_sig = extract_de_signal(path)
        N = len(raw_sig)

        # Chronological partition before windowing
        train_end = int(0.70 * N)
        test_start = int(0.80 * N)  # 10% guard interval discarded

        raw_train = raw_sig[:train_end]
        raw_test  = raw_sig[test_start:]

        # Slice each partition independently
        w_train = normalize_to_int8(slice_windows(raw_train))
        w_test  = normalize_to_int8(slice_windows(raw_test))

        X_train_list.append(w_train)
        y_train_list.append(np.full(len(w_train), label, dtype=np.int64))

        X_test_list.append(w_test)
        y_test_list.append(np.full(len(w_test), label, dtype=np.int64))

        print(f"[*] {desc:<20} | Train: {len(w_train):4d} win | Test: {len(w_test):4d} win | Guard: {test_start - train_end} pts")

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_test  = np.concatenate(X_test_list, axis=0)
    y_test  = np.concatenate(y_test_list, axis=0)

    # Shuffle training pool only
    np.random.seed(42)
    perm = np.random.permutation(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]

    # Split 15% validation from the train pool
    val_idx = int(0.85 * len(X_train))
    X_tr, y_tr = X_train[:val_idx], y_train[:val_idx]
    X_val, y_val = X_train[val_idx:], y_train[val_idx:]

    out_file = os.path.join(BASE_PATH, "models", "cwru_temporal_dataset.npz")
    np.savez_compressed(
        out_file,
        X_train=X_tr, y_train=y_tr,
        X_val=X_val,   y_val=y_val,
        X_test=X_test, y_test=y_test
    )

    print("-" * 70)
    print(f"Compiled to: {out_file}")
    print(f"  Training Windows   : {len(X_tr)}")
    print(f"  Validation Windows : {len(X_val)}")
    print(f"  Held-Out Test      : {len(X_test)}")
    print("=" * 70)

if __name__ == "__main__":
    build_temporal_dataset()