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
    if num_windows <= 0:
        return np.empty((0, window_size), dtype=np.float32)
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
    print("=" * 80)
    print(" Path B: Strict Chronological Holdout (Train / Val / Test separated by Guard Bands)")
    print("=" * 80)

    files = [
        (os.path.join(BASE_PATH, "normal", "Normal_1.mat"), 0, "Normal"),
        (os.path.join(BASE_PATH, "0.007", "IR007_1.mat"),   1, "Inner Race 0.007"),
        (os.path.join(BASE_PATH, "0.007", "B007_1.mat"),     2, "Ball 0.007"),
        (os.path.join(BASE_PATH, "0.007", "OR007@6_1.mat"), 3, "Outer Race 0.007"),
        (os.path.join(BASE_PATH, "0.014", "IR014_1.mat"),   1, "Inner Race 0.014"),
        (os.path.join(BASE_PATH, "0.014", "B014_1.mat"),     2, "Ball 0.014"),
        (os.path.join(BASE_PATH, "0.014", "OR014@6_1.mat"), 3, "Outer Race 0.014"),
        (os.path.join(BASE_PATH, "0.021", "IR021_1.mat"),   1, "Inner Race 0.021"),
        (os.path.join(BASE_PATH, "0.021", "B021_1.mat"),     2, "Ball 0.021"),
        (os.path.join(BASE_PATH, "0.021", "OR021@6_1.mat"), 3, "Outer Race 0.021"),
    ]

    X_tr_list, y_tr_list = [], []
    X_val_list, y_val_list = [], []
    X_te_list, y_te_list = [], []

    for path, label, desc in files:
        raw_sig = extract_de_signal(path)
        N = len(raw_sig)

        # 60% Train | 5% Guard | 15% Val | 5% Guard | 15% Test
        train_end = int(0.60 * N)
        val_start = int(0.65 * N)
        val_end   = int(0.80 * N)
        test_start= int(0.85 * N)

        raw_train = raw_sig[:train_end]
        raw_val   = raw_sig[val_start:val_end]
        raw_test  = raw_sig[test_start:]

        w_train = normalize_to_int8(slice_windows(raw_train))
        w_val   = normalize_to_int8(slice_windows(raw_val))
        w_test  = normalize_to_int8(slice_windows(raw_test))

        X_tr_list.append(w_train)
        y_tr_list.append(np.full(len(w_train), label, dtype=np.int64))
        
        X_val_list.append(w_val)
        y_val_list.append(np.full(len(w_val), label, dtype=np.int64))

        X_te_list.append(w_test)
        y_te_list.append(np.full(len(w_test), label, dtype=np.int64))

        guard1 = val_start - train_end
        guard2 = test_start - val_end
        print(f"[*] {desc:<16} | Tr: {len(w_train):4d} | Val: {len(w_val):4d} | Te: {len(w_test):4d} | G1/G2: {guard1}/{guard2} pts")

    X_train = np.concatenate(X_tr_list, axis=0)
    y_train = np.concatenate(y_tr_list, axis=0)
    X_val   = np.concatenate(X_val_list, axis=0)
    y_val   = np.concatenate(y_val_list, axis=0)
    X_test  = np.concatenate(X_te_list, axis=0)
    y_test  = np.concatenate(y_te_list, axis=0)

    # Shuffle only the training pool
    np.random.seed(42)
    perm = np.random.permutation(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]

    out_file = os.path.join(BASE_PATH, "models", "cwru_temporal_dataset.npz")
    np.savez_compressed(
        out_file,
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test
    )

    print("-" * 80)
    print(f"Dataset compiled to: {out_file}")
    print(f"  Strict Training Windows   : {len(X_train)}")
    print(f"  Strict Validation Windows : {len(X_val)}")
    print(f"  Strict Held-Out Test      : {len(X_test)}")
    print("=" * 80)

if __name__ == "__main__":
    build_temporal_dataset()