import os
import scipy.io
import numpy as np

# Adjust BASE_DIR if your folders (0.007, normal) are in another folder.
# "." means the current folder where this script is executed.
BASE_DIR = "."

FILES = {
    "normal": os.path.join(BASE_DIR, "normal", "Normal_1.mat"),
    "inner_race": os.path.join(BASE_DIR, "0.007", "IR007_1.mat"),
    "ball": os.path.join(BASE_DIR, "0.007", "B007_1.mat"),
    "outer_race": os.path.join(BASE_DIR, "0.007", "OR007@6_1.mat"),
}

WINDOW_SIZE = 128  # 128 consecutive sensor readings per 1D input window
STEP_SIZE = 64     # 50% overlap between consecutive windows

def extract_drive_end_signal(mat_filepath):
    """
    Extracts the Drive End (DE) acceleration array from the MATLAB file.
    CWRU files store this under keys like 'X098_DE_time' or 'X106_DE_time'.
    """
    mat = scipy.io.loadmat(mat_filepath)
    for key in mat.keys():
        if "DE_time" in key:
            return mat[key].flatten()
    raise KeyError(f"No variable containing '_DE_time' found in {mat_filepath}")

def slice_into_windows(signal_1d, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """
    Converts a continuous 1D stream of ~120,000 readings into 
    discrete 128-element windows that mimic an MCU's streaming input buffer.
    """
    windows = []
    total_len = len(signal_1d)
    for start in range(0, total_len - window_size + 1, step_size):
        windows.append(signal_1d[start : start + window_size])
    return np.array(windows, dtype=np.float32)

def main():
    print("=== Step 1: Ingesting Real CWRU 1D Sensor Data ===")
    
    X_list = []
    y_list = []
    class_names = ["normal", "inner_race", "ball", "outer_race"]

    for class_id, label in enumerate(class_names):
        path = FILES[label]
        
        # Check for .mat extension if Windows saved it without extension
        if not os.path.exists(path) and os.path.exists(path.replace(".mat", "")):
            path = path.replace(".mat", "")

        if not os.path.exists(path):
            print(f"[ERROR] Could not find file at: {path}")
            print("Please check that the script is in the same folder as 'normal' and '0.007'.")
            return

        print(f"Loading {label} from {path}...")
        raw_signal = extract_drive_end_signal(path)
        windows = slice_into_windows(raw_signal)

        # Standardize each window (zero mean, unit variance)
        # Prevents numerical saturation when converted to INT8 later
        mean = np.mean(windows, axis=1, keepdims=True)
        std = np.std(windows, axis=1, keepdims=True) + 1e-8
        windows = (windows - mean) / std

        labels = np.full(len(windows), class_id, dtype=np.int64)
        
        X_list.append(windows)
        y_list.append(labels)
        print(f"  -> Generated {len(windows)} windows (shape: {windows.shape})")

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # Save processed authentic dataset
    output_path = "cwru_1d_dataset.npz"
    np.savez_compressed(output_path, X=X, y=y)

    print("\n--- Dataset Ingestion Summary ---")
    print(f"Output File:       {output_path}")
    print(f"Total Samples:     {X.shape[0]}")
    print(f"Feature Dimension: {X.shape[1]} readings per sample")
    print(f"Class Breakdown:   Class 0 (Normal): {np.sum(y == 0)}")
    print(f"                   Class 1 (IR):     {np.sum(y == 1)}")
    print(f"                   Class 2 (Ball):   {np.sum(y == 2)}")
    print(f"                   Class 3 (OR):     {np.sum(y == 3)}")
    print("Dataset generation complete. Ready for baseline model training.")

if __name__ == "__main__":
    main()