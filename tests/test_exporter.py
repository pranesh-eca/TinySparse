import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_adaptive_model import export_model  # noqa: E402


class ExporterTest(unittest.TestCase):
    def test_mixed_storage_executes_reference_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "tiny.npz"
            header = root / "model.h"
            weights1 = np.zeros((8, 3), dtype=np.int8)
            weights1[0, 0] = 1
            weights1[1, 1:] = [1, -1]
            weights2 = np.zeros((2, 8), dtype=np.int8)
            weights2[0, 0] = 3
            weights2[1, 0] = -2
            np.savez(source, fc1_w=weights1, fc1_b=np.zeros(8, dtype=np.int32),
                     fc1_shift=np.array(0, dtype=np.int32), fc2_w=weights2,
                     fc2_b=np.array([1, 4], dtype=np.int32),
                     ref_input=np.array([2, 3, 4], dtype=np.int8),
                     ref_logits=np.array([7, 0], dtype=np.int32))
            summary = export_model(source, header)
            self.assertEqual([item[1] for item in summary], ["dense", "CSR"])
            first = header.read_bytes()
            export_model(source, header)
            self.assertEqual(header.read_bytes(), first)

            runner = root / "verify.cpp"
            runner.write_text('''#include "model.h"
int main() {
    using namespace TinySparseExport;
    if (!TinySparse::validate_model(model)) return 1;
    int8_t workspace[16] = {};
    int32_t output[2] = {};
    if (TinySparse::run_model(model, golden_input, workspace, sizeof(workspace), output)
        != TinySparse::ModelStatus::Ok) return 2;
    return output[0] == golden_logits[0] && output[1] == golden_logits[1] ? 0 : 3;
}
''', encoding="utf-8")
            executable = root / "verify.exe"
            subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Wpedantic",
                            "-I", str(ROOT / "include"), "-I", str(root),
                            str(runner), "-o", str(executable)], check=True)
            subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    unittest.main()
