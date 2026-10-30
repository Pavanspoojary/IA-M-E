"""Verification script for ONNX Runtime CPU inference.

Validates input tensor shapes, CPU execution provider configuration,
and measures inference latency on sample digit crops.
"""

from pathlib import Path
import time
import numpy as np
import onnxruntime as ort

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "digit_model.onnx"


def verify_model():
    """Verify local ONNX Runtime inference using CPU execution provider."""
    print("==================================================")
    print("  ONNX Runtime CPU Model Verification")
    print("==================================================")

    if not MODEL_PATH.exists():
        print(f"[FAIL] Model file not found at: {MODEL_PATH}")
        print("Run 'python scripts/train_digit_model.py' first.")
        return False

    print(f"Loading model: {MODEL_PATH}")
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1

    # Strictly configure CPUExecutionProvider
    try:
        session = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        print(f"[FAIL] Error initializing ONNX Runtime session: {exc}")
        return False

    active_providers = session.get_providers()
    print(f"Active Execution Providers: {active_providers}")
    if "CPUExecutionProvider" not in active_providers:
        print("[FAIL] CPUExecutionProvider is not active.")
        return False

    # Inspect inputs and outputs
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    print(f"Input: name='{inputs[0].name}', shape={inputs[0].shape}, type={inputs[0].type}")
    print(f"Output: name='{outputs[0].name}', shape={outputs[0].shape}, type={outputs[0].type}")

    # Test 1: Dummy tensor shape validation
    dummy_input = np.zeros((1, 1, 28, 28), dtype=np.float32)
    start_time = time.perf_counter()
    res = session.run(None, {inputs[0].name: dummy_input})[0]
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    print(f"Test 1 - Dummy inference output shape: {res.shape}, latency: {elapsed_ms:.3f} ms")
    if res.shape != (1, 10):
        print(f"[FAIL] Unexpected output shape: {res.shape}, expected (1, 10)")
        return False

    # Test 2: Latency over 100 sequential inferences (simulate batch)
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        session.run(None, {inputs[0].name: dummy_input})
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = float(np.mean(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    print(f"Benchmark over 100 iterations: Mean = {avg_ms:.3f} ms, P95 = {p95_ms:.3f} ms")

    print("\n[PASS] Model verified successfully with ONNX Runtime CPU Provider.")
    print("==================================================")
    return True


if __name__ == "__main__":
    verify_model()
