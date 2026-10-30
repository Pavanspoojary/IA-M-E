"""Train and export a compact handwritten digit classifier to ONNX format.

Generates an augmented training dataset from standard typography, thickness variations,
and bluebook digit styles, trains neural network weights, and exports a verified
ONNX model ready for local CPU execution in ONNX Runtime.
"""

from pathlib import Path
import numpy as np
import cv2
import onnx
from onnx import helper, TensorProto

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "digit_model.onnx"


def generate_digit_dataset(samples_per_digit: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize an augmented training dataset of 28x28 normalized digits.

    Applies affine transformations, rotations (-14 to +14 deg), thickness variations,
    and multiple font styles.
    """
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_PLAIN,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
    ]

    x_list = []
    y_list = []

    np.random.seed(42)

    for digit in range(10):
        digit_str = str(digit)
        count = 0

        while count < samples_per_digit:
            canvas = np.zeros((48, 48), dtype=np.uint8)
            thickness = np.random.choice([1, 2])

            # For 4: include handwritten open-top 4 variation
            if digit == 4 and np.random.rand() < 0.5:
                x1 = np.random.randint(24, 29)
                y1 = np.random.randint(8, 14)
                x2 = np.random.randint(13, 17)
                y2 = np.random.randint(24, 29)
                cv2.line(canvas, (x1, y1), (x2, y2), 255, thickness, cv2.LINE_AA)
                x3 = np.random.randint(31, 36)
                y3 = y2 + np.random.randint(-1, 2)
                cv2.line(canvas, (x2 - 1, y2), (x3, y3), 255, thickness, cv2.LINE_AA)
                x4 = x1
                y4 = np.random.randint(7, 12)
                x5 = x4 + np.random.randint(-1, 2)
                y5 = np.random.randint(35, 41)
                cv2.line(canvas, (x4, y4), (x5, y5), 255, thickness, cv2.LINE_AA)
            else:
                font = fonts[np.random.randint(0, len(fonts))]
                scale = np.random.uniform(0.75, 1.15)
                (tw, th), baseline = cv2.getTextSize(digit_str, font, scale, thickness)
                tx = (48 - tw) // 2 + np.random.randint(-2, 3)
                ty = (48 + th) // 2 + np.random.randint(-2, 3)
                cv2.putText(canvas, digit_str, (tx, ty), font, scale, 255, thickness, cv2.LINE_AA)

            # Random rotation
            angle = np.random.uniform(-14.0, 14.0)
            rot_mat = cv2.getRotationMatrix2D((24, 24), angle, 1.0)
            rotated = cv2.warpAffine(canvas, rot_mat, (48, 48), flags=cv2.INTER_LINEAR)

            # Crop tightly around bounding box
            coords = cv2.findNonZero(rotated)
            if coords is None:
                continue
            bx, by, bw, bh = cv2.boundingRect(coords)
            if bw <= 0 or bh <= 0:
                continue
            digit_crop = rotated[by : by + bh, bx : bx + bw]

            # Scale so max dimension is 20 pixels (standard MNIST centering convention)
            max_dim = max(bw, bh)
            scale_factor = 20.0 / max_dim
            new_w = max(1, int(round(bw * scale_factor)))
            new_h = max(1, int(round(bh * scale_factor)))
            resized = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Center inside 28x28 canvas
            final_img = np.zeros((28, 28), dtype=np.float32)
            pad_x = (28 - new_w) // 2
            pad_y = (28 - new_h) // 2
            final_img[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized / 255.0

            x_list.append(final_img)
            y_list.append(digit)
            count += 1

    x_arr = np.array(x_list, dtype=np.float32)[:, np.newaxis, :, :]  # Shape: (N, 1, 28, 28)
    y_arr = np.array(y_list, dtype=np.int64)

    # Shuffle dataset
    indices = np.arange(len(x_arr))
    np.random.shuffle(indices)
    return x_arr[indices], y_arr[indices]


def train_and_export_model(output_path: Path = MODEL_PATH):
    """Train a 3-layer neural network classifier and export directly to ONNX format."""
    print("Generating training dataset with affine augmentations...")
    x_train, y_train = generate_digit_dataset(samples_per_digit=450)
    num_samples = len(x_train)
    print(f"Dataset generated: {num_samples} samples across 10 digit classes.")

    # Flatten inputs for training: (N, 784)
    x_flat = x_train.reshape(num_samples, 784)

    # One-hot encode targets
    y_onehot = np.zeros((num_samples, 10), dtype=np.float32)
    y_onehot[np.arange(num_samples), y_train] = 1.0

    # Initialize weights
    np.random.seed(1337)
    w1 = np.random.randn(784, 128).astype(np.float32) * np.sqrt(2.0 / 784)
    b1 = np.zeros(128, dtype=np.float32)
    w2 = np.random.randn(128, 64).astype(np.float32) * np.sqrt(2.0 / 128)
    b2 = np.zeros(64, dtype=np.float32)
    w3 = np.random.randn(64, 10).astype(np.float32) * np.sqrt(2.0 / 64)
    b3 = np.zeros(10, dtype=np.float32)

    # Training parameters
    learning_rate = 0.002
    epochs = 40
    batch_size = 64
    num_batches = num_samples // batch_size

    # Adam optimizer moments
    m_w1, v_w1 = np.zeros_like(w1), np.zeros_like(w1)
    m_b1, v_b1 = np.zeros_like(b1), np.zeros_like(b1)
    m_w2, v_w2 = np.zeros_like(w2), np.zeros_like(w2)
    m_b2, v_b2 = np.zeros_like(b2), np.zeros_like(b2)
    m_w3, v_w3 = np.zeros_like(w3), np.zeros_like(w3)
    m_b3, v_b3 = np.zeros_like(b3), np.zeros_like(b3)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t_step = 0

    print("Beginning model optimization...")
    for epoch in range(epochs):
        perm = np.random.permutation(num_samples)
        x_shuff = x_flat[perm]
        y_shuff = y_onehot[perm]

        epoch_loss = 0.0

        for b in range(num_batches):
            t_step += 1
            bx = x_shuff[b * batch_size : (b + 1) * batch_size]
            by = y_shuff[b * batch_size : (b + 1) * batch_size]

            # Forward pass
            z1 = np.dot(bx, w1) + b1
            a1 = np.maximum(0.0, z1)

            z2 = np.dot(a1, w2) + b2
            a2 = np.maximum(0.0, z2)

            z3 = np.dot(a2, w3) + b3
            # Stable softmax
            shift_z3 = z3 - np.max(z3, axis=1, keepdims=True)
            exp_z3 = np.exp(shift_z3)
            probs = exp_z3 / np.sum(exp_z3, axis=1, keepdims=True)

            loss = -np.mean(np.sum(by * np.log(probs + 1e-12), axis=1))
            epoch_loss += loss

            # Backward pass
            dz3 = (probs - by) / batch_size
            dw3 = np.dot(a2.T, dz3)
            db3 = np.sum(dz3, axis=0)

            da2 = np.dot(dz3, w3.T)
            dz2 = da2 * (z2 > 0)
            dw2 = np.dot(a1.T, dz2)
            db2 = np.sum(dz2, axis=0)

            da1 = np.dot(dz2, w2.T)
            dz1 = da1 * (z1 > 0)
            dw1 = np.dot(bx.T, dz1)
            db1 = np.sum(dz1, axis=0)

            # Adam update
            for p, g, m, v in [
                (w1, dw1, m_w1, v_w1), (b1, db1, m_b1, v_b1),
                (w2, dw2, m_w2, v_w2), (b2, db2, m_b2, v_b2),
                (w3, dw3, m_w3, v_w3), (b3, db3, m_b3, v_b3),
            ]:
                m[:] = beta1 * m + (1.0 - beta1) * g
                v[:] = beta2 * v + (1.0 - beta2) * (g ** 2)
                m_hat = m / (1.0 - beta1 ** t_step)
                v_hat = v / (1.0 - beta2 ** t_step)
                p -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            # Measure training accuracy
            z1_all = np.maximum(0.0, np.dot(x_flat, w1) + b1)
            z2_all = np.maximum(0.0, np.dot(z1_all, w2) + b2)
            logits_all = np.dot(z2_all, w3) + b3
            preds = np.argmax(logits_all, axis=1)
            acc = np.mean(preds == y_train) * 100.0
            print(f"Epoch {epoch + 1:2d}/{epochs:2d} - Loss: {epoch_loss / num_batches:.4f} - Accuracy: {acc:.2f}%")

    # Export to ONNX format
    print(f"\nExporting trained model to ONNX: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ONNX Graph Definition
    x_val_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 1, 28, 28])
    y_val_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 10])

    flatten_node = helper.make_node("Flatten", ["input"], ["flat"], axis=1)

    w1_init = helper.make_tensor("W1", TensorProto.FLOAT, [784, 128], w1.flatten())
    b1_init = helper.make_tensor("b1", TensorProto.FLOAT, [128], b1)
    gemm1_node = helper.make_node("Gemm", ["flat", "W1", "b1"], ["fc1"], alpha=1.0, beta=1.0)
    relu1_node = helper.make_node("Relu", ["fc1"], ["relu1"])

    w2_init = helper.make_tensor("W2", TensorProto.FLOAT, [128, 64], w2.flatten())
    b2_init = helper.make_tensor("b2", TensorProto.FLOAT, [64], b2)
    gemm2_node = helper.make_node("Gemm", ["relu1", "W2", "b2"], ["fc2"], alpha=1.0, beta=1.0)
    relu2_node = helper.make_node("Relu", ["fc2"], ["relu2"])

    w3_init = helper.make_tensor("W3", TensorProto.FLOAT, [64, 10], w3.flatten())
    b3_init = helper.make_tensor("b3", TensorProto.FLOAT, [10], b3)
    gemm3_node = helper.make_node("Gemm", ["relu2", "W3", "b3"], ["logits"], alpha=1.0, beta=1.0)
    softmax_node = helper.make_node("Softmax", ["logits"], ["output"], axis=1)

    graph = helper.make_graph(
        [flatten_node, gemm1_node, relu1_node, gemm2_node, relu2_node, gemm3_node, softmax_node],
        "IA_Bluebook_DigitClassifier",
        [x_val_info],
        [y_val_info],
        [w1_init, b1_init, w2_init, b2_init, w3_init, b3_init],
    )

    opset = helper.make_opsetid("", 17)
    onnx_model = helper.make_model(
        graph,
        producer_name="ia_bluebook_digit_classifier",
        ir_version=10,
        opset_imports=[opset],
    )
    onnx.checker.check_model(onnx_model)

    onnx.save(onnx_model, str(output_path))
    file_size_kb = output_path.stat().st_size / 1024.0
    print(f"Model successfully saved ({file_size_kb:.1f} KB).")


if __name__ == "__main__":
    train_and_export_model()
