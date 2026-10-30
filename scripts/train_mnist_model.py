"""Train a high-accuracy Convolutional Neural Network on MNIST handwritten digits.

Exports a compact, robust ONNX model to `models/digit_model.onnx` optimized for CPUExecutionProvider.
Strict Zero Emojis Policy Enforced.
"""

from pathlib import Path
import gzip
import struct
import urllib.request
import os
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "digit_model.onnx"
DATA_DIR = BASE_DIR / "models" / "mnist_data"


def ensure_mnist_downloaded():
    """Download MNIST raw binary files if not present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    for f in files:
        dest = DATA_DIR / f
        if not dest.exists():
            print(f"Downloading {f}...")
            urllib.request.urlretrieve(base_url + f, dest)
            print(f"Downloaded {f}: {dest.stat().st_size} bytes")


def load_mnist_images(filename: Path) -> np.ndarray:
    """Load and parse IDX3-ubyte gzip image file."""
    with gzip.open(filename, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image magic number: {magic}")
        buf = f.read(num_images * rows * cols)
        data = np.frombuffer(buf, dtype=np.uint8)
        return data.reshape(num_images, 1, rows, cols).astype(np.float32) / 255.0


def load_mnist_labels(filename: Path) -> np.ndarray:
    """Load and parse IDX1-ubyte gzip label file."""
    with gzip.open(filename, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label magic number: {magic}")
        buf = f.read(num_labels)
        return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)


def train_and_export():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    ensure_mnist_downloaded()

    print("Loading MNIST training and testing sets...")
    train_x = load_mnist_images(DATA_DIR / "train-images-idx3-ubyte.gz")
    train_y = load_mnist_labels(DATA_DIR / "train-labels-idx1-ubyte.gz")
    test_x = load_mnist_images(DATA_DIR / "t10k-images-idx3-ubyte.gz")
    test_y = load_mnist_labels(DATA_DIR / "t10k-labels-idx1-ubyte.gz")

    print(f"Loaded {len(train_x)} training samples and {len(test_x)} testing samples.")

    # Create PyTorch DataLoaders
    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    test_ds = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # Compact CNN Architecture
    class CompactDigitCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # 14x14
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # 7x7
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(32 * 7 * 7, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 10),
                nn.Softmax(dim=1),
            )

        def forward(self, x):
            f = self.features(x)
            return self.classifier(f)

    model = CompactDigitCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    print("Beginning CNN model training on MNIST...")
    device = torch.device("cpu")
    model.to(device)

    for epoch in range(1, 4):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_acc = (correct / total) * 100.0
        print(f"Epoch {epoch}/3 - Training Accuracy: {epoch_acc:.2f}%")

    # Evaluate on test set
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_acc = (test_correct / test_total) * 100.0
    print(f"Final Test Accuracy on unseen handwriting: {test_acc:.2f}%")

    # Export to ONNX
    print(f"Exporting model to ONNX: {MODEL_PATH}")
    dummy_input = torch.randn(1, 1, 28, 28, requires_grad=False)
    torch.onnx.export(
        model,
        dummy_input,
        str(MODEL_PATH),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"ONNX model saved successfully: {MODEL_PATH.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    train_and_export()
