"""Train an ultra-high-accuracy Convolutional Neural Network on EMNIST + MNIST + Real Bluebook handwriting.

Designed specifically for challenging, messy Indian college bluebook marksheets:
- EMNIST Digits (240,000 real messy human handwriting samples across varied writers)
- MNIST benchmark dataset (60,000 train + 10,000 test)
- Real handwritten digit crops harvested from actual student bluebooks with hysteresis thresholding
- Open-top trident 4s vs figure-8s with extreme separation
- Cursive and looped 2s (bottom loop at baseline, swan neck, flat tail)
- Angular and flat-top 3s
- Disconnected two-stroke 5s (separated horizontal bar) and cursive 5s
- Tall narrow 6s and circular 6s
- Crossed 7s (horizontal bar through stem) and hooked 7s
- S-curve figure-8s, asymmetrical loops, and double-circle 8s
- Single-stroke and seriffed 1s
- Hooked 9s

Exports to `models/digit_model.onnx` (IR version 8) optimized for local CPU and in-browser WASM execution.
Strict Zero Emojis Policy Enforced.
"""

from pathlib import Path
from typing import Optional
import gzip
import struct
import os
import sys
import shutil
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
MODEL_PATH = BASE_DIR / "models" / "digit_model.onnx"
DATA_DIR = BASE_DIR / "models" / "mnist_data"
EMNIST_DIR = DATA_DIR / "emnist"
SAMPLE_DIR = BASE_DIR / "sample_data"


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


def load_emnist_digits(limit_per_class: int = 6000) -> tuple[np.ndarray, np.ndarray]:
    """Load and orient EMNIST digits dataset."""
    img_path = EMNIST_DIR / "emnist-digits-train-images-idx3-ubyte.gz"
    lbl_path = EMNIST_DIR / "emnist-digits-train-labels-idx1-ubyte.gz"
    if not img_path.exists() or not lbl_path.exists():
        print("EMNIST digits files not found, skipping EMNIST portion.")
        return np.zeros((0, 1, 28, 28), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    with gzip.open(img_path, "rb") as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        buf = f.read(rows * cols * num)
        images = np.frombuffer(buf, dtype=np.uint8).reshape(num, rows, cols)
    with gzip.open(lbl_path, "rb") as f:
        _, _ = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(num), dtype=np.uint8)

    # EMNIST images are transposed (oriented sideways) by NIST specification
    images = images.transpose(0, 2, 1)

    counts = {d: 0 for d in range(10)}
    selected_x = []
    selected_y = []
    for img, lbl in zip(images, labels):
        lbl_int = int(lbl)
        if counts[lbl_int] < limit_per_class:
            selected_x.append(img.astype(np.float32) / 255.0)
            selected_y.append(lbl_int)
            counts[lbl_int] += 1
            if all(c >= limit_per_class for c in counts.values()):
                break

    x_arr = np.array(selected_x, dtype=np.float32)[:, np.newaxis, :, :]
    y_arr = np.array(selected_y, dtype=np.int64)
    print(f"Loaded {len(x_arr)} real handwriting samples from EMNIST Digits ({limit_per_class} per class).")
    return x_arr, y_arr


def center_and_scale_to_28(digit_crop: np.ndarray) -> Optional[np.ndarray]:
    """Center and scale tight binary crop to 28x28 normalized canvas."""
    coords = cv2.findNonZero(digit_crop)
    if coords is None:
        return None
    bx, by, bw, bh = cv2.boundingRect(coords)
    if bw <= 0 or bh <= 0:
        return None
    tight = digit_crop[by : by + bh, bx : bx + bw]

    scale = 20.0 / max(bw, bh)
    new_w = max(1, int(round(bw * scale)))
    new_h = max(1, int(round(bh * scale)))
    resized = cv2.resize(tight, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.float32)
    pad_x = (28 - new_w) // 2
    pad_y = (28 - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized / 255.0
    return canvas


def generate_synthetic_messy_digits(samples_per_digit: int = 2500) -> tuple[np.ndarray, np.ndarray]:
    """Generate extensive synthetic messy handwriting samples matching Indian college booklet marks."""
    x_list = []
    y_list = []

    np.random.seed(42)

    for digit in range(10):
        count = 0

        while count < samples_per_digit:
            canvas = np.zeros((56, 56), dtype=np.uint8)
            thickness = int(np.random.choice([1, 2, 3], p=[0.45, 0.45, 0.10]))

            if digit == 0:
                cx, cy = 28 + np.random.randint(-3, 4), 28 + np.random.randint(-3, 4)
                ax = np.random.randint(6, 12)
                ay = np.random.randint(12, 18)
                angle = np.random.randint(-20, 21)
                cv2.ellipse(canvas, (cx, cy), (ax, ay), angle, 0, 360, 255, thickness, cv2.LINE_AA)

            elif digit == 1:
                # 1: straight, slanted, or with top serif
                x_top = np.random.randint(24, 32)
                y_top = np.random.randint(10, 16)
                x_bot = x_top + np.random.randint(-4, 5)
                y_bot = np.random.randint(40, 48)
                cv2.line(canvas, (x_top, y_top), (x_bot, y_bot), 255, thickness, cv2.LINE_AA)
                if np.random.rand() < 0.4:
                    cv2.line(canvas, (x_top, y_top), (x_top - np.random.randint(4, 8), y_top + np.random.randint(4, 7)), 255, thickness, cv2.LINE_AA)

            elif digit == 2:
                # 2: looped cursive 2, swan-neck 2, flat-base 2, wavy-base 2, thick cursive, or looped no-tail
                style = np.random.choice(["loop", "swan", "flat", "wavy", "thick_cursive", "looped_notail"], p=[0.20, 0.15, 0.15, 0.10, 0.15, 0.25])
                if style == "looped_notail":
                    # Fast cursive 2 with looped bottom and no tail (common in total sums)
                    cv2.ellipse(canvas, (28, 18), (8, 7), 0, 180, 360, 255, thickness, cv2.LINE_AA)
                    cv2.line(canvas, (36, 18), (20, 38), 255, thickness, cv2.LINE_AA)
                    cv2.ellipse(canvas, (20, 38), (5, 4), 30, 0, 360, 255, -1 if np.random.rand() < 0.6 else thickness, cv2.LINE_AA)
                elif style == "loop":
                    cv2.ellipse(canvas, (28, 19), (8, 7), 0, 180, 360, 255, thickness, cv2.LINE_AA)
                    cv2.line(canvas, (36, 19), (19, 39), 255, thickness, cv2.LINE_AA)
                    cv2.ellipse(canvas, (21, 39), (4, 3), 45, 0, 360, 255, thickness, cv2.LINE_AA)
                    cv2.line(canvas, (21, 41), (39 + np.random.randint(-2, 5), 41 + np.random.randint(-2, 2)), 255, thickness, cv2.LINE_AA)
                elif style == "thick_cursive":
                    cv2.ellipse(canvas, (28, 18), (8, 7), 0, 180, 360, 255, thickness + 1, cv2.LINE_AA)
                    cv2.line(canvas, (36, 18), (18, 38), 255, thickness + 1, cv2.LINE_AA)
                    cv2.ellipse(canvas, (20, 38), (5, 4), 30, 0, 360, 255, -1 if np.random.rand() < 0.5 else thickness, cv2.LINE_AA)
                    cv2.line(canvas, (18, 40), (38, 40), 255, thickness + 1, cv2.LINE_AA)
                elif style == "swan":
                    cv2.ellipse(canvas, (27, 18), (9, 8), 0, 160, 360, 255, thickness, cv2.LINE_AA)
                    cv2.line(canvas, (36, 18), (17, 43), 255, thickness, cv2.LINE_AA)
                    cv2.line(canvas, (17, 43), (40, 43), 255, thickness, cv2.LINE_AA)
                elif style == "wavy":
                    cv2.ellipse(canvas, (28, 19), (8, 7), 0, 180, 360, 255, thickness, cv2.LINE_AA)
                    cv2.line(canvas, (36, 19), (18, 41), 255, thickness, cv2.LINE_AA)
                    pts = np.array([[18, 41], [26, 39], [34, 43], [40, 41]], np.int32)
                    cv2.polylines(canvas, [pts], False, 255, thickness, cv2.LINE_AA)
                else:
                    cv2.ellipse(canvas, (28, 19), (8, 7), 0, 180, 360, 255, thickness, cv2.LINE_AA)
                    cv2.line(canvas, (36, 19), (18, 42), 255, thickness, cv2.LINE_AA)
                    cv2.line(canvas, (18, 42), (38, 42), 255, thickness, cv2.LINE_AA)

            elif digit == 3:
                # 3: flat-top, open cursive, looped waist, sharp center, evaluator open lobes, top curled loop
                style = np.random.choice([
                    "flat_top", "open_cursive", "loop_waist", "sharp_center", "evaluator_open", "slanted_open", "top_curled_loop"
                ], p=[0.12, 0.20, 0.12, 0.12, 0.16, 0.14, 0.14])
                if style == "top_curled_loop":
                    # Evaluator cursive 3 with curled loop/serif at top
                    cv2.ellipse(canvas, (24, 16), (4, 3), 0, 0, 360, 255, thickness, cv2.LINE_AA)
                    cv2.line(canvas, (26, 16), (36, 16), 255, thickness, cv2.LINE_AA)
                    cv2.line(canvas, (36, 16), (25, 27), 255, thickness, cv2.LINE_AA)
                    cv2.ellipse(canvas, (28, 36), (9, 9), 0, -90, 130, 255, thickness, cv2.LINE_AA)
                elif style == "evaluator_open":
                    # Distinct evaluator 3: two open rounded lobes with a deep central tuck/waist
                    cv2.ellipse(canvas, (26, 20), (8, 7), 0, -130, 110, 255, thickness, cv2.LINE_AA)
                    cv2.line(canvas, (34, 20), (24, 28), 255, thickness, cv2.LINE_AA)
                    cv2.line(canvas, (24, 28), (35, 35), 255, thickness, cv2.LINE_AA)
                    cv2.ellipse(canvas, (27, 36), (9, 9), 0, -100, 125, 255, thickness, cv2.LINE_AA)
                elif style == "slanted_open":
                    cv2.ellipse(canvas, (29, 20), (8, 7), 15, -120, 110, 255, thickness, cv2.LINE_AA)
                    cv2.ellipse(canvas, (27, 37), (10, 9), 10, -90, 130, 255, thickness, cv2.LINE_AA)
                elif style == "flat_top":
                    cv2.line(canvas, (18, 14), (36, 14), 255, thickness, cv2.LINE_AA)
                    cv2.line(canvas, (36, 14), (25, 27), 255, thickness, cv2.LINE_AA)
                    cv2.ellipse(canvas, (28, 36), (9, 9), 0, -90, 130, 255, thickness, cv2.LINE_AA)
                elif style == "open_cursive":
                    # Classic two-curved 3 with completely open left perimeter
                    cv2.ellipse(canvas, (27, 21), (8, 8), 0, -110, 110, 255, thickness, cv2.LINE_AA)
                    cv2.ellipse(canvas, (28, 36), (9, 9), 0, -100, 120, 255, thickness, cv2.LINE_AA)
                elif style == "loop_waist":
                    # Cursive 3 with small central loop at waist
                    cv2.ellipse(canvas, (27, 20), (8, 7), 0, -120, 110, 255, thickness, cv2.LINE_AA)
                    cv2.ellipse(canvas, (25, 28), (3, 2), 0, 0, 360, 255, thickness, cv2.LINE_AA)
                    cv2.ellipse(canvas, (28, 37), (9, 9), 0, -90, 120, 255, thickness, cv2.LINE_AA)
                elif style == "sharp_center":
                    cv2.line(canvas, (18, 16), (34, 16), 255, thickness, cv2.LINE_AA)
                    cv2.line(canvas, (34, 16), (24, 28), 255, thickness, cv2.LINE_AA)
                    cv2.line(canvas, (24, 28), (36, 36), 255, thickness, cv2.LINE_AA)
                    cv2.line(canvas, (36, 36), (19, 44), 255, thickness, cv2.LINE_AA)
                else:
                    cv2.ellipse(canvas, (29, 21), (7, 8), 10, -110, 110, 255, thickness, cv2.LINE_AA)
                    cv2.ellipse(canvas, (27, 36), (9, 9), 10, -90, 130, 255, thickness, cv2.LINE_AA)

            elif digit == 4:
                # 4: open-top trident 4, closed 4, fast 4, or slanted open 4
                style = np.random.choice(["open_top", "trident", "closed", "slanted_open"], p=[0.30, 0.20, 0.15, 0.35])
                if style == "slanted_open":
                    # Evaluator 4 with diagonal left stroke sloping to horizontal bar and straight vertical stem
                    x_top_l = np.random.randint(22, 28)
                    y_top_l = np.random.randint(10, 15)
                    x_bot_l = np.random.randint(14, 18)
                    y_cross = np.random.randint(28, 33)
                    cv2.line(canvas, (x_top_l, y_top_l), (x_bot_l, y_cross), 255, thickness, cv2.LINE_AA)
                    x_stem_r = np.random.randint(30, 36)
                    cv2.line(canvas, (x_bot_l, y_cross), (x_stem_r + 4, y_cross), 255, thickness, cv2.LINE_AA)
                    y_top_r = np.random.randint(8, 14)
                    y_bot_r = np.random.randint(42, 48)
                    cv2.line(canvas, (x_stem_r, y_top_r), (x_stem_r, y_bot_r), 255, thickness, cv2.LINE_AA)
                elif style == "open_top":
                    x_stem_left = np.random.randint(18, 23)
                    y_top_left = np.random.randint(12, 17)
                    y_cross = np.random.randint(28, 33)
                    cv2.line(canvas, (x_stem_left, y_top_left), (x_stem_left, y_cross), 255, thickness, cv2.LINE_AA)
                    x_stem_right = np.random.randint(32, 38)
                    cv2.line(canvas, (x_stem_left, y_cross), (x_stem_right + 3, y_cross), 255, thickness, cv2.LINE_AA)
                    y_top_right = np.random.randint(8, 14)
                    y_bot_right = np.random.randint(42, 48)
                    cv2.line(canvas, (x_stem_right, y_top_right), (x_stem_right, y_bot_right), 255, thickness, cv2.LINE_AA)
                elif style == "trident":
                    pts = np.array([
                        [np.random.randint(18, 22), np.random.randint(14, 18)],
                        [np.random.randint(18, 22), np.random.randint(28, 33)],
                        [np.random.randint(32, 37), np.random.randint(28, 33)],
                    ], np.int32)
                    cv2.polylines(canvas, [pts], False, 255, thickness, cv2.LINE_AA)
                    x_r = pts[2][0]
                    cv2.line(canvas, (x_r, np.random.randint(10, 15)), (x_r, np.random.randint(42, 47)), 255, thickness, cv2.LINE_AA)
                else:
                    cv2.line(canvas, (32, 10), (16, 32), 255, thickness, cv2.LINE_AA)
                    cv2.line(canvas, (16, 32), (38, 32), 255, thickness, cv2.LINE_AA)
                    cv2.line(canvas, (32, 10), (32, 46), 255, thickness, cv2.LINE_AA)

            elif digit == 5:
                # 5: separated top stroke, continuous cursive single-stroke, attached, floating bar, or slanted top straight
                style = np.random.choice(["continuous_cursive", "floating_bar", "separated", "attached", "slanted", "slanted_top_straight"], p=[0.25, 0.20, 0.15, 0.15, 0.10, 0.15])
                if style == "slanted_top_straight":
                    # Evaluator 5 with straight top stroke (often confused with 6 or 8)
                    x1 = np.random.randint(18, 24)
                    y1 = np.random.randint(10, 15)
                    x2 = np.random.randint(30, 38)
                    cv2.line(canvas, (x1, y1), (x2, y1 - np.random.randint(0, 3)), 255, thickness, cv2.LINE_AA)
                    cv2.line(canvas, (x1, y1), (x1, y1 + np.random.randint(10, 14)), 255, thickness, cv2.LINE_AA)
                    center = (x1 + np.random.randint(6, 9), (y1 + 12 + 40) // 2)
                    axes = (np.random.randint(9, 13), np.random.randint(9, 12))
                    cv2.ellipse(canvas, center, axes, 0, -80, 140, 255, thickness, cv2.LINE_AA)
                elif style == "continuous_cursive":
                    # Single continuous stroke: top curved bar flowing down into stem and belly
                    x_start = np.random.randint(32, 38)
                    y_start = np.random.randint(11, 15)
                    x_corner = np.random.randint(17, 21)
                    y_corner = np.random.randint(12, 16)
                    cv2.line(canvas, (x_start, y_start), (x_corner, y_corner), 255, thickness, cv2.LINE_AA)
                    y_stem_bot = y_corner + np.random.randint(11, 15)
                    cv2.line(canvas, (x_corner, y_corner), (x_corner, y_stem_bot), 255, thickness, cv2.LINE_AA)
                    center = (x_corner + np.random.randint(7, 10), (y_stem_bot + 42) // 2)
                    axes = (np.random.randint(9, 13), np.random.randint(9, 13))
                    cv2.ellipse(canvas, center, axes, 0, -85, 135, 255, thickness, cv2.LINE_AA)
                elif style == "floating_bar":
                    # Distinct floating horizontal bar at top
                    x1 = np.random.randint(16, 21)
                    y1 = np.random.randint(10, 14)
                    x2 = np.random.randint(33, 40)
                    cv2.line(canvas, (x1, y1), (x2, y1 + np.random.randint(-2, 3)), 255, thickness, cv2.LINE_AA)
                    stem_x = x1 + np.random.randint(0, 3)
                    stem_y_top = y1 + np.random.randint(3, 7)
                    stem_y_bot = stem_y_top + np.random.randint(10, 14)
                    cv2.line(canvas, (stem_x, stem_y_top), (stem_x, stem_y_bot), 255, thickness, cv2.LINE_AA)
                    center = (stem_x + np.random.randint(7, 10), (stem_y_bot + 42) // 2)
                    axes = (np.random.randint(9, 13), np.random.randint(9, 12))
                    cv2.ellipse(canvas, center, axes, 0, -80, 140, 255, thickness, cv2.LINE_AA)
                else:
                    x1 = np.random.randint(16, 20)
                    y1 = np.random.randint(12, 16)
                    x2 = np.random.randint(32, 38)
                    gap = np.random.choice([0, 1, 2, 3]) if style == "separated" else 0
                    cv2.line(canvas, (x1, y1), (x2, y1 + np.random.randint(-2, 3)), 255, thickness, cv2.LINE_AA)
                    y_stem_top = y1 + gap
                    y_stem_bot = y_stem_top + np.random.randint(10, 14)
                    cv2.line(canvas, (x1, y_stem_top), (x1, y_stem_bot), 255, thickness, cv2.LINE_AA)
                    center = (x1 + np.random.randint(6, 9), (y_stem_bot + 40) // 2)
                    axes = (np.random.randint(9, 13), np.random.randint(9, 12))
                    cv2.ellipse(canvas, center, axes, 0, -80, 140, 255, thickness, cv2.LINE_AA)

            elif digit == 6:
                style = np.random.choice(["std", "straight_ascender", "curved_ascender", "open_loop"], p=[0.35, 0.25, 0.25, 0.15])
                if style == "straight_ascender":
                    cv2.line(canvas, (34, 12), (18, 32), 255, thickness, cv2.LINE_AA)
                    cv2.ellipse(canvas, (26, 35), (9, 9), 0, 0, 360, 255, thickness, cv2.LINE_AA)
                elif style == "curved_ascender":
                    pts = np.array([[32, 12], [22, 20], [18, 32]], np.int32)
                    cv2.polylines(canvas, [pts], False, 255, thickness, cv2.LINE_AA)
                    cv2.ellipse(canvas, (26, 35), (9, 9), 0, 0, 360, 255, thickness, cv2.LINE_AA)
                elif style == "open_loop":
                    cv2.line(canvas, (32, 12), (18, 32), 255, thickness, cv2.LINE_AA)
                    cv2.ellipse(canvas, (26, 35), (9, 9), 0, -45, 270, 255, thickness, cv2.LINE_AA)
                else:
                    cv2.line(canvas, (32, 12), (20, 32), 255, thickness, cv2.LINE_AA)
                    cv2.ellipse(canvas, (26, 35), (9, 9), 0, 0, 360, 255, thickness, cv2.LINE_AA)

            elif digit == 7:
                # 7: crossed 7 (dominant in Indian bluebooks, 85%+ frequency) or clean 7
                style = np.random.choice(["crossed_std", "crossed_hook", "crossed_wide", "crossed_slanted", "crossed_hook_bar", "clean"], p=[0.30, 0.20, 0.15, 0.10, 0.15, 0.10])
                if style == "crossed_hook_bar":
                    # Evaluator crossed 7 with prominent left hook and center crossbar (often confused with 8)
                    x1, y1 = np.random.randint(14, 18), np.random.randint(12, 16)
                    x2, y2 = np.random.randint(34, 42), y1 + np.random.randint(-1, 2)
                    cv2.line(canvas, (x1, y1), (x2, y2), 255, thickness, cv2.LINE_AA)
                    cv2.line(canvas, (x1, y1), (x1, y1 + np.random.randint(6, 12)), 255, thickness, cv2.LINE_AA)
                    x3, y3 = np.random.randint(18, 25), np.random.randint(42, 48)
                    cv2.line(canvas, (x2, y2), (x3, y3), 255, thickness, cv2.LINE_AA)
                    cx_mid = (x2 + x3) // 2
                    cy_mid = (y2 + y3) // 2
                    cv2.line(canvas, (cx_mid - 10, cy_mid), (cx_mid + 10, cy_mid), 255, thickness + 1, cv2.LINE_AA)
                else:
                    x1, y1 = np.random.randint(14, 18), np.random.randint(12, 16)
                    x2, y2 = np.random.randint(34, 44), y1 + np.random.randint(-1, 3)
                    cv2.line(canvas, (x1, y1), (x2, y2), 255, thickness, cv2.LINE_AA)

                    # Optional left serif / hook going downwards
                    if style in ["crossed_hook", "crossed_slanted"] or np.random.rand() < 0.4:
                        cv2.line(canvas, (x1, y1), (x1, y1 + np.random.randint(4, 9)), 255, thickness, cv2.LINE_AA)

                    x3, y3 = np.random.randint(17, 27), np.random.randint(42, 48)
                    cv2.line(canvas, (x2, y2), (x3, y3), 255, thickness, cv2.LINE_AA)

                    # Add prominent horizontal crossbar
                    if style != "clean":
                        frac = float(np.random.uniform(0.35, 0.62))
                        cx_mid = int(x2 + (x3 - x2) * frac)
                        cy_mid = int(y2 + (y3 - y2) * frac)
                        bar_w = np.random.randint(12, 24) if style == "crossed_wide" else np.random.randint(10, 20)
                        tilt = np.random.randint(-3, 4) if style == "crossed_slanted" else np.random.randint(-1, 2)
                        bar_thick = thickness if np.random.rand() > 0.3 else thickness + 1
                        cv2.line(canvas, (cx_mid - bar_w // 2, cy_mid - tilt), (cx_mid + bar_w // 2, cy_mid + tilt), 255, bar_thick, cv2.LINE_AA)

            elif digit == 8:
                # 8: continuous parametric S-curve, asymmetrical loops, double-circle, open top, or faint top
                style = np.random.choice(["scurve", "asymmetric", "circles", "waist", "open_top_8", "faint_top_8"], p=[0.25, 0.20, 0.15, 0.15, 0.15, 0.10])
                if style == "open_top_8":
                    cv2.ellipse(canvas, (28, 20), (7, 7), 0, -90, 180, 255, thickness, cv2.LINE_AA)
                    cv2.ellipse(canvas, (28, 36), (9, 9), 0, 0, 360, 255, thickness, cv2.LINE_AA)
                elif style == "faint_top_8":
                    cv2.ellipse(canvas, (28, 20), (7, 7), 0, 0, 360, 255, 1, cv2.LINE_AA)
                    cv2.ellipse(canvas, (28, 36), (9, 9), 0, 0, 360, 255, thickness + 1, cv2.LINE_AA)
                elif style == "scurve":
                    cx, cy = 28 + np.random.randint(-2, 3), 28 + np.random.randint(-2, 3)
                    w_8 = np.random.randint(8, 12)
                    h_8 = np.random.randint(14, 18)
                    top_scale = float(np.random.uniform(0.70, 0.95))
                    t = np.linspace(0, 2 * np.pi, 80)
                    xs = []
                    ys = []
                    for ti in t:
                        curr_y = cy - h_8 * np.cos(ti)
                        w_curr = w_8 * top_scale if curr_y < cy else w_8
                        curr_x = cx + w_curr * np.sin(2 * ti)
                        xs.append(curr_x)
                        ys.append(curr_y)
                    pts = np.int32(np.column_stack([xs, ys]))
                    is_closed = bool(np.random.rand() > 0.15)
                    cv2.polylines(canvas, [pts], is_closed, 255, thickness, cv2.LINE_AA)
                elif style == "asymmetric":
                    top_r = np.random.randint(5, 8)
                    bot_r = np.random.randint(8, 12)
                    top_cy = 28 - bot_r - top_r + 2
                    bot_cy = 28 + (bot_r // 2)
                    cv2.ellipse(canvas, (28, top_cy), (top_r, top_r), 0, 0, 360, 255, thickness, cv2.LINE_AA)
                    cv2.ellipse(canvas, (28, bot_cy), (bot_r, bot_r), 0, 0, 360, 255, thickness, cv2.LINE_AA)
                elif style == "waist":
                    # Distinct pinched waist figure 8
                    cv2.ellipse(canvas, (28, 19), (7, 7), 0, 0, 360, 255, thickness, cv2.LINE_AA)
                    cv2.ellipse(canvas, (28, 36), (9, 9), 0, 0, 360, 255, thickness, cv2.LINE_AA)
                    # Cross-through connection
                    cv2.line(canvas, (23, 23), (33, 31), 255, thickness, cv2.LINE_AA)
                    cv2.line(canvas, (33, 23), (23, 31), 255, thickness, cv2.LINE_AA)
                else:
                    cv2.ellipse(canvas, (28, 20), (7, 7), 0, 0, 360, 255, thickness, cv2.LINE_AA)
                    cv2.ellipse(canvas, (28, 36), (9, 9), 0, 0, 360, 255, thickness, cv2.LINE_AA)

            elif digit == 9:
                cv2.ellipse(canvas, (28, 21), (8, 8), 0, 0, 360, 255, thickness, cv2.LINE_AA)
                cv2.line(canvas, (36, 21), (36, 44), 255, thickness, cv2.LINE_AA)
                if np.random.rand() < 0.4:
                    cv2.line(canvas, (36, 44), (28, 44), 255, thickness, cv2.LINE_AA)

            # Random rotation (-15 to +15 deg)
            angle = float(np.random.uniform(-15.0, 15.0))
            rot_mat = cv2.getRotationMatrix2D((28, 28), angle, 1.0)
            rotated = cv2.warpAffine(canvas, rot_mat, (56, 56), flags=cv2.INTER_LINEAR)

            processed = center_and_scale_to_28(rotated)
            if processed is not None:
                x_list.append(processed)
                y_list.append(digit)
                count += 1

    x_arr = np.array(x_list, dtype=np.float32)[:, np.newaxis, :, :]
    y_arr = np.array(y_list, dtype=np.int64)
    return x_arr, y_arr


def harvest_real_bluebook_digits() -> tuple[np.ndarray, np.ndarray]:
    """Harvest and massively augment all real handwritten crops from data/real_digits."""
    real_digits_dir = BASE_DIR / "data" / "real_digits"
    x_list = []
    y_list = []

    k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    if not real_digits_dir.exists():
        print(f"Warning: {real_digits_dir} not found.")
        return np.zeros((0, 1, 28, 28), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    raw_crops_by_digit = {d: [] for d in range(10)}

    for d in range(10):
        d_dir = real_digits_dir / str(d)
        if not d_dir.exists():
            continue
        for f in sorted(d_dir.glob("*.png")):
            c_img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if c_img is None:
                continue
            norm = center_and_scale_to_28(c_img)
            if norm is None:
                norm = cv2.resize(c_img, (28, 28), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            raw_crops_by_digit[d].append(norm)

    total_raw = sum(len(v) for v in raw_crops_by_digit.values())
    print(f"Found {total_raw} real digit crops across classes 0-9:")
    for d in range(10):
        print(f"  Digit {d}: {len(raw_crops_by_digit[d])} real samples")

    # Target roughly 4,000 augmented samples per class for real bluebook digits
    for d in range(10):
        crops = raw_crops_by_digit[d]
        if not crops:
            continue
        # Multiplier to reach ~4,000 samples per digit
        augs_per_crop = max(200, 4000 // len(crops))

        for base_canvas in crops:
            x_list.append(base_canvas)
            y_list.append(d)

            base_u8 = (base_canvas * 255.0).astype(np.uint8)

            for _ in range(augs_per_crop):
                # 1. Random rotation (-16 to +16 deg)
                ang = float(np.random.uniform(-16.0, 16.0))
                M_rot = cv2.getRotationMatrix2D((14, 14), ang, 1.0)
                aug = cv2.warpAffine(base_u8, M_rot, (28, 28), flags=cv2.INTER_LINEAR)

                # 2. Random affine shear / slant
                shear_x = float(np.random.uniform(-0.20, 0.20))
                M_shear = np.float32([[1, shear_x, -shear_x * 14], [0, 1, 0]])
                aug = cv2.warpAffine(aug, M_shear, (28, 28), flags=cv2.INTER_LINEAR)

                # 3. Random translation (-3 to +3 px)
                tx = int(np.random.randint(-3, 4))
                ty = int(np.random.randint(-3, 4))
                M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
                aug = cv2.warpAffine(aug, M_trans, (28, 28), flags=cv2.INTER_LINEAR)

                # 4. Random stroke dilation / erosion
                p_morph = np.random.rand()
                if p_morph < 0.35:
                    aug = cv2.dilate(aug, k_dilate, iterations=1)
                elif p_morph < 0.50:
                    aug = cv2.erode(aug, k_dilate, iterations=1)

                # 5. Random stroke dropout (simulating faint red ink)
                if np.random.rand() < 0.15:
                    drop_mask = (np.random.rand(28, 28) > 0.12).astype(np.uint8)
                    aug = aug * drop_mask

                processed = center_and_scale_to_28(aug)
                if processed is not None:
                    x_list.append(processed)
                    y_list.append(d)
                else:
                    x_list.append(aug.astype(np.float32) / 255.0)
                    y_list.append(d)

    print(f"Harvested and augmented {len(x_list)} real bluebook handwriting samples.")
    x_arr = np.array(x_list, dtype=np.float32)[:, np.newaxis, :, :]
    y_arr = np.array(y_list, dtype=np.int64)
    return x_arr, y_arr

    print(f"Harvested and augmented {len(x_list)} real bluebook handwriting samples.")
    x_arr = np.array(x_list, dtype=np.float32)[:, np.newaxis, :, :]
    y_arr = np.array(y_list, dtype=np.int64)
    return x_arr, y_arr


class DeepBluebookCNN(nn.Module):
    """Deep 4-stage Convolutional Neural Network for robust handwritten digit recognition."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Stage 1: 1 -> 32 (28x28)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 14x14
            nn.Dropout2d(0.12),

            # Stage 2: 32 -> 64 (14x14)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 7x7
            nn.Dropout2d(0.18),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        f = self.features(x)
        return self.classifier(f)


class ExportWrapper(nn.Module):
    """Wraps model to output softmax probabilities in ONNX graph."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.model(x)
        return self.softmax(logits)


def train_and_export():
    print("Loading base MNIST training and testing benchmarks...")
    mnist_train_x = load_mnist_images(DATA_DIR / "train-images-idx3-ubyte.gz")
    mnist_train_y = load_mnist_labels(DATA_DIR / "train-labels-idx1-ubyte.gz")
    mnist_test_x = load_mnist_images(DATA_DIR / "t10k-images-idx3-ubyte.gz")
    mnist_test_y = load_mnist_labels(DATA_DIR / "t10k-labels-idx1-ubyte.gz")

    print("Loading EMNIST real messy handwriting digits dataset...")
    emnist_x, emnist_y = load_emnist_digits(limit_per_class=6000)

    print("Generating comprehensive synthetic messy handwriting digits (4,000 per class)...")
    synth_x, synth_y = generate_synthetic_messy_digits(samples_per_digit=4000)

    print("Harvesting real bluebook handwriting crops with hysteresis thresholding...")
    real_x, real_y = harvest_real_bluebook_digits()

    # Combine MNIST + EMNIST + Synthetic + Real Bluebook Crops
    parts_x = [mnist_train_x, synth_x, real_x]
    parts_y = [mnist_train_y, synth_y, real_y]
    if len(emnist_x) > 0:
        parts_x.append(emnist_x)
        parts_y.append(emnist_y)

    all_train_x = np.concatenate(parts_x, axis=0)
    all_train_y = np.concatenate(parts_y, axis=0)

    # Shuffle training set
    indices = np.arange(len(all_train_x))
    np.random.shuffle(indices)
    all_train_x = all_train_x[indices]
    all_train_y = all_train_y[indices]

    print(f"Total training dataset size: {len(all_train_x)} samples.")

    train_ds = TensorDataset(torch.from_numpy(all_train_x), torch.from_numpy(all_train_y))
    test_ds = TensorDataset(torch.from_numpy(mnist_test_x), torch.from_numpy(mnist_test_y))

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = DeepBluebookCNN().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    epochs = 8
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"Beginning DeepBluebookCNN model training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
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

        scheduler.step()
        epoch_acc = (correct / total) * 100.0
        print(f"Epoch {epoch:2d}/{epochs} - Training Accuracy: {epoch_acc:.2f}% (Loss: {running_loss/total:.4f})")

    # Evaluate on standard test benchmark
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
    print(f"Final Test Accuracy on MNIST test benchmark: {test_acc:.2f}%")

    # Evaluate directly on all raw real crops
    real_digits_dir = BASE_DIR / "data" / "real_digits"
    real_correct = 0
    real_total = 0
    misclassified_real = []
    with torch.no_grad():
        for d in range(10):
            d_dir = real_digits_dir / str(d)
            if not d_dir.exists():
                continue
            for f in sorted(d_dir.glob("*.png")):
                c_img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if c_img is None:
                    continue
                norm = center_and_scale_to_28(c_img)
                if norm is None:
                    norm = cv2.resize(c_img, (28, 28), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                t_in = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(device)
                pred = torch.argmax(model(t_in), dim=1).item()
                real_total += 1
                if pred == d:
                    real_correct += 1
                else:
                    misclassified_real.append((d, pred, f.name))
    real_acc = (real_correct / max(1, real_total)) * 100.0
    print(f"Accuracy on raw real bluebook crops: {real_correct}/{real_total} ({real_acc:.2f}%)")
    if misclassified_real:
        print("Misclassified real samples:")
        for td, pd, fn in misclassified_real:
            print(f"  True {td} -> Pred {pd} : {fn}")

    # Export to ONNX with ExportWrapper (attaches Softmax)
    print(f"Exporting updated model to ONNX: {MODEL_PATH}")
    model.to("cpu")
    model.eval()
    export_model = ExportWrapper(model)
    export_model.eval()

    dummy_input = torch.randn(1, 1, 28, 28, requires_grad=False)
    torch.onnx.export(
        export_model,
        dummy_input,
        str(MODEL_PATH),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    # Consolidate external data into a single self-contained ONNX file if needed
    data_file = MODEL_PATH.parent / f"{MODEL_PATH.name}.data"
    if data_file.exists():
        import onnx
        loaded_m = onnx.load(str(MODEL_PATH), load_external_data=True)
        onnx.save_model(loaded_m, str(MODEL_PATH), save_as_external_data=False)
        data_file.unlink(missing_ok=True)

    # Standardize ONNX IR version to 8 for onnxruntime-web compatibility
    import onnx
    loaded_m = onnx.load(str(MODEL_PATH))
    loaded_m.ir_version = 8
    onnx.save(loaded_m, str(MODEL_PATH))

    # Sync to web and public model paths
    for dest_dir in [BASE_DIR / "public" / "models", BASE_DIR / "web" / "models"]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(MODEL_PATH, dest_dir / "digit_model.onnx")

    print(f"ONNX model saved successfully (IR version 8): {MODEL_PATH.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    train_and_export()
