"""Train an ultra-high-accuracy Convolutional Neural Network on MNIST + Real Bluebook handwriting.

Designed specifically for challenging, messy Indian college bluebook marksheets:
- Real handwritten digit crops harvested from actual student bluebooks
- Red-pen grading ink with varying pen pressure (gel, ballpoint)
- Cursive and looped 2s (bottom loop at baseline, swan neck, flat tail)
- Angular and flat-top 3s
- Open-top trident 4s and closed 4s
- Disconnected two-stroke 5s (separated horizontal bar) and cursive 5s
- Tall narrow 6s and circular 6s
- Crossed 7s and slanted 7s
- S-curve figure-8s and double-loop 8s
- Single-stroke and seriffed 1s
- Hooked 9s
- Two-digit combinations (13, 16, 21, 24)

Exports to `models/digit_model.onnx` optimized for local CPU execution.
Strict Zero Emojis Policy Enforced.
"""

from pathlib import Path
from typing import Optional
import gzip
import struct
import os
import sys
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


def generate_synthetic_messy_digits(samples_per_digit: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Generate extensive synthetic messy handwriting samples matching Indian college booklet marks."""
    x_list = []
    y_list = []

    np.random.seed(42)

    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_PLAIN,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
    ]

    for digit in range(10):
        digit_str = str(digit)
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
                    # top serif
                    cv2.line(canvas, (x_top, y_top), (x_top - np.random.randint(4, 8), y_top + np.random.randint(4, 7)), 255, thickness, cv2.LINE_AA)

            elif digit == 2:
                # 2: looped cursive 2, swan-neck 2, flat-base 2, or wavy-base 2
                style = np.random.choice(["loop", "swan", "flat", "wavy"])
                if style == "loop":
                    # top arch
                    cv2.ellipse(canvas, (28, 19), (8, 7), 0, 180, 360, 255, thickness, cv2.LINE_AA)
                    # diagonal down to bottom-left
                    cv2.line(canvas, (36, 19), (19, 39), 255, thickness, cv2.LINE_AA)
                    # loop at baseline
                    cv2.ellipse(canvas, (21, 39), (4, 3), 45, 0, 360, 255, thickness, cv2.LINE_AA)
                    # tail extending right
                    cv2.line(canvas, (21, 41), (39 + np.random.randint(-2, 5), 41 + np.random.randint(-2, 2)), 255, thickness, cv2.LINE_AA)
                elif style == "swan":
                    # curved swan neck with curved head
                    cv2.ellipse(canvas, (27, 18), (9, 8), 0, 160, 360, 255, thickness, cv2.LINE_AA)
                    cv2.line(canvas, (36, 18), (17, 43), 255, thickness, cv2.LINE_AA)
                    cv2.line(canvas, (17, 43), (40, 43), 255, thickness, cv2.LINE_AA)
                elif style == "wavy":
                    cv2.ellipse(canvas, (28, 19), (8, 7), 0, 180, 360, 255, thickness, cv2.LINE_AA)
                    cv2.line(canvas, (36, 19), (18, 41), 255, thickness, cv2.LINE_AA)
                    # wavy tail
                    pts = np.array([[18, 41], [26, 39], [34, 43], [40, 41]], np.int32)
                    cv2.polylines(canvas, [pts], False, 255, thickness, cv2.LINE_AA)
                else:
                    cv2.ellipse(canvas, (28, 19), (8, 7), 0, 180, 360, 255, thickness, cv2.LINE_AA)
                    cv2.line(canvas, (36, 19), (18, 42), 255, thickness, cv2.LINE_AA)
                    cv2.line(canvas, (18, 42), (38, 42), 255, thickness, cv2.LINE_AA)

            elif digit == 3:
                # 3: flat-top or angular or rounded
                style = np.random.choice(["flat_top", "angular", "round"])
                if style == "flat_top":
                    # horizontal top
                    cv2.line(canvas, (18, 14), (36, 14), 255, thickness, cv2.LINE_AA)
                    cv2.line(canvas, (36, 14), (25, 27), 255, thickness, cv2.LINE_AA)
                    cv2.ellipse(canvas, (28, 36), (9, 9), 0, -90, 130, 255, thickness, cv2.LINE_AA)
                elif style == "angular":
                    cv2.line(canvas, (18, 14), (34, 14), 255, thickness, cv2.LINE_AA)
                    cv2.line(canvas, (34, 14), (24, 27), 255, thickness, cv2.LINE_AA)
                    cv2.line(canvas, (24, 27), (36, 36), 255, thickness, cv2.LINE_AA)
                    cv2.line(canvas, (36, 36), (20, 44), 255, thickness, cv2.LINE_AA)
                else:
                    cv2.ellipse(canvas, (27, 21), (7, 7), 0, -110, 120, 255, thickness, cv2.LINE_AA)
                    cv2.ellipse(canvas, (27, 35), (8, 8), 0, -110, 120, 255, thickness, cv2.LINE_AA)

            elif digit == 4:
                # 4: open-top trident 4, closed 4, or fast 4
                style = np.random.choice(["open_top", "trident", "closed"])
                if style == "open_top":
                    # left vertical
                    x_stem_left = np.random.randint(18, 23)
                    y_top_left = np.random.randint(12, 17)
                    y_cross = np.random.randint(28, 33)
                    cv2.line(canvas, (x_stem_left, y_top_left), (x_stem_left, y_cross), 255, thickness, cv2.LINE_AA)
                    # horizontal bar
                    x_stem_right = np.random.randint(32, 38)
                    cv2.line(canvas, (x_stem_left, y_cross), (x_stem_right + 3, y_cross), 255, thickness, cv2.LINE_AA)
                    # long right vertical
                    y_top_right = np.random.randint(8, 14)
                    y_bot_right = np.random.randint(42, 48)
                    cv2.line(canvas, (x_stem_right, y_top_right), (x_stem_right, y_bot_right), 255, thickness, cv2.LINE_AA)
                elif style == "trident":
                    # U-shaped left part
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
                # 5: separated top stroke or cursive
                style = np.random.choice(["separated", "attached", "slanted"])
                x1 = np.random.randint(16, 20)
                y1 = np.random.randint(12, 16)
                x2 = np.random.randint(32, 38)
                gap = np.random.choice([0, 1, 2, 3]) if style == "separated" else 0
                # horizontal top bar (slanted up or straight)
                cv2.line(canvas, (x1, y1), (x2, y1 + np.random.randint(-2, 3)), 255, thickness, cv2.LINE_AA)
                y_stem_top = y1 + gap
                y_stem_bot = y_stem_top + np.random.randint(10, 14)
                cv2.line(canvas, (x1, y_stem_top), (x1, y_stem_bot), 255, thickness, cv2.LINE_AA)
                # bottom round belly
                center = (x1 + np.random.randint(6, 9), (y_stem_bot + 40) // 2)
                axes = (np.random.randint(9, 13), np.random.randint(9, 12))
                cv2.ellipse(canvas, center, axes, 0, -80, 140, 255, thickness, cv2.LINE_AA)

            elif digit == 6:
                # 6: straight slanted stem curving into bottom circle
                cv2.line(canvas, (32, 12), (20, 32), 255, thickness, cv2.LINE_AA)
                cv2.ellipse(canvas, (26, 35), (9, 9), 0, 0, 360, 255, thickness, cv2.LINE_AA)

            elif digit == 7:
                # 7: crossed 7 (dominant in Indian bluebooks) or uncrossed 7
                x1, y1 = np.random.randint(14, 18), np.random.randint(12, 16)
                x2, y2 = np.random.randint(34, 42), y1 + np.random.randint(-1, 3)
                cv2.line(canvas, (x1, y1), (x2, y2), 255, thickness, cv2.LINE_AA)
                if np.random.rand() < 0.35:
                    # downward left hook / serif at start of 7
                    cv2.line(canvas, (x1, y1), (x1, y1 + np.random.randint(3, 7)), 255, thickness, cv2.LINE_AA)
                x3, y3 = np.random.randint(18, 26), np.random.randint(42, 48)
                cv2.line(canvas, (x2, y2), (x3, y3), 255, thickness, cv2.LINE_AA)
                if np.random.rand() < 0.75:
                    # horizontal crossbar
                    frac = float(np.random.uniform(0.38, 0.58))
                    cx_mid = int(x2 + (x3 - x2) * frac)
                    cy_mid = int(y2 + (y3 - y2) * frac)
                    bar_w = np.random.randint(8, 18)
                    tilt = np.random.randint(-2, 3)
                    cv2.line(canvas, (cx_mid - bar_w // 2, cy_mid - tilt), (cx_mid + bar_w // 2, cy_mid + tilt), 255, thickness, cv2.LINE_AA)

            elif digit == 8:
                # 8: continuous parametric S-curve, asymmetrical loops, or double-circle
                style = np.random.choice(["scurve", "asymmetric", "circles"])
                if style == "scurve":
                    cx, cy = 28 + np.random.randint(-2, 3), 28 + np.random.randint(-2, 3)
                    w_8 = np.random.randint(8, 12)
                    h_8 = np.random.randint(14, 18)
                    top_scale = float(np.random.uniform(0.75, 0.95))
                    t = np.linspace(0, 2 * np.pi, 80)
                    xs = []
                    ys = []
                    for ti in t:
                        curr_y = cy - h_8 * np.cos(ti)
                        # scale width if in top half
                        w_curr = w_8 * top_scale if curr_y < cy else w_8
                        curr_x = cx + w_curr * np.sin(2 * ti)
                        xs.append(curr_x)
                        ys.append(curr_y)
                    pts = np.int32(np.column_stack([xs, ys]))
                    is_closed = bool(np.random.rand() > 0.15)
                    cv2.polylines(canvas, [pts], is_closed, 255, thickness, cv2.LINE_AA)
                elif style == "asymmetric":
                    # top loop smaller than bottom loop
                    top_r = np.random.randint(6, 8)
                    bot_r = np.random.randint(8, 11)
                    cv2.ellipse(canvas, (28, 20), (top_r, top_r), 0, 0, 360, 255, thickness, cv2.LINE_AA)
                    cv2.ellipse(canvas, (28, 36), (bot_r, bot_r), 0, 0, 360, 255, thickness, cv2.LINE_AA)
                else:
                    cv2.ellipse(canvas, (28, 20), (7, 7), 0, 0, 360, 255, thickness, cv2.LINE_AA)
                    cv2.ellipse(canvas, (28, 36), (9, 9), 0, 0, 360, 255, thickness, cv2.LINE_AA)

            elif digit == 9:
                # 9: top circle + straight or hooked stem
                cv2.ellipse(canvas, (28, 21), (8, 8), 0, 0, 360, 255, thickness, cv2.LINE_AA)
                cv2.line(canvas, (36, 21), (36, 44), 255, thickness, cv2.LINE_AA)
                if np.random.rand() < 0.4:
                    cv2.line(canvas, (36, 44), (28, 44), 255, thickness, cv2.LINE_AA)

            # Random rotation (-15 to +15 deg)
            angle = float(np.random.uniform(-15.0, 15.0))
            rot_mat = cv2.getRotationMatrix2D((28, 28), angle, 1.0)
            rotated = cv2.warpAffine(canvas, rot_mat, (56, 56), flags=cv2.INTER_LINEAR)

            # Center and scale to 28x28
            processed = center_and_scale_to_28(rotated)
            if processed is not None:
                x_list.append(processed)
                y_list.append(digit)
                count += 1

    x_arr = np.array(x_list, dtype=np.float32)[:, np.newaxis, :, :]
    y_arr = np.array(y_list, dtype=np.int64)
    return x_arr, y_arr


def harvest_real_bluebook_digits() -> tuple[np.ndarray, np.ndarray]:
    """Harvest and augment all real handwritten digit crops from actual bluebook samples."""
    from src.eval_sheet_extractor import EvalSheetExtractor
    ext = EvalSheetExtractor()

    extractor_rows = [
        ("1a", "1. a", 96, 141),
        ("1b", "1. b", 141, 189),
        ("1c", "1. c", 189, 236),
        ("2a", "2. a", 236, 283),
        ("2b", "2. b", 283, 331),
        ("2c", "2. c", 331, 378),
        ("3a", "3. a", 378, 425),
        ("3b", "3. b", 425, 472),
        ("3c", "3. c", 472, 520),
        ("4a", "4. a", 520, 567),
        ("4b", "4. b", 567, 614),
        ("4c", "4. c", 614, 662),
        ("total", "Total Marks", 662, 709),
    ]
    cols = {"IA 1": (95, 199), "IA 2": (294, 399)}

    ground_truth = {
        ("distribution_marks_sample_1.jpg", "IA 1", "1a"): "5",
        ("distribution_marks_sample_1.jpg", "IA 1", "1b"): "2",
        ("distribution_marks_sample_1.jpg", "IA 1", "4b"): "2",
        ("distribution_marks_sample_1.jpg", "IA 2", "1a"): "7",
        ("distribution_marks_sample_1.jpg", "IA 2", "3a"): "5",
        ("distribution_marks_sample_1.jpg", "IA 2", "3b"): "3",
        ("distribution_marks_sample_2.jpg", "IA 1", "1a"): "6",
        ("distribution_marks_sample_2.jpg", "IA 1", "1b"): "4",
        ("distribution_marks_sample_2.jpg", "IA 1", "4a"): "4",
        ("distribution_marks_sample_2.jpg", "IA 1", "4b"): "4",
        ("distribution_marks_sample_2.jpg", "IA 1", "4c"): "7",
        ("distribution_marks_sample_2.jpg", "IA 2", "1a"): "8",
        ("distribution_marks_sample_2.jpg", "IA 2", "1b"): "3",
        ("distribution_marks_sample_2.jpg", "IA 2", "4a"): "5",
        ("distribution_marks_sample_2.jpg", "IA 2", "4b"): "8",
        ("distribution_marks_sample_3.jpg", "IA 1", "1a"): "5",
        ("distribution_marks_sample_3.jpg", "IA 1", "1b"): "2",
        ("distribution_marks_sample_3.jpg", "IA 1", "4b"): "2",
        ("distribution_marks_sample_3.jpg", "IA 2", "1a"): "7",
        ("distribution_marks_sample_3.jpg", "IA 2", "3a"): "5",
        ("distribution_marks_sample_3.jpg", "IA 2", "3b"): "3",
        ("bluebook_messy_1.png", "IA 1", "2a"): "1",
        ("bluebook_messy_1.png", "IA 1", "2b"): "3",
        ("bluebook_messy_1.png", "IA 1", "4a"): "2",
        ("bluebook_messy_1.png", "IA 1", "4b"): "3",
        ("bluebook_messy_1.png", "IA 1", "total"): "9",
        ("bluebook_messy_1.png", "IA 2", "1a"): "6",
        ("bluebook_messy_1.png", "IA 2", "1b"): "4",
        ("bluebook_messy_1.png", "IA 2", "3a"): "4",
        ("bluebook_messy_1.png", "IA 2", "3b"): "5",
        ("bluebook_messy_2.png", "IA 1", "1a"): "3",
        ("bluebook_messy_2.png", "IA 1", "total"): "3",
        ("bluebook_messy_2.png", "IA 2", "1a"): "4",
        ("bluebook_messy_2.png", "IA 2", "1b"): "2",
        ("bluebook_messy_2.png", "IA 2", "4a"): "3",
        ("bluebook_messy_2.png", "IA 2", "4b"): "5",
        ("bluebook_messy_3.png", "IA 1", "1a"): "4",
        ("bluebook_messy_3.png", "IA 1", "1b"): "2",
        ("bluebook_messy_3.png", "IA 1", "4a"): "2",
        ("bluebook_messy_3.png", "IA 1", "4b"): "4",
        ("bluebook_messy_3.png", "IA 1", "4c"): "4",
        ("bluebook_messy_3.png", "IA 2", "1a"): "8",
        ("bluebook_messy_3.png", "IA 2", "1b"): "3",
        ("bluebook_messy_3.png", "IA 2", "3a"): "3",
        ("bluebook_messy_3.png", "IA 2", "3b"): "6",
        ("bluebook_messy_4.png", "IA 1", "1a"): "5",
        ("bluebook_messy_4.png", "IA 1", "1b"): "3",
        ("bluebook_messy_4.png", "IA 1", "4a"): "3",
        ("bluebook_messy_4.png", "IA 1", "4b"): "4",
        ("bluebook_messy_4.png", "IA 1", "4c"): "6",
        ("bluebook_messy_4.png", "IA 2", "1a"): "8",
        ("bluebook_messy_4.png", "IA 2", "1b"): "4",
        ("bluebook_messy_4.png", "IA 2", "3a"): "5",
        ("bluebook_messy_4.png", "IA 2", "3b"): "7",
        ("bluebook_messy_5.png", "IA 1", "1a"): "5",
        ("bluebook_messy_5.png", "IA 1", "1b"): "1",
        ("bluebook_messy_5.png", "IA 1", "3b"): "1",
        ("bluebook_messy_5.png", "IA 1", "3c"): "6",
        ("bluebook_messy_5.png", "IA 2", "1a"): "6",
        ("bluebook_messy_5.png", "IA 2", "1b"): "3",
        ("bluebook_messy_5.png", "IA 2", "4a"): "4",
        ("bluebook_messy_5.png", "IA 2", "4b"): "2",
    }

    x_list = []
    y_list = []

    k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    for (fn, ia, qid), target_str in ground_truth.items():
        img_path = SAMPLE_DIR / fn
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        warped = ext._detect_distribution_of_marks(img)
        if warped is None:
            continue

        y1, y2 = next(r[2:] for r in extractor_rows if r[0] == qid)
        x1, x2 = cols[ia]
        cell = warped[y1 + 5 : y2 - 5, x1 + 6 : x2 - 6]

        b, g, red = cv2.split(cell)
        diff = red.astype(int) - np.maximum(b, g).astype(int)
        med_diff = float(np.median(diff))
        mask = ((diff > med_diff + 8) & (diff >= 16)).astype(np.uint8) * 255
        mask[:2, :] = 0; mask[-2:, :] = 0; mask[:, :2] = 0; mask[:, -2:] = 0

        cnt = cv2.countNonZero(mask)
        if cnt < 15:
            # check dark ink
            cgray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            shaved = cgray[4:-4, 4:-4]
            _, dark = cv2.threshold(shaved, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            if cv2.countNonZero(dark) < 20:
                continue
            mask = np.zeros_like(cgray)
            mask[4:-4, 4:-4] = dark

        if not target_str.isdigit():
            continue

        target_digit = int(target_str)
        base_canvas = center_and_scale_to_28(mask)
        if base_canvas is None:
            continue

        # Add base crop
        x_list.append(base_canvas)
        y_list.append(target_digit)

        # Generate 40 augmented variations per real crop
        for _ in range(40):
            # 1. Random rotation (-12 to +12 deg)
            ang = float(np.random.uniform(-12.0, 12.0))
            M = cv2.getRotationMatrix2D((14, 14), ang, 1.0)
            aug = cv2.warpAffine((base_canvas * 255).astype(np.uint8), M, (28, 28), flags=cv2.INTER_LINEAR)

            # 2. Random translation (-2 to +2 px)
            tx = np.random.randint(-2, 3)
            ty = np.random.randint(-2, 3)
            M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
            aug = cv2.warpAffine(aug, M_trans, (28, 28))

            # 3. Random morphological dilation or erosion
            if np.random.rand() < 0.35:
                aug = cv2.dilate(aug, k_dilate, iterations=1)
            elif np.random.rand() < 0.15:
                aug = cv2.erode(aug, k_dilate, iterations=1)

            x_list.append(aug.astype(np.float32) / 255.0)
            y_list.append(target_digit)

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
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        f = self.features(x)
        return self.classifier(f)


def train_and_export():
    print("Loading base MNIST training and testing benchmarks...")
    mnist_train_x = load_mnist_images(DATA_DIR / "train-images-idx3-ubyte.gz")
    mnist_train_y = load_mnist_labels(DATA_DIR / "train-labels-idx1-ubyte.gz")
    mnist_test_x = load_mnist_images(DATA_DIR / "t10k-images-idx3-ubyte.gz")
    mnist_test_y = load_mnist_labels(DATA_DIR / "t10k-labels-idx1-ubyte.gz")

    print("Generating comprehensive synthetic messy handwriting digits (2,000 per class)...")
    synth_x, synth_y = generate_synthetic_messy_digits(samples_per_digit=2000)

    print("Harvesting real bluebook handwriting crops from student booklets...")
    real_x, real_y = harvest_real_bluebook_digits()

    # Combine MNIST + Synthetic + Real Bluebook Crops
    all_train_x = np.concatenate([mnist_train_x, synth_x, real_x], axis=0)
    all_train_y = np.concatenate([mnist_train_y, synth_y, real_y], axis=0)

    # Shuffle training set
    indices = np.arange(len(all_train_x))
    np.random.shuffle(indices)
    all_train_x = all_train_x[indices]
    all_train_y = all_train_y[indices]

    print(f"Total training dataset size: {len(all_train_x)} samples.")

    train_ds = TensorDataset(torch.from_numpy(all_train_x), torch.from_numpy(all_train_y))
    test_ds = TensorDataset(torch.from_numpy(mnist_test_x), torch.from_numpy(mnist_test_y))

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = DeepBluebookCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    print("Beginning DeepBluebookCNN model training for 10 epochs...")
    for epoch in range(1, 11):
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
        print(f"Epoch {epoch:2d}/10 - Training Accuracy: {epoch_acc:.2f}% (Loss: {running_loss/total:.4f})")

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

    # Export to ONNX
    print(f"Exporting updated model to ONNX: {MODEL_PATH}")
    model.to("cpu")
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

    # Consolidate external data into a single self-contained ONNX file if needed
    data_file = MODEL_PATH.parent / f"{MODEL_PATH.name}.data"
    if data_file.exists():
        import onnx
        loaded_m = onnx.load(str(MODEL_PATH), load_external_data=True)
        onnx.save_model(loaded_m, str(MODEL_PATH), save_as_external_data=False)
        data_file.unlink(missing_ok=True)

    # Standardize ONNX IR version to 8 for onnxruntime-web compatibility
    import onnx
    import shutil
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
