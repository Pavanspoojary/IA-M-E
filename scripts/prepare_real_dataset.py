"""Prepare and harvest labeled real digit crops from all 8 user evaluation sheets.
Strict Zero Emojis Policy Enforced.
"""

from pathlib import Path
import sys
import shutil
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.eval_sheet_extractor import EvalSheetExtractor
from scripts.grid_align_utils import get_snapped_grid, clean_cell_ink

OUTPUT_DIR = BASE_DIR / "data" / "real_digits"
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
for d in range(10):
    (OUTPUT_DIR / str(d)).mkdir(parents=True, exist_ok=True)

CUTS_DIR = BASE_DIR / "data" / "real_cuts"
if CUTS_DIR.exists():
    shutil.rmtree(CUTS_DIR)
CUTS_DIR.mkdir(parents=True, exist_ok=True)

extractor = EvalSheetExtractor()

# Ground truth mapping: (sheet_filename, IA_name, row_id) -> label
GROUND_TRUTH = {
    # Sheet 1: media_1794306996842_2913c1dd.jpg
    ("media_1794306996842_2913c1dd.jpg", "IA 1", "1a"): "3",
    ("media_1794306996842_2913c1dd.jpg", "IA 1", "1b"): "1",
    ("media_1794306996842_2913c1dd.jpg", "IA 1", "1c"): "-",
    ("media_1794306996842_2913c1dd.jpg", "IA 1", "2c"): "-",
    ("media_1794306996842_2913c1dd.jpg", "IA 1", "4a"): "1",
    ("media_1794306996842_2913c1dd.jpg", "IA 1", "4b"): "3",
    ("media_1794306996842_2913c1dd.jpg", "IA 1", "4c"): "4",
    ("media_1794306996842_2913c1dd.jpg", "IA 1", "total"): "12",
    ("media_1794306996842_2913c1dd.jpg", "IA 2", "1a"): "4",
    ("media_1794306996842_2913c1dd.jpg", "IA 2", "1b"): "5",
    ("media_1794306996842_2913c1dd.jpg", "IA 2", "1c"): "-",
    ("media_1794306996842_2913c1dd.jpg", "IA 2", "2c"): "-",
    ("media_1794306996842_2913c1dd.jpg", "IA 2", "3a"): "3",
    ("media_1794306996842_2913c1dd.jpg", "IA 2", "3b"): "7",
    ("media_1794306996842_2913c1dd.jpg", "IA 2", "3c"): "-",
    ("media_1794306996842_2913c1dd.jpg", "IA 2", "4c"): "-",
    ("media_1794306996842_2913c1dd.jpg", "IA 2", "total"): "19",

    # Sheet 2: media_1794307003666_298da81b.jpg
    ("media_1794307003666_298da81b.jpg", "IA 1", "1a"): "5",
    ("media_1794307003666_298da81b.jpg", "IA 1", "1b"): "3",
    ("media_1794307003666_298da81b.jpg", "IA 1", "1c"): "-",
    ("media_1794307003666_298da81b.jpg", "IA 1", "2c"): "-",
    ("media_1794307003666_298da81b.jpg", "IA 1", "4a"): "3",
    ("media_1794307003666_298da81b.jpg", "IA 1", "4b"): "4",
    ("media_1794307003666_298da81b.jpg", "IA 1", "4c"): "0",
    ("media_1794307003666_298da81b.jpg", "IA 1", "total"): "15",
    ("media_1794307003666_298da81b.jpg", "IA 2", "1a"): "7",
    ("media_1794307003666_298da81b.jpg", "IA 2", "1b"): "3",
    ("media_1794307003666_298da81b.jpg", "IA 2", "1c"): "-",
    ("media_1794307003666_298da81b.jpg", "IA 2", "2c"): "-",
    ("media_1794307003666_298da81b.jpg", "IA 2", "3a"): "5",
    ("media_1794307003666_298da81b.jpg", "IA 2", "3b"): "8",
    ("media_1794307003666_298da81b.jpg", "IA 2", "3c"): "-",
    ("media_1794307003666_298da81b.jpg", "IA 2", "4c"): "-",
    ("media_1794307003666_298da81b.jpg", "IA 2", "total"): "23",

    # Sheet 3: media_1794307009346_92303e9f.jpg
    ("media_1794307009346_92303e9f.jpg", "IA 1", "1a"): "5",
    ("media_1794307009346_92303e9f.jpg", "IA 1", "1b"): "1",
    ("media_1794307009346_92303e9f.jpg", "IA 1", "1c"): "-",
    ("media_1794307009346_92303e9f.jpg", "IA 1", "2c"): "-",
    ("media_1794307009346_92303e9f.jpg", "IA 1", "3b"): "1",
    ("media_1794307009346_92303e9f.jpg", "IA 1", "3c"): "6",
    ("media_1794307009346_92303e9f.jpg", "IA 1", "total"): "13",
    ("media_1794307009346_92303e9f.jpg", "IA 2", "1a"): "6",
    ("media_1794307009346_92303e9f.jpg", "IA 2", "1b"): "3",
    ("media_1794307009346_92303e9f.jpg", "IA 2", "1c"): "-",
    ("media_1794307009346_92303e9f.jpg", "IA 2", "2c"): "-",
    ("media_1794307009346_92303e9f.jpg", "IA 2", "3c"): "-",
    ("media_1794307009346_92303e9f.jpg", "IA 2", "4a"): "4",
    ("media_1794307009346_92303e9f.jpg", "IA 2", "4b"): "2",
    ("media_1794307009346_92303e9f.jpg", "IA 2", "4c"): "-",

    # Sheet 4: media_1794307014468_5c571ebd.jpg
    ("media_1794307014468_5c571ebd.jpg", "IA 1", "1a"): "5",
    ("media_1794307014468_5c571ebd.jpg", "IA 1", "1b"): "3",
    ("media_1794307014468_5c571ebd.jpg", "IA 1", "1c"): "-",
    ("media_1794307014468_5c571ebd.jpg", "IA 1", "2c"): "-",
    ("media_1794307014468_5c571ebd.jpg", "IA 1", "4a"): "3",
    ("media_1794307014468_5c571ebd.jpg", "IA 1", "4b"): "4",
    ("media_1794307014468_5c571ebd.jpg", "IA 1", "4c"): "6",
    ("media_1794307014468_5c571ebd.jpg", "IA 1", "total"): "21",
    ("media_1794307014468_5c571ebd.jpg", "IA 2", "1a"): "8",
    ("media_1794307014468_5c571ebd.jpg", "IA 2", "1b"): "4",
    ("media_1794307014468_5c571ebd.jpg", "IA 2", "1c"): "-",
    ("media_1794307014468_5c571ebd.jpg", "IA 2", "2c"): "-",
    ("media_1794307014468_5c571ebd.jpg", "IA 2", "3a"): "5",
    ("media_1794307014468_5c571ebd.jpg", "IA 2", "3b"): "7",
    ("media_1794307014468_5c571ebd.jpg", "IA 2", "3c"): "-",
    ("media_1794307014468_5c571ebd.jpg", "IA 2", "4c"): "-",

    # Sheet 5: media_1794307019700_d8f47c5f.jpg
    ("media_1794307019700_d8f47c5f.jpg", "IA 1", "1a"): "4",
    ("media_1794307019700_d8f47c5f.jpg", "IA 1", "1b"): "2",
    ("media_1794307019700_d8f47c5f.jpg", "IA 1", "1c"): "-",
    ("media_1794307019700_d8f47c5f.jpg", "IA 1", "2c"): "-",
    ("media_1794307019700_d8f47c5f.jpg", "IA 1", "4a"): "2",
    ("media_1794307019700_d8f47c5f.jpg", "IA 1", "4b"): "4",
    ("media_1794307019700_d8f47c5f.jpg", "IA 1", "4c"): "4",
    ("media_1794307019700_d8f47c5f.jpg", "IA 1", "total"): "16",
    ("media_1794307019700_d8f47c5f.jpg", "IA 2", "1a"): "8",
    ("media_1794307019700_d8f47c5f.jpg", "IA 2", "1b"): "3",
    ("media_1794307019700_d8f47c5f.jpg", "IA 2", "1c"): "-",
    ("media_1794307019700_d8f47c5f.jpg", "IA 2", "2c"): "-",
    ("media_1794307019700_d8f47c5f.jpg", "IA 2", "3a"): "3",
    ("media_1794307019700_d8f47c5f.jpg", "IA 2", "3b"): "6",
    ("media_1794307019700_d8f47c5f.jpg", "IA 2", "3c"): "-",
    ("media_1794307019700_d8f47c5f.jpg", "IA 2", "4c"): "-",

    # Sheet 6: media_1794307025472_2477aa51.jpg
    ("media_1794307025472_2477aa51.jpg", "IA 1", "1a"): "3",
    ("media_1794307025472_2477aa51.jpg", "IA 1", "1c"): "-",
    ("media_1794307025472_2477aa51.jpg", "IA 1", "2c"): "-",
    ("media_1794307025472_2477aa51.jpg", "IA 1", "total"): "3",
    ("media_1794307025472_2477aa51.jpg", "IA 2", "1a"): "4",
    ("media_1794307025472_2477aa51.jpg", "IA 2", "1b"): "2",
    ("media_1794307025472_2477aa51.jpg", "IA 2", "1c"): "-",
    ("media_1794307025472_2477aa51.jpg", "IA 2", "2c"): "-",
    ("media_1794307025472_2477aa51.jpg", "IA 2", "3c"): "-",
    ("media_1794307025472_2477aa51.jpg", "IA 2", "4a"): "3",
    ("media_1794307025472_2477aa51.jpg", "IA 2", "4b"): "5",
    ("media_1794307025472_2477aa51.jpg", "IA 2", "4c"): "-",

    # Sheet 7: media_1794307033634_b4069efa.jpg
    ("media_1794307033634_b4069efa.jpg", "IA 1", "1c"): "-",
    ("media_1794307033634_b4069efa.jpg", "IA 1", "2a"): "1",
    ("media_1794307033634_b4069efa.jpg", "IA 1", "2b"): "3",
    ("media_1794307033634_b4069efa.jpg", "IA 1", "2c"): "-",
    ("media_1794307033634_b4069efa.jpg", "IA 1", "4a"): "2",
    ("media_1794307033634_b4069efa.jpg", "IA 1", "4b"): "3",
    ("media_1794307033634_b4069efa.jpg", "IA 1", "total"): "9",
    ("media_1794307033634_b4069efa.jpg", "IA 2", "1a"): "6",
    ("media_1794307033634_b4069efa.jpg", "IA 2", "1b"): "4",
    ("media_1794307033634_b4069efa.jpg", "IA 2", "1c"): "-",
    ("media_1794307033634_b4069efa.jpg", "IA 2", "2c"): "-",
    ("media_1794307033634_b4069efa.jpg", "IA 2", "3a"): "4",
    ("media_1794307033634_b4069efa.jpg", "IA 2", "3b"): "5",
    ("media_1794307033634_b4069efa.jpg", "IA 2", "3c"): "-",
    ("media_1794307033634_b4069efa.jpg", "IA 2", "4c"): "-",

    # Sheet 8: media_1794307057019_57e455aa.jpg
    ("media_1794307057019_57e455aa.jpg", "IA 1", "1a"): "6",
    ("media_1794307057019_57e455aa.jpg", "IA 1", "1b"): "4",
    ("media_1794307057019_57e455aa.jpg", "IA 1", "1c"): "-",
    ("media_1794307057019_57e455aa.jpg", "IA 1", "2c"): "-",
    ("media_1794307057019_57e455aa.jpg", "IA 1", "4a"): "4",
    ("media_1794307057019_57e455aa.jpg", "IA 1", "4b"): "4",
    ("media_1794307057019_57e455aa.jpg", "IA 1", "4c"): "6",
    ("media_1794307057019_57e455aa.jpg", "IA 1", "total"): "24",
    ("media_1794307057019_57e455aa.jpg", "IA 2", "1a"): "8",
    ("media_1794307057019_57e455aa.jpg", "IA 2", "1b"): "3",
    ("media_1794307057019_57e455aa.jpg", "IA 2", "1c"): "-",
    ("media_1794307057019_57e455aa.jpg", "IA 2", "2c"): "-",
    ("media_1794307057019_57e455aa.jpg", "IA 2", "3c"): "-",
    ("media_1794307057019_57e455aa.jpg", "IA 2", "4a"): "5",
    ("media_1794307057019_57e455aa.jpg", "IA 2", "4b"): "8",
    ("media_1794307057019_57e455aa.jpg", "IA 2", "4c"): "-",
}

ROWS_GEOM = [
    ("1a", 96, 142), ("1b", 142, 189), ("1c", 189, 236),
    ("2a", 236, 283), ("2b", 283, 331), ("2c", 331, 378),
    ("3a", 378, 425), ("3b", 425, 472), ("3c", 472, 520),
    ("4a", 520, 567), ("4b", 567, 614), ("4c", 614, 662),
    ("total", 662, 710)
]

COLS_GEOM = {
    "IA 1": (95, 198),
    "IA 2": (295, 399),
    "IA 3": (497, 599)
}

def extract_adaptive_ink(cell_bgr):
    h, w, _ = cell_bgr.shape
    inner = cell_bgr[3:h-3, 3:w-3]
    if inner.size == 0:
        return np.zeros((h, w), dtype=np.uint8)

    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    bg_lum = float(np.median(gray))

    b = inner[:, :, 0].astype(float)
    g = inner[:, :, 1].astype(float)
    r = inner[:, :, 2].astype(float)
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    contrast = bg_lum - lum
    diff = r - np.maximum(g, b)
    red_bias = r - b

    is_red = (diff >= 7) | ((contrast >= 8) & (red_bias >= 2) & (r >= g - 3))
    is_dark = (contrast >= 22) & (lum < 140)

    ink_inner = is_red | is_dark

    kernel = np.ones((2, 2), np.uint8)
    ink_cleaned = cv2.morphologyEx(ink_inner.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[3:h-3, 3:w-3] = ink_cleaned * 255
    return mask

def normalize_to_28(binary_crop):
    coords = cv2.findNonZero(binary_crop)
    if coords is None:
        return None
    bx, by, bw, bh = cv2.boundingRect(coords)
    if bw <= 0 or bh <= 0:
        return None
    tight = binary_crop[by:by+bh, bx:bx+bw]

    scale = 20.0 / max(bw, bh)
    new_w = max(1, int(round(bw * scale)))
    new_h = max(1, int(round(bh * scale)))
    resized = cv2.resize(tight, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    px = (28 - new_w) // 2
    py = (28 - new_h) // 2
    canvas[py:py+new_h, px:px+new_w] = resized
    return canvas

counts = {d: 0 for d in range(10)}
cut_count = 0

# Cache rectified sheets and their snapped grids
rectified_cache = {}
grid_cache = {}

row_ids = ["1a", "1b", "1c", "2a", "2b", "2c", "3a", "3b", "3c", "4a", "4b", "4c", "total"]

for (fn, ia, qid), target in GROUND_TRUTH.items():
    if fn not in rectified_cache:
        img_p = BASE_DIR / "sample_data" / "user_sheets" / fn
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        rectified = extractor._detect_distribution_of_marks(img)
        rectified_cache[fn] = rectified
        if rectified is not None:
            grid_cache[fn] = get_snapped_grid(rectified)

    rectified = rectified_cache.get(fn)
    if rectified is None or fn not in grid_cache:
        continue

    rows, cols = grid_cache[fn]
    r_idx = row_ids.index(qid)
    y1, y2 = rows[r_idx], rows[r_idx + 1]
    x1, x2 = cols[ia]
    cell = rectified[y1:y2, x1:x2]
    mask = clean_cell_ink(cell)

    if target == "-":
        # Cut mark
        coords = cv2.findNonZero(mask)
        if coords is not None:
            bx, by, bw, bh = cv2.boundingRect(coords)
            tight = mask[by:by+bh, bx:bx+bw]
            cut_count += 1
            cv2.imwrite(str(CUTS_DIR / f"cut_{fn.split('.')[0]}_{ia.replace(' ', '_')}_{qid}.png"), tight)
        continue

    # Numeric label: single or multi-digit
    if len(target) == 1:
        d = int(target)
        norm = normalize_to_28(mask)
        if norm is not None:
            counts[d] += 1
            cv2.imwrite(str(OUTPUT_DIR / str(d) / f"digit_{d}_{fn.split('.')[0]}_{ia.replace(' ', '_')}_{qid}.png"), norm)
    elif len(target) == 2:
        # Multi-digit total e.g. "12", "19", "23", "24"
        d1 = int(target[0])
        d2 = int(target[1])

        coords = cv2.findNonZero(mask)
        if coords is not None:
            bx, by, bw, bh = cv2.boundingRect(coords)
            crop = mask[by:by+bh, bx:bx+bw]

            # Split near center
            col_hist = np.sum(crop > 0, axis=0)
            mid = bw // 2
            split_x = mid
            min_c = 999999
            for x in range(int(bw * 0.30), int(bw * 0.70)):
                score = col_hist[x] + abs(x - mid) * 0.4
                if score < min_c:
                    min_c = score
                    split_x = x

            left_part = crop[:, :split_x]
            right_part = crop[:, split_x:]

            n1 = normalize_to_28(left_part)
            n2 = normalize_to_28(right_part)
            if n1 is not None:
                counts[d1] += 1
                cv2.imwrite(str(OUTPUT_DIR / str(d1) / f"digit_{d1}_tot_left_{fn.split('.')[0]}_{ia.replace(' ', '_')}_{qid}.png"), n1)
            if n2 is not None:
                counts[d2] += 1
                cv2.imwrite(str(OUTPUT_DIR / str(d2) / f"digit_{d2}_tot_right_{fn.split('.')[0]}_{ia.replace(' ', '_')}_{qid}.png"), n2)

print("Harvested real digits summary:")
for d in range(10):
    print(f"  Digit {d}: {counts[d]} real samples")
print(f"  Cut marks: {cut_count} real samples")
