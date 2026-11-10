"""Test improved ink extraction across all cells of all 8 user sheets.
Strict Zero Emojis Policy Enforced.
"""

from pathlib import Path
import sys
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.eval_sheet_extractor import EvalSheetExtractor

extractor = EvalSheetExtractor()

ROWS = [
    ("1a", 96, 142), ("1b", 142, 189), ("1c", 189, 236),
    ("2a", 236, 283), ("2b", 283, 331), ("2c", 331, 378),
    ("3a", 378, 425), ("3b", 425, 472), ("3c", 472, 520),
    ("4a", 520, 567), ("4b", 567, 614), ("4c", 614, 662),
    ("total", 662, 710)
]

COLS = {
    "IA 1": (95, 198),
    "IA 2": (295, 399),
    "IA 3": (497, 599)
}

def extract_adaptive_ink_bfs(cell_bgr):
    h, w, _ = cell_bgr.shape
    inner = cell_bgr[3:h-3, 3:w-3]
    if inner.size == 0:
        return 0, np.zeros((h, w), dtype=np.uint8)

    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    bg_lum = float(np.median(gray))

    b = inner[:, :, 0].astype(float)
    g = inner[:, :, 1].astype(float)
    r = inner[:, :, 2].astype(float)
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    contrast = bg_lum - lum
    diff = r - np.maximum(g, b)
    red_bias = r - b

    # Median diff on inner
    bg_diff = float(np.median(diff))
    high_th = max(8.0, bg_diff + 6.0)
    low_th = max(3.0, bg_diff + 2.5)

    is_strong = (
        (diff >= high_th) |
        ((contrast >= 10.0) & (red_bias >= 3.0) & (r >= g - 3.0)) |
        ((contrast >= 25.0) & (lum < 135.0))
    )

    is_weak = (
        (diff >= low_th) |
        ((contrast >= 6.0) & (red_bias >= 1.5) & (r >= g - 5.0)) |
        ((contrast >= 16.0) & (lum < 145.0))
    )

    num_labels, labels = cv2.connectedComponents(is_weak.astype(np.uint8))
    mask_inner = np.zeros_like(is_weak, dtype=np.uint8)

    for lbl in range(1, num_labels):
        comp = (labels == lbl)
        if np.any(is_strong[comp]):
            mask_inner[comp] = 255

    # Check for border line spanning almost entire width with tiny height
    coords = cv2.findNonZero(mask_inner)
    if coords is not None:
        bx, by, bw, bh = cv2.boundingRect(coords)
        if bw >= (w - 10) and bh <= 4:
            return 0, np.zeros((h, w), dtype=np.uint8)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[3:h-3, 3:w-3] = mask_inner
    cnt = cv2.countNonZero(mask)
    return cnt, mask

sheet_files = sorted(Path(BASE_DIR / "sample_data" / "user_sheets").glob("media_1794*.jpg"))

for s_idx, fpath in enumerate(sheet_files, 1):
    img = cv2.imread(str(fpath))
    dom = extractor._detect_distribution_of_marks(img)
    if dom is None:
        print(f"Sheet {s_idx}: FAILED")
        continue

    print(f"\n--- SHEET {s_idx} ({fpath.name}) ---")
    for ia, (x1, x2) in COLS.items():
        found = []
        for rid, y1, y2 in ROWS:
            c = dom[y1:y2, x1:x2]
            cnt, mask = extract_adaptive_ink_bfs(c)
            if cnt >= 20:
                found.append(f"{rid}:{cnt}")
        print(f"  {ia}: {len(found)} ink cells -> {', '.join(found)}")
