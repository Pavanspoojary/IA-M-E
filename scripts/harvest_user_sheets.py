"""Harvest real handwriting crops from the 8 user evaluation sheets.
Strict Zero Emojis Policy Enforced.
"""

from pathlib import Path
import sys
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

SHEETS_DIR = BASE_DIR / "sample_data" / "user_sheets"
OUTPUT_DIR = BASE_DIR / "sample_data" / "harvested_crops"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from src.eval_sheet_extractor import EvalSheetExtractor

extractor = EvalSheetExtractor()

# Template geometry (DOM_ROWS and DOM_COLUMNS)
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

def extract_adaptive_ink(cell_bgr):
    h, w, _ = cell_bgr.shape
    # Margin crop: 3px from borders to avoid printed lines
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

    # A pixel is ink if:
    # 1. Obvious red: diff >= 10
    # 2. Dim/shadow red: contrast >= 10 and red_bias >= 3 and r >= g - 2
    # 3. Dark pen / high contrast stroke: contrast >= 25 and lum < 130
    is_red = (diff >= 8) | ((contrast >= 10) & (red_bias >= 3) & (r >= g - 3))
    is_dark = (contrast >= 25) & (lum < 135)

    ink_inner = is_red | is_dark

    # Remove isolated noise specks using morphological opening
    kernel = np.ones((2, 2), np.uint8)
    ink_cleaned = cv2.morphologyEx(ink_inner.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[3:h-3, 3:w-3] = ink_cleaned * 255
    return mask

sheet_files = sorted(SHEETS_DIR.glob("media_1794*.jpg"))
total_harvested = 0

for s_idx, fpath in enumerate(sheet_files, 1):
    img = cv2.imread(str(fpath))
    if img is None:
        continue

    dom = extractor._detect_distribution_of_marks(img)
    if dom is None:
        print(f"Sheet {s_idx} ({fpath.name}): DOM detection failed")
        continue

    cv2.imwrite(str(OUTPUT_DIR / f"rectified_sheet_{s_idx}.png"), dom)

    for ia_name, (x1, x2) in COLS.items():
        ia_tag = ia_name.replace(" ", "_")
        for rid, y1, y2 in ROWS:
            cell = dom[y1:y2, x1:x2]
            mask = extract_adaptive_ink(cell)
            ink_count = np.sum(mask > 0)

            # If ink count >= 20, save as candidate
            if ink_count >= 20:
                coords = cv2.findNonZero(mask)
                if coords is not None:
                    bx, by, bw, bh = cv2.boundingRect(coords)
                    # Exclude border lines (spanning entire width)
                    if bw >= (x2 - x1 - 8) and bh <= 4:
                        continue
                    fname_base = f"s{s_idx}_{ia_tag}_{rid}"
                    cv2.imwrite(str(OUTPUT_DIR / f"{fname_base}_cell.png"), cell)
                    cv2.imwrite(str(OUTPUT_DIR / f"{fname_base}_mask.png"), mask)
                    crop_tight = mask[by:by+bh, bx:bx+bw]
                    cv2.imwrite(str(OUTPUT_DIR / f"{fname_base}_tight.png"), crop_tight)
                    total_harvested += 1

print(f"Harvested {total_harvested} ink cell samples from {len(sheet_files)} user sheets.")
