"""Grid snapping and cell ink extraction utilities for Evaluation Sheets.
Strict Zero Emojis Policy Enforced.
"""

from typing import Dict, List, Tuple
import cv2
import numpy as np


def get_snapped_grid(warped_table: np.ndarray) -> Tuple[List[int], Dict[str, Tuple[int, int]]]:
    """Snap horizontal rows and vertical columns to exact grid lines.
    
    Returns:
        rows: List of 14 y-coordinates bounding 13 rows (1a..4c and Total Marks).
        cols: Dict mapping IA names ('IA 1', 'IA 2', 'IA 3') to (x1, x2) column boundaries.
    """
    gray = cv2.cvtColor(warped_table, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)

    # 1. Snap horizontal rows
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
    h_proj = np.sum(h_lines[:, 50:150], axis=1) / 255.0

    y_start = 80 + int(np.argmax(h_proj[80:110]))
    y_end = 690 + int(np.argmax(h_proj[690:725]))
    avg_h = (y_end - y_start) / 13.0

    rows = [y_start]
    for k in range(1, 13):
        exp_y = int(round(y_start + k * avg_h))
        w_low = max(0, exp_y - 10)
        w_high = min(warped_table.shape[0], exp_y + 11)
        sub = h_proj[w_low:w_high]
        rows.append(w_low + int(np.argmax(sub)) if len(sub) > 0 and np.max(sub) > 20 else exp_y)
    rows.append(y_end)

    # 2. Snap vertical columns
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)
    v_proj = np.sum(v_lines[150:650, :], axis=0) / 255.0

    expected_v = [0, 95, 199, 295, 399, 495, 599]
    v_snapped = []
    for exp_x in expected_v:
        w_low = max(0, exp_x - 12)
        w_high = min(warped_table.shape[1], exp_x + 13)
        sub = v_proj[w_low:w_high]
        v_snapped.append(w_low + int(np.argmax(sub)) if len(sub) > 0 and np.max(sub) > 50 else exp_x)

    cols = {
        "IA 1": (v_snapped[1], v_snapped[2]),
        "IA 2": (v_snapped[3], v_snapped[4]),
        "IA 3": (v_snapped[5], v_snapped[6]),
    }
    return rows, cols


def clean_cell_ink(cell_bgr: np.ndarray) -> np.ndarray:
    """Extract clean handwritten ink from cell, eliminating border lines and background noise."""
    h, w = cell_bgr.shape[:2]
    inner = cell_bgr[2:h-2, 3:w-3]
    if inner.size == 0:
        return np.zeros((h, w), dtype=np.uint8)

    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8)

    # Calculate color metrics
    b = inner[:, :, 0].astype(float)
    g = inner[:, :, 1].astype(float)
    r = inner[:, :, 2].astype(float)
    diff = r - np.maximum(g, b)
    is_red = (diff >= 4.0) | (r >= g + 2.0)

    # Stroke is local contrast or reddish
    stroke = th & ((is_red.astype(np.uint8) * 255) | (th > 0))

    # Eliminate border lines and artifacts via connected component analysis
    num_l, labels, stats, _ = cv2.connectedComponentsWithStats(stroke)
    for l in range(1, num_l):
        cw = stats[l, cv2.CC_STAT_WIDTH]
        ch = stats[l, cv2.CC_STAT_HEIGHT]
        cx = stats[l, cv2.CC_STAT_LEFT]
        cy = stats[l, cv2.CC_STAT_TOP]

        # Vertical border touching edge
        if (cx <= 3 or (cx + cw) >= w - 5) and ch >= 10:
            stroke[labels == l] = 0
        # Horizontal border touching top/bottom
        if (cy <= 2 or (cy + ch) >= h - 4) and cw >= 12:
            stroke[labels == l] = 0
        # Tiny specks
        if stats[l, cv2.CC_STAT_AREA] < 8:
            stroke[labels == l] = 0

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[2:h-2, 3:w-3] = stroke
    return mask
