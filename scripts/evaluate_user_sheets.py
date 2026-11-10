"""Evaluate current bluebook extractor against the 8 real user-uploaded sheets.
Strict Zero Emojis Policy Enforced.
"""

from pathlib import Path
import json
import os
import sys
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.digit_recognizer import DigitRecognizer
from src.eval_sheet_extractor import EvalSheetExtractor
CONFIG_PATH = BASE_DIR / "config" / "template.json"
MODEL_PATH = BASE_DIR / "models" / "digit_model.onnx"
SHEETS_DIR = BASE_DIR / "sample_data" / "user_sheets"

images = sorted([
    f for f in SHEETS_DIR.glob("media_1794*.jpg")
])

recognizer = DigitRecognizer(model_path=str(MODEL_PATH))
extractor = EvalSheetExtractor(recognizer=recognizer)

print(f"Loaded {len(images)} user sheets for evaluation.")

for idx, img_path in enumerate(images, 1):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"Sheet {idx}: FAILED TO READ IMAGE")
        continue

    res = extractor.extract_from_image(img_bgr, target_ia="ALL")
    sheets = res.get("sheets", [])
    if not sheets:
        # Check single sheet result structure
        sheets = [{
            "results_by_ia": res.get("results_by_ia", {}),
            "summary_totals": res.get("summary_totals", {}),
        }]
    if not sheets:
        print(f"Sheet {idx}: NO SHEETS DETECTED")
        continue

    sheet = sheets[0]
    fname = img_path.name
    print(f"\n================ SHEET {idx}: {fname} ================")

    for ia in ["IA 1", "IA 2", "IA 3"]:
        marks = sheet.get("results_by_ia", {}).get(ia, [])
        non_blank = []
        for m in marks:
            mid = m.get("id")
            val = m.get("detected_str")
            conf = m.get("confidence", 0)
            num = m.get("numeric_val", 0)
            if num > 0 or val in ["-", "0"]:
                non_blank.append(f"{mid}:{val}({conf:.2f})")

        totals = sheet.get("summary_totals", {}).get(ia, {})
        calc = totals.get("calculated_sum", 0)
        det = totals.get("detected_total")
        matches = totals.get("matches", False)
        print(f"  {ia} [Calc: {calc}, Det: {det}, Match: {matches}]")
        print(f"    Marks: {', '.join(non_blank)}")
