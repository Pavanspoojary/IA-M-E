"""Evaluation Sheet marks extraction and column cropping module.

Designed specifically for 3-column college Marks Evaluation Sheets (IA 1, IA 2, IA 3)
containing question rows 1a, 1b, 2a, 2b, 3a, 3b, 3b-alt, 4a, 4b, and Total.
Strict Zero Emojis Policy Enforced.
"""

from typing import List, Dict, Any, Tuple, Optional
import io
import csv
import base64
import cv2
import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

from src.digit_recognizer import DigitRecognizer


class EvalSheetExtractor:
    """Extractor for 3-column IA Marks Evaluation Sheets."""

    REF_WIDTH = 764
    REF_HEIGHT = 1024

    # Column boundary geometries on reference resolution
    COLUMNS = {
        "IA 1": {
            "name": "IA 1",
            "x_box": 175,
            "crop_x": 50,
            "crop_w": 215,
            "crop_y": 108,
            "crop_h": 828,
        },
        "IA 2": {
            "name": "IA 2",
            "x_box": 402,
            "crop_x": 277,
            "crop_w": 215,
            "crop_y": 108,
            "crop_h": 828,
        },
        "IA 3": {
            "name": "IA 3",
            "x_box": 624,
            "crop_x": 502,
            "crop_w": 215,
            "crop_y": 108,
            "crop_h": 828,
        },
    }

    # 10 Questions and Total rows (Legacy Marks Evaluation Sheet)
    ROWS = [
        ("1a", "Q1 (a)", 235),
        ("1b", "Q1 (b)", 307),
        ("2a", "Q2 (a)", 379),
        ("2b", "Q2 (b)", 449),
        ("3a", "Q3 (a)", 518),
        ("3b", "Q3 (b)", 588),
        ("3b_alt", "Q3 (b-alt)", 658),
        ("4a", "Q4 (a)", 729),
        ("4b", "Q4 (b)", 800),
        ("total", "Total", 872),
    ]

    # 13 Rows for VTU/Engineering College 'DISTRIBUTION OF MARKS' template
    DOM_ROWS = [
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

    # Columns configuration for 'DISTRIBUTION OF MARKS' (600x780 rectified resolution)
    DOM_COLUMNS = {
        "IA 1": {"col_range": (0, 200), "marks_x": (95, 199)},
        "IA 2": {"col_range": (199, 400), "marks_x": (294, 399)},
        "IA 3": {"col_range": (399, 600), "marks_x": (495, 599)},
    }

    def __init__(self, recognizer: Optional[DigitRecognizer] = None):
        self.recognizer = recognizer or DigitRecognizer()

    def _encode_base64(self, img_bgr: np.ndarray) -> str:
        """Encode OpenCV BGR image as base64 PNG data URL."""
        if img_bgr is None or img_bgr.size == 0:
            return ""
        success, buf = cv2.imencode(".png", img_bgr)
        if not success:
            return ""
        b64 = base64.b64encode(buf).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """Order 4 polygon points as top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _detect_distribution_of_marks(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Detect and rectify VTU/College 'DISTRIBUTION OF MARKS' grid table."""
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            area = cv2.contourArea(c)
            if area > (w * h * 0.35):
                bx, by, bw, bh = cv2.boundingRect(c)
                if bw / float(w) >= 0.65 and bh / float(h) >= 0.55:
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                    if len(approx) == 4:
                        pts = self._order_points(approx.reshape(4, 2).astype("float32"))
                        dest_w, dest_h = 600, 780
                        dst = np.array([
                            [0, 0],
                            [dest_w - 1, 0],
                            [dest_w - 1, dest_h - 1],
                            [0, dest_h - 1]
                        ], dtype="float32")
                        m = cv2.getPerspectiveTransform(pts, dst)
                        warped = cv2.warpPerspective(img_bgr, m, (dest_w, dest_h))
                        return warped

        return None

    def _extract_distribution_of_marks(
        self,
        warped_table: np.ndarray,
        target_ia: str,
        selected_ias: List[str],
    ) -> Dict[str, Any]:
        """Extract marks from rectified 'DISTRIBUTION OF MARKS' grid table."""
        if len(selected_ias) == 1:
            col_x1, col_x2 = self.DOM_COLUMNS[selected_ias[0]]["col_range"]
            overview_crop = warped_table[0:715, col_x1:col_x2]
        else:
            overview_crop = warped_table[0:715, 0:600]
        overview_b64 = self._encode_base64(overview_crop)

        results_by_ia: Dict[str, List[Dict[str, Any]]] = {}
        kernel_2x2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        for ia_name in selected_ias:
            x1, x2 = self.DOM_COLUMNS[ia_name]["marks_x"]
            ia_marks: List[Dict[str, Any]] = []

            for q_id, q_lbl, y1, y2 in self.DOM_ROWS:
                cell = warped_table[y1 + 5 : y2 - 5, x1 + 6 : x2 - 6]
                cgray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

                if cgray.std() < 5.0 or (int(cgray.max()) - int(cgray.min())) < 22:
                    ia_marks.append({
                        "id": q_id,
                        "label": q_lbl,
                        "detected_str": "-",
                        "numeric_val": 0.0,
                        "verified_val": 0.0,
                        "confidence": 1.0,
                        "status": "Blank",
                        "is_blank": True,
                        "crop_base64": self._encode_base64(cell),
                    })
                    continue

                _, bw = cv2.threshold(cgray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                bw[:2, :] = 0
                bw[-2:, :] = 0
                bw[:, :2] = 0
                bw[:, -2:] = 0

                pts = cv2.findNonZero(bw)
                if pts is None or cv2.countNonZero(bw) < 18:
                    ia_marks.append({
                        "id": q_id,
                        "label": q_lbl,
                        "detected_str": "-",
                        "numeric_val": 0.0,
                        "verified_val": 0.0,
                        "confidence": 1.0,
                        "status": "Blank",
                        "is_blank": True,
                        "crop_base64": self._encode_base64(cell),
                    })
                    continue

                rx, ry, rw, rh = cv2.boundingRect(pts)
                if (rh <= 12 and rw >= 18) or (rw > cell.shape[1] * 0.42 and rh < 15):
                    ia_marks.append({
                        "id": q_id,
                        "label": q_lbl,
                        "detected_str": "-",
                        "numeric_val": 0.0,
                        "verified_val": 0.0,
                        "confidence": 0.95,
                        "status": "Valid",
                        "is_blank": False,
                        "crop_base64": self._encode_base64(cell),
                    })
                    continue

                max_m = 50.0 if q_id == "total" else 10.0
                bw_dilated = cv2.dilate(bw, kernel_2x2)
                res = self.recognizer.recognize_region(bw_dilated, max_marks=max_m, allow_decimal=False)

                if res.value_str is None or res.value_str == "-":
                    det_str = "-"
                    num_val = 0.0
                    status_text = "Valid"
                else:
                    det_str = res.value_str
                    num_val = res.numeric_value if res.numeric_value is not None else 0.0
                    status_text = "Valid" if res.confidence >= 0.65 and not res.is_ambiguous else "Needs review"

                ia_marks.append({
                    "id": q_id,
                    "label": q_lbl,
                    "detected_str": det_str,
                    "numeric_val": num_val,
                    "verified_val": num_val,
                    "confidence": round(res.confidence, 3),
                    "status": status_text,
                    "is_blank": False,
                    "crop_base64": self._encode_base64(cell),
                })

            results_by_ia[ia_name] = ia_marks

        summary_totals = {}
        for ia_name, marks in results_by_ia.items():
            calc_sum = sum(
                m["verified_val"] for m in marks
                if m["id"] != "total" and m["verified_val"] is not None and not m["is_blank"]
            )
            detected_tot_mark = next((m for m in marks if m["id"] == "total"), None)
            detected_val = detected_tot_mark["numeric_val"] if detected_tot_mark else None

            summary_totals[ia_name] = {
                "calculated_sum": round(calc_sum, 1),
                "detected_total": detected_val,
                "matches": detected_val is not None and abs(calc_sum - detected_val) < 0.01,
            }

        return {
            "template_type": "DISTRIBUTION_OF_MARKS",
            "column_crop_base64": overview_b64,
            "target_ia": target_ia,
            "selected_ias": selected_ias,
            "results_by_ia": results_by_ia,
            "summary_totals": summary_totals,
            "total_sheets": 1,
            "sheets": [{
                "sheet_index": 1,
                "page_number": 1,
                "sheet_label": "Sheet 1 (Page 1)",
                "template_type": "DISTRIBUTION_OF_MARKS",
                "column_crop_base64": overview_b64,
                "target_ia": target_ia,
                "selected_ias": selected_ias,
                "results_by_ia": results_by_ia,
                "summary_totals": summary_totals,
            }],
        }

    def _detect_printed_boxes(self, img_bgr: np.ndarray) -> Optional[Dict[str, List[Tuple[int, int, int, int]]]]:
        """Detect printed marks square boxes across IA columns on camera photos and scans."""
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Scale expected box dimensions relative to image resolution (calibrated on 896x1200 / 764x1024)
        min_bw = max(35, int(w * 0.055))
        max_bw = min(120, int(w * 0.125))
        min_bh = max(30, int(h * 0.038))
        max_bh = min(100, int(h * 0.075))

        raw_boxes = []
        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            if min_bw <= bw <= max_bw and min_bh <= bh <= max_bh:
                raw_boxes.append((bx, by, bw, bh))

        if len(raw_boxes) < 20:
            return None

        # Deduplicate inner/outer contour pairs
        raw_boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        unique_boxes = []
        for b in raw_boxes:
            bx, by, bw, bh = b
            bcx, bcy = bx + bw / 2, by + bh / 2
            if not any(abs(bcx - (u[0] + u[2] / 2)) < 15 and abs(bcy - (u[1] + u[3] / 2)) < 15 for u in unique_boxes):
                unique_boxes.append(b)

        # Cluster boxes into columns by horizontal position
        cols: Dict[float, List[Tuple[int, int, int, int]]] = {}
        for b in unique_boxes:
            cx = b[0] + b[2] / 2
            assigned = False
            for k in list(cols.keys()):
                if abs(cx - k) < (w * 0.06):
                    cols[k].append(b)
                    assigned = True
                    break
            if not assigned:
                cols[cx] = [b]

        sorted_col_keys = sorted([k for k in cols.keys() if len(cols[k]) >= 8])
        if len(sorted_col_keys) < 3:
            return None

        ia_names = ["IA 1", "IA 2", "IA 3"]
        result_cols = {}
        for idx in range(3):
            col_key = sorted_col_keys[idx]
            col_boxes = sorted(cols[col_key], key=lambda b: b[1])
            # Exactly 10 rows per column
            if len(col_boxes) > 10:
                col_boxes = col_boxes[-10:]
            elif len(col_boxes) < 10:
                return None
            result_cols[ia_names[idx]] = col_boxes

        return result_cols

    def _is_strikeout_or_dash(self, b_thresh: np.ndarray) -> bool:
        """Detect unattempted indicators: horizontal dashes, diagonal slashes, and full-box X cross-outs."""
        h, w = b_thresh.shape[:2]
        cnts, _ = cv2.findContours(b_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return True

        pts = cv2.findNonZero(b_thresh)
        if pts is None:
            return True

        rx, ry, rw, rh = cv2.boundingRect(pts)
        density = cv2.countNonZero(b_thresh) / float(max(1, rw * rh))

        # 1. Dash: short and wide
        if rh <= max(14, int(h * 0.28)) and rw >= max(18, int(w * 0.35)):
            return True
        # 2. Large diagonal slash or cross-out X across the box
        if rw >= int(w * 0.70) and rh >= int(h * 0.65) and density < 0.22:
            return True
        # 3. Very low fill
        if cv2.countNonZero(b_thresh) < 25:
            return True

        return False

    def _extract_total_numerator(self, b_thresh: np.ndarray) -> Tuple[str, float]:
        """Extract numerator score from the bottom total box, filtering denominators like /25."""
        res = self.recognizer.recognize_region(b_thresh, max_marks=100.0, allow_decimal=False)
        val_str = res.value_str or "0"

        # If evaluator wrote score/25, take first 2 digits if <= 25
        if len(val_str) > 2 and val_str[:2].isdigit() and int(val_str[:2]) <= 25:
            val_str = val_str[:2]

        try:
            num_val = float(val_str)
        except ValueError:
            num_val = 0.0

        return val_str, num_val

    def extract_from_image(
        self,
        img_bgr: np.ndarray,
        target_ia: str = "IA 1",
    ) -> Dict[str, Any]:
        """Extract question marks and crop selected column(s).

        Args:
            img_bgr: Loaded input image in BGR format.
            target_ia: 'IA 1', 'IA 2', 'IA 3', or 'ALL'.

        Returns:
            Dictionary containing:
                - column_crop_base64: Image of the cropped IA column.
                - target_ia: Normalized target name.
                - results: List of extracted question records per IA.
                - total_calculated: Computed sum of question marks.
                - total_detected: Value detected in the sheet Total box.
        """
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Invalid or empty image provided.")

        target_clean = target_ia.strip().upper()
        if target_clean in ("ALL", "ALL IAS", "ALL IA"):
            selected_ias = ["IA 1", "IA 2", "IA 3"]
        else:
            ia_key = "IA 1"
            for k in self.COLUMNS:
                if k.upper() in target_clean:
                    ia_key = k
                    break
            selected_ias = [ia_key]

        # 0. Attempt VTU / Bluebook 'DISTRIBUTION OF MARKS' detection
        warped_dom = self._detect_distribution_of_marks(img_bgr)
        if warped_dom is not None:
            return self._extract_distribution_of_marks(warped_dom, target_ia, selected_ias)

        # 1. Attempt adaptive printed box detection directly on image
        detected_boxes = self._detect_printed_boxes(img_bgr)
        results_by_ia: Dict[str, List[Dict[str, Any]]] = {}

        if detected_boxes is not None:
            # Compute overview crop boundaries
            all_selected_boxes = []
            for ia_name in selected_ias:
                all_selected_boxes.extend(detected_boxes[ia_name])

            min_bx = min(b[0] for b in all_selected_boxes)
            max_bx = max(b[0] + b[2] for b in all_selected_boxes)
            min_by = min(b[1] for b in all_selected_boxes)
            max_by = max(b[1] + b[3] for b in all_selected_boxes)

            pad_left = int(img_bgr.shape[1] * 0.14)
            pad_top = int(img_bgr.shape[0] * 0.11)
            crop_x1 = max(0, min_bx - pad_left)
            crop_x2 = min(img_bgr.shape[1], max_bx + 15)
            crop_y1 = max(0, min_by - pad_top)
            crop_y2 = min(img_bgr.shape[0], max_by + 20)

            overview_crop = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
            overview_b64 = self._encode_base64(overview_crop)

            labels_map = [r[1] for r in self.ROWS]
            ids_map = [r[0] for r in self.ROWS]

            for ia_name in selected_ias:
                col_boxes = detected_boxes[ia_name]
                ia_marks: List[Dict[str, Any]] = []

                for row_idx, box in enumerate(col_boxes):
                    q_id = ids_map[row_idx]
                    q_lbl = labels_map[row_idx]
                    bx, by, bw, bh = box

                    m = max(5, int(bw * 0.09))
                    box_crop = img_bgr[by + m : by + bh - m, bx + m : bx + bw - m]
                    gray = cv2.cvtColor(box_crop, cv2.COLOR_BGR2GRAY)

                    is_blank = gray.std() < 5.0 or (int(gray.max()) - int(gray.min())) < 25

                    if is_blank:
                        ia_marks.append({
                            "id": q_id,
                            "label": q_lbl,
                            "detected_str": "-",
                            "numeric_val": 0.0,
                            "verified_val": 0.0,
                            "confidence": 1.0,
                            "status": "Valid",
                            "is_blank": True,
                            "crop_base64": self._encode_base64(box_crop),
                        })
                    else:
                        _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)

                        if q_id != "total":
                            is_so = self._is_strikeout_or_dash(thresh)
                            if is_so:
                                ia_marks.append({
                                    "id": q_id,
                                    "label": q_lbl,
                                    "detected_str": "-",
                                    "numeric_val": 0.0,
                                    "verified_val": 0.0,
                                    "confidence": 0.95,
                                    "status": "Valid",
                                    "is_blank": False,
                                    "crop_base64": self._encode_base64(box_crop),
                                })
                            else:
                                res = self.recognizer.recognize_region(thresh, max_marks=100.0, allow_decimal=False)
                                det_str = res.value_str or "-"
                                num_val = res.numeric_value if res.numeric_value is not None else 0.0
                                ia_marks.append({
                                    "id": q_id,
                                    "label": q_lbl,
                                    "detected_str": det_str,
                                    "numeric_val": num_val,
                                    "verified_val": num_val,
                                    "confidence": round(res.confidence, 3),
                                    "status": "Valid" if res.confidence >= 0.65 and not res.is_ambiguous else "Needs review",
                                    "is_blank": False,
                                    "crop_base64": self._encode_base64(box_crop),
                                })
                        else:
                            val_str, num_val = self._extract_total_numerator(thresh)
                            ia_marks.append({
                                "id": "total",
                                "label": "Total",
                                "detected_str": val_str,
                                "numeric_val": num_val,
                                "verified_val": num_val,
                                "confidence": 0.98,
                                "status": "Valid",
                                "is_blank": False,
                                "crop_base64": self._encode_base64(box_crop),
                            })

                results_by_ia[ia_name] = ia_marks

        else:
            # Fallback to reference template geometry
            h, w = img_bgr.shape[:2]
            if (w, h) != (self.REF_WIDTH, self.REF_HEIGHT):
                img_resized = cv2.resize(img_bgr, (self.REF_WIDTH, self.REF_HEIGHT), interpolation=cv2.INTER_AREA)
            else:
                img_resized = img_bgr.copy()

            if len(selected_ias) > 1:
                overview_crop = img_resized[108:936, 50:717]
            else:
                info = self.COLUMNS[selected_ias[0]]
                overview_crop = img_resized[info["crop_y"]:info["crop_y"] + info["crop_h"], info["crop_x"]:info["crop_x"] + info["crop_w"]]

            overview_b64 = self._encode_base64(overview_crop)

            for ia_name in selected_ias:
                col_info = self.COLUMNS[ia_name]
                xb = col_info["x_box"]
                ia_marks = []

                for q_id, q_lbl, y in self.ROWS:
                    box_crop = img_resized[y + 8 : y + 56 - 8, xb + 9 : xb + 68 - 9]
                    gray = cv2.cvtColor(box_crop, cv2.COLOR_BGR2GRAY)

                    is_blank = gray.std() < 5.0 or (int(gray.max()) - int(gray.min())) < 25

                    if is_blank:
                        ia_marks.append({
                            "id": q_id,
                            "label": q_lbl,
                            "detected_str": "",
                            "numeric_val": None,
                            "verified_val": 0.0,
                            "confidence": 1.0,
                            "status": "Blank",
                            "is_blank": True,
                            "crop_base64": self._encode_base64(box_crop),
                        })
                    else:
                        _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)
                        res = self.recognizer.recognize_region(
                            thresh,
                            max_marks=100.0,
                            allow_decimal=(q_id != "total")
                        )
                        if res.value_str == "-" or (res.value_str is None and res.confidence >= 0.80):
                            det_str = "-"
                            num_val = 0.0
                            ver_val = 0.0
                            status_text = "Valid"
                        else:
                            det_str = res.value_str or ""
                            num_val = res.numeric_value
                            ver_val = num_val if num_val is not None else 0.0
                            status_text = "Valid" if res.confidence >= 0.65 and not res.is_ambiguous else "Needs review"

                        ia_marks.append({
                            "id": q_id,
                            "label": q_lbl,
                            "detected_str": det_str,
                            "numeric_val": num_val,
                            "verified_val": ver_val,
                            "confidence": round(res.confidence, 3),
                            "status": status_text,
                            "is_blank": False,
                            "crop_base64": self._encode_base64(box_crop),
                        })

                results_by_ia[ia_name] = ia_marks

        # Compute calculated totals
        summary_totals = {}
        for ia_name, marks in results_by_ia.items():
            calc_sum = sum(
                m["verified_val"] for m in marks
                if m["id"] != "total" and m["verified_val"] is not None and not m["is_blank"]
            )
            detected_tot_mark = next((m for m in marks if m["id"] == "total"), None)
            detected_val = detected_tot_mark["numeric_val"] if detected_tot_mark else None

            summary_totals[ia_name] = {
                "calculated_sum": round(calc_sum, 1),
                "detected_total": detected_val,
                "matches": detected_val is not None and abs(calc_sum - detected_val) < 0.01,
            }

        sheet_payload = {
            "template_type": "MARKS_EVALUATION_SHEET",
            "column_crop_base64": overview_b64,
            "target_ia": target_ia,
            "selected_ias": selected_ias,
            "results_by_ia": results_by_ia,
            "summary_totals": summary_totals,
        }

        # Include single-sheet representation in sheets list for uniform multi-sheet contract
        sheet_payload["total_sheets"] = 1
        sheet_payload["sheets"] = [{
            "sheet_index": 1,
            "page_number": 1,
            "sheet_label": "Sheet 1 (Page 1)",
            "template_type": "MARKS_EVALUATION_SHEET",
            "column_crop_base64": overview_b64,
            "target_ia": target_ia,
            "selected_ias": selected_ias,
            "results_by_ia": results_by_ia,
            "summary_totals": summary_totals,
        }]

        return sheet_payload

    def extract_multipage(
        self,
        images: List[Tuple[int, np.ndarray]],
        target_ia: str = "IA 1",
    ) -> Dict[str, Any]:
        """Extract marks across multiple sheets/pages in order.

        Args:
            images: List of (page_number, cv2_bgr_image) tuples.
            target_ia: 'IA 1', 'IA 2', 'IA 3', or 'ALL'.

        Returns:
            Dictionary with multi-sheet payload:
                - total_sheets: count of sheets
                - sheets: list of individual sheet extraction dicts
                - Top-level keys corresponding to sheet 1 for backward compatibility
        """
        if not images:
            raise ValueError("No images provided for multipage extraction.")

        sheets = []
        for page_idx, img_bgr in images:
            sheet_res = self.extract_from_image(img_bgr, target_ia=target_ia)
            sheets.append({
                "sheet_index": page_idx,
                "page_number": page_idx,
                "sheet_label": f"Sheet {page_idx} (Page {page_idx})",
                "template_type": sheet_res.get("template_type", "MARKS_EVALUATION_SHEET"),
                "column_crop_base64": sheet_res["column_crop_base64"],
                "target_ia": sheet_res["target_ia"],
                "selected_ias": sheet_res["selected_ias"],
                "results_by_ia": sheet_res["results_by_ia"],
                "summary_totals": sheet_res["summary_totals"],
            })

        first = sheets[0]
        return {
            "total_sheets": len(sheets),
            "template_type": first.get("template_type", "MARKS_EVALUATION_SHEET"),
            "target_ia": first["target_ia"],
            "selected_ias": first["selected_ias"],
            "column_crop_base64": first["column_crop_base64"],
            "results_by_ia": first["results_by_ia"],
            "summary_totals": first["summary_totals"],
            "sheets": sheets,
        }

    def generate_excel_bytes(self, extraction_data: Dict[str, Any]) -> bytes:
        """Create formatted .xlsx workbook from extraction results (single or multi-sheet)."""
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "IA Marks Summary"

        header_font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
        title_font = Font(name="Segoe UI", size=14, bold=True, color="1F4E79")
        bold_font = Font(name="Segoe UI", size=11, bold=True)
        regular_font = Font(name="Segoe UI", size=11)
        center_align = Alignment(horizontal="center", vertical="center")
        left_align = Alignment(horizontal="left", vertical="center")
        thin_border = Border(
            left=Side(style="thin", color="DDE2E5"),
            right=Side(style="thin", color="DDE2E5"),
            top=Side(style="thin", color="DDE2E5"),
            bottom=Side(style="thin", color="DDE2E5"),
        )

        sheets = extraction_data.get("sheets") if extraction_data else None
        if not sheets and extraction_data and "results_by_ia" in extraction_data:
            sheets = [{
                "sheet_index": 1,
                "page_number": 1,
                "sheet_label": extraction_data.get("sheet_label", "Sheet 1 (Page 1)"),
                "results_by_ia": extraction_data["results_by_ia"],
            }]
        if not sheets:
            sheets = []

        q_cols = self._get_question_columns(sheets)

        # Sheet 1: Marks Summary Matrix
        last_col_letter = openpyxl.utils.get_column_letter(len(q_cols) + 4)
        ws.merge_cells(f"A1:{last_col_letter}1")
        ws["A1"] = "INTERNAL ASSESSMENT EVALUATION MARKS MATRIX"
        ws["A1"].font = title_font
        ws["A1"].alignment = left_align
        ws.row_dimensions[1].height = 28

        headers = ["Sheet / Page", "Assessment"] + [lbl for _, lbl in q_cols] + ["Total", "Status"]
        col_widths = [18, 14] + [8] * len(q_cols) + [10, 16]

        ws.append([])
        ws.append(headers)
        ws.row_dimensions[3].height = 24

        for col_idx in range(1, len(headers) + 1):
            cell = ws.cell(row=3, column=col_idx)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = center_align

        current_row = 4
        for sheet in sheets:
            sheet_label = sheet.get("sheet_label") or f"Sheet {sheet.get('sheet_index', 1)}"
            for ia_name, marks in sheet.get("results_by_ia", {}).items():
                m_map = {m.get("id"): m for m in marks}
                q_vals = [self._format_mark_display(m_map.get(qid)) for qid, _ in q_cols]
                tot_m = m_map.get("total")
                tot_val = self._format_mark_display(tot_m)

                statuses = [m.get("status", "") for m in marks]
                if any("Needs Review" in s or "Needs review" in s for s in statuses):
                    row_status = "Needs Review"
                elif any("Corrected" in s for s in statuses):
                    row_status = "Corrected manually"
                elif all(self._is_mark_unattempted(m) for m in marks if m.get("id") != "total"):
                    row_status = "Blank"
                else:
                    row_status = "Valid"

                row_vals = [sheet_label, ia_name] + q_vals + [tot_val, row_status]
                ws.append(row_vals)
                ws.row_dimensions[current_row].height = 20

                for col_idx in range(1, len(row_vals) + 1):
                    cell = ws.cell(row=current_row, column=col_idx)
                    cell.font = bold_font if col_idx == len(row_vals) - 1 else regular_font
                    cell.border = thin_border
                    cell.alignment = left_align if col_idx == 1 else center_align

                current_row += 1

        for idx, width in enumerate(col_widths, start=1):
            col_letter = openpyxl.utils.get_column_letter(idx)
            ws.column_dimensions[col_letter].width = width

        # Sheet 2: Detailed Question-Wise Breakdown
        ws_det = wb.create_sheet(title="Detailed Extraction")
        ws_det.merge_cells("A1:F1")
        ws_det["A1"] = "DETAILED QUESTION-WISE EXTRACTION AUDIT"
        ws_det["A1"].font = title_font
        ws_det["A1"].alignment = left_align
        ws_det.row_dimensions[1].height = 28

        det_headers = ["Sheet / Page", "Assessment", "Question", "Detected Mark", "Verified Mark", "Status"]
        det_col_widths = [18, 14, 16, 16, 16, 18]

        ws_det.append([])
        ws_det.append(det_headers)
        ws_det.row_dimensions[3].height = 24

        for col_idx in range(1, len(det_headers) + 1):
            cell = ws_det.cell(row=3, column=col_idx)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = center_align

        det_row = 4
        for sheet in sheets:
            sheet_label = sheet.get("sheet_label") or f"Sheet {sheet.get('sheet_index', 1)}"
            for ia_name, marks in sheet.get("results_by_ia", {}).items():
                for m in marks:
                    det_str = m.get("detected_str", "")
                    ver_val = m.get("verified_val")
                    ver_str = self._format_number_val(ver_val) if ver_val is not None else ""
                    st = m.get("status", "")

                    ws_det.append([
                        sheet_label,
                        ia_name,
                        m.get("label", m.get("id", "")),
                        det_str,
                        ver_str,
                        st,
                    ])
                    ws_det.row_dimensions[det_row].height = 20
                    for col_idx in range(1, 7):
                        cell = ws_det.cell(row=det_row, column=col_idx)
                        cell.font = bold_font if m.get("id") == "total" else regular_font
                        cell.border = thin_border
                        cell.alignment = left_align if col_idx in (1, 3) else center_align
                    det_row += 1

        for idx, width in enumerate(det_col_widths, start=1):
            col_letter = openpyxl.utils.get_column_letter(idx)
            ws_det.column_dimensions[col_letter].width = width

        # For multi-sheet workbooks, include dedicated tabs per sheet
        if len(sheets) > 1:
            for s_idx, sheet in enumerate(sheets, start=1):
                sheet_title = f"Sheet {s_idx}"
                ws_sub = wb.create_sheet(title=sheet_title)
                ws_sub.merge_cells("A1:E1")
                ws_sub["A1"] = f"MARKS EVALUATION - {sheet.get('sheet_label', sheet_title)}"
                ws_sub["A1"].font = title_font
                ws_sub["A1"].alignment = left_align
                ws_sub.row_dimensions[1].height = 28

                sub_headers = ["Assessment", "Question", "Detected Mark", "Verified Mark", "Status"]
                ws_sub.append([])
                ws_sub.append(sub_headers)
                ws_sub.row_dimensions[3].height = 24

                for col_idx in range(1, 6):
                    cell = ws_sub.cell(row=3, column=col_idx)
                    cell.font = header_font
                    cell.fill = header_fill
                    cell.alignment = center_align

                sub_row = 4
                for ia_name, marks in sheet.get("results_by_ia", {}).items():
                    for m in marks:
                        det_str = m.get("detected_str", "")
                        ver_val = m.get("verified_val")
                        ver_str = self._format_number_val(ver_val) if ver_val is not None else ""
                        st = m.get("status", "")
                        ws_sub.append([
                            ia_name,
                            m.get("label", m.get("id", "")),
                            det_str,
                            ver_str,
                            st,
                        ])
                        ws_sub.row_dimensions[sub_row].height = 20
                        for col_idx in range(1, 6):
                            cell = ws_sub.cell(row=sub_row, column=col_idx)
                            cell.font = bold_font if m.get("id") == "total" else regular_font
                            cell.border = thin_border
                            cell.alignment = center_align if col_idx != 2 else left_align
                        sub_row += 1

                sub_col_widths = [16, 18, 16, 16, 16]
                for idx, width in enumerate(sub_col_widths, start=1):
                    col_letter = openpyxl.utils.get_column_letter(idx)
                    ws_sub.column_dimensions[col_letter].width = width

        out_buffer = io.BytesIO()
        wb.save(out_buffer)
        return out_buffer.getvalue()

    @staticmethod
    def _format_number_val(val: Any) -> str:
        """Format number cleanly: whole numbers as integers, decimals as float."""
        try:
            f = float(val)
            if f.is_integer():
                return str(int(f))
            return f"{f:.1f}"
        except (ValueError, TypeError):
            return str(val)

    @classmethod
    def _is_mark_unattempted(cls, m: Optional[Dict[str, Any]]) -> bool:
        """Check if question was unattempted / dash / blank."""
        if not m:
            return True
        if m.get("is_blank", False) or m.get("status") == "Blank":
            return True
        det = str(m.get("detected_str", "")).strip()
        return det in ("-", "—", "–", "")

    @classmethod
    def _format_mark_display(cls, m: Optional[Dict[str, Any]]) -> str:
        """Format mark for table/CSV cell: '-' for unattempted, clean number for attempted."""
        if not m:
            return "-"
        if cls._is_mark_unattempted(m):
            v = m.get("verified_val")
            if v is not None and v > 0:
                return cls._format_number_val(v)
            return "-"
        v = m.get("verified_val")
        if v is not None:
            return cls._format_number_val(v)
        v = m.get("numeric_val")
        if v is not None:
            return cls._format_number_val(v)
        det = str(m.get("detected_str", "")).strip()
        return "-" if det in ("", "-") else det

    @classmethod
    def _get_question_columns(cls, sheets: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """Dynamically detect question columns based on whether sheets contain 12-question (VTU) or 10-question layout."""
        for s in sheets:
            for ia_name, marks in s.get("results_by_ia", {}).items():
                if any(m.get("id") in ("1c", "2c", "3c", "4c") for m in marks):
                    return [
                        ("1a", "1.a"),
                        ("1b", "1.b"),
                        ("1c", "1.c"),
                        ("2a", "2.a"),
                        ("2b", "2.b"),
                        ("2c", "2.c"),
                        ("3a", "3.a"),
                        ("3b", "3.b"),
                        ("3c", "3.c"),
                        ("4a", "4.a"),
                        ("4b", "4.b"),
                        ("4c", "4.c"),
                    ]
        return [
            ("1a", "1a"),
            ("1b", "1b"),
            ("2a", "2a"),
            ("2b", "2b"),
            ("3a", "3a"),
            ("3b", "3b"),
            ("3b_alt", "3b-alt"),
            ("4a", "4a"),
            ("4b", "4b"),
        ]

    def generate_csv_string(self, extraction_data: Dict[str, Any]) -> str:
        """Create formatted gradebook matrix CSV string from extraction results."""
        out = io.StringIO()
        writer = csv.writer(out)

        sheets = extraction_data.get("sheets") if extraction_data else None
        if not sheets and extraction_data and "results_by_ia" in extraction_data:
            sheets = [{
                "sheet_index": 1,
                "page_number": 1,
                "sheet_label": extraction_data.get("sheet_label", "Sheet 1 (Page 1)"),
                "results_by_ia": extraction_data["results_by_ia"],
            }]
        if not sheets:
            sheets = []

        q_cols = self._get_question_columns(sheets)

        headers = ["Sheet", "Assessment"] + [lbl for _, lbl in q_cols] + ["Total", "Status"]
        writer.writerow(headers)

        for sheet in sheets:
            sheet_label = sheet.get("sheet_label") or f"Sheet {sheet.get('sheet_index', 1)}"
            for ia_name, marks in sheet.get("results_by_ia", {}).items():
                m_map = {m.get("id"): m for m in marks}
                q_vals = [self._format_mark_display(m_map.get(qid)) for qid, _ in q_cols]
                tot_m = m_map.get("total")
                tot_val = self._format_mark_display(tot_m)

                statuses = [m.get("status", "") for m in marks]
                if any("Needs Review" in s or "Needs review" in s for s in statuses):
                    row_status = "Needs Review"
                elif any("Corrected" in s for s in statuses):
                    row_status = "Corrected manually"
                elif all(self._is_mark_unattempted(m) for m in marks if m.get("id") != "total"):
                    row_status = "Blank"
                else:
                    row_status = "Valid"

                writer.writerow([sheet_label, ia_name] + q_vals + [tot_val, row_status])

        return out.getvalue()
