# metrics_ocr.py
from typing import Dict, Tuple, List
import math
import re

MONTH_MAP = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
    "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
    "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"
}

def normalize_date(text: str) -> str:
    """
    Normalisasi berbagai format tanggal menjadi YYYY-MM-DD
    Contoh:
    - 07 Jul 1972 -> 1972-07-07
    - 01 FEB 1985 -> 1985-02-01
    - 1972-07-07 -> tetap
    """
    if not text:
        return ""

    t = text.strip()

    # Sudah ISO
    if re.match(r"\d{4}-\d{2}-\d{2}", t):
        return t

    # Format: 07 Jul 1972 / 7 July 1972
    m = re.match(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", t)
    if m:
        d, mon, y = m.groups()
        mon = mon[:3].upper()
        if mon in MONTH_MAP:
            return f"{y}-{MONTH_MAP[mon]}-{int(d):02d}"

    # Format: 01 FEB 1985
    m = re.match(r"(\d{1,2})\s+([A-Z]{3})\s+(\d{4})", t.upper())
    if m:
        d, mon, y = m.groups()
        if mon in MONTH_MAP:
            return f"{y}-{MONTH_MAP[mon]}-{int(d):02d}"

    # Jika gagal, kembalikan teks asli
    return t


# ---------- Levenshtein utk WER ----------
def levenshtein(a: List[str], b: List[str]) -> int:
    """Hitung jarak Levenshtein sederhana (untuk WER)."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )
    return dp[m][n]

# ---------- CAR & WAR ----------
def compute_car(pred: str, gt: str) -> float:
    """Character Accuracy Rate dalam persen."""
    if gt is None:
        gt = ""
    if pred is None:
        pred = ""

    gt = gt.strip()
    pred = pred.strip()

    if len(gt) == 0:
        return 100.0 if len(pred) == 0 else 0.0

    total = len(gt)
    benar = 0
    for i, c in enumerate(gt):
        if i < len(pred) and pred[i] == c:
            benar += 1
    return benar / total * 100.0


def compute_war(pred: str, gt: str) -> float:
    """Word Accuracy Rate dalam persen (berbasis split spasi)."""
    if gt is None:
        gt = ""
    if pred is None:
        pred = ""

    gt_words = gt.strip().split()
    pred_words = pred.strip().split()

    if len(gt_words) == 0:
        return 100.0 if len(pred_words) == 0 else 0.0

    total = len(gt_words)
    benar = 0
    for i, w in enumerate(gt_words):
        if i < len(pred_words) and pred_words[i] == w:
            benar += 1
    return benar / total * 100.0


def compute_wer(pred: str, gt: str) -> float:
    """Word Error Rate (0–1)."""
    if gt is None:
        gt = ""
    if pred is None:
        pred = ""

    gt_words = gt.strip().split()
    pred_words = pred.strip().split()

    if len(gt_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0

    dist = levenshtein(pred_words, gt_words)
    return dist / float(len(gt_words))


# ---------- TP/FP/FN/TN berbasis field ----------
def compare_field(pred: str, gt: str) -> Tuple[int, int, int, int]:
    """
    Definisi sederhana:
    - TP: pred == gt != "" (match)
    - FP: pred != "" dan pred != gt
    - FN: gt != "" dan pred == ""
    - TN: keduanya kosong
    """
    pred = (pred or "").strip()
    gt = (gt or "").strip()

    if gt == "" and pred == "":
        return 0, 0, 0, 1  # TN
    if gt != "" and pred == "":
        return 0, 0, 1, 0  # FN
    if gt == "" and pred != "":
        return 0, 1, 0, 0  # FP
    if pred == gt:
        return 1, 0, 0, 0  # TP
    else:
        return 0, 1, 0, 0  # FP


def aggregate_classification_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    """Hitung Accuracy, Precision, Recall, F1 (dalam persen)."""
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total * 100.0 if total > 0 else 0.0
    precision = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def compute_metrics_for_passport(pred: Dict[str, str], gt: Dict[str, str]) -> Dict[str, float]:
    """
    pred & gt: dict field paspor, contoh:
      {
        "surname": "...",
        "given_names": "...",
        "passport_number": "...",
        "nationality": "...",
        "date_of_birth": "...",
        ...
      }
    """
    fields = [
        "surname",
        "given_names",
        "passport_number",
        "nationality",
        "place_of_birth",
        "date_of_birth",
        "date_of_issue",
        "date_of_expiry",
        "sex",
        "mrz",
    ]

    # CAR/WAR/WER agregat: rata-rata semua field teks
    car_list, war_list, wer_list = [], [], []
    tp = fp = fn = tn = 0

    DATE_FIELDS = {
    "date_of_birth",
    "date_of_issue",
    "date_of_expiry"
}

    for f in fields:
        p = (pred.get(f) or "").strip()
        g = (gt.get(f) or "").strip()

        # 🔥 NORMALISASI TANGGAL OTOMATIS
        if f in DATE_FIELDS:
            p = normalize_date(p)
            g = normalize_date(g)

        car_list.append(compute_car(p, g))
        war_list.append(compute_war(p, g))
        wer_list.append(compute_wer(p, g))

        ctp, cfp, cfn, ctn = compare_field(p, g)
        tp += ctp; fp += cfp; fn += cfn; tn += ctn


    # rata-rata
    car_avg = sum(car_list) / len(car_list) if car_list else 0.0
    war_avg = sum(war_list) / len(war_list) if war_list else 0.0
    wer_avg = sum(wer_list) / len(wer_list) if wer_list else 0.0

    cls_metrics = aggregate_classification_metrics(tp, fp, fn, tn)

    return {
    "car": car_avg,
    "war": war_avg,
    "wer": wer_avg,

    "accuracy": cls_metrics["accuracy"],
    "precision": cls_metrics["precision"],
    "recall": cls_metrics["recall"],
    "f1": cls_metrics["f1"],

    "ground_truth": gt,
    "compare": {
        key: (pred.get(key) or "")
        for key in gt.keys()
    }
}


"""
Passport OCR using YOLOv12 and Faster R-CNN

Copyright (c) 2026 Masniari Samosir

All Rights Reserved.

This source code is made publicly available only for
academic review, research reference, and portfolio evaluation.

Any reproduction, modification, redistribution,
or commercial use without written permission
from the copyright holder is prohibited.
"""
