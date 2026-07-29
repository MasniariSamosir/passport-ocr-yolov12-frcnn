#!/usr/bin/env python3
"""
frcnn_paspor_ocr.py
Faster R-CNN + TrOCR + MRZ Tesseract (FIX MRZ FULL)

Kelas (1-based) model Faster R-CNN:
    1: photo
    2: surname
    3: given_names
    4: passport_number
    5: nationality
    6: date_of_birth
    7: place_of_birth
    8: sex
    9: date_of_issue
    10: date_of_expiry
    11: mrz
"""

import os
import cv2
import re
import json
import glob
import argparse
import traceback
from pathlib import Path
from datetime import datetime
import time

import numpy as np
from PIL import Image

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T

import pytesseract

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# ----------------------------------
# Import preprocessing
# ----------------------------------
try:
    from detection.frcnn.preprocessing import global_preprocess_for_frcnn
except ImportError:
    from preprocessing import global_preprocess_for_frcnn


# =========================================================
# KONFIGURASI GLOBAL
# =========================================================

LABELS = [
    "photo", "surname", "given_names", "passport_number",
    "nationality", "date_of_birth", "place_of_birth",
    "sex", "date_of_issue", "date_of_expiry", "mrz"
]

DEFAULT_CONF = 0.05

OCR_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TrOCR Base
TROCR_MODEL_NAME = "microsoft/trocr-base-printed"
trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_NAME).to(OCR_DEVICE)
trocr_processor = ViTImageProcessor.from_pretrained(TROCR_MODEL_NAME)
trocr_tokenizer = AutoTokenizer.from_pretrained(TROCR_MODEL_NAME)
trocr_model.eval()


# =========================================================
# UTIL UMUM
# =========================================================

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def clean_text_basic(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\n", " ").replace("\t", " ")
    t = re.sub(r"[^A-Za-z0-9<\s\/\-\:\(\)']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_name(raw: str) -> str:
    if not raw:
        return ""
    s = clean_text_basic(raw).upper()
    s = s.replace("'", "")
    s = re.sub(r"[^A-Z\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def fix_passport_number(raw: str) -> str:
    if not raw:
        return ""
    s = clean_text_basic(raw).upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    s = (s
         .replace("O", "0")
         .replace("I", "1")
         .replace("S", "5")
         .replace("B", "8"))
    return s


# =========================================================
# PAD CROPPING PER LABEL
# =========================================================

def get_padding_for_label(label):
    if label == "mrz":
        return 0, 5, 0, 0
    if label == "photo":
        return 5, 5, 5, 5
    
    # [FIX] Padding EKSTRA BESAR untuk Sex
    # Kiri-Kanan ditambah 10px agar huruf "F" atau "P/F" tidak terpotong
    if label == "sex":
        return 2, 2, 10, 10
        
    return 2, 2, 2, 2

def min_conf_for_label(label: str) -> float:
    # KITA PAKSA SEMUA JADI 0.01 atau 0.05
    # Tujuannya: Biarkan semua deteksi lolos dulu ke UI
    if label == "photo":
        return 0.01 
    return 0.05 # 5% Confidence sudah dianggap lolos

def auto_tight_crop(img, threshold=245, margin=2):
    if img is None or img.size == 0:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    row_has_text = np.any(gray < threshold, axis=1)
    col_has_text = np.any(gray < threshold, axis=0)

    if not row_has_text.any() or not col_has_text.any():
        return img

    y_idx = np.where(row_has_text)[0]
    x_idx = np.where(col_has_text)[0]

    y1 = max(0, y_idx[0] - margin)
    y2 = min(gray.shape[0] - 1, y_idx[-1] + margin)
    x1 = max(0, x_idx[0] - margin)
    x2 = min(gray.shape[1] - 1, x_idx[-1] + margin)

    if (y2 - y1) < 4 or (x2 - x1) < 4:
        return img

    return img[y1:y2 + 1, x1:x2 + 1]


# =========================================================
# DATE NORMALIZATION (ringkas)
# =========================================================

MONTH_FIX_NUM = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
    "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
    "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"
}


def frcnn_clean_date(raw):
    if not raw:
        return ""
    s = raw.upper().strip()
    s = re.sub(r"[^A-Z0-9\-\s]", "", s)
    s = s.replace("O", "0").replace("I", "1").replace("S", "5").replace("B", "8")
    s = s.lstrip("-").strip()
    return s


def frcnn_parse_date(s: str) -> str:
    if not s:
        return ""
    s = frcnn_clean_date(s)
    m = re.match(r"(\d{1,2})\s+([A-Z]{3})\s+(\d{4})", s)
    if m:
        d, mon, y = m.groups()
        if mon in MONTH_FIX_NUM:
            return f"{y}-{MONTH_FIX_NUM[mon]}-{int(d):02d}"
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    return ""

# ===========================================
# FINAL DATE NORMALIZER FOR FRCNN — 100% FIXED
# ===========================================
MONTH_MAP = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
    "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
    "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"
}

def normalize_frcnn_date(raw):
    """
    Normalisasi tanggal hasil OCR FRCNN ke format YYYY-MM-DD.
    Mendukung berbagai format:
    - 23 NOV 1999
    - 23 Nov 1999
    - 23-Nov-1999
    - 23/11/1999
    - 19730824
    - 24.08.1973
    - 24 Aug 73
    """
    if not raw:
        return ""

    text = fix_ocr_day_errors(raw.strip())
    text = text.replace(".", "-").replace("/", "-")

    # Case 1: DD MON YYYY
    m = re.match(r"(\d{1,2})[- ]([A-Z]{3})[- ](\d{4})", text)
    if m:
        d, mon, y = m.groups()
        if mon in MONTH_MAP:
            return f"{y}-{MONTH_MAP[mon]}-{int(d):02d}"

    # Case 2: Numeric format YYYYMMDD
    m = re.match(r"^(\d{4})(\d{2})(\d{2})$", text)
    if m:
        y, mm, dd = m.groups()
        return f"{y}-{mm}-{dd}"

    # Case 3: DD-MM-YYYY
    m = re.match(r"(\d{2})-(\d{2})-(\d{4})", text)
    if m:
        d, mm, y = m.groups()
        return f"{y}-{mm}-{d}"

    # Case 4: DD MON YY (2-digit year)
    m = re.match(r"(\d{1,2})[- ]([A-Z]{3})[- ](\d{2})$", text)
    if m:
        d, mon, yy = m.groups()
        if mon in MONTH_MAP:
            y = "19" + yy if int(yy) > 30 else "20" + yy
            return f"{y}-{MONTH_MAP[mon]}-{int(d):02d}"

    # Default: return raw cleaned
    return text
def fix_ocr_day_errors(text: str) -> str:
    """
    Perbaiki error OCR khusus di bagian HARI:
    OI → 01, IO → 10, II → 11
    """
    if not text:
        return text

    # Normalisasi spasi
    s = re.sub(r"\s+", " ", text.strip())

    # Pattern: HARI BULAN TAHUN
    m = re.match(r"^(OI|IO|II|\d{1,2})\s+([A-Za-z]{3})\s+(\d{4})$", s, re.IGNORECASE)
    if m:
        d, mon, y = m.groups()
        day_fix = {
            "OI": "01",
            "IO": "10",
            "II": "11"
        }.get(d.upper(), d)
        return f"{day_fix} {mon} {y}"

    return text


def postprocess_ocr_field(field, raw):
    # Bersihkan karakter aneh di awal
    raw = clean_text_basic(raw)

    if field in ("surname", "given_names", "nationality", "place_of_birth"):
        return raw.upper()
        
    if field in ("date_of_birth", "date_of_issue", "date_of_expiry"):
        return normalize_frcnn_date(raw)
        
    # [FIX] Logika Smart untuk Sex
    if field == "sex":
        val = raw.upper().strip()
        
        # Cek Perempuan (F atau P/F atau Female)
        # Kadang OCR baca 'F' jadi 'E' atau 'P' jadi 'D'
        if any(x in val for x in ["F", "P/F", "FEMALE", "WANITA", "PEREMPUAN"]):
            return "F"
            
        # Cek Laki-laki (M atau L/M atau Male)
        if any(x in val for x in ["M", "L/M", "MALE", "PRIA", "LAKI"]):
            return "M"
            
        # Fallback jika hanya 1 huruf dan mirip
        if val == "P": return "F" # P = Perempuan
        if val == "L": return "M" # L = Laki-laki
        
        return "" # Gagal deteksi

    if field == "mrz":
        return raw.upper().replace(" ", "")
        
    return raw

# =========================================================
# OCR: TrOCR UNTUK FIELD TEKS
# =========================================================

def ocr_trocr(img_bgr, max_length: int = 64) -> str:
    if img_bgr is None or img_bgr.size == 0:
        return ""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pixel_values = trocr_processor(pil_img, return_tensors="pt").pixel_values.to(OCR_DEVICE)
    with torch.no_grad():
        generated_ids = trocr_model.generate(pixel_values, max_length=max_length)
    text = trocr_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return text.strip()


# =========================================================
# OCR: MRZ (TESSERACT + NORMALIZER ICAO)
# =========================================================

def ocr_mrz(img_bgr) -> str:
    """
    OCR khusus MRZ (2 baris) menggunakan Tesseract.
    Output: raw text maks 2 baris (tanpa normalisasi format).
    """
    if img_bgr is None or img_bgr.size == 0:
        return ""

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # MRZ: teks hitam di atas latar abu-abu
    _, th = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    config = (
        "--oem 1 "
        "--psm 4 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
    )

    text = pytesseract.image_to_string(th, config=config)

    text = text.upper()
    text = re.sub(r"[^A-Z0-9<\n]", "", text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    if len(lines) >= 2:
        return "\n".join(lines[:2])

    if len(lines) == 1:
        s = lines[0].replace(" ", "")
        if len(s) >= 88:
            return s[:44] + "\n" + s[44:88]
        return s

    return ""

MRZ_FILLER_CONFUSIONS = {"C", "E", "S", "T", "R", "K", "N", "F"}

def fix_mrz_filler_errors(line: str) -> str:
    """
    Tesseract sering membaca '<' sebagai C/E/S/T/R/K.
    Fungsi ini mengembalikan semua huruf tersebut menjadi '<'
    DI AREA FILLER tempat nama sudah selesai.
    """
    fixed = []
    for ch in line:
        if ch in MRZ_FILLER_CONFUSIONS:
            fixed.append("<")
        else:
            fixed.append(ch)
    return "".join(fixed)


def normalize_mrz(raw: str) -> str:
    """
    Normalisasi MRZ ke format 2 baris × 44 karakter (ICAO 9303).
    Perbaikan:
    - Hapus noise
    - Split 2 baris
    - Fix kesalahan OCR '<' → C/E/S/T/R/K
    - Perbaikan negara USA
    - Perbaikan angka/huruf umum
    - Perapian panjang 44 char
    """
    if not raw:
        return ""

    # --- BASIC CLEAN ---
    s = raw.upper()
    s = s.replace(" ", "")
    s = re.sub(r"[^A-Z0-9<\n]", "", s)

    # Ambil baris valid
    lines = [ln for ln in s.splitlines() if ln.strip()]

    # Jika hanya ada 1 baris → coba split
    if len(lines) == 1:
        one = lines[0]
        if len(one) >= 88:
            lines = [one[:44], one[44:88]]
        else:
            lines = [one[:44], one[44:]]

    if not lines:
        return ""

    line1 = lines[0]
    line2 = lines[1] if len(lines) > 1 else ""

    # ======================================================
    # LINE 1 PROCESSING  (Nama + doc type)
    # ======================================================
    # Hapus angka (line1 tidak boleh punya digit)
    line1 = re.sub(r"[0-9]", "", line1)

    # Only A–Z and '<'
    line1 = re.sub(r"[^A-Z<]", "", line1)

    # Standarisasi separator
    line1 = re.sub(r"<{2,}", "<<", line1)

    # ---------- Perbaiki kode negara USA (OCR error) ----------
    line1 = line1.replace("U<A", "USA")
    line1 = line1.replace("U5A", "USA")
    line1 = line1.replace("U<SA", "USA")
    line1 = line1.replace("<USA", "USA")     # fallback
    line1 = line1.replace("U<A", "USA")      # double fallback

    # Perbaiki prefix dokumen US
    if line1.startswith("PCUSA"):    # "<" dibaca "C"
        line1 = "P<USA" + line1[5:]
    elif line1.startswith("P<USA"):
        pass

    # Panjangkan ke 44 karakter
    line1 = line1.ljust(44, "<")[:44]

    # FIX KESALAHAN OCR UNTUK FILLER '<'
    line1 = fix_mrz_filler_errors(line1)

    # ======================================================
    # LINE 2 PROCESSING  (Nomor paspor, negara, sex, DOB)
    # ======================================================
    line2 = re.sub(r"[^A-Z0-9<]", "", line2)

    # Common OCR confusion:
    line2 = (line2
             .replace("O", "0")
             .replace("I", "1")
             .replace("S", "5")
             .replace("B", "8"))

    # ---------- Perbaiki kode negara line2 ----------
    line2 = line2.replace("U5A", "USA")
    line2 = line2.replace("U<A", "USA")
    line2 = line2.replace("USA0", "USA")   # OCR salah baca 0

    # Panjang wajib 44
    line2 = line2.ljust(44, "<")[:44]

    # FIX filler error juga di line2
    line2 = fix_mrz_filler_errors(line2)

    # ======================================================
    # FINAL
    # ======================================================
    return f"{line1}\n{line2}"




def parse_mrz_names(mrz_clean: str):
    """
    Ekstrak surname & given_names dari MRZ line 1 yang sudah dinormalisasi.
    Format: P<CCCSURNAME<<GIVEN<<... 
    """
    if not mrz_clean or "\n" not in mrz_clean:
        return "", ""
    line1 = mrz_clean.splitlines()[0]
    if len(line1) < 6:
        return "", ""
    body = line1[5:]  # setelah 'P<CC'
    parts = body.split("<<")
    surname = parts[0].replace("<", "").strip()
    given = ""
    if len(parts) > 1:
        given = parts[1].replace("<", " ").strip()
    surname = normalize_name(surname)
    given = normalize_name(given)
    return surname, given


def parse_mrz_for_sex(mrz_clean: str) -> str:
    """
    Ambil gender (M/F) dari line 2 posisi ke-21 (indeks 20) sesuai ICAO.
    """
    if not mrz_clean or "\n" not in mrz_clean:
        return ""
    line2 = mrz_clean.splitlines()[1]
    if len(line2) < 21:
        return ""
    c = line2[20]
    return c if c in ("M", "F") else ""


# =========================================================
# Faster R-CNN MODEL
# =========================================================

def build_frcnn(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    return model


def load_model(weights_path, device):
    num_classes = len(LABELS) + 1
    model = build_frcnn(num_classes)
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    elif isinstance(state, dict):
        model.load_state_dict(state)
    else:
        model = state
    model.to(device)
    model.eval()
    print("[INFO] Faster R-CNN model loaded.")
    return model


# =========================================================
# PIPELINE: SINGLE IMAGE
# =========================================================

# =========================================================
# FUNGSI run_single (DIPERBAIKI TIMER & SAVE JSON)
# =========================================================

def run_single(model, device, img_path, out_dir, conf, reader=None, save_visuals=False):
    # [FIX 1] MULAI TIMER DI SINI (PALING ATAS)
    # Agar menghitung waktu total dari load gambar sampai selesai OCR
    start_time = time.time()
    
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Gambar & Preprocessing
    pil = Image.open(img_path).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    img_proc = global_preprocess_for_frcnn(img_bgr)
    vis_img = img_proc.copy()

    # 2. Inferensi Model Faster R-CNN
    pil_proc = Image.fromarray(cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB))
    tensor = T.ToTensor()(pil_proc).to(device).unsqueeze(0)

    with torch.no_grad():
        pred = model(tensor)[0]

    boxes = pred["boxes"].cpu().numpy()
    labels_pred = pred["labels"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()

    detections = {}
    H, W = img_proc.shape[:2]

    # Inisialisasi JSON Output
    json_output = {
        "image": str(img_path),
        "raw": {},
        "cleaned": {},
        "confidence": {},
        "paths": {},
        "debug": { "boxes": [] },
        "photo_detected": False,
        "speed": 0.0  # Placeholder, nanti diupdate di bawah
    }

    # 3. LOOP DETEKSI & CROP
    for box, lab, score in zip(boxes, labels_pred, scores):
        cls = int(lab) - 1
        if cls < 0 or cls >= len(LABELS): continue

        label = LABELS[cls]
        if float(score) < min_conf_for_label(label): continue

        # Ambil skor tertinggi jika duplikat
        if label in detections and score < detections[label]["score"]: continue

        x1, y1, x2, y2 = [int(x) for x in box]
        pt, pb, pl, pr = get_padding_for_label(label)

        if label == "mrz": x1, x2 = 0, W

        x1p = clamp(x1 - pl, 0, W - 1)
        y1p = clamp(y1 - pt, 0, H - 1)
        x2p = clamp(x2 + pr, 0, W)
        y2p = clamp(y2 + pb, 0, H)

        # Crop & Auto Tight
        crop = img_proc[y1p:y2p, x1p:x2p].copy()
        if label not in ("sex", "date_of_birth", "date_of_issue", "date_of_expiry", "place_of_birth", "surname", "given_names", "nationality"):
             crop = auto_tight_crop(crop)

        # Simpan Gambar Crop
        crop_filename = f"{label}.png"
        crop_path = os.path.join(out_dir, crop_filename)
        cv2.imwrite(crop_path, crop)

        # Simpan Data Deteksi
        detections[label] = {
            "bbox": [int(x1p), int(y1p), int(x2p), int(y2p)],
            "score": float(score),
            "crop_path": crop_path,
            "crop_rel_path": crop_path
        }
        
        # Update JSON Output
        json_output["confidence"][label] = float(score)
        json_output["paths"][label] = str(crop_path)
        json_output["debug"]["boxes"].append({
            "class": label,
            "box": [float(x1p), float(y1p), float(x2p), float(y2p)],
            "conf": float(score)
        })

        if label == "photo": json_output["photo_detected"] = True

        if save_visuals:
            cv2.rectangle(vis_img, (x1p, y1p), (x2p, y2p), (0, 255, 0), 2)
            cv2.putText(vis_img, f"{label} {score:.2f}", (x1p, max(0, y1p - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if save_visuals:
        cv2.imwrite(os.path.join(out_dir, "preproc_vis.png"), vis_img)

    # 4. OCR PROCESSING
    ocr_raw = {}
    ocr_norm = {}

    # -- MRZ Processing --
    mrz_raw, mrz_clean, mrz_surname, mrz_given, mrz_sex = "", "", "", "", ""
    if "mrz" in detections:
        try:
            m_img = cv2.imread(detections["mrz"]["crop_path"])
            mrz_raw = ocr_mrz(m_img)
            mrz_clean = normalize_mrz(mrz_raw)
            mrz_surname, mrz_given = parse_mrz_names(mrz_clean)
            mrz_sex = parse_mrz_for_sex(mrz_clean or mrz_raw)
        except Exception: pass

    # -- Loop Per Label (OCR) --
    for label, info in detections.items():
        crop_path = info["crop_path"]
        img_crop = cv2.imread(crop_path)
        if img_crop is None: continue

        # Logic per field (Photo, MRZ, Sex, Text Lain)
        if label == "photo":
            ocr_raw[label] = "<PHOTO_DETECTED>"
            ocr_norm[label] = "<PHOTO_DETECTED>"
        elif label == "mrz":
            ocr_raw[label] = mrz_clean or mrz_raw
            ocr_norm[label] = mrz_clean or mrz_raw
        elif label == "sex":
            if mrz_sex:
                raw, norm = mrz_sex, mrz_sex
            else:
                raw = ocr_trocr(img_crop)
                val = raw.upper().strip()
                if "F" in val or "P" in val: norm = "F"
                elif "M" in val: norm = "M"
                else: norm = "" 
            ocr_raw[label] = raw
            ocr_norm[label] = norm
        else:
            raw = ocr_trocr(img_crop)
            ocr_raw[label] = raw
            ocr_norm[label] = postprocess_ocr_field(label, raw)

        # Update JSON Realtime
        json_output["raw"][label] = ocr_raw[label]
        json_output["cleaned"][label] = ocr_norm[label]

    # -- Override Nama dari MRZ --
    if mrz_surname and (len(ocr_norm.get("surname", "")) < 3):
        ocr_norm["surname"] = mrz_surname
        ocr_raw["surname"] = mrz_surname
        json_output["raw"]["surname"] = mrz_surname
        json_output["cleaned"]["surname"] = mrz_surname
        
    if mrz_given and (len(ocr_norm.get("given_names", "")) < 3):
        ocr_norm["given_names"] = mrz_given
        ocr_raw["given_names"] = mrz_given
        json_output["raw"]["given_names"] = mrz_given
        json_output["cleaned"]["given_names"] = mrz_given
        
    if mrz_sex:
        ocr_norm["sex"] = mrz_sex
        json_output["cleaned"]["sex"] = mrz_sex

    # ========================================================
    # [FIX 2] HITUNG SPEED & SIMPAN KE JSON (BAGIAN KRUSIAL)
    # ========================================================
    end_time = time.time()
    inference_time = end_time - start_time
    
    # Masukkan ke dalam dictionary JSON SEBELUM didump ke file
    json_output["speed"] = round(inference_time, 4)
    
    # Hitung FPS (Hanya untuk log print)
    fps = 1.0 / inference_time if inference_time > 0 else 0.0
    print(f"🐢 FRCNN Speed: {inference_time:.4f} detik ({fps:.2f} FPS)")

    # Simpan file result.json
    out_json_path = os.path.join(out_dir, "result.json")
    with open(out_json_path, "w", encoding='utf-8') as f:
        json.dump(json_output, f, indent=4)

    # Simpan file ocr_result.json (Backward Compatibility)
    old_json_path = os.path.join(out_dir, "ocr_result.json")
    simple_json = json_output["cleaned"].copy()
    simple_json["speed"] = round(inference_time, 4) # Tambahkan speed juga di sini
    with open(old_json_path, "w", encoding='utf-8') as f:
        json.dump(simple_json, f, indent=4)

    return detections, ocr_raw, ocr_norm

# =========================================================
# BATCH PROCESS
# =========================================================

def run_batch(images_dir, weights, out_root, conf, save_visuals=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    model = load_model(weights, device)
    print("[INFO] OCR: Tesseract (MRZ) + TrOCR (fields).")

    os.makedirs(out_root, exist_ok=True)

    imgs = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg")) +
        glob.glob(os.path.join(images_dir, "*.jpeg")) +
        glob.glob(os.path.join(images_dir, "*.png"))
    )

    for p in imgs:
        name = Path(p).stem
        print(f"[INFO] Processing {name} ...")
        out_dir = os.path.join(out_root, name)
        try:
            run_single(model, device, p, out_dir, conf, reader=None,
                       save_visuals=save_visuals)
        except Exception as e:
            print(f"[ERROR] Failed processing {name}: {e}")
            traceback.print_exc()
            continue

    print("[INFO] ALL DONE.")


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--save_visuals", action="store_true")
    args = parser.parse_args()

    run_batch(
        args.images_dir,
        args.model,
        args.out_root,
        args.conf,
        args.save_visuals,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[FATAL ERROR]", e)
        traceback.print_exc()


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
