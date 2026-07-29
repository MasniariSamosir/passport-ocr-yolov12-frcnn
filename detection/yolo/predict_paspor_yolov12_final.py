#!/usr/bin/env python3
"""
predict_paspor_yolov12_final.py
BESTI EDITION — FINAL (With Speed Timer & process_image Fix)
"""
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import logging
import re
import cv2
import numpy as np
import sys
import time  # [PENTING] Import time untuk stopwatch

# Pastikan path root project terbaca
sys.path.append(os.getcwd())

from ultralytics import YOLO

# optional tesseract fallback for MRZ
try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    import easyocr
except Exception:
    easyocr = None

# Import preprocessing (sesuaikan path jika perlu)
try:
    from detection.yolo.preprocessing import (
        ensure_dir,
        expand_box_safe,
        crop_from_box,
        auto_tight_crop,
        generic_preprocess_for_text,
        normalize_mrz_text,
    )
except ImportError:
    # Fallback import jika dijalankan dari folder root
    from preprocessing import (
        ensure_dir,
        expand_box_safe,
        crop_from_box,
        auto_tight_crop,
        generic_preprocess_for_text,
        normalize_mrz_text,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("PASPOR_YOLO_FINAL")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_sex_from_mrz(mrz_text):
    if not mrz_text: return ""
    if len(mrz_text) >= 88:
        line2 = mrz_text[44:88]
        if len(line2) > 20:
            sx = line2[20]
            if sx in ("M", "F"): return sx
    return ""

def fix_mrz_ocr_errors(mrz: str) -> str:
    if not mrz: return mrz
    mrz = mrz.upper().replace("«", "<").replace(" ", "")
    mrz = re.sub(r"[^A-Z0-9<]", "", mrz)
    mrz = mrz.replace("U5A", "USA").replace("UA", "USA")
    mrz = re.sub(r"P<UA", "P<USA", mrz)
    return mrz

def parse_date_freeform(text):
    if not text: return None
    t = re.sub(r"[^A-Za-z0-9 ]", " ", text)
    t = re.sub(r"\s+", " ", t).strip()
    for fmt in ("%d %b %Y", "%d %B %Y", "%Y-%m-%d", "%d %m %Y", "%d %b %y", "%d %m %y"):
        try:
            dt = datetime.strptime(t, fmt)
            if dt.year < 100:
                dt = dt.replace(year=(1900 + dt.year) if dt.year > 30 else (2000 + dt.year))
            return dt.strftime("%Y-%m-%d")
        except:
            pass
    return None

def format_date_ground_truth(text):
    """
    Normalisasi tanggal ke format Ground Truth:
    DD Mon YYYY (contoh: 10 Mar 1981)
    """
    if not text:
        return None

    # Perbaiki kesalahan OCR umum (I → 1)
    t = fix_date_leading_I(text)

    # Bersihkan karakter aneh
    t = re.sub(r"[^A-Za-z0-9 ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    for fmt in (
        "%d %b %Y",   # 10 Mar 1981
        "%d %B %Y",   # 10 March 1981
        "%Y-%m-%d",   # 1981-03-10
        "%d-%m-%Y",   # 10-03-1981
        "%d %m %Y",   # 10 03 1981
        "%d %b %y",   # 10 Mar 81
        "%d %m %y",   # 10 03 81
    ):
        try:
            dt = datetime.strptime(t, fmt)

            # Normalisasi tahun 2 digit
            if dt.year < 100:
                dt = dt.replace(
                    year=(1900 + dt.year) if dt.year > 30 else (2000 + dt.year)
                )

            # OUTPUT FINAL SESUAI GROUND TRUTH
            return dt.strftime("%d %b %Y")

        except Exception:
            continue

    return None

def fix_month_capitalization(date_str):
    """
    Mengubah bulan dari ALL CAPS menjadi Title Case
    Contoh: '14 JAN 2016' -> '14 Jan 2016'
    """
    if not date_str:
        return date_str

    MONTHS = {
        "JAN": "Jan", "FEB": "Feb", "MAR": "Mar",
        "APR": "Apr", "MAY": "May", "JUN": "Jun",
        "JUL": "Jul", "AUG": "Aug", "SEP": "Sep",
        "OCT": "Oct", "NOV": "Nov", "DEC": "Dec"
    }

    parts = date_str.split()
    if len(parts) == 3 and parts[1].upper() in MONTHS:
        parts[1] = MONTHS[parts[1].upper()]
        return " ".join(parts)

    return date_str


def clean_passport_number(s):
    if not s: return None
    return re.sub(r"[^A-Z0-9]", "", s.upper())

def init_ocr(gpu=False):
    if easyocr is None:
        log.warning("easyocr unavailable")
        return None
    try:
        return easyocr.Reader(["en"], gpu=gpu)
    except:
        return easyocr.Reader(["en"], gpu=False)

def ocr_read(reader, img):
    if reader is None: return "", 0.0
    try:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = reader.readtext(rgb, detail=1)
        if not res: return "", 0.0
        texts = [r[1] for r in res]; confs = [float(r[2]) for r in res]
        return " ".join(texts), float(np.mean(confs)) if confs else 0.0
    except Exception as e:
        log.debug(f"ocr_read fail: {e}")
        return "", 0.0

def ocr_multi(reader, crop):
    if crop is None or crop.size == 0: return "", 0.0
    best_txt, best_conf = ocr_read(reader, crop)
    return best_txt, best_conf

def ocr_mrz_safe(reader, crop):
    try:
        if crop is None or crop.size == 0: return "", 0.0
        gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        res = reader.readtext(cv2.cvtColor(thr, cv2.COLOR_GRAY2RGB), detail=1)
        if not res: return "", 0.0
        txt = "".join([r[1] for r in res])
        txt = re.sub(r"[^A-Z0-9<]", "", txt.upper().replace(" ", ""))
        confs = [r[2] for r in res]
        conf = float(sum(confs) / len(confs)) if confs else 0.0
        return txt, conf
    except Exception:
        return "", 0.0

def iou_np(a,b):
    xa1,ya1,xa2,ya2 = a
    xb1,yb1,xb2,yb2 = b
    inter = max(0, min(xa2,xb2)-max(xa1,xb1)) * max(0, min(ya2,yb2)-max(ya1,yb1))
    areaA = max(0,(xa2-xa1)) * max(0,(ya2-ya1))
    areaB = max(0,(xb2-xb1)) * max(0,(yb2-yb1))
    denom = areaA + areaB - inter
    return inter/denom if denom>0 else 0

def nms_numpy(boxes, scores, thr=0.45):
    if not boxes: return []
    boxes = np.array(boxes); scores=np.array(scores)
    idxs = scores.argsort()[::-1]
    keep=[]
    while len(idxs):
        cur=idxs[0]; keep.append(int(cur))
        if len(idxs)==1: break
        rest=idxs[1:]
        ious = np.array([iou_np(boxes[cur], boxes[i]) for i in rest])
        idxs = rest[ious <= thr]
    return keep

def resolve_cross_class_overlaps(final_list, overlap_thr=0.6):
    out = []
    used = set()
    for i,a in enumerate(final_list):
        if i in used: continue
        ax = a["box"]; ac = a["conf"]
        keep = a
        for j,b in enumerate(final_list):
            if j==i or j in used: continue
            bx = b["box"]; bc = b["conf"]
            if iou_np(ax,bx) > overlap_thr:
                if (b["class"]=="photo" and a["class"]!="photo"): keep = b
                elif bc > ac: keep = b
        out.append(keep)
        for j,b in enumerate(final_list):
            if j in used: continue
            if iou_np(keep["box"], b["box"]) > overlap_thr: used.add(j)
    
    byclass = {}
    for d in out:
        k = canonical_name(d["class"])
        if k not in byclass or d["conf"] > byclass[k]["conf"]:
            byclass[k] = d
    return list(byclass.values())

PER_CLASS_SCALE = {
    "passport_number": 1.5, "given_names": 1.4, "surname": 1.4,
    "date_of_birth": 1.35, "date_of_issue": 1.35, "date_of_expiry": 1.35,
    "mrz": 1.28, "nationality": 1.25, "place_of_birth": 1.25, 
    "sex": 1.8, 
    "photo": 1.05
}

PER_CLASS_NMS = {"mrz": 0.25, "photo": 0.5, "sex": 0.3}
ALIASES = {"name":"given_names"}

def canonical_name(cls):
    cls = cls.lower().strip().replace(" ","_")
    return ALIASES.get(cls, cls)

def fix_date_leading_I(text):
    if not text: return text
    text = re.sub(r"\bI(\d)", r"1\1", text)
    return text.replace("I0","10").replace("IO","10").replace("I1","11").replace("I2","12")

def force_month_title_case(text):
    """
    Paksa penulisan bulan menjadi Title Case
    Contoh:
    'JAN 2001' -> 'Jan 2001'
    '03 MAR 2011' -> '03 Mar 2011'
    """
    if not text:
        return text

    MONTHS = {
        "JAN": "Jan", "FEB": "Feb", "MAR": "Mar",
        "APR": "Apr", "MAY": "May", "JUN": "Jun",
        "JUL": "Jul", "AUG": "Aug", "SEP": "Sep",
        "OCT": "Oct", "NOV": "Nov", "DEC": "Dec"
    }

    parts = text.strip().split()
    for i, p in enumerate(parts):
        up = p.upper()
        if up in MONTHS:
            parts[i] = MONTHS[up]

    return " ".join(parts)

# ============================================================
# [FIXED] PROCESS IMAGE FUNCTION (WITH DEFINITION & TIMER)
# ============================================================
def process_image(path, model, outdir, conf_th, save_crops, reader):
    # [START TIMER]
    start_time = time.time()

    log.info(f"Processing {path}")
    img = cv2.imread(str(path))
    if img is None: 
        log.warning(f"Could not read {path}")
        return None
    H,W = img.shape[:2]

    vis_img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    # 1. YOLO INFERENCE
    try:
        res = model(img, verbose=False)[0]
    except Exception as e:
        log.exception(f"Model infer failed: {e}")
        return None

    boxes=[]
    if getattr(res,"boxes",None) is not None:
        xy = res.boxes.xyxy.cpu().numpy()
        cfs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        names = getattr(model,"names",None)
        for box,conf,cid in zip(xy,cfs,clss):
            if conf < conf_th: continue
            cname = names[int(cid)] if names else str(cid)
            boxes.append({"class":cname,"box":[float(x) for x in box],"conf":float(conf)})

    # NMS & Grouping
    grouped={}
    for b in boxes:
        fld = canonical_name(b["class"])
        grouped.setdefault(fld,[]).append(b)

    final=[]
    for fld, arr in grouped.items():
        bxs=[a["box"] for a in arr]; scs=[a["conf"] for a in arr]
        thr = PER_CLASS_NMS.get(fld, 0.45)
        keep = nms_numpy(bxs, scs, thr=thr)
        if not keep and arr:
            arr.sort(key=lambda x:x["conf"], reverse=True)
            final.append(arr[0])
        else:
            kept = [arr[i] for i in keep]
            if kept:
                kept.sort(key=lambda x:x["conf"], reverse=True)
                final.append(kept[0])

    final = resolve_cross_class_overlaps(final, overlap_thr=0.6)

    # Draw Vis
    vis = vis_img.copy()
    for d in final:
        x1,y1,x2,y2 = map(int,d["box"])
        cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(vis,f"{d['class']}:{d['conf']:.2f}",(x1,y1-6), cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),1)

    vis_dir = outdir/"preproc_vis"; ensure_dir(vis_dir)
    vis_path = vis_dir/f"{path.stem}_preproc_vis.png"
    cv2.imwrite(str(vis_path),vis)

    js = {"image":str(path),"raw":{},"cleaned":{},"confidence":{},"paths":{},"debug":{"boxes":final}}

    # ---------------------------------------------------------
    # 2. CROP & OCR & SAVING
    # ---------------------------------------------------------
    for det in final:
        fld = canonical_name(det["class"])
        x1,y1,x2,y2 = det["box"]
        
        # Atur Scale
        scale = PER_CLASS_SCALE.get(fld, 1.15)
        
        ex = expand_box_safe([x1,y1,x2,y2], W, H, scale)
        x1e,y1e,x2e,y2e = map(int,ex)
        crop = crop_from_box(img, [x1e,y1e,x2e,y2e])

        # Safety check size
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            crop = crop_from_box(img, [int(x1), int(y1), int(x2), int(y2)])

        # ========================================================
        # [FIX BESTI] VISUAL PREPROCESSING (HITAM PUTIH UNTUK SEMUA FIELD TEKS)
        # ========================================================
        vis_crop = crop.copy()

        # Terapkan preprocessing ke SEMUA field teks.
        # KECUALI:
        # - 'photo': Agar tetap berwarna.
        # - 'sex': Karena 'sex' punya logika preprocessing + saving khusus di bawah nanti.
        if fld != "photo" and fld != "sex":
            try:
                # 1. Robust Grayscale Conversion (Cegah error channel OpenCV)
                if len(vis_crop.shape) == 3 and vis_crop.shape[2] == 3:
                    gray_vis = cv2.cvtColor(vis_crop, cv2.COLOR_BGR2GRAY)
                elif len(vis_crop.shape) == 3 and vis_crop.shape[2] == 1:
                    gray_vis = vis_crop[:, :, 0]
                else:
                    gray_vis = vis_crop

                # 2. Perbaiki Kontras (CLAHE)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced_vis = clahe.apply(gray_vis)

                # 3. Binarization (Otsu) -> Mengubah jadi HITAM PUTIH PEKAT
                # Inilah yang membuat hasil crop jadi bersih.
                _, vis_crop = cv2.threshold(enhanced_vis, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            except Exception as e:
                # Jika preprocessing gagal, baru fallback ke crop asli
                # (Harusnya jarang terjadi dengan robust grayscale di atas)
                vis_crop = crop.copy() 

        # SIMPAN GAMBAR
        # Untuk field teks biasa (nama, tanggal, mrz), 'vis_crop' sudah hitam putih.
        # Untuk 'photo', masih original.
        # Untuk 'sex', nanti akan ditimpa oleh logika khususnya di bawah.
        pd = outdir / "preprocessed_crops" / path.stem
        ensure_dir(pd)

        out_crop = pd / f"{fld}_pre.png"
        ok = cv2.imwrite(str(out_crop), vis_crop)

        print(f"[SAVE CROP] {fld} -> {out_crop} | status={ok}")




        # Logic per Class
        if fld == "photo":
            js["raw"][fld] = "PHOTO_DETECTED"
            js["cleaned"][fld] = "PHOTO_DETECTED"
            js["confidence"][fld] = 1.0
            js["photo_detected"] = True
            continue

        if fld == "mrz":
            mrz_txt, mrz_conf = ocr_mrz_safe(reader, crop)
            norm = normalize_mrz_text(mrz_txt)
            norm_fixed = fix_mrz_ocr_errors(norm)
            mrz_clean = norm_fixed.replace(" ", "").replace("\n", "")
            if len(mrz_clean) >= 88: mrz_clean = mrz_clean[:44] + "\n" + mrz_clean[44:88]
            mrz_fixed = fix_mrz_ocr_errors(mrz_clean)
            js["raw"]["mrz"] = mrz_fixed
            js["cleaned"]["mrz"] = mrz_fixed
            js["confidence"]["mrz"] = float(mrz_conf)
            continue

        # SEX - ENHANCED LOGIC (Support L/M & P/F)
        if fld == "sex":
            try:
                # Preprocessing khusus SEX (Binarization)
                gray_s = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                _, th = cv2.threshold(gray_s, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                
                # Baca OCR
                sex_txt, sex_conf = ocr_multi(reader, cv2.cvtColor(th, cv2.COLOR_GRAY2BGR))
                sex_txt = sex_txt.strip().upper()
                
                # Logic Mapping: L/M, P/F -> M/F
                final_sex = ""
                if "L" in sex_txt or "M" in sex_txt:
                    final_sex = "M"
                elif "P" in sex_txt or "F" in sex_txt:
                    final_sex = "F"
                elif "K" in sex_txt: # Kadang "LAKI" terbaca "K"
                    final_sex = "M"
                
                if final_sex:
                    js["raw"]["sex"] = sex_txt
                    js["cleaned"]["sex"] = final_sex
                    js["confidence"]["sex"] = float(sex_conf)
                else:
                    # Fallback kosong (nanti diisi MRZ)
                    js["raw"]["sex"] = sex_txt
                    js["confidence"]["sex"] = float(sex_conf)

                if save_crops:
                    pd = outdir/"preprocessed_crops"/path.stem; ensure_dir(pd)
                    cv2.imwrite(str(pd/"sex_pre.png"), th)
                    js["paths"][fld] = str(pd/"sex_pre.png")
            except Exception as e:
                js["raw"]["sex"] = ""; js["confidence"]["sex"] = 0.0
            continue

        # Text Fields
        text, conf = ocr_multi(reader, crop)
        text = text.strip()
        # Simpan raw
        if fld.startswith("date"):
            js["raw"][fld] = force_month_title_case(text)
        else:
            js["raw"][fld] = text

        js["confidence"][fld] = float(conf)

        
        if fld.startswith("date"):
            # 1. Coba normalisasi tanggal ke format Ground Truth
            gt_date = format_date_ground_truth(text)

            if gt_date:
                # Jika parsing sukses
                js["cleaned"][fld] = force_month_title_case(gt_date)
            else:
                # Jika parsing gagal (JAN 2001, dll)
                js["cleaned"][fld] = force_month_title_case(text)

        elif fld == "passport_number":
            js["cleaned"][fld] = clean_passport_number(text)

        else:
            js["cleaned"][fld] = text


    # Fallback Sex from MRZ
    if "raw" in js and "mrz" in js["raw"]:
        s = extract_sex_from_mrz(js["raw"]["mrz"])
        if s:
            js["raw"]["sex"] = s; js["cleaned"]["sex"] = s; js["confidence"]["sex"] = 1.0

    # [END TIMER]
    end_time = time.time()
    inference_time = end_time - start_time
    
    # Save Speed
    js["speed"] = round(inference_time, 4)
    print(f"🚀 Kecepatan Total (YOLO+OCR): {inference_time:.4f} detik")

    # Save JSON
    out = outdir/f"{path.stem}_ocr_final.json"
    with open(out,"w",encoding="utf-8") as f:
        json.dump(js,f,indent=4,ensure_ascii=False)
    log.info(f"SAVED JSON -> {out}")
    return js


# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--save_crops", action="store_true")
    ap.add_argument("--gpu_ocr", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.out); ensure_dir(outdir)
    log.info("Loading YOLO model...")
    model = YOLO(args.model)
    log.info("Model loaded.")

    reader = init_ocr(args.gpu_ocr)
    src = Path(args.source)
    imgs=[]
    if src.is_dir():
        imgs = sorted([p for p in src.glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".tif",".tiff"]])
    else:
        imgs = [src]
    if not imgs:
        log.warning("No images found!")
        return
    for p in imgs:
        try:
            process_image(p, model, outdir, args.conf, args.save_crops, reader)
        except Exception as e:
            log.exception(f"Error processing {p}: {e}")

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        print("[FATAL ERROR]", e)