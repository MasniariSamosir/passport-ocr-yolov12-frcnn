#!/usr/bin/env python3
"""
app.py — Paspor OCR AI (YOLOv12 + Faster R-CNN)
FINAL VERSION — FIXED:
✓ YOLO → UI mapping lengkap
✓ MRZ muncul di UI
✓ DATE OF ISSUE muncul
✓ Import YOLO tidak error
✓ Format output YOLO sama seperti FRCNN
"""

import os
import re
import traceback
from datetime import datetime
from pathlib import Path
from werkzeug.utils import secure_filename

import torch
import easyocr
import json
from types import SimpleNamespace
from flask import send_file
from fpdf import FPDF
import hashlib
from flask import session

import mysql.connector
from mysql.connector import Error
from datetime import datetime
from PIL import Image
from metrics_ocr import compute_metrics_for_passport 
from dataset_evaluation import compute_speed_metrics
from flask import (
    Flask, render_template, request,
    redirect, url_for, flash,
    session, send_file
)

import secrets
from datetime import timedelta


# ============================================================
# KONFIGURASI TAMPILAN UI (Urutan Kolom)
# ============================================================
UI_FIELDS_ORDER = [
    {"key": "mrz", "label": "MRZ Code"},
    {"key": "surname", "label": "Surname"},
    {"key": "given_names", "label": "Given Names"},
    {"key": "passport_number", "label": "Passport Number"},
    {"key": "nationality", "label": "Nationality"},
    {"key": "date_of_birth", "label": "Date of Birth"},
    {"key": "date_of_issue", "label": "Date of Issue"},
    {"key": "date_of_expiry", "label": "Date of Expiry"},
    {"key": "sex", "label": "Sex"},
    {"key": "place_of_birth", "label": "Place of Birth"},
    # "photo" kita handle terpisah atau di akhir jika mau
]


app = Flask(__name__)
app.secret_key = "super_secret_key_passport_ai"

@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        username = request.form.get("username").strip()

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
        user = cursor.fetchone()

        if not user:
            cursor.close()
            conn.close()
            return render_template(
                "forgot_password.html",
                error="Username tidak ditemukan."
            )

        # Generate token
        token = secrets.token_hex(32)
        expired = datetime.now() + timedelta(minutes=15)

        cursor.execute(
            "UPDATE users SET reset_token=%s, reset_expired=%s WHERE id=%s",
            (token, expired, user["id"])
        )
        conn.commit()
        cursor.close()
        conn.close()

        return redirect(url_for("reset_password", token=token))

    return render_template("forgot_password.html")

@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute(
        "SELECT * FROM users WHERE reset_token=%s AND reset_expired > NOW()",
        (token,)
    )
    user = cursor.fetchone()

    if not user:
        cursor.close()
        conn.close()
        return "Token tidak valid atau sudah kedaluwarsa", 403

    if request.method == "POST":
        new_password = request.form.get("password").strip()
        hashed = hashlib.sha256(new_password.encode()).hexdigest()

        cursor.execute(
            "UPDATE users SET password=%s, reset_token=NULL, reset_expired=NULL WHERE id=%s",
            (hashed, user["id"])
        )
        conn.commit()
        cursor.close()
        conn.close()

        flash("Password berhasil direset. Silakan login.")
        return redirect(url_for("login"))

    cursor.close()
    conn.close()
    return render_template("reset_password.html")

@app.route("/")
def home():
    return redirect(url_for("login"))


@app.before_request
def require_login():
    # Endpoint yang BOLEH diakses tanpa login
    allowed_routes = (
        "login",
        "forgot_password",
        "reset_password",
        "static",
    )

    if request.endpoint in allowed_routes:
        return

    if request.endpoint and request.endpoint.startswith("static"):
        return

    if "user_id" not in session:
        return redirect(url_for("login"))


# ---------------------------------------------------------------------
# FIX IMPORT ERROR (Pylance)
# Allow import detection.yolo.predict_paspor_yolov12_final
# ---------------------------------------------------------------------
import sys
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def load_ground_truth(passport_id_or_filename: str):
    import json, os
    base = "ground_truth"
    fname = os.path.splitext(passport_id_or_filename)[0] + ".json"
    path = os.path.join(base, fname)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compress_image(path, max_width=900, quality=70):
    try:
        img = Image.open(path)
        img = img.convert("RGB")

        if img.width > max_width:
            ratio = max_width / img.width
            new_size = (max_width, int(img.height * ratio))
            img = img.resize(new_size)

        img.save(path, "JPEG", optimize=True, quality=quality)
    except Exception as e:
        print("Image compression failed:", e)


def fix_date_format(date_str):
    """
    Convert OCR output (24-NOV-1987) → DB format (1987-11-24)
    """
    if not date_str:
        return None

    s = date_str.strip().replace(".", "-").replace("/", "-").upper()

    fmts = [
        "%d-%b-%Y",   # 23-MAR-1990
        "%d-%B-%Y",   # 23-MARCH-1990
        "%Y-%m-%d",   # 1990-03-23
        "%d-%m-%Y",   # 23-03-1990
    ]

    for fmt in fmts:
        try:
            d = datetime.strptime(s, fmt)
            return d.strftime("%Y-%m-%d")  # format MySQL compatible
        except:
            pass

    return None


# ============================================================
# SAFE DATE PARSER (Tambahan agar tanggal YOLO tidak hilang)
# ============================================================
def fix_date_safe(s):
    """
    Wrapper aman untuk parsing tanggal:
    - Jika sudah format YYYY-MM-DD → langsung dipakai
    - Jika tidak cocok format standar, fallback ke nilai asli
    - Tidak mengembalikan None supaya UI tidak kosong
    """
    if not s:
        return ""

    s = str(s).strip()

    # Jika sudah format yyyy-mm-dd (contoh: 2006-10-21)
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s

    # Pakai parser lama
    out = fix_date_format(s)

    # Jika gagal parse → kembalikan string asli
    return out if out else s


def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Root123!",   # ← isi password root kamu di sini
            database="paspor_ocr_db"
        )
        return conn
    except Error as e:
        print("Error connecting to MySQL:", e)
        return None

# ============================================================
# UNIVERSAL DB QUERY FUNCTION — untuk pagination / records
# ============================================================
def query_db(query, args=(), one=False):
    conn = get_db_connection()
    if conn is None:
        print("DB connection failed in query_db()")
        return None

    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute(query, args)
        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        if one:
            return rows[0] if rows else None
        return rows

    except Error as e:
        print("query_db Error:", e)
        cursor.close()
        conn.close()
        return None


def save_ocr_to_db(model_name, ocr_data, raw_json_text, photo_path):
    conn = get_db_connection()
    if not conn:
        print("Database connection failed.")
        return

    try:
        cursor = conn.cursor()

        sql = """
        INSERT INTO passport_ocr 
        (model, mrz, surname, given_names, passport_number, nationality,
        place_of_birth, sex, date_of_birth, date_of_issue, date_of_expiry,
        speed, fps, raw_json, photo_path)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        values = (
            model_name,
            ocr_data.get("mrz"),
            ocr_data.get("surname"),
            ocr_data.get("given_names"),
            ocr_data.get("passport_number"),
            ocr_data.get("nationality"),
            ocr_data.get("place_of_birth"),
            ocr_data.get("sex"),

            fix_date_format(ocr_data.get("date_of_birth")),
            fix_date_format(ocr_data.get("date_of_issue")),
            fix_date_format(ocr_data.get("date_of_expiry")),

            ocr_data.get("speed", 0),   # ⬅️ SIMPAN SPEED
            ocr_data.get("fps", 0),     # ⬅️ SIMPAN FPS

            raw_json_text,
            photo_path
        )



        cursor.execute(sql, values)
        conn.commit()
        cursor.close()
        conn.close()
        print("OCR data saved to database successfully.")

    except Error as e:
        print("Error:", e)

# ============================
# IMPORT YOLO & FRCNN
# ============================
try:
    from detection.yolo.predict_paspor_yolov12_final import (
        process_image as yolo_process_image,
        init_ocr as yolo_init_ocr,
    )
except Exception as e:
    print("YOLO Import Error:", e)
    yolo_process_image = None
    yolo_init_ocr = None

try:
    from detection.frcnn.frcnn_paspor_ocr import (
        run_single as frcnn_run_single,
        load_model as frcnn_load_model,
        DEFAULT_CONF as FRCNN_DEFAULT_CONF,
    )
except Exception as e:
    print("FRCNN Import Error:", e)
    frcnn_run_single = None
    frcnn_load_model = None
    FRCNN_DEFAULT_CONF = 0.40


# ============================
# CONFIG
# ============================
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

YOLO_MODEL_PATH = BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
FRCNN_MODEL_PATH = BASE_DIR / "detection" / "frcnn" / "faster_rcnn_paspor_final.pth"

YOLO_OUT_DIR = BASE_DIR / "output_json" / "yolo_web"
FRCNN_OUT_DIR = BASE_DIR / "output_json" / "frcnn_web"
YOLO_OUT_DIR.mkdir(parents=True, exist_ok=True)
FRCNN_OUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".jpg", ".jpeg", ".png"}

# Static metrics
YOLO_MAP50 = 0.95
YOLO_MAP5095 = 0.90
FRCNN_MAP50 = 0.93
FRCNN_MAP5095 = 0.89

app.config["SECRET_KEY"] = "masniari-secret"
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username").strip()
        password = request.form.get("password").strip()

        # hash input pengguna
        hashed = hashlib.sha256(password.encode()).hexdigest()

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user and user["password"] == hashed:
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("dashboard_page"))
        else:
            return render_template("login.html", error="Username atau password salah.")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))



# ============================
# GLOBAL MODELS
# ============================
yolo_model = None
yolo_reader = None

frcnn_model = None
frcnn_reader = None
frcnn_device = None


# ============================
# HELPERS
# ============================
def allowed(fname: str) -> bool:
    return Path(fname).suffix.lower() in ALLOWED_EXT


def build_metrics():
    return SimpleNamespace(
        yolo=SimpleNamespace(map50=YOLO_MAP50, map5095=YOLO_MAP5095),
        frcnn=SimpleNamespace(map50=FRCNN_MAP50, map5095=FRCNN_MAP5095),
    )


def auto_insight(m, mode):
    y = m.yolo.map50
    f = m.frcnn.map50

    if mode == "yolo":
        return f"YOLOv12 unggul (mAP@50 = {y*100:.2f}%). Cepat dan stabil."
    if mode == "frcnn":
        return f"Faster R-CNN unggul (mAP@50 = {f*100:.2f}%). Lebih akurat pada teks kecil."

    if y > f:
        return "YOLO lebih unggul pada mAP@50 setelah perbandingan."
    if f > y:
        return "Faster R-CNN lebih unggul untuk detil kecil."

    return "Kedua model memberikan performa serupa."

def force_month_title_case(text):
    """
    Paksa penulisan bulan menjadi Title Case
    Contoh:
    '25 SEP 1970' -> '25 Sep 1970'
    '07 APR 2005' -> '07 Apr 2005'
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

# ============================
# FIX: FORMAT TANGGAL & YOLO OUTPUT
# ============================
def _parse_date_ui(s):
    if not s:
        return ""

    s = str(s).strip()

    fmts = [
        "%Y-%m-%d",
        "%d %b %Y",
        "%d %B %Y",
        "%d-%b-%Y",
        "%d-%B-%Y",
        "%d-%m-%Y",
        "%Y/%m/%d",
    ]

    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            # ✅ Title Case bulan (Sep, Apr, Jan)
            return dt.strftime("%d %b %Y")
        except:
            pass

    # Fallback: paksa hanya bulan jadi Title Case
    return force_month_title_case(s)

def iso_to_ui_date(s):
    """
    ISO (YYYY-MM-DD) → UI (DD Mon YYYY)
    """
    if not s:
        return ""

    try:
        dt = datetime.strptime(s, "%Y-%m-%d")
        return dt.strftime("%d %b %Y")
    except:
        return s


def format_yolo_for_ui(js):
    """
    FORMAT OUTPUT YOLO — FINAL FIXED VERSION (WITH SMART MRZ FALLBACK)
    Perbaikan: Memastikan field SEX terisi menggunakan data MRZ jika visual gagal.
    """
    if not isinstance(js, dict):
        return {}

    raw = js.get("raw", {}) or {}
    clean = js.get("cleaned", {}) or {}

    out = {}

    def pick(k):
        # Prioritas: Cleaned -> Raw -> String Kosong
        val = clean.get(k) or raw.get(k) or ""
        return str(val).strip()

    # === 1. AMBIL DATA DASAR ===
    mrz_raw = pick("mrz").replace("\n", "").replace(" ", "")
    
    # === 2. LOGIKA SEX / JENIS KELAMIN (SUPER ROBUST) ===
    sex_visual = pick("sex").upper()
    sex_final = ""

    # A. Coba dari Visual dulu (Handle P/F, WANITA, dll)
    if any(x in sex_visual for x in ["F", "P/F", "FEMALE", "WANITA", "PEREMPUAN"]):
        sex_final = "F"
    elif any(x in sex_visual for x in ["M", "L/M", "MALE", "PRIA", "LAKI"]):
        sex_final = "M"
    elif sex_visual == "P": 
        sex_final = "F" # P sering terbaca dari P/F
    elif sex_visual == "L": 
        sex_final = "M" # L sering terbaca dari L/M
    elif len(sex_visual) == 1 and sex_visual in ["M", "F"]:
        sex_final = sex_visual

    # B. [FIX UTAMA] MRZ FALLBACK AUTHORITY
    # Jika visual kosong atau aneh, AMBIL DARI MRZ.
    # Standar ICAO: Baris 2, Karakter ke-21 (Index 20) adalah Sex.
    if (not sex_final or sex_final == "-") and len(mrz_raw) > 44:
        try:
            # Jika MRZ digabung (88 char), baris 2 mulai dari index 44
            # Posisi sex di baris 2 adalah index 20. Jadi total index = 44 + 20 = 64
            mrz_part2 = mrz_raw[44:] if len(mrz_raw) >= 88 else mrz_raw
            
            # Cari karakter di posisi standar (Index 20 di baris 2)
            if len(mrz_part2) > 20:
                char_code = mrz_part2[20] # Karakter ke-21
                if char_code == "F": 
                    sex_final = "F"
                elif char_code == "M": 
                    sex_final = "M"
                else:
                    # Coba cari mundur/maju dikit jika MRZ geser (Heuristic)
                    # Cari pola tanggal lahir diikuti angka dan M/F
                    # Contoh: 9809077F (YYMMDD + Check + Sex)
                    import re
                    match = re.search(r"\d{6}\d([MF])", mrz_part2)
                    if match:
                        sex_final = match.group(1)
        except Exception:
            pass

    # Set hasil akhir Sex
    out["sex"] = sex_final

    # === 3. FIELD LAINNYA ===
    if pick("photo"):
        out["photo"] = "<PHOTO_DETECTED>"

    out["nationality"] = pick("nationality").upper()
    out["place_of_birth"] = pick("place_of_birth").upper()
    out["surname"] = pick("surname").upper()
    out["given_names"] = pick("given_names").upper()
    out["passport_number"] = pick("passport_number")

    # Dates
    out["date_of_birth"] = _parse_date_ui(pick("date_of_birth"))
    out["date_of_expiry"] = _parse_date_ui(pick("date_of_expiry"))
    out["date_of_issue"] = _parse_date_ui(pick("date_of_issue"))

    # Format Tampilan MRZ (Split 2 baris agar rapi di UI)
    if len(mrz_raw) == 88:
        out["mrz"] = mrz_raw[:44] + "\n" + mrz_raw[44:]
    else:
        out["mrz"] = pick("mrz") # Fallback raw

    # === 4. METRICS (Speed & FPS) ===
    speed = js.get("speed", 0)
    out["speed"] = speed
    fps = round(1.0 / speed, 2) if speed > 0 else 0
    out["fps"] = fps
    out["fps_percent"] = 100 if fps >= 60 else (fps / 60 * 100)

    return out

# ============================
# LOAD MODELS
# ============================
def init_models():
    global yolo_model, yolo_reader, frcnn_model, frcnn_reader, frcnn_device

    # ---- YOLO ----
    if yolo_model is None:
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(str(YOLO_MODEL_PATH))
            yolo_reader = (
                yolo_init_ocr(gpu=torch.cuda.is_available())
                if yolo_init_ocr else easyocr.Reader(["en"])
            )
            print("YOLO loaded.")
        except Exception as e:
            print("YOLO load failed:", e)
            yolo_model = None

    # ---- FRCNN ----
    if frcnn_model is None:
        frcnn_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            frcnn_model = frcnn_load_model(str(FRCNN_MODEL_PATH), frcnn_device)
            frcnn_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
            print("FRCNN loaded.")
        except Exception as e:
            print("FRCNN load failed:", e)
            frcnn_model = None

def iso_to_ui_date(s):
    if not s:
        return ""
    try:
        dt = datetime.strptime(s, "%Y-%m-%d")
        return dt.strftime("%d %b %Y")
    except:
        return s

# ============================
# ROUTES
# ============================
@app.route("/upload_paspor", methods=["GET", "POST"])
def handle_upload():
    init_models()

    if request.method == "GET":
        return render_template("upload_paspor.html", metrics=build_metrics())

    # POST
    if "image" not in request.files:
        flash("File tidak ditemukan.", "error")
        return redirect(url_for("handle_upload"))

    file = request.files["image"]
    if file.filename == "":
        flash("Tidak ada file dipilih.", "error")
        return redirect(url_for("handle_upload"))

    if not allowed(file.filename):
        flash("Format file tidak didukung.", "error")
        return redirect(url_for("handle_upload"))

    filename = secure_filename(file.filename)
    save = UPLOAD_DIR / filename
    file.save(str(save))

    photo_path = f"static/uploads/{filename}"
    image_url = filename

    mode = (request.form.get("mode") or "both").lower()

    ocr_yolo = {}
    ocr_frcnn = {}

    # ======================
    # YOLO
    # ======================
    js = {}
    if mode in ("yolo", "both"):
        if yolo_model and yolo_process_image:
            try:
                from pathlib import Path as _P
                js = yolo_process_image(
                    _P(str(save)), yolo_model, _P(str(YOLO_OUT_DIR)), 0.25, False, yolo_reader
                )
            except Exception:
                traceback.print_exc()
                ocr_yolo = {"error": "YOLO gagal berjalan"}

        ocr_yolo = format_yolo_for_ui(js) if isinstance(js, dict) else {}

        # --- FIX TANGGAL LENGKAP (NO MORE EMPTY UI!) ---
        ocr_yolo["date_of_birth"]  = fix_date_safe(ocr_yolo.get("date_of_birth"))
        ocr_yolo["date_of_issue"]  = fix_date_safe(ocr_yolo.get("date_of_issue"))
        ocr_yolo["date_of_expiry"] = fix_date_safe(ocr_yolo.get("date_of_expiry"))

        # Simpan DB
        try:
            save_ocr_to_db("YOLOv12", ocr_yolo, str(js), photo_path)
        except Exception as e:
            print("Save YOLO error:", e)


    # ======================
    # FRCNN (DENGAN TIMER APP.PY)
    # ======================
    norm = {}
    raw = {}
    frcnn_exec_time = 0  # Variabel untuk menampung waktu

    if mode in ("frcnn", "both"):
        if frcnn_model:
            try:
                import time
                start_frcnn = time.time()
                
                # [FIX] Pastikan parameter conf=0.05
                det, raw, norm = frcnn_run_single(
                    model=frcnn_model,
                    device=frcnn_device,
                    img_path=str(save),
                    out_dir=str(FRCNN_OUT_DIR / Path(filename).stem),
                    conf=0.05,  # <--- INI PENTING, JANGAN PAKAI DEFAULT
                    reader=frcnn_reader,
                    save_visuals=True,
                )
                
                frcnn_exec_time = round(time.time() - start_frcnn, 4)
                
            except Exception as e:
                print(f"[ERROR FRCNN]: {e}") # Print error di terminal biar ketahuan
                traceback.print_exc()
                ocr_frcnn = {"error": "FRCNN Gagal"}
                norm = {}
                frcnn_exec_time = 0

        # ============================
        # 100% FINAL FIX — NO OVERRIDE
        # gunakan HANYA hasil normalisasi FRCNN
        # ============================
        
        # Ambil speed final
        speed_final = norm.get("speed", frcnn_exec_time)
        if speed_final == 0: speed_final = frcnn_exec_time

        # Hitung FPS di sini
        fps_val = round(1.0 / speed_final, 2) if speed_final > 0 else 0

        ocr_frcnn = {
            "surname":         norm.get("surname", ""),
            "given_names":     norm.get("given_names", ""),
            "passport_number": norm.get("passport_number", ""),
            "nationality":     norm.get("nationality", ""),
            "place_of_birth":  norm.get("place_of_birth", ""),

            "date_of_birth":  iso_to_ui_date(norm.get("date_of_birth")),
            "date_of_issue":  iso_to_ui_date(norm.get("date_of_issue")),
            "date_of_expiry": iso_to_ui_date(norm.get("date_of_expiry")),


            "sex":             norm.get("sex", ""),
            "mrz":             norm.get("mrz", ""),

            "speed": speed_final,
            "fps": fps_val,
            "fps_percent": 100 if fps_val >= 60 else (fps_val / 60 * 100)
        }


        # ======================================================
        # FINAL UI DATE FORMAT — SAMA DENGAN YOLO
        # ======================================================
        ocr_frcnn["date_of_birth"]  = _parse_date_ui(ocr_frcnn.get("date_of_birth"))
        ocr_frcnn["date_of_issue"]  = _parse_date_ui(ocr_frcnn.get("date_of_issue"))
        ocr_frcnn["date_of_expiry"] = _parse_date_ui(ocr_frcnn.get("date_of_expiry"))


        # Simpan DB
        try:
            save_ocr_to_db("FRCNN", ocr_frcnn, str(raw), photo_path)
        except Exception as e:
            print("Save FRCNN error:", e)


    # ================
    # HITUNG METRICS
    # ================
    # ================
    # HITUNG METRICS
    # ================
    gt = load_ground_truth(filename)
    
    metrics_yolo_full = None
    metrics_frcnn_full = None

    if gt:
        if mode in ("yolo", "both"):
            metrics_yolo_full = compute_metrics_for_passport(ocr_yolo, gt)
        if mode in ("frcnn", "both"):
            metrics_frcnn_full = compute_metrics_for_passport(ocr_frcnn, gt)

        # Simpan session untuk halaman Metrics
        session["metrics_yolo"] = metrics_yolo_full
        session["metrics_frcnn"] = metrics_frcnn_full

    # Build panel insight YOLO/FRCNN
    metrics = build_metrics()
    insights_auto = auto_insight(metrics, mode)

    # ---------------------------------------------------------
    # [FIX] SINKRONISASI DATA UNTUK UI
    # Pastikan semua key di UI_FIELDS_ORDER ada di kedua dict
    # ---------------------------------------------------------
    for field in UI_FIELDS_ORDER:
        k = field["key"]
        
        # Jika YOLO tidak punya key ini, isi strip
        if k not in ocr_yolo or not ocr_yolo[k]:
            ocr_yolo[k] = "-"
            
        # Jika FRCNN tidak punya key ini, isi strip
        if k not in ocr_frcnn or not ocr_frcnn[k]:
            ocr_frcnn[k] = "-"

    # RETURN DITARUH PALING AKHIR
    return render_template(
        "result_compare.html",
        image_url=image_url,
        ocr_yolo=ocr_yolo,
        ocr_frcnn=ocr_frcnn,
        metrics=metrics,
        
        # [PENTING] Kirim urutan field ke HTML
        ui_fields=UI_FIELDS_ORDER, 

        metrics_yolo_full=metrics_yolo_full,
        metrics_frcnn_full=metrics_frcnn_full,

        insights_auto=insights_auto,
        insights_user="",
    )
    
# ============================
# SAVE USER INSIGHT
# ============================
from flask import session

@app.post("/save_insight")
def save_insight():
    note = request.form.get("insights_user", "").strip()
    session["insights_user"] = note
    return redirect(request.referrer)

@app.route("/dashboard")
def dashboard_page():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT COUNT(*) AS total FROM passport_ocr")
    total = cursor.fetchone()["total"]

    cursor.execute("SELECT COUNT(*) AS yolo FROM passport_ocr WHERE model='YOLOv12'")
    yolo = cursor.fetchone()["yolo"]

    cursor.execute("SELECT COUNT(*) AS frcnn FROM passport_ocr WHERE model='FRCNN'")
    frcnn = cursor.fetchone()["frcnn"]

    cursor.execute("SELECT COUNT(*) AS compare FROM passport_ocr WHERE model='COMPARE'")
    compare = cursor.fetchone()["compare"]

    cursor.execute("""
        SELECT id, surname, given_names, model, passport_number, timestamp
        FROM passport_ocr ORDER BY timestamp DESC LIMIT 5
    """)
    latest = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template("dashboard.html",
                           stats={"total": total, "yolo": yolo, "frcnn": frcnn, "compare": compare},
                           latest=latest)

@app.route("/records")
def records():
    page = int(request.args.get("page", 1))
    per_page = 10
    offset = (page - 1) * per_page

    base_query = "FROM passport_ocr WHERE 1=1"
    params = []

    # Hitung total
    total_query = f"SELECT COUNT(*) {base_query}"
    total_row = query_db(total_query, params, one=True)
    total = total_row["COUNT(*)"] if total_row else 0

    # Pagination query
    page_query = f"""
        SELECT *
        {base_query}
        ORDER BY id DESC
        LIMIT %s OFFSET %s
    """
    records = query_db(page_query, params + [per_page, offset])

    total_pages = (total + per_page - 1) // per_page

    return render_template(
    "records.html",
    records=records,
    page=page,
    per_page=per_page,
    total=total,
    total_pages=total_pages,
)


from flask import Response


@app.post("/delete_record/<int:rec_id>")
def delete_record(rec_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        # 1. Cari path foto sebelum data dihapus
        cursor.execute("SELECT photo_path FROM passport_ocr WHERE id=%s", (rec_id,))
        row = cursor.fetchone()

        # 2. Hapus file fisik foto jika ada
        if row and row["photo_path"] and os.path.exists(row["photo_path"]):
            try:
                os.remove(row["photo_path"])
            except OSError as e:
                print(f"Error deleting file: {e}")

        # 3. Hapus data dari database
        cursor.execute("DELETE FROM passport_ocr WHERE id=%s", (rec_id,))
        conn.commit()

    except Exception as e:
        print(f"Database error: {e}")
        
    finally:
        cursor.close()
        conn.close()

    # [PERBAIKAN] Ganti "list_records" menjadi "records" (sesuai nama fungsi route halaman tabel Anda)
    return redirect(url_for("records"))


@app.route("/record/<int:rec_id>/edit", methods=["GET", "POST"])
def edit_record(rec_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT * FROM passport_ocr WHERE id=%s", (rec_id,))
    row = cursor.fetchone()

    if not row:
        return "Record not found", 404

    if request.method == "POST":
        surname = request.form.get("surname")
        given_names = request.form.get("given_names")
        passport_number = request.form.get("passport_number")

        new_photo_path = row["photo_path"]

        # ---- CHECK IF NEW PHOTO IS UPLOADED ----
        if "photo" in request.files:
            new_photo = request.files["photo"]
            if new_photo.filename != "":
                filename = secure_filename(new_photo.filename)
                saved_path = os.path.join("static/uploads", filename)

                # Save original
                new_photo.save(saved_path)

                # Compress automatically
                compress_image(saved_path)

                new_photo_path = saved_path.replace("\\", "/")

                # Delete old file
                if row["photo_path"] and os.path.exists(row["photo_path"]):
                    os.remove(row["photo_path"])

        # ---- UPDATE RECORD ----
        sql = """
            UPDATE passport_ocr
            SET surname=%s, given_names=%s, passport_number=%s, photo_path=%s
            WHERE id=%s
        """

        cursor.execute(sql, (surname, given_names, passport_number,
                             new_photo_path, rec_id))
        conn.commit()

        cursor.close()
        conn.close()

        return redirect(url_for("view_record", rec_id=rec_id))

    cursor.close()
    conn.close()
    return render_template("record_edit.html", data=row)

@app.post("/record/<int:rec_id>/update")
def update_record(rec_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    sql = """
        UPDATE passport_ocr SET
            surname=%s,
            given_names=%s,
            passport_number=%s,
            nationality=%s,
            place_of_birth=%s,
            sex=%s,
            date_of_birth=%s,
            date_of_issue=%s,
            date_of_expiry=%s
        WHERE id=%s
    """

    vals = (
        request.form.get("surname"),
        request.form.get("given_names"),
        request.form.get("passport_number"),
        request.form.get("nationality"),
        request.form.get("place_of_birth"),
        request.form.get("sex"),
        request.form.get("date_of_birth"),
        request.form.get("date_of_issue"),
        request.form.get("date_of_expiry"),
        rec_id
    )

    cursor.execute(sql, vals)
    conn.commit()
    cursor.close()
    conn.close()

    return redirect(f"/record/{rec_id}")

@app.route("/record/<int:rec_id>", methods=["GET"])
def view_record(rec_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT * FROM passport_ocr WHERE id=%s", (rec_id,))
    row = cursor.fetchone()

    cursor.close()
    conn.close()

    if not row:
        return "Record not found", 404

    return render_template("record_detail.html", data=row)

# ==============================
# EXPORT: JSON SEMUA RECORD
# ==============================
@app.route("/export/json")
def export_json_all():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT * FROM passport_ocr ORDER BY timestamp DESC")
    records = cursor.fetchall()

    cursor.close()
    conn.close()

    # Convert datetime → string
    def fix_types(o):
        if isinstance(o, datetime):
            return o.strftime("%Y-%m-%d %H:%M:%S")
        return str(o)

    output_file = "export_all_records.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, default=fix_types)

    return send_file(output_file, as_attachment=True)



# ==============================
# EXPORT: EXCEL
# ==============================
@app.route("/export/excel")
def export_excel():
    import pandas as pd

    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM passport_ocr ORDER BY timestamp DESC", conn)
    conn.close()

    output_file = "export_records.xlsx"
    df.to_excel(output_file, index=False)

    return send_file(output_file, as_attachment=True)



# ==============================
# EXPORT: PDF
# ==============================
@app.route("/export/pdf")
def export_pdf():
    from fpdf import FPDF
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM passport_ocr ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for row in rows:
        for k, v in row.items():
            if isinstance(v, datetime):
                v = v.strftime("%Y-%m-%d")
            pdf.cell(0, 8, txt=f"{k}: {v}", ln=1)
        pdf.ln(5)

    output_file = "export_records.pdf"
    pdf.output(output_file)

    return send_file(output_file, as_attachment=True)


@app.route("/list_records")
def list_records_redirect():
    return redirect(url_for("records"))

@app.route("/metrics")
def metrics_page():
    if "user_id" not in session:
        return redirect(url_for("login"))

    metrics_yolo = session.get("metrics_yolo")
    metrics_frcnn = session.get("metrics_frcnn")

    return render_template(
        "metrics_overview.html",
        metrics_yolo=metrics_yolo,
        metrics_frcnn=metrics_frcnn
    )

from dataset_evaluation import (
    compute_dataset_metrics,
    conclude_best_model
)
from metrics_ocr import compute_metrics_for_passport

@app.route("/dataset_evaluation")
def dataset_evaluation_page():
    """
    Menampilkan evaluasi keseluruhan dataset (31 paspor)
    untuk YOLOv12 dan Faster R-CNN
    """

    # Ambil data dari DB
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT * FROM passport_ocr WHERE model='YOLOv12'")
    yolo_rows = cursor.fetchall()

    cursor.execute("SELECT * FROM passport_ocr WHERE model='FRCNN'")
    frcnn_rows = cursor.fetchall()

    cursor.close()
    conn.close()

    metrics_yolo_list = []
    metrics_frcnn_list = []

    # Hitung metrik per paspor
    for y, f in zip(yolo_rows, frcnn_rows):
        gt = load_ground_truth(y["photo_path"].split("/")[-1])
        if not gt:
            continue

        metrics_yolo_list.append(
            compute_metrics_for_passport(y, gt)
        )
        metrics_frcnn_list.append(
            compute_metrics_for_passport(f, gt)
        )

    # HITUNG DATASET METRICS (INI PENTING)
    dataset_yolo = compute_dataset_metrics(metrics_yolo_list)
    dataset_frcnn = compute_dataset_metrics(metrics_frcnn_list)

    # ===============================
    # HITUNG RATA-RATA KECEPATAN
    # ===============================
    speed_yolo  = compute_speed_metrics(yolo_rows)
    speed_frcnn = compute_speed_metrics(frcnn_rows)

    # GABUNGKAN KE METRIK DATASET
    dataset_yolo.update(speed_yolo)
    dataset_frcnn.update(speed_frcnn)


    # TENTUKAN MODEL TERBAIK
    best_model = conclude_best_model(dataset_yolo, dataset_frcnn)

    # KIRIM KE UI
    return render_template(
        "dataset_evaluation.html",
        yolo=dataset_yolo,
        frcnn=dataset_frcnn,
        best_model=best_model
    )

# ============================
# MAIN
# ============================
if __name__ == "__main__":
    init_models()
    app.run(host="0.0.0.0", port=5000, debug=True)




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
