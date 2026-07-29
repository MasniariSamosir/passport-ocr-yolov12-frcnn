import cv2
import numpy as np
from pathlib import Path
import imutils


# ============================================================
#  UTIL
# ============================================================
def ensure_dir(path):
    """Create directory if not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


# ============================================================
#  DESKEW
# ============================================================
def deskew_image(image, delta=1, limit=5):
    """Find optimal rotation angle by minimizing histogram variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    best_angle = 0
    min_score = 999999

    for angle in np.arange(-limit, limit + delta, delta):
        rotated = imutils.rotate_bound(thresh, angle)
        histogram = np.sum(rotated, axis=1)
        score = np.var(histogram)

        if score < min_score:
            min_score = score
            best_angle = angle

    rotated = imutils.rotate_bound(image, best_angle)
    return rotated, best_angle


# ============================================================
#  GLOBAL PREPROCESSING (FRCNN STYLE)
# ============================================================
def global_preprocess_for_frcnn(image):
    """Preprocessing ala FRCNN output (deskew + CLAHE + denoise)."""
    deskewed, angle = deskew_image(image)

    # Grayscale
    gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    clahe_img = clahe.apply(denoised)

    return clahe_img, deskewed, angle


# ============================================================
#  BOUNDING BOX HELPERS
# ============================================================
def expand_box_safe(box, img_w, img_h, scale=1.15):
    """
    Perbesar bounding box secara aman (tidak keluar frame).
    box: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = box
    w = (x2 - x1)
    h = (y2 - y1)

    new_w = w * scale
    new_h = h * scale

    cx = x1 + w / 2
    cy = y1 + h / 2

    nx1 = int(max(0, cx - new_w / 2))
    ny1 = int(max(0, cy - new_h / 2))
    nx2 = int(min(img_w - 1, cx + new_w / 2))
    ny2 = int(min(img_h - 1, cy + new_h / 2))

    return [nx1, ny1, nx2, ny2]


def crop_from_box(img, box):
    """Crop image dengan bounding box XYXY."""
    x1, y1, x2, y2 = map(int, box)
    return img[y1:y2, x1:x2]


def auto_tight_crop(img):
    """Crop otomatis area non-white (berguna untuk text crop)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.threshold(gray, 0, 255,
                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    coords = cv2.findNonZero(th)
    if coords is None:
        return img

    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y+h, x:x+w]


# ============================================================
#  TEXT CROP PREPROCESSING
# ============================================================
def generic_preprocess_for_text(crop):
    """Preprocessing untuk OCR (EasyOCR)."""
    if crop is None or crop.size == 0:
        return crop

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0,
                            tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    # Threshold
    thr = cv2.threshold(clahe_img, 0, 255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thr


# ============================================================
#  MRZ SPECIAL PREPROCESSING
# ============================================================
def preprocess_crop_mrz(crop):
    """MRZ special preprocessing."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Morph gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)

    _, thr = cv2.threshold(grad, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thr


def normalize_mrz_text(text):
    """Cleanup MRZ characters."""
    return (
        text.replace(" ", "")
            .replace("\n", "")
            .replace("\t", "")
            .replace("«", "<")
            .strip()
    )


# ============================================================
#  MRZ PARSER
# ============================================================
def parse_mrz_to_fields(mrz_text):
    """Parse MRZ menjadi dictionary."""
    mrz = normalize_mrz_text(mrz_text)

    # Minimal validation
    if len(mrz) < 80:
        return {
            "mrz_raw": mrz,
            "mrz_valid": False
        }

    # MRZ 2 lines (44 chars each)
    line1 = mrz[:44]
    line2 = mrz[44:88]

    return {
        "mrz_raw": mrz,
        "mrz_valid": True,
        "line1": line1,
        "line2": line2,
    }
