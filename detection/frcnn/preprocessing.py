# detection/frcnn/preprocessing.py
import cv2
import numpy as np

def global_preprocess_for_frcnn(img_bgr):
    """
    SAFE preprocessing for Faster R-CNN:
      - NO aggressive deskew (we only correct small skew if found elsewhere)
      - Light denoise
      - Convert to grayscale and back to BGR (model expects 3 channels)
      - Resize only if extremely large
    """
    if img_bgr is None:
        return img_bgr

    # convert to gray
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # light denoise
    try:
        den = cv2.fastNlMeansDenoising(gray, h=7, templateWindowSize=7, searchWindowSize=21)
    except Exception:
        den = cv2.GaussianBlur(gray, (3,3), 0)

    # optional small contrast stretch (keeps features natural)
    p2, p98 = np.percentile(den, (2, 98))
    if p98 - p2 > 10:
        den = cv2.normalize(den, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    final = cv2.cvtColor(den, cv2.COLOR_GRAY2BGR)

    # limit max size for speed; keep aspect ratio
    max_dim = max(final.shape[0], final.shape[1])
    target = 2000
    if max_dim > target:
        scale = target / float(max_dim)
        final = cv2.resize(final, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    return final
