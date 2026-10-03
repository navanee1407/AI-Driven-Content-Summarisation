"""step5_contrast.py - STEP 5: Contrast enhancement so text separates from background."""
import cv2
import numpy as np
import config


def _stretch(gray):
    lo, hi = np.percentile(gray, (config.STRETCH_LOW_PERCENT, config.STRETCH_HIGH_PERCENT))
    if hi <= lo:
        return gray
    out = (gray.astype(np.float32) - lo) * 255.0 / (hi - lo)
    return np.clip(out, 0, 255).astype(np.uint8)


def _clahe(gray):
    return cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT,
                           tileGridSize=config.CLAHE_TILE_GRID).apply(gray)


def enhance_contrast(gray):
    m = config.CONTRAST_METHOD
    if m == "stretch":
        return _stretch(gray)
    if m == "clahe":
        return _clahe(gray)
    return gray
