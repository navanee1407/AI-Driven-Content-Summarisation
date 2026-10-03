"""step7_border_cleanup.py - STEP 7: Detect the page with contours and crop away the surroundings."""
import cv2
import numpy as np
import config


def find_page_rect(img):
    """Return (x, y, w, h) of the largest page-like contour, or None."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(g, (7, 7), 0)
    # page is usually brighter than its surroundings
    mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < config.MIN_PAGE_AREA_RATIO * g.shape[0] * g.shape[1]:
        return None
    return cv2.boundingRect(biggest)


def crop_borders(img):
    if not config.ENABLE_BORDER_CLEANUP:
        return img
    rect = find_page_rect(img)
    if rect is None:
        return img  # nothing reliable found: keep full image
    x, y, w, h = rect
    p = config.BORDER_PADDING
    H, W = img.shape[:2]
    return img[max(0, y - p):min(H, y + h + p), max(0, x - p):min(W, x + w + p)]
