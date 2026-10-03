"""step6_binarize.py - STEP 6: Binarization (Otsu or adaptive) to clean black-and-white."""
import cv2
import config


def binarize(gray):
    if not config.ENABLE_BINARIZATION:
        return gray
    if config.BINARIZE_METHOD == "otsu":
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    block = config.ADAPTIVE_BLOCK_SIZE | 1
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block, config.ADAPTIVE_C)
