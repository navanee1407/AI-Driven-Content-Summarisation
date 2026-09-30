"""Step 2.4 - lightweight text-region detection.

Default: contour/morphology based (no model files, no OCR engine, pure OpenCV).
Optional: EAST detector if `config.east_model_path` points to
frozen_east_text_detection.pb (you must download that model file yourself).
"""
from __future__ import annotations
from typing import List, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # x, y, w, h


def detect_text_regions(img_bgr: np.ndarray, cfg) -> List[Box]:
    if cfg.east_model_path:
        try:
            return _east(img_bgr, cfg)
        except Exception:               # noqa: BLE001 - fall back gracefully
            pass
    return _morphological(img_bgr)


def _morphological(img_bgr: np.ndarray) -> List[Box]:
    # Work at a normalised size so kernel sizes behave consistently
    h0, w0 = img_bgr.shape[:2]
    scale = 1200.0 / max(h0, w0)
    img = cv2.resize(img_bgr, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_AREA) if scale < 1 else img_bgr
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Local (adaptive) threshold: ink -> white, robust to uneven lighting
    binar = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 25, 15)
    binar = cv2.morphologyEx(binar, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    # Connect letters into words / short lines horizontally
    kx = max(9, w // 90)
    joined = cv2.morphologyEx(binar, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 3)))
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[Box] = []
    inv = 1.0 / scale if scale < 1 else 1.0
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bh < 6 or bh > 0.15 * h:              # too tiny / too tall for a text line
            continue
        if bw < 12 or bw > 0.98 * w:
            continue
        if bw / float(bh) < 1.2:                 # text lines/words are wider than tall
            continue
        fill = cv2.countNonZero(binar[y:y + bh, x:x + bw]) / float(bw * bh)
        if fill < 0.08 or fill > 0.85:           # too empty (noise) / solid blob
            continue
        boxes.append((int(x * inv), int(y * inv), int(bw * inv), int(bh * inv)))
    return boxes


def _east(img_bgr: np.ndarray, cfg) -> List[Box]:
    net = cv2.dnn.readNet(cfg.east_model_path)
    h0, w0 = img_bgr.shape[:2]
    W, H = 640, 640
    blob = cv2.dnn.blobFromImage(img_bgr, 1.0, (W, H), (123.68, 116.78, 103.94), True, False)
    net.setInput(blob)
    scores, geometry = net.forward(["feature_fusion/Conv_7/Sigmoid",
                                    "feature_fusion/concat_3"])
    rects, confs = [], []
    for y in range(scores.shape[2]):
        for x in range(scores.shape[3]):
            s = float(scores[0, 0, y, x])
            if s < cfg.east_conf_threshold:
                continue
            ox, oy = x * 4.0, y * 4.0
            top, right, bottom, left = (float(geometry[0, i, y, x]) for i in range(4))
            w_, h_ = left + right, top + bottom
            ex, ey = ox + right, oy + bottom
            rects.append([int(ex - w_), int(ey - h_), int(w_), int(h_)])
            confs.append(s)
    if not rects:
        return []
    idx = cv2.dnn.NMSBoxes(rects, confs, cfg.east_conf_threshold, 0.4)
    rx, ry = w0 / W, h0 / H
    out = []
    for i in np.array(idx).flatten():
        x, y, w_, h_ = rects[i]
        out.append((int(x * rx), int(y * ry), int(w_ * rx), int(h_ * ry)))
    return out
