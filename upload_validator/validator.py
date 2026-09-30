"""
Validate_Upload(file)  --  implementation of the algorithm.

STEP 1  file-level checks   (extension/MIME, size, decode)
STEP 2  image-level checks  (dimensions, blur, exposure, text regions, doc classifier)
STEP 3  decision            (Case 1 valid -> preprocessing, Case 2 invalid -> rejection)
"""
from __future__ import annotations

import io
import os
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import ValidatorConfig
from .result import CASE_INVALID, CASE_VALID, ValidationResult
from . import text_detection

FileInput = Union[str, os.PathLike, bytes]

# Magic-byte signatures used to check the *real* type (not just the extension)
_SIGNATURES = {
    "pdf": [b"%PDF-"],
    "png": [b"\x89PNG\r\n\x1a\n"],
    "jpeg": [b"\xff\xd8\xff"],
    "tiff": [b"II*\x00", b"MM\x00*"],
    "bmp": [b"BM"],
}
_EXT_TO_KIND = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png",
                ".tiff": "tiff", ".tif": "tiff", ".bmp": "bmp", ".pdf": "pdf"}


class _Reject(Exception):
    """Internal control-flow exception carrying a reason code + explanation."""
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code, self.detail = code, detail


# --------------------------------------------------------------------------- #
# STEP 1 - file-level checks
# --------------------------------------------------------------------------- #
def _detect_kind(head: bytes) -> Optional[str]:
    for kind, sigs in _SIGNATURES.items():
        if any(head.startswith(s) for s in sigs):
            return kind
    return None


def _read_input(file: FileInput, filename: Optional[str]) -> Tuple[bytes, str]:
    if isinstance(file, (bytes, bytearray)):
        return bytes(file), (filename or "")
    path = os.fspath(file)
    try:
        with open(path, "rb") as fh:
            return fh.read(), (filename or os.path.basename(path))
    except OSError as exc:
        raise _Reject("unreadable_file", f"Cannot read file: {exc}")


def _step1_file_checks(data: bytes, filename: str, cfg: ValidatorConfig, metrics: dict) -> str:
    ext = os.path.splitext(filename)[1].lower()
    metrics["extension"] = ext
    metrics["file_size_bytes"] = len(data)

    # 1.2 (empty check first so an empty file gets a clear message)
    if len(data) < cfg.min_file_size_bytes:
        raise _Reject("empty_file", "File is empty (0 KB).")

    # 1.1 extension / MIME
    if ext not in cfg.allowed_extensions:
        raise _Reject("unsupported_format", f"Extension '{ext or '(none)'}' is not supported.")
    kind = _detect_kind(data[:16])
    metrics["detected_type"] = kind
    if kind is None:
        raise _Reject("unsupported_format", "File content is not a supported image/PDF type.")
    if _EXT_TO_KIND[ext] != kind:
        raise _Reject("mime_mismatch",
                      f"File extension '{ext}' does not match its real content type '{kind}'.")

    # 1.2 upper size bound
    if len(data) > cfg.max_file_size_bytes:
        raise _Reject("file_too_large",
                      f"File is {len(data)/1_048_576:.1f} MB; limit is "
                      f"{cfg.max_file_size_bytes/1_048_576:.0f} MB.")
    return kind


def _decode_image(data: bytes, cfg: ValidatorConfig) -> np.ndarray:
    """1.3 decode with PIL (verify + load) and return a BGR uint8 array."""
    try:
        with Image.open(io.BytesIO(data)) as probe:
            w, h = probe.size
            if w * h > cfg.max_pixels:
                raise _Reject("image_too_large_pixels",
                              f"Image has {w*h:,} pixels, above the safety limit.")
            probe.verify()                       # catches corrupt headers/chunks
        with Image.open(io.BytesIO(data)) as img:
            img = ImageOps.exif_transpose(img)   # honour phone-camera rotation
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGBA")
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[-1])
                img = bg
            else:
                img = img.convert("RGB")
            arr = np.asarray(img)                # forces full pixel decode (catches truncation)
    except _Reject:
        raise
    except Exception as exc:                     # noqa: BLE001 - any decode failure = corrupt
        raise _Reject("decode_failed", f"Image could not be decoded (corrupted?): {exc}")
    if arr.size == 0:
        raise _Reject("decode_failed", "Decoded image is empty.")
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _decode_pdf(data: bytes, cfg: ValidatorConfig) -> np.ndarray:
    """1.3 for PDFs: open and rasterise page 1 (PyMuPDF, else pdf2image)."""
    pages = []
    try:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            fitz = None
        if fitz is not None:
            with fitz.open(stream=data, filetype="pdf") as doc:
                if doc.needs_pass:
                    raise _Reject("pdf_encrypted", "PDF is password-protected.")
                if doc.page_count == 0:
                    raise _Reject("decode_failed", "PDF has no pages.")
                zoom = cfg.pdf_render_dpi / 72.0
                for i in range(min(doc.page_count, cfg.pdf_pages_to_check)):
                    pix = doc[i].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
                    pages.append(cv2.cvtColor(a[:, :, :3], cv2.COLOR_RGB2BGR))
        else:
            try:
                from pdf2image import convert_from_bytes
            except ImportError:
                raise _Reject("pdf_support_missing",
                              "PDF support needs PyMuPDF (pip install pymupdf) or pdf2image+poppler.")
            for pil in convert_from_bytes(data, dpi=cfg.pdf_render_dpi,
                                          first_page=1, last_page=cfg.pdf_pages_to_check):
                pages.append(cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR))
    except _Reject:
        raise
    except Exception as exc:                     # noqa: BLE001
        raise _Reject("decode_failed", f"PDF could not be opened (corrupted?): {exc}")
    if not pages:
        raise _Reject("decode_failed", "PDF produced no renderable pages.")
    return pages[0]


# --------------------------------------------------------------------------- #
# STEP 2 - image-level checks
# --------------------------------------------------------------------------- #
def _check_dimensions(img, cfg, metrics):                                  # 2.1
    h, w = img.shape[:2]
    metrics["width"], metrics["height"] = w, h
    if w < cfg.min_width or h < cfg.min_height:
        raise _Reject("too_small",
                      f"Image is {w}x{h}px; minimum is {cfg.min_width}x{cfg.min_height}px.")


def _check_blur(gray, cfg, metrics):                                       # 2.2
    h, w = gray.shape
    # A perfectly uniform image has zero Laplacian variance, but "blurry" would be a
    # misleading message: leave it to the exposure/contrast check (2.3) to report it.
    if float(gray.std()) < cfg.min_gray_std:
        metrics["laplacian_variance"] = 0.0
        return
    scale = cfg.blur_normalize_long_side / max(h, w)
    g = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))),
                   interpolation=cv2.INTER_AREA) if scale < 1 else gray
    var = float(cv2.Laplacian(g, cv2.CV_64F).var())
    metrics["laplacian_variance"] = round(var, 2)
    if var < cfg.blur_threshold:
        raise _Reject("too_blurry",
                      f"Image is too blurry (sharpness {var:.1f} < {cfg.blur_threshold}).")


def _check_exposure(gray, cfg, metrics):                                   # 2.3
    mean, std = float(gray.mean()), float(gray.std())
    metrics["mean_intensity"], metrics["intensity_std"] = round(mean, 2), round(std, 2)
    if mean < cfg.too_dark_mean:
        raise _Reject("under_exposed", f"Image is almost entirely black (mean {mean:.0f}/255).")
    if mean > cfg.too_bright_mean and std < cfg.min_gray_std:
        raise _Reject("over_exposed", f"Image is almost entirely white (mean {mean:.0f}/255).")
    if std < cfg.min_gray_std:
        raise _Reject("no_contrast",
                      f"Image has almost no contrast (std {std:.1f}); likely a solid colour.")


def _check_text_regions(img, cfg, metrics):                                # 2.4
    boxes = text_detection.detect_text_regions(img, cfg)
    metrics["text_regions"] = len(boxes)
    if len(boxes) < cfg.min_text_regions:
        raise _Reject("no_text_detected",
                      f"Only {len(boxes)} text region(s) found (need >= {cfg.min_text_regions}); "
                      "this does not look like a document.")
    return boxes


def _document_confidence(gray, boxes, cfg, metrics) -> float:              # 2.5
    """Rule-based score in [0,1]: aspect ratio + edge density + text layout."""
    h, w = gray.shape
    aspect = w / h
    metrics["aspect_ratio"] = round(aspect, 3)
    aspect_score = 1.0 if cfg.doc_aspect_min <= aspect <= cfg.doc_aspect_max else 0.0

    edges = cv2.Canny(gray, 80, 200)
    edge_density = float((edges > 0).mean())
    metrics["edge_density"] = round(edge_density, 4)
    # documents: some edges (text) but not the dense clutter of natural photos
    edge_score = float(np.clip(1.0 - abs(edge_density - 0.05) / 0.12, 0.0, 1.0))

    text_area = sum(bw * bh for (_, _, bw, bh) in boxes)
    coverage = text_area / float(w * h)
    metrics["text_coverage"] = round(coverage, 4)
    coverage_score = float(np.clip(coverage / 0.06, 0.0, 1.0))
    count_score = float(np.clip(len(boxes) / 25.0, 0.0, 1.0))

    # documents are mostly light background (paper) with dark ink
    light_ratio = float((gray > 150).mean())
    metrics["light_pixel_ratio"] = round(light_ratio, 3)
    paper_score = float(np.clip((light_ratio - 0.35) / 0.35, 0.0, 1.0))

    # text lines tend to have similar heights
    if len(boxes) >= 3:
        hs = np.array([b[3] for b in boxes], dtype=float)
        uniform = float(np.clip(1.0 - (hs.std() / (hs.mean() + 1e-6)) / 1.2, 0.0, 1.0))
    else:
        uniform = 0.0

    score = (0.10 * aspect_score + 0.15 * edge_score + 0.25 * coverage_score +
             0.20 * count_score + 0.15 * paper_score + 0.15 * uniform)
    metrics["document_confidence"] = round(score, 3)
    return score


# --------------------------------------------------------------------------- #
# STEP 3 - public entry point
# --------------------------------------------------------------------------- #
def validate_upload(file: FileInput, filename: Optional[str] = None,
                    config: Optional[ValidatorConfig] = None) -> ValidationResult:
    """Run the full algorithm. Returns CASE 1 (valid) or CASE 2 (invalid)."""
    cfg = config or ValidatorConfig()
    metrics: dict = {}
    try:
        # ---- STEP 1 ----
        data, name = _read_input(file, filename)
        kind = _step1_file_checks(data, name, cfg, metrics)
        img = _decode_pdf(data, cfg) if kind == "pdf" else _decode_image(data, cfg)

        # ---- STEP 2 ----
        _check_dimensions(img, cfg, metrics)                    # 2.1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _check_blur(gray, cfg, metrics)                         # 2.2
        _check_exposure(gray, cfg, metrics)                     # 2.3
        boxes = _check_text_regions(img, cfg, metrics)          # 2.4
        if cfg.enable_document_classifier:                      # 2.5
            conf = _document_confidence(gray, boxes, cfg, metrics)
            if conf < cfg.doc_confidence_threshold:
                raise _Reject("not_a_document",
                              f"Document confidence {conf:.2f} < {cfg.doc_confidence_threshold}.")
    except _Reject as rej:
        # ---- STEP 3: ELSE branch -> CASE 2 ----
        return ValidationResult(case=CASE_INVALID, valid=False, message=cfg.rejection_message,
                                reasons=[rej.code], details=[rej.detail], metrics=metrics)

    # ---- STEP 3: all checks passed -> CASE 1 ----
    return ValidationResult(case=CASE_VALID, valid=True,
                            message="Valid document image. Proceeding to preprocessing.",
                            metrics=metrics, image=img)
