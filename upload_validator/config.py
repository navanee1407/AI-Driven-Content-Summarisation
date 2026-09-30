"""All tunable thresholds live here so they can be calibrated on real data."""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ValidatorConfig:
    # ---- Step 1: file-level ----
    allowed_extensions: Tuple[str, ...] = (
        ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".pdf")
    min_file_size_bytes: int = 1                    # > 0 KB
    max_file_size_bytes: int = 25 * 1024 * 1024     # 25 MB
    pdf_pages_to_check: int = 1                     # pages rasterised for Step 2
    pdf_render_dpi: int = 150

    # ---- Step 2.1: dimensions ----
    min_width: int = 100
    min_height: int = 100
    max_pixels: int = 200_000_000                   # decompression-bomb guard

    # ---- Step 2.2: blur (variance of Laplacian on a normalised-size image) ----
    blur_normalize_long_side: int = 1000
    blur_threshold: float = 60.0                    # below => too blurry

    # ---- Step 2.3: brightness / exposure ----
    too_dark_mean: float = 25.0                     # mean gray below => under-exposed
    too_bright_mean: float = 250.0                  # mean gray above => over-exposed
    min_gray_std: float = 6.0                       # almost no contrast => solid colour

    # ---- Step 2.4: text-region detection ----
    min_text_regions: int = 5
    east_model_path: Optional[str] = None           # optional frozen_east_text_detection.pb
    east_conf_threshold: float = 0.5

    # ---- Step 2.5: document-vs-non-document heuristic ----
    doc_confidence_threshold: float = 0.45
    doc_aspect_min: float = 0.3                     # width/height; A4 ~0.71, Letter ~0.77
    doc_aspect_max: float = 3.5
    enable_document_classifier: bool = True

    rejection_message: str = (
        "Invalid image. Please upload a clear, well-lit photo or scan of a "
        "document (JPG, PNG, TIFF, BMP or PDF).")
