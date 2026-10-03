"""
Integrated preprocessing pipeline for AI-Driven-Content-Summarisation.

Pipeline:
1. Upload validation
2. Orientation correction
3. Border cleanup
4. Illumination correction (DocTr IllTr model)
5. AI upscaling (FSRCNN x2)
6. Binarization
7. Contrast enhancement

Run from the project root:
    python main.py

The input image is image.jpeg. All outputs are saved in output/.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------
# Project paths and imports
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"
DOCTR_DIR = PROJECT_ROOT / "DocTr"
OUTPUT_DIR = PROJECT_ROOT / "output"
INPUT_IMAGE = PROJECT_ROOT / "image.jpeg"
ILLTR_MODEL_PATH = DOCTR_DIR / "model_pretrained" / "illtr.pth"
FSRCNN_MODEL_PATH = PREPROCESSING_DIR / "models" / "FSRCNN_x2.pb"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# The supplied preprocessing modules use `import config`, so expose their
# directory before importing them.
sys.path.insert(0, str(PREPROCESSING_DIR))
sys.path.insert(0, str(DOCTR_DIR))

from upload_validator import validate_upload
from preprocessing.orientation import correct_orientation
import config
from border_cleanup import crop_borders
from binarize import binarize
from contrast import enhance_contrast
from IllTr import IllTr


# ---------------------------------------------------------------------
# Illumination correction helpers (DocTr / IllTr)
# ---------------------------------------------------------------------

_DEVICE = torch.device("cpu")
_ILLTR_MODEL = None


def load_illumination_model():
    """Load the IllTr model once, on CPU."""
    global _ILLTR_MODEL

    if _ILLTR_MODEL is not None:
        return _ILLTR_MODEL

    if not ILLTR_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Illumination model not found: {ILLTR_MODEL_PATH}"
        )

    print("Loading DocTr IllTr model on CPU...")
    model = IllTr()
    checkpoint = torch.load(
        ILLTR_MODEL_PATH,
        map_location=_DEVICE,
        weights_only=True,
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    cleaned_checkpoint = {}
    for key, value in checkpoint.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned_checkpoint[key] = value

    missing, unexpected = model.load_state_dict(
        cleaned_checkpoint,
        strict=False,
    )
    if missing:
        print(f"IllTr: {len(missing)} checkpoint keys missing.")
    if unexpected:
        print(f"IllTr: {len(unexpected)} unexpected checkpoint keys.")

    model.to(_DEVICE)
    model.eval()
    _ILLTR_MODEL = model
    print("IllTr model loaded.")
    return _ILLTR_MODEL


def make_illumination_patches(rgb_image):
    """Split an RGB image into overlapping 128x128 patches."""
    h, w = rgb_image.shape[:2]
    patch_size = 128
    overlap = int(patch_size * 0.125)
    step = patch_size - overlap

    # Pad first so every patch has the required dimensions, including
    # images smaller than 128 pixels in either dimension.
    pad_h = max(0, (int(np.ceil(max(0, h - patch_size) / step)) * step
                    + patch_size) - h)
    pad_w = max(0, (int(np.ceil(max(0, w - patch_size) / step)) * step
                    + patch_size) - w)
    padded = cv2.copyMakeBorder(
        rgb_image, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE
    )

    y_num = (padded.shape[0] - patch_size) // step + 1
    x_num = (padded.shape[1] - patch_size) // step + 1
    patches = np.zeros(
        (y_num, x_num, patch_size, patch_size, 3), dtype=np.uint8
    )

    for j in range(y_num):
        for i in range(x_num):
            y, x = j * step, i * step
            patches[j, i] = padded[
                y:y + patch_size, x:x + patch_size
            ]

    return patches, y_num, x_num


def run_illumination_inference(patches):
    """Run IllTr on patches and return corrected uint8 RGB patches."""
    model = load_illumination_model()
    normalized = patches.astype(np.float32) / 255.0
    y_num, x_num = normalized.shape[:2]
    results = np.zeros_like(patches, dtype=np.uint8)
    total = y_num * x_num
    count = 0

    with torch.inference_mode():
        for j in range(y_num):
            for i in range(x_num):
                patch = torch.from_numpy(normalized[j, i])
                patch = patch.permute(2, 0, 1).unsqueeze(0).to(_DEVICE)
                output = model(patch)
                output = (
                    output.squeeze(0)
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                )
                results[j, i] = np.clip(
                    output * 255.0, 0, 255
                ).astype(np.uint8)
                count += 1
                if count == total or count % 25 == 0:
                    print(f"  IllTr patches: {count}/{total}")

    return results


def combine_illumination_patches(results, original_shape):
    """Blend overlapping patches into an image of the original size."""
    y_num, x_num = results.shape[:2]
    patch_size = 128
    overlap = int(patch_size * 0.125)
    step = patch_size - overlap
    h, w = original_shape[:2]

    # Weighted overlap averaging avoids seams between adjacent patches.
    accum = np.zeros((h + patch_size, w + patch_size, 3), dtype=np.float32)
    weights = np.zeros((h + patch_size, w + patch_size, 1), dtype=np.float32)

    ramp = np.ones(patch_size, dtype=np.float32)
    ramp[:overlap] = np.linspace(0.15, 1.0, overlap, dtype=np.float32)
    ramp[-overlap:] = np.linspace(1.0, 0.15, overlap, dtype=np.float32)
    weight_map = np.outer(ramp, ramp)[..., None]

    for j in range(y_num):
        for i in range(x_num):
            y, x = j * step, i * step
            patch = results[j, i].astype(np.float32)
            accum[y:y + patch_size, x:x + patch_size] += patch * weight_map
            weights[y:y + patch_size, x:x + patch_size] += weight_map

    combined = accum / np.maximum(weights, 1e-6)
    return np.clip(combined[:h, :w], 0, 255).astype(np.uint8)


def correct_illumination_image(bgr_image):
    """Correct illumination; accepts and returns an OpenCV BGR image."""
    if config.ILLUMINATION_METHOD == "none":
        return bgr_image

    # IllTr was trained for RGB inputs.
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    patches, _, _ = make_illumination_patches(rgb_image)
    corrected_patches = run_illumination_inference(patches)
    corrected_rgb = combine_illumination_patches(
        corrected_patches, rgb_image.shape
    )
    return cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------
# AI upscaling
# ---------------------------------------------------------------------

def upscale_image(image):
    """Upscale an OpenCV BGR image by 2x using the FSRCNN model."""
    if not FSRCNN_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"FSRCNN model not found: {FSRCNN_MODEL_PATH}"
        )

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(FSRCNN_MODEL_PATH))
    sr.setModel("fsrcnn", 2)
    return sr.upsample(image)


# ---------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------

def save_image(filename, image):
    path = OUTPUT_DIR / filename
    if image is None or image.size == 0:
        raise RuntimeError(f"Cannot save an empty image: {filename}")
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not save image: {path}")
    print(f"Saved: {path}")
    return path


def to_grayscale(image):
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main():
    print("=" * 68)
    print("AI-DRIVEN CONTENT SUMMARISATION - PREPROCESSING")
    print("=" * 68)
    print(f"Input: {INPUT_IMAGE}")

    if not INPUT_IMAGE.is_file():
        raise FileNotFoundError(f"Input image not found: {INPUT_IMAGE}")

    # 1. Upload validation
    print("\n[1/7] Upload validation")
    validation = validate_upload(INPUT_IMAGE)
    if not validation.valid:
        print("\nUPLOAD REJECTED")
        print(f"Reason: {validation.message}")
        return

    print("Upload accepted.")
    print(validation.message)
    if getattr(validation, "metrics", None):
        for key, value in validation.metrics.items():
            print(f"  {key}: {value}")

    image = validation.image
    if image is None:
        raise RuntimeError(
            "Validator accepted the upload but returned no decoded image."
        )
    save_image("validated.png", image)

    # 2. Orientation
    print("\n[2/7] Orientation correction")
    image = correct_orientation(image)
    if image is None:
        raise RuntimeError("Orientation correction returned no image.")
    save_image("oriented.jpeg", image)

    # 3. Border cleanup
    print("\n[3/7] Border cleanup")
    image = crop_borders(image)
    save_image("tight_cropped.png", image)

    # 4. Illumination correction
    print("\n[4/7] Illumination correction")
    started = time.time()
    image = correct_illumination_image(image)
    save_image("illumination_corrected.png", image)
    print(f"Illumination correction time: {time.time() - started:.1f}s")

    # 5. AI upscaling
    print("\n[5/7] AI upscaling (FSRCNN x2)")
    image = upscale_image(image)
    save_image("ai_upscaled.png", image)
    print(f"Upscaled dimensions: {image.shape[1]} x {image.shape[0]}")

    # 6. Binarization
    print("\n[6/7] Binarization")
    gray = to_grayscale(image)
    binary = binarize(gray)
    save_image("binarized.png", binary)

    # 7. Contrast enhancement
    print("\n[7/7] Contrast enhancement")
    # The supplied contrast module expects a grayscale image. If binarization
    # is enabled, contrast may have little visible effect; this preserves the
    # requested order and uses the configured method.
    final_image = enhance_contrast(binary)
    save_image("final_preprocessed.png", final_image)

    print("\n" + "=" * 68)
    print("PREPROCESSING COMPLETED")
    print(f"Final image: {OUTPUT_DIR / 'final_preprocessed.png'}")
    print(f"All generated images are in: {OUTPUT_DIR}")
    print("=" * 68)


if __name__ == "__main__":
    main()
