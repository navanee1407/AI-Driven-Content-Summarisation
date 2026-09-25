import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCTR_PATH = PROJECT_ROOT / "DocTr"

INPUT_PATH = PROJECT_ROOT / "preprocessing" / "paper_boundary.png"
MODEL_PATH = DOCTR_PATH / "model_pretrained" / "illtr.pth"
OUTPUT_PATH = PROJECT_ROOT / "preprocessing" / "illumination_corrected.png"

# Allow Python to import IllTr.py from the DocTr folder
sys.path.insert(0, str(DOCTR_PATH))

from IllTr import IllTr


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cpu")

print("Device:", device)
print("Input :", INPUT_PATH)
print("Model :", MODEL_PATH)
print("Output:", OUTPUT_PATH)


# --------------------------------------------------
# Load model
# --------------------------------------------------

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Input image not found: {INPUT_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

print("\nLoading IllTr model...")

model = IllTr()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

# Handle different checkpoint formats
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

# Remove "module." prefix when the model was saved using DataParallel
cleaned_checkpoint = {}

for key, value in checkpoint.items():
    if key.startswith("module."):
        key = key[7:]
    cleaned_checkpoint[key] = value

missing, unexpected = model.load_state_dict(
    cleaned_checkpoint,
    strict=False
)

print("Missing keys   :", len(missing))
print("Unexpected keys:", len(unexpected))

model.to(device)
model.eval()

print("Model loaded successfully.")


# --------------------------------------------------
# Split image into patches
# --------------------------------------------------

def pad_crop_img(img):
    h, w = img.shape[:2]

    patch_size = 128
    overlap = int(patch_size * 0.125)
    step = patch_size - overlap

    pad_h = (
        int((h - patch_size) / step + 1) * step
        + patch_size
        - h
    )

    pad_w = (
        int((w - patch_size) / step + 1) * step
        + patch_size
        - w
    )

    padded = cv2.copyMakeBorder(
        img,
        0,
        pad_h,
        0,
        pad_w,
        cv2.BORDER_REPLICATE
    )

    y_num = int((padded.shape[0] - patch_size) / step) + 1
    x_num = int((padded.shape[1] - patch_size) / step) + 1

    patches = np.zeros(
        (y_num, x_num, patch_size, patch_size, 3),
        dtype=np.uint8
    )

    for j in range(y_num):
        for i in range(x_num):

            x = i * step
            y = j * step

            if j == y_num - 1 and i == x_num - 1:
                patches[j, i] = img[-patch_size:, -patch_size:]

            elif j == y_num - 1:
                patches[j, i] = img[
                    -patch_size:,
                    x:x + patch_size
                ]

            elif i == x_num - 1:
                patches[j, i] = img[
                    y:y + patch_size,
                    -patch_size:
                ]

            else:
                patches[j, i] = padded[
                    y:y + patch_size,
                    x:x + patch_size
                ]

    return patches, y_num, x_num


# --------------------------------------------------
# Illumination correction
# --------------------------------------------------

def correct_illumination(patches):

    patches = patches.astype(np.float32) / 255.0

    y_num = patches.shape[0]
    x_num = patches.shape[1]

    results = np.zeros(
        (y_num, x_num, 128, 128, 3),
        dtype=np.uint8
    )

    total = y_num * x_num
    count = 0

    with torch.inference_mode():

        for j in range(y_num):
            for i in range(x_num):

                count += 1

                patch = torch.from_numpy(
                    patches[j, i]
                ).permute(2, 0, 1)

                patch = patch.unsqueeze(0).to(device)

                output = model(patch)

                output = (
                    output
                    .squeeze(0)
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                )

                output = np.clip(
                    output * 255.0,
                    0,
                    255
                ).astype(np.uint8)

                results[j, i] = output

                print(
                    f"Processed patch {count}/{total}"
                )

    return results


# --------------------------------------------------
# Reconstruct image
# --------------------------------------------------

def compose_patches(results, original_shape):

    y_num = results.shape[0]
    x_num = results.shape[1]

    patch_size = 128
    overlap = int(patch_size * 0.125)
    step = patch_size - overlap

    h, w = original_shape[:2]

    reconstructed = np.zeros(
        (h + 200, w + 200, 3),
        dtype=np.uint8
    )

    for j in range(y_num):
        for i in range(x_num):

            sy = j * step
            sx = i * step

            patch = results[j, i]

            if j == 0 and i != x_num - 1:

                reconstructed[
                    sy:sy + patch_size,
                    sx:sx + patch_size
                ] = patch

            elif i == 0 and j != y_num - 1:

                reconstructed[
                    sy + 10:sy + patch_size,
                    sx:sx + patch_size
                ] = patch[10:]

            elif j == y_num - 1 and i == x_num - 1:

                reconstructed[
                    -patch_size + 10:,
                    -patch_size + 10:
                ] = patch[10:, 10:]

            elif j == y_num - 1 and i == 0:

                reconstructed[
                    -patch_size + 10:,
                    sx:sx + patch_size
                ] = patch[10:]

            elif j == y_num - 1 and i != 0:

                reconstructed[
                    -patch_size + 10:,
                    sx + 10:sx + patch_size
                ] = patch[10:, 10:]

            elif i == x_num - 1 and j == 0:

                reconstructed[
                    sy:sy + patch_size,
                    -patch_size + 10:
                ] = patch[:, 10:]

            elif i == x_num - 1 and j != 0:

                reconstructed[
                    sy + 10:sy + patch_size,
                    -patch_size + 10:
                ] = patch[10:, 10:]

            else:

                reconstructed[
                    sy + 10:sy + patch_size,
                    sx + 10:sx + patch_size
                ] = patch[10:, 10:]

    return reconstructed[:h, :w]


# --------------------------------------------------
# Main
# --------------------------------------------------

print("\nReading image...")

# PIL reads the PNG as RGB, which matches the expected
# color-channel ordering used by the original DocTr code.
image = np.array(Image.open(INPUT_PATH).convert("RGB"))

print("Image size:", image.shape[1], "x", image.shape[0])

print("\nCreating patches...")

patches, y_num, x_num = pad_crop_img(image)

print(
    f"Patch grid: {y_num} rows x {x_num} columns "
    f"= {y_num * x_num} patches"
)

print("\nStarting illumination correction...")
print("This may take some time because your computer is CPU-only.\n")

start_time = time.time()

results = correct_illumination(patches)

corrected = compose_patches(
    results,
    image.shape
)

Image.fromarray(corrected).save(OUTPUT_PATH)

elapsed = time.time() - start_time

print("\n----------------------------------")
print("Illumination correction completed!")
print("Saved to:", OUTPUT_PATH)
print(f"Processing time: {elapsed:.1f} seconds")
print("----------------------------------")


# --------------------------------------------------
# Display result
# --------------------------------------------------

display_image = cv2.cvtColor(
    corrected,
    cv2.COLOR_RGB2BGR
)

cv2.imshow(
    "Illumination Corrected",
    display_image
)

cv2.waitKey(0)
cv2.destroyAllWindows()