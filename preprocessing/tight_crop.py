import sys
from pathlib import Path

import cv2
import numpy as np
import torch


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "output" / "geometric_unwarped.png"
SEG_MODEL_PATH = PROJECT_ROOT / "DocTr" / "model_pretrained" / "seg.pth"

OUTPUT_DIR = PROJECT_ROOT / "output"

MASK_OUTPUT = OUTPUT_DIR / "document_mask.png"
SEGMENTED_OUTPUT = OUTPUT_DIR / "segmented_document.png"
CROPPED_OUTPUT = OUTPUT_DIR / "tight_cropped.png"


# --------------------------------------------------
# Add DocTr folder to Python path
# --------------------------------------------------

DOCTR_DIR = PROJECT_ROOT / "DocTr"

if str(DOCTR_DIR) not in sys.path:
    sys.path.insert(0, str(DOCTR_DIR))


from seg import U2NETP


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Load segmentation model
# --------------------------------------------------

model = U2NETP(3, 1)

checkpoint = torch.load(
    SEG_MODEL_PATH,
    map_location=device
)

# DocTr checkpoint keys contain "stage..." prefix.
# Remove the first 6 characters, matching the
# official DocTr inference.py.
model_dict = model.state_dict()

pretrained_dict = {
    k[6:]: v
    for k, v in checkpoint.items()
    if k[6:] in model_dict
}

model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

model = model.to(device)
model.eval()

print("Segmentation model loaded successfully.")


# --------------------------------------------------
# Read input image
# --------------------------------------------------

image = cv2.imread(str(INPUT_PATH))

if image is None:
    raise FileNotFoundError(
        f"Could not read input image:\n{INPUT_PATH}"
    )

original_height, original_width = image.shape[:2]

print(
    f"Input size: "
    f"({original_width}, {original_height})"
)


# --------------------------------------------------
# Prepare image for U2NETP
# --------------------------------------------------

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

# DocTr segmentation model works with 288 x 288 input
resized = cv2.resize(
    image_rgb,
    (288, 288)
)

# Convert HWC -> CHW
input_tensor = resized.transpose(2, 0, 1)

input_tensor = torch.from_numpy(
    input_tensor
).float() / 255.0

input_tensor = input_tensor.unsqueeze(0).to(device)


# --------------------------------------------------
# Run segmentation
# --------------------------------------------------

with torch.no_grad():

    outputs = model(input_tensor)

    # U2NETP returns multiple outputs.
    # The first output is the main segmentation map.
    prediction = outputs[0]

    prediction = prediction.squeeze()

    prediction = prediction.cpu().numpy()


# --------------------------------------------------
# Normalize segmentation result
# --------------------------------------------------

prediction = prediction - prediction.min()

if prediction.max() > 0:
    prediction = prediction / prediction.max()


# --------------------------------------------------
# Resize mask back to original image size
# --------------------------------------------------

mask = cv2.resize(
    prediction,
    (original_width, original_height)
)


# --------------------------------------------------
# Threshold
# --------------------------------------------------

mask_uint8 = (
    mask * 255
).astype(np.uint8)

_, mask_uint8 = cv2.threshold(
    mask_uint8,
    127,
    255,
    cv2.THRESH_BINARY
)


# --------------------------------------------------
# Keep only the largest connected component
# --------------------------------------------------

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    mask_uint8,
    connectivity=8
)

if num_labels > 1:

    # Ignore label 0 because it is the background.
    largest_label = (
        1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )
    )

    mask_uint8 = np.where(
        labels == largest_label,
        255,
        0
    ).astype(np.uint8)


# --------------------------------------------------
# Save cleaned document mask
# --------------------------------------------------

cv2.imwrite(
    str(MASK_OUTPUT),
    mask_uint8
)

print("Document mask saved to:")
print(MASK_OUTPUT)


# --------------------------------------------------
# Create segmented document
# --------------------------------------------------

segmented_document = cv2.bitwise_and(
    image,
    image,
    mask=mask_uint8
)


# --------------------------------------------------
# Save segmented document
# --------------------------------------------------

cv2.imwrite(
    str(SEGMENTED_OUTPUT),
    segmented_document
)

print("Segmented document saved to:")
print(SEGMENTED_OUTPUT)


# --------------------------------------------------
# Find document bounding box
# --------------------------------------------------

coords = cv2.findNonZero(mask_uint8)

if coords is None:
    raise RuntimeError(
        "No document was detected in the segmentation mask."
    )

x, y, w, h = cv2.boundingRect(coords)


# --------------------------------------------------
# Add a small safety margin
# --------------------------------------------------

margin = 5

x1 = max(
    0,
    x - margin
)

y1 = max(
    0,
    y - margin
)

x2 = min(
    original_width,
    x + w + margin
)

y2 = min(
    original_height,
    y + h + margin
)


# --------------------------------------------------
# Tight crop
# --------------------------------------------------

tight_cropped = image[
    y1:y2,
    x1:x2
]


# --------------------------------------------------
# Save tight crop
# --------------------------------------------------

cv2.imwrite(
    str(CROPPED_OUTPUT),
    tight_cropped
)

print("Tight cropped document saved to:")
print(CROPPED_OUTPUT)

print()
print("Segmentation completed successfully.")