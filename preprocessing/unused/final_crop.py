import cv2
import numpy as np
from pathlib import Path


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "output" / "ai_upscaled.png"
OUTPUT_PATH = PROJECT_ROOT / "output" / "final_crop.png"


# --------------------------------------------------
# LOAD IMAGE
# --------------------------------------------------

image = cv2.imread(str(INPUT_PATH))

if image is None:
    raise FileNotFoundError(f"Could not find input image: {INPUT_PATH}")

print(f"Input image: {INPUT_PATH}")
print(f"Original size: {image.shape[1]} x {image.shape[0]}")


# --------------------------------------------------
# REMOVE BLACK / EMPTY BORDER
# --------------------------------------------------

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a mask for pixels that are not black.
# Pixels above this threshold are considered part of the page.
_, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)


# --------------------------------------------------
# FIND PAGE CONTENT
# --------------------------------------------------

# Find all non-black pixels
coords = cv2.findNonZero(mask)

if coords is None:
    raise RuntimeError("Could not detect the page content.")


# Get bounding rectangle around the detected content
x, y, w, h = cv2.boundingRect(coords)

print(f"Detected crop: x={x}, y={y}, width={w}, height={h}")


# --------------------------------------------------
# ADD SMALL SAFETY MARGIN
# --------------------------------------------------

margin = 5

x1 = max(0, x - margin)
y1 = max(0, y - margin)

x2 = min(image.shape[1], x + w + margin)
y2 = min(image.shape[0], y + h + margin)


# --------------------------------------------------
# CROP IMAGE
# --------------------------------------------------

cropped = image[y1:y2, x1:x2]


# --------------------------------------------------
# SAVE FINAL IMAGE
# --------------------------------------------------

cv2.imwrite(str(OUTPUT_PATH), cropped)

print(f"Final cropped size: {cropped.shape[1]} x {cropped.shape[0]}")
print(f"Saved final image to: {OUTPUT_PATH}")