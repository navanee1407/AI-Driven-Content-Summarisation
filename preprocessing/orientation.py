from paddleocr import DocImgOrientationClassification
import cv2
from pathlib import Path


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Original input image is in the project root
INPUT_PATH = PROJECT_ROOT / "image.jpeg"

# All generated images go into output/
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_PATH = OUTPUT_DIR / "oriented.jpeg"

# Create output folder if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Load orientation model
# --------------------------------------------------

print("Loading orientation model...")

model = DocImgOrientationClassification(
    model_name="PP-LCNet_x1_0_doc_ori",
    device="cpu"
)

print("Model loaded successfully!")


# --------------------------------------------------
# Predict orientation
# --------------------------------------------------

output = model.predict(
    str(INPUT_PATH),
    batch_size=1
)

# Get the first result
res = output[0]

# Extract orientation information
orientation = int(
    res.json["res"]["class_ids"][0][0]
)

confidence = float(
    res.json["res"]["scores"][0]
)

print("Detected orientation:", orientation)
print("Confidence:", confidence)


# --------------------------------------------------
# Read image
# --------------------------------------------------

image = cv2.imread(str(INPUT_PATH))

if image is None:
    raise FileNotFoundError(
        f"Could not read input image: {INPUT_PATH}"
    )


# --------------------------------------------------
# Correct orientation
# --------------------------------------------------

if orientation == 0:
    corrected = image

elif orientation == 1:
    corrected = cv2.rotate(
        image,
        cv2.ROTATE_90_COUNTERCLOCKWISE
    )

elif orientation == 2:
    corrected = cv2.rotate(
        image,
        cv2.ROTATE_180
    )

elif orientation == 3:
    corrected = cv2.rotate(
        image,
        cv2.ROTATE_90_CLOCKWISE
    )

else:
    raise ValueError(
        f"Unknown orientation class: {orientation}"
    )


# --------------------------------------------------
# Save corrected image
# --------------------------------------------------

cv2.imwrite(
    str(OUTPUT_PATH),
    corrected
)

print("Corrected image saved as:", OUTPUT_PATH)