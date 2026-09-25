from paddleocr import DocImgOrientationClassification
import cv2
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
input_path = project_root / "image.jpeg"
output_dir = project_root / "output"
output_path = "oriented.jpeg"

print("Loading orientation model...")

model = DocImgOrientationClassification(
    model_name="PP-LCNet_x1_0_doc_ori",
    device="cpu"
)

print("Model loaded successfully!")

# Predict orientation
output = model.predict(
    input_path,
    batch_size=1
)

# Get the first result
res = output[0]

# Extract orientation information
orientation = int(res.json["res"]["class_ids"][0][0])
confidence = float(res.json["res"]["scores"][0])

print("Detected orientation:", orientation)
print("Confidence:", confidence)

# Read image
image = cv2.imread(input_path)

# Correct orientation
if orientation == 0:
    corrected = image

elif orientation == 1:
    corrected = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

elif orientation == 2:
    corrected = cv2.rotate(image, cv2.ROTATE_180)

elif orientation == 3:
    corrected = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

else:
    raise ValueError(f"Unknown orientation class: {orientation}")

# Save corrected image
cv2.imwrite(output_path, corrected)

print("Corrected image saved as:", output_path)