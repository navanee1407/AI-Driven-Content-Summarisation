import cv2

# Input image
input_path = "output/illumination_corrected.png"

# AI model
model_path = "preprocessing/models/FSRCNN_x2.pb"

# Output image
output_path = "preprocessing/ai_upscaled.png"

# Load image
img = cv2.imread(input_path)

if img is None:
    raise FileNotFoundError(f"Could not read image: {input_path}")

# Create super-resolution model
sr = cv2.dnn_superres.DnnSuperResImpl_create()

# Load FSRCNN model
sr.readModel(model_path)

# Set model and scale
sr.setModel("fsrcnn", 2)

# Perform AI super-resolution
upscaled = sr.upsample(img)

# Save result
success = cv2.imwrite(output_path, upscaled)

if not success:
    raise RuntimeError(f"Could not save image: {output_path}")

print("AI super-resolution completed successfully.")
print(f"Original size : {img.shape[1]} x {img.shape[0]}")
print(f"Upscaled size : {upscaled.shape[1]} x {upscaled.shape[0]}")
print(f"Saved to      : {output_path}")