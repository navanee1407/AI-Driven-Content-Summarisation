import cv2
import numpy as np

# Read the oriented image
img = cv2.imread("oriented.jpeg")

if img is None:
    print("Error: Could not read the image.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Otsu thresholding
_, thresh = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Find external contours
contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Find the largest contour
largest_contour = max(contours, key=cv2.contourArea)

# Get its bounding rectangle
x, y, w, h = cv2.boundingRect(largest_contour)

print("Detected boundary:")
print("x =", x)
print("y =", y)
print("width =", w)
print("height =", h)

# Crop the detected paper region
cropped = img[y:y+h, x:x+w]

# Save result
cv2.imwrite("paper_boundary.png", cropped)

# Display result
cv2.imshow("Paper Boundary", cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()