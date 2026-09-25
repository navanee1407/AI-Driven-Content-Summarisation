import cv2
import numpy as np

def flatten_channel(ch, ksize=51):
    """ksize should be larger than the thickest text stroke."""
    ksize |= 1  # must be odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    # Closing removes dark text, leaving only the background/illumination
    bg = cv2.morphologyEx(ch, cv2.MORPH_CLOSE, kernel)
    bg = cv2.medianBlur(bg, ksize)
    bg = cv2.GaussianBlur(bg, (0, 0), ksize / 3)
    # Divide out the illumination -> background becomes ~white
    return cv2.divide(ch, bg, scale=255)

img = cv2.imread(r"D:\Coding\Reverse Engineering Project\image.jpeg")

# Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("corrected_gray.png", flatten_channel(gray))

# Colour (also removes colour casts)
corrected = cv2.merge([flatten_channel(c) for c in cv2.split(img)])
cv2.imwrite("corrected_color.png", corrected)

cv2.imshow("Corrected Color Image", corrected)
cv2.waitKey(0)
cv2.destroyAllWindows()