"""config.py - ALL tunable settings live here. Change values here, not in step files."""

# ---- STEP 1: Orientation ----
USE_TESSERACT_OSD = True        # needs Tesseract installed + pytesseract; falls back automatically
MAX_FINE_DESKEW_ANGLE = 15.0    # only correct residual skew within +/- this many degrees
MIN_DESKEW_ANGLE = 0.1          # ignore tiny angles below this (degrees)

# ---- STEP 2: Illumination ----
ILLUMINATION_METHOD = "flatfield"   # "flatfield" | "clahe" | "both" | "none"
BACKGROUND_BLUR_FRACTION = 0.08     # blur kernel = this fraction of image's longer side
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# ---- STEP 4: Noise reduction ----
DENOISE_METHOD = "nlm"          # "nlm" | "gaussian" | "median" | "none"
NLM_STRENGTH = 10               # higher = smoother (may blur thin text)
GAUSSIAN_KERNEL = 3             # odd number
MEDIAN_KERNEL = 3               # odd number

# ---- STEP 5: Contrast ----
CONTRAST_METHOD = "stretch"     # "stretch" | "clahe" | "none"
STRETCH_LOW_PERCENT = 1         # clip darkest 1%
STRETCH_HIGH_PERCENT = 99       # clip brightest 1%

# ---- STEP 6: Binarization ----
ENABLE_BINARIZATION = True      # set False for photos with complex backgrounds
BINARIZE_METHOD = "adaptive"    # "otsu" | "adaptive"
ADAPTIVE_BLOCK_SIZE = 31        # odd number
ADAPTIVE_C = 15

# ---- STEP 7: Border cleanup ----
ENABLE_BORDER_CLEANUP = True
MIN_PAGE_AREA_RATIO = 0.30      # only crop if detected page covers >= 30% of image
BORDER_PADDING = 10             # pixels kept around the detected page

# ---- Debug ----
SAVE_INTERMEDIATE_STEPS = False # True = save an image after every step into debug_output/
DEBUG_DIR = "debug_output"
