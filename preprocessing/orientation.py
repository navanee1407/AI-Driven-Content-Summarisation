from paddleocr import DocImgOrientationClassification
import cv2
from pathlib import Path
import tempfile

# --------------------------------------------------

# Project paths

# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_PATH = OUTPUT_DIR / "oriented.jpeg"

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

# Orientation function

# --------------------------------------------------

def correct_orientation(image):
    """
    Detect and correct document orientation.

    ```
    Parameters
    ----------
    image : numpy.ndarray
        BGR image returned by the upload validator.

    Returns
    -------
    numpy.ndarray
        Orientation-corrected BGR image.
    """

    if image is None:
        raise ValueError("Input image is None.")

    if len(image.shape) != 3:
        raise ValueError(
            f"Expected a color image with 3 dimensions, got: {image.shape}"
    )

    # --------------------------------------------------
    # PaddleOCR orientation model currently receives
    # the image through a temporary file.
    # --------------------------------------------------

    with tempfile.NamedTemporaryFile(
        suffix=".jpeg",
        delete=False
    ) as temp_file:

        temp_path = Path(temp_file.name)

    try:
        # Save validated BGR image temporarily
        success = cv2.imwrite(
            str(temp_path),
            image
        )

        if not success:
            raise RuntimeError(
                "Could not create temporary image for orientation detection."
            )

        # --------------------------------------------------
        # Predict orientation
        # --------------------------------------------------

        output = model.predict(
            str(temp_path),
            batch_size=1
        )

        res = output[0]

        orientation = int(
            res.json["res"]["class_ids"][0][0]
        )

        confidence = float(
            res.json["res"]["scores"][0]
        )

        print("Detected orientation:", orientation)
        print("Confidence:", confidence)

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
        # Save output
        # --------------------------------------------------

        success = cv2.imwrite(
            str(OUTPUT_PATH),
            corrected
        )

        if not success:
            raise RuntimeError(
                f"Could not save oriented image to: {OUTPUT_PATH}"
            )

        print(
            "Corrected image saved as:",
            OUTPUT_PATH
        )

        return corrected

    finally:

        # Remove temporary image
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass

