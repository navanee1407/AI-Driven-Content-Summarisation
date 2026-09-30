from pathlib import Path

from upload_validator import validate_upload
from preprocessing.orientation import correct_orientation


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_IMAGE = PROJECT_ROOT / "image.jpeg"


def main():
    print("=" * 60)
    print("AI-DRIVEN CONTENT SUMMARISATION")
    print("=" * 60)

    print(f"\nInput file: {INPUT_IMAGE}")

    # Step 1: Upload validation
    print("\n[1/3] Running upload validator...")

    result = validate_upload(INPUT_IMAGE)

    if not result.valid:
        print("\nUPLOAD REJECTED")
        print(f"Reason: {result.message}")
        print("\nProcessing stopped.")
        return

    print("\nUPLOAD VALIDATED")
    print(result.message)

    print("\nValidator metrics:")

    for key, value in result.metrics.items():
        print(f"  {key}: {value}")

    validated_image = result.image

    print("\nValidated image ready for preprocessing.")
    print(f"Image shape: {validated_image.shape}")
    print(f"Image dtype: {validated_image.dtype}")

    # Step 2: Orientation correction
    print("\n[2/3] Running orientation correction...")

    oriented_image = correct_orientation(validated_image)

    print("\nOrientation correction completed.")
    print(f"Oriented image shape: {oriented_image.shape}")

    # Step 3: Remaining stages
    print("\n[3/3] Remaining preprocessing:")

    print("  → GeoTr")
    print("  → IllTr")
    print("  → Upscaling")
    print("  → OCR")
    print("  → English summarisation")

    print("\nGeoTr and later stages are not connected yet.")


if __name__ == "__main__":
    main()
