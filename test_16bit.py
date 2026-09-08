import cv2
import numpy as np
import sys
from pathlib import Path

# Allow importing from src/
sys.path.append(
    str(Path(__file__).parent / "src")
)

from pipeline import create_enhanced_image


IMAGE_PATH = Path(
    "dataset/arcade/syntax/train/images/1.png"
)


def main():

    # ---------------------------------------------
    # Read normal 8-bit image
    # ---------------------------------------------

    image_8bit = cv2.imread(
        str(IMAGE_PATH),
        cv2.IMREAD_GRAYSCALE
    )

    if image_8bit is None:
        print("Could not read image.")
        return

    print("Original image")
    print("---------------")
    print("Shape:", image_8bit.shape)
    print("Dtype:", image_8bit.dtype)
    print("Min:", image_8bit.min())
    print("Max:", image_8bit.max())

    # ---------------------------------------------
    # Create simulated 16-bit image
    # ---------------------------------------------

    image_16bit = (
        image_8bit.astype(np.uint16)
        * 257
    )

    print()
    print("16-bit test image")
    print("-----------------")
    print("Shape:", image_16bit.shape)
    print("Dtype:", image_16bit.dtype)
    print("Min:", image_16bit.min())
    print("Max:", image_16bit.max())

    # ---------------------------------------------
    # Run pipeline
    # ---------------------------------------------

    output = create_enhanced_image(
        image_16bit
    )

    # ---------------------------------------------
    # Validate output
    # ---------------------------------------------

    print()
    print("Pipeline output")
    print("----------------")
    print("Shape:", output.shape)
    print("Dtype:", output.dtype)
    print("Min:", output.min())
    print("Max:", output.max())

    # ---------------------------------------------
    # Checks
    # ---------------------------------------------

    assert output is not None

    assert output.shape == image_16bit.shape

    assert output.dtype == np.uint8

    assert output.min() >= 0

    assert output.max() <= 255

    print()
    print("===================================")
    print("16-BIT SUPPORT TEST: PASS")
    print("===================================")
    print(
        "The pipeline successfully accepted "
        "uint16 input and produced a valid "
        "uint8 enhanced image."
    )


if __name__ == "__main__":
    main()