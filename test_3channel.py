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
    "dataset/arcade/syntax/train/images/10.png"
)


def main():

    # ---------------------------------------------
    # Read 3-channel image
    # ---------------------------------------------

    image = cv2.imread(
        str(IMAGE_PATH),
        cv2.IMREAD_COLOR
    )

    if image is None:
        print("Could not read image.")
        return

    print("Input image")
    print("-----------")
    print("Shape:", image.shape)
    print("Dtype:", image.dtype)
    print("Min:", image.min())
    print("Max:", image.max())

    # ---------------------------------------------
    # Run pipeline
    # ---------------------------------------------

    output = create_enhanced_image(
        image
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

    assert output.ndim == 2

    assert output.shape == (
        image.shape[0],
        image.shape[1]
    )

    assert output.dtype == np.uint8

    assert output.min() >= 0

    assert output.max() <= 255

    print()
    print("===================================")
    print("3-CHANNEL INPUT TEST: PASS")
    print("===================================")
    print(
        "The pipeline successfully accepted "
        "3-channel input, converted it to "
        "grayscale, and produced a valid "
        "enhanced image."
    )


if __name__ == "__main__":
    main()