import cv2
import numpy as np
from pathlib import Path
from skimage.filters import frangi


IMAGE_DIR = Path(
    "dataset/arcade/syntax/train/images"
)

MASK_DIR = Path(
    "outputs/masks"
)


image_names = [
    "922.png",
    "793.png",
    "782.png",
    "660.png",
    "708.png"
]


for name in image_names:

    image = cv2.imread(
        str(IMAGE_DIR / name),
        cv2.IMREAD_UNCHANGED
    )

    if image is None:
        print(f"Could not read {name}")
        continue

    if image.ndim == 3:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    # Same preprocessing as our pipeline
    image = image.astype(np.float32)

    image = cv2.normalize(
        image,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    image = clahe.apply(image)

    image_float = image.astype(
        np.float32
    ) / 255.0

    response = frangi(
        image_float,
        sigmas=[1.0, 1.5, 2.0, 2.5],
        alpha=0.5,
        beta=0.5,
        gamma=0.5,
        black_ridges=True
    )

    response = np.nan_to_num(response)

    print(f"\n===== {name} =====")
    print("Image min:", image_float.min())
    print("Image max:", image_float.max())

    print("Frangi min:", response.min())
    print("Frangi max:", response.max())
    print("Frangi mean:", response.mean())

    print(
        "Percentiles:",
        np.percentile(
            response,
            [50, 75, 90, 95, 99, 99.5, 99.9]
        )
    )