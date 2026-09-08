import cv2
import numpy as np
from pathlib import Path
from skimage.filters import frangi


IMAGE_DIR = Path(
    "dataset/arcade/syntax/train/images"
)

MASK_DIR = Path(
    "outputs/all_masks"
)

def preprocess(image):
    """
    Apply the same preprocessing used
    by the main image-processing pipeline.
    """

    # Convert RGB/BGR to grayscale
    if image.ndim == 3:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    # Convert to float
    image = image.astype(
        np.float32
    )

    # Percentile normalization
    low = np.percentile(
        image,
        1
    )

    high = np.percentile(
        image,
        99
    )

    if high <= low:
        low = image.min()
        high = image.max()

    image = np.clip(
        image,
        low,
        high
    )

    image = (
        (image - low)
        / (high - low + 1e-6)
        * 255.0
    )

    image = image.astype(
        np.uint8
    )

    # Denoising
    image = cv2.GaussianBlur(
        image,
        (5, 5),
        0
    )

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    image = clahe.apply(image)

    return image


def get_frangi_response(image):

    image = preprocess(image)

    image_float = (
        image.astype(np.float32)
        / 255.0
    )

    response = frangi(
        image_float,
        sigmas=[
            1.0,
            1.5,
            2.0,
            2.5
        ],
        alpha=0.5,
        beta=0.5,
        gamma=0.5,
        black_ridges=True
    )

    response = np.nan_to_num(
        response,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return response


# --------------------------------------------------
# Thresholds based on actual Frangi response values
# --------------------------------------------------

thresholds = [
    0.0005,
    0.001,
    0.002,
    0.003,
    0.005,
    0.008,
    0.010,
    0.015,
    0.020,
    0.030
]


results = {
    threshold: {
        "tp": 0,
        "fp": 0,
        "fn": 0
    }
    for threshold in thresholds
}


image_paths = list(
    MASK_DIR.glob("*.png")
)

print(
    "Evaluating",
    len(image_paths),
    "images..."
)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

for mask_path in image_paths:

    image_path = (
        IMAGE_DIR / mask_path.name
    )

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_UNCHANGED
    )

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        print(
            f"Could not read image: "
            f"{image_path}"
        )
        continue

    if mask is None:
        print(
            f"Could not read mask: "
            f"{mask_path}"
        )
        continue

    response = get_frangi_response(
        image
    )

    ground_truth = mask > 0

    for threshold in thresholds:

        prediction = (
            response >= threshold
        )

        tp = np.sum(
            prediction & ground_truth
        )

        fp = np.sum(
            prediction & ~ground_truth
        )

        fn = np.sum(
            ~prediction & ground_truth
        )

        results[threshold]["tp"] += tp
        results[threshold]["fp"] += fp
        results[threshold]["fn"] += fn


# --------------------------------------------------
# Results
# --------------------------------------------------

print(
    "\n===== FRANGI THRESHOLD EVALUATION ====="
)

best_threshold = None
best_dice = -1


for threshold in thresholds:

    tp = results[threshold]["tp"]
    fp = results[threshold]["fp"]
    fn = results[threshold]["fn"]

    precision = tp / (
        tp + fp + 1e-8
    )

    recall = tp / (
        tp + fn + 1e-8
    )

    dice = (
        2 * tp
        / (
            2 * tp
            + fp
            + fn
            + 1e-8
        )
    )

    print(
        f"Threshold: {threshold:.4f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"Dice: {dice:.4f}"
    )

    if dice > best_dice:

        best_dice = dice
        best_threshold = threshold


print(
    "\n===== BEST RESULT ====="
)

print(
    f"Best threshold: "
    f"{best_threshold:.4f}"
)

print(
    f"Best Dice: "
    f"{best_dice:.4f}"
)