import cv2
import matplotlib.pyplot as plt
from pathlib import Path


IMAGE_DIR = Path("dataset/arcade/syntax/train/images")
MASK_DIR = Path("outputs/masks")

image_names = [
    "922.png",
    "793.png",
    "782.png",
    "660.png",
    "708.png",
    "207.png",
    "439.png",
    "783.png",
    "149.png",
]

fig, axes = plt.subplots(3, 3, figsize=(12, 12))

for ax, name in zip(axes.ravel(), image_names):

    image_path = IMAGE_DIR / name
    mask_path = MASK_DIR / name

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_UNCHANGED
    )

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None or mask is None:
        ax.set_title(f"{name} - ERROR")
        ax.axis("off")
        continue

    # Convert RGB/BGR images to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    # Normalize image for display
    image_display = cv2.normalize(
        image,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype("uint8")

    # Create RGB image
    overlay = cv2.cvtColor(
        image_display,
        cv2.COLOR_GRAY2RGB
    )

    # Highlight ground-truth vessel pixels
    overlay[mask > 0] = [255, 0, 0]

    # Blend original + mask
    blended = cv2.addWeighted(
        cv2.cvtColor(
            image_display,
            cv2.COLOR_GRAY2RGB
        ),
        0.75,
        overlay,
        0.25,
        0
    )

    ax.imshow(blended)
    ax.set_title(name)
    ax.axis("off")


plt.tight_layout()

output_path = "outputs/mask_overlays.png"

plt.savefig(
    output_path,
    dpi=150
)

print(f"Saved: {output_path}")

plt.show()