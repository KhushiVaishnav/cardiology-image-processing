import cv2
import matplotlib.pyplot as plt
from pathlib import Path


IMAGE_DIR = Path(
    "dataset/arcade/syntax/train/images"
)

RESULT_DIR = Path(
    "outputs/v1"
)

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


fig, axes = plt.subplots(
    len(image_names),
    2,
    figsize=(10, 35)
)


for row, name in enumerate(image_names):

    original = cv2.imread(
        str(IMAGE_DIR / name),
        cv2.IMREAD_GRAYSCALE
    )

    result = cv2.imread(
        str(RESULT_DIR / name),
        cv2.IMREAD_GRAYSCALE
    )

    axes[row, 0].imshow(
        original,
        cmap="gray"
    )

    axes[row, 0].set_title(
        f"{name} - Raw"
    )

    axes[row, 0].axis("off")


    axes[row, 1].imshow(
        result,
        cmap="gray"
    )

    axes[row, 1].set_title(
        f"{name} - V1 Enhanced"
    )

    axes[row, 1].axis("off")


plt.tight_layout()

output_path = "outputs/v1_comparison.png"

plt.savefig(
    output_path,
    dpi=150
)

print(
    f"Saved comparison to: {output_path}"
)

plt.show()