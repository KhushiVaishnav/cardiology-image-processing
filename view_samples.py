import cv2
import matplotlib.pyplot as plt
from pathlib import Path

IMAGE_DIR = Path("dataset/arcade/syntax/train/images")

# Pick some images
image_names = [
    "1.png",
    "10.png",
    "100.png",
    "101.png",
    "102.png",
    "103.png",
    "104.png",
    "105.png",
    "106.png",
]

fig, axes = plt.subplots(3, 3, figsize=(12, 12))

for ax, name in zip(axes.ravel(), image_names):

    path = IMAGE_DIR / name

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image is None:
        ax.set_title(f"{name} - NOT FOUND")
        ax.axis("off")
        continue

    # Convert BGR → RGB if image has 3 channels
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ax.imshow(image, cmap="gray")
    ax.set_title(name)
    ax.axis("off")

plt.tight_layout()

output_path = "outputs/sample_images.png"
plt.savefig(output_path, dpi=150)

print(f"Saved sample visualization to: {output_path}")

plt.show()