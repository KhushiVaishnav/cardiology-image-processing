import cv2
import numpy as np
from pathlib import Path

IMAGE_DIR = Path("dataset/arcade/syntax/train/images")

images = list(IMAGE_DIR.glob("*.png"))

print("Number of images:", len(images))

print("\n===== IMAGE INFORMATION =====")

# Inspect first 10 images
for image_path in images[:10]:

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if image is None:
        print(f"Could not read: {image_path}")
        continue

    print(f"\nImage: {image_path.name}")
    print("Shape:", image.shape)
    print("Data type:", image.dtype)
    print("Minimum:", image.min())
    print("Maximum:", image.max())
    print("Mean:", image.mean())
    print("Unique values:", len(np.unique(image)))