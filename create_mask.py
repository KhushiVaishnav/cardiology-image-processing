import json
import cv2
import numpy as np
from pathlib import Path
from pycocotools import mask as mask_utils


ANNOTATION_FILE = Path(
    "dataset/arcade/syntax/train/annotations/train.json"
)

IMAGE_DIR = Path(
    "dataset/arcade/syntax/train/images"
)

OUTPUT_DIR = Path("outputs/masks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load COCO annotations
# --------------------------------------------------

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


# --------------------------------------------------
# Build image lookup
# --------------------------------------------------

images = {
    image["id"]: image
    for image in data["images"]
}


# --------------------------------------------------
# Build annotation lookup
# --------------------------------------------------

annotations_by_image = {}

for annotation in data["annotations"]:

    image_id = annotation["image_id"]

    if image_id not in annotations_by_image:
        annotations_by_image[image_id] = []

    annotations_by_image[image_id].append(annotation)


# --------------------------------------------------
# Create masks for first 10 images
# --------------------------------------------------

count = 0

for image_id, image_info in images.items():

    if count >= 10:
        break

    width = image_info["width"]
    height = image_info["height"]
    file_name = image_info["file_name"]

    # Empty binary mask
    combined_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    image_annotations = annotations_by_image.get(
        image_id,
        []
    )

    for annotation in image_annotations:

        # Ignore stenosis category (26)
        if annotation["category_id"] == 26:
            continue

        segmentation = annotation.get("segmentation")

        if not segmentation:
            continue

        # Convert polygon(s) to binary mask
        for polygon in segmentation:

            polygon = np.array(
                polygon,
                dtype=np.float32
            ).reshape(-1, 2)

            polygon = np.round(polygon).astype(np.int32)

            cv2.fillPoly(
                combined_mask,
                [polygon],
                255
            )

    # Save mask
    output_path = OUTPUT_DIR / file_name

    cv2.imwrite(
        str(output_path),
        combined_mask
    )

    print(
        f"Created: {output_path} "
        f"| annotations: {len(image_annotations)}"
    )

    count += 1


print("\nDone.")