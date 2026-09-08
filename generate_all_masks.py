import json
import cv2
import numpy as np
from pathlib import Path


ANNOTATION_FILE = Path(
    "dataset/arcade/syntax/train/annotations/train.json"
)

OUTPUT_DIR = Path(
    "outputs/all_masks"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Load annotations
# --------------------------------------------------

with open(
    ANNOTATION_FILE,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


# --------------------------------------------------
# Image lookup
# --------------------------------------------------

images = {
    image["id"]: image
    for image in data["images"]
}


# --------------------------------------------------
# Generate masks
# --------------------------------------------------

count = 0


for image_id, image_info in images.items():

    width = image_info["width"]
    height = image_info["height"]

    file_name = image_info["file_name"]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )


    # Find annotations belonging to this image
    for annotation in data["annotations"]:

        if annotation["image_id"] != image_id:
            continue

        # Ignore stenosis category
        if annotation["category_id"] == 26:
            continue

        segmentation = annotation.get(
            "segmentation"
        )

        if not segmentation:
            continue


        for polygon in segmentation:

            points = np.array(
                polygon,
                dtype=np.float32
            ).reshape(-1, 2)

            points = np.round(
                points
            ).astype(np.int32)


            cv2.fillPoly(
                mask,
                [points],
                255
            )


    output_path = OUTPUT_DIR / file_name

    cv2.imwrite(
        str(output_path),
        mask
    )

    count += 1

    if count % 100 == 0:
        print(
            f"Generated {count}/{len(images)} masks"
        )


print(
    f"\nDone. Generated {count} masks."
)

print(
    f"Saved to: {OUTPUT_DIR}"
)