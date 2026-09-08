import cv2
from pathlib import Path

from src.pipeline import create_enhanced_image


IMAGE_DIR = Path(
    "dataset/arcade/syntax/train/images"
)

OUTPUT_DIR = Path("outputs/final")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
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


for name in image_names:

    image_path = IMAGE_DIR / name

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_UNCHANGED
    )

    if image is None:
        print(f"Could not read {name}")
        continue

    enhanced = create_enhanced_image(image)

    output_path = OUTPUT_DIR / name

    cv2.imwrite(
        str(output_path),
        enhanced
    )

    print(f"Processed: {name}")


print("\nAll images processed.")