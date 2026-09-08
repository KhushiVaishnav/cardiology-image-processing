import json
from pathlib import Path

ANNOTATION_FILE = Path(
    "dataset/arcade/syntax/train/annotations/train.json"
)

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("\n===== TOP LEVEL KEYS =====")
print(data.keys())

print("\n===== NUMBER OF IMAGES =====")
print(len(data.get("images", [])))

print("\n===== NUMBER OF ANNOTATIONS =====")
print(len(data.get("annotations", [])))

print("\n===== CATEGORIES =====")

for category in data.get("categories", []):
    print(category)

print("\n===== FIRST IMAGE =====")

if data.get("images"):
    print(data["images"][0])

print("\n===== FIRST ANNOTATION =====")

if data.get("annotations"):
    annotation = data["annotations"][0]

    # Print only important fields
    for key, value in annotation.items():
        if key == "segmentation":
            print(
                key,
                "type =", type(value).__name__,
                "length =", len(value)
            )
        else:
            print(key, "=", value)