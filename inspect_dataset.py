import os
from pathlib import Path

DATASET_DIR = Path("dataset")

print("\n===== DATASET STRUCTURE =====\n")

for root, dirs, files in os.walk(DATASET_DIR):
    level = root.replace(str(DATASET_DIR), "").count(os.sep)
    indent = "    " * level

    print(f"{indent}{Path(root).name}/")

    for file in files[:5]:
        print(f"{indent}    {file}")

print("\n===== SUMMARY =====\n")

extensions = {}

for root, dirs, files in os.walk(DATASET_DIR):
    for file in files:
        ext = Path(file).suffix.lower()

        if ext:
            extensions[ext] = extensions.get(ext, 0) + 1

for ext, count in extensions.items():
    print(f"{ext}: {count} files")