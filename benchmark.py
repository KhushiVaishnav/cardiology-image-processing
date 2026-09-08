import cv2
cv2.setNumThreads(1)
import time
import json
import platform
import numpy as np
import sys
from pathlib import Path

# Allow importing from src/
sys.path.append(str(Path(__file__).parent / "src"))

from pipeline import create_enhanced_image


IMAGE_DIR = Path(
    "dataset/arcade/syntax/train/images"
)

REPORT_PATH = Path(
    "outputs/latency_report.json"
)

NUM_IMAGES = 100
NUM_WARMUP = 5
NUM_RUNS = 20

MAX_LATENCY_LIMIT = 36.0


def benchmark_image(image):
    """
    Measure processing latency for one image.
    """

    # Warm-up runs
    for _ in range(NUM_WARMUP):
        create_enhanced_image(image)

    start = time.perf_counter()

    for _ in range(NUM_RUNS):
        create_enhanced_image(image)

    end = time.perf_counter()

    average_ms = (
        (end - start)
        / NUM_RUNS
        * 1000
    )

    return average_ms


def main():

    image_paths = sorted(
        IMAGE_DIR.glob("*.png")
    )

    if not image_paths:
        print(
            "No images found in:",
            IMAGE_DIR
        )
        return

    # Use up to 100 images
    test_images = image_paths[:NUM_IMAGES]

    print(
        f"Benchmarking {len(test_images)} images..."
    )

    print(
        f"Each image: "
        f"{NUM_WARMUP} warm-up + "
        f"{NUM_RUNS} measured runs"
    )

    print()

    latencies = []

    for index, image_path in enumerate(
        test_images,
        start=1
    ):

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_UNCHANGED
        )

        if image is None:
            print(
                f"Could not read: "
                f"{image_path}"
            )
            continue

        latency = benchmark_image(
            image
        )

        latencies.append(
            latency
        )

        print(
            f"{index:3d}/{len(test_images)} "
            f"{image_path.name:>10} : "
            f"{latency:7.2f} ms/frame"
        )

    if not latencies:
        print(
            "No valid images benchmarked."
        )
        return

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    average_latency = float(
        np.mean(latencies)
    )

    minimum_latency = float(
        np.min(latencies)
    )

    maximum_latency = float(
        np.max(latencies)
    )

    std_latency = float(
        np.std(latencies)
    )

    average_fps = (
        1000.0
        / average_latency
    )

    maximum_latency_pass = (
        maximum_latency
        <= MAX_LATENCY_LIMIT
    )

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    report = {
        "pipeline": "V8 Fast Classical Coronary Enhancement",

        "dataset": "ARCADE syntax train",

        "image_count": len(latencies),

        "image_size": "512x512",

        "warmup_runs_per_image": NUM_WARMUP,

        "measured_runs_per_image": NUM_RUNS,

        "latency_ms_per_frame": {
            "average": round(
                average_latency,
                4
            ),
            "minimum": round(
                minimum_latency,
                4
            ),
            "maximum": round(
                maximum_latency,
                4
            ),
            "standard_deviation": round(
                std_latency,
                4
            )
        },

        "average_fps": round(
            average_fps,
            4
        ),

        "maximum_latency_requirement_ms": (
            MAX_LATENCY_LIMIT
        ),

        "maximum_latency_pass": (
            maximum_latency_pass
        ),

        "hardware": {
            "processor": platform.processor(),
            "system": platform.system(),
            "python": platform.python_version()
        },

        "per_image_latency_ms": {
            image_paths[i].name: round(
                latencies[i],
                4
            )
            for i in range(
                len(latencies)
            )
        }
    }

    # Create outputs directory
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    # --------------------------------------------------
    # Console summary
    # --------------------------------------------------

    print()
    print("=" * 55)
    print("FINAL LATENCY REPORT")
    print("=" * 55)

    print(
        f"Images benchmarked: "
        f"{len(latencies)}"
    )

    print(
        f"Average latency: "
        f"{average_latency:.2f} ms/frame"
    )

    print(
        f"Minimum latency: "
        f"{minimum_latency:.2f} ms/frame"
    )

    print(
        f"Maximum latency: "
        f"{maximum_latency:.2f} ms/frame"
    )

    print(
        f"Standard deviation: "
        f"{std_latency:.2f} ms"
    )

    print(
        f"Average FPS: "
        f"{average_fps:.2f}"
    )

    print(
        f"Maximum latency limit: "
        f"{MAX_LATENCY_LIMIT:.2f} ms/frame"
    )

    print()

    if maximum_latency_pass:

        print(
            "STATUS: PASS"
        )

        print(
            "Maximum measured latency is "
            "within the 36 ms/frame requirement."
        )

    else:

        print(
            "STATUS: FAIL"
        )

        print(
            "Maximum measured latency exceeds "
            "the 36 ms/frame requirement."
        )

    print()

    print(
        "Report saved to:",
        REPORT_PATH
    )


if __name__ == "__main__":
    main()