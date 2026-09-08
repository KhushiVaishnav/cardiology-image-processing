# Coronary X-Ray Image Processing Pipeline

## Overview

This project implements a fast classical image-processing pipeline for enhancing coronary arteries in X-ray angiography images.

The main objective is to improve the visibility of coronary vessels while reducing the influence of background anatomical structures and image noise.

The pipeline is designed for low-latency processing and supports:

- 8-bit grayscale images
- 16-bit grayscale images
- 3-channel images
- 512×512 images

The final pipeline uses classical image-processing techniques rather than a trained deep-learning model.

## Objectives

The system is designed to:

- Enhance coronary artery visibility
- Improve local image contrast
- Reduce image noise
- Reduce the influence of distracting background structures
- Preserve coronary vessel structures
- Support 8-bit grayscale input
- Support 16-bit grayscale input
- Support 3-channel image input
- Maintain the original 512×512 image resolution
- Provide low-latency processing suitable for real-time applications
- Measure and report processing latency and FPS

## Dataset

The project uses the ARCADE coronary X-ray angiography dataset.

The dataset contains coronary angiography images with annotations for coronary vessel segments and stenosis.

The ARCADE dataset is not included in this repository because of dataset size.

Download it separately and place it under:

```text
dataset/arcade/
```

## Pipeline Architecture

The processing pipeline consists of the following stages:

1. Input image handling
2. Grayscale conversion
3. Intensity normalization
4. Gaussian denoising
5. CLAHE-based contrast enhancement
6. Dark-structure enhancement using black-hat morphology
7. Vessel-response normalization
8. Final coronary vessel enhancement

The pipeline uses classical image-processing techniques and does not require a trained deep-learning model.

## Processing Stages

**1. Grayscale Conversion**

Input images are converted to grayscale when required.

This allows the same processing pipeline to handle both grayscale and 3-channel input images.

**2. Intensity Normalization**

The input intensity range is normalized to improve consistency between images with different intensity distributions.

The implementation also supports 16-bit input images.

**3. Gaussian Denoising**

A small Gaussian filter is applied to reduce high-frequency noise while preserving important vessel structures.

**4. Contrast Enhancement**

CLAHE (Contrast Limited Adaptive Histogram Equalization) is used to improve local contrast.

This helps make low-contrast coronary vessel structures more visible.

**5. Dark-Structure Enhancement**

Black-hat morphological filtering is used to enhance dark line-like structures against a brighter background.

This is useful for highlighting vessel-like structures in coronary angiography images.

**6. Vessel Response Normalization**

The enhanced structural response is normalized so that it can be combined with the original image.

**7. Final Image Enhancement**

The vessel response is combined with the normalized image to produce the final enhanced output.

The enhancement is intentionally conservative to improve coronary vessel visibility without excessively suppressing surrounding anatomy.

## Input Compatibility

The pipeline supports:

- 8-bit grayscale images
- 16-bit grayscale images
- 3-channel images
- 512×512 images

The output is an 8-bit grayscale image with the same spatial resolution as the input.

Input compatibility was validated using dedicated test scripts.

## Performance Benchmark

The pipeline was benchmarked on 100 ARCADE 512×512 training images.

Each image was processed using 5 warm-up runs followed by 20 measured runs.

| Metric | Result |
|---|---:|
| Images benchmarked | 100 |
| Average latency | 11.35 ms/frame |
| Minimum latency | 9.62 ms/frame |
| Maximum latency | 28.32 ms/frame |
| Standard deviation | 2.47 ms |
| Average FPS | 88.08 |
| Required maximum latency | 36 ms/frame |
| Status | PASS |

The maximum measured latency was 28.32 ms/frame, which is below the required 36 ms/frame limit.

Therefore, the optimized pipeline satisfies the latency requirement on the benchmarked ARCADE images.

The detailed benchmark results are saved in:

```text
outputs/latency_report.json
```

## 16-bit Support

A dedicated test was performed using a converted 16-bit grayscale image.

Result:

```text
16-BIT SUPPORT TEST: PASS
```

The pipeline successfully processed the 16-bit input and produced a valid enhanced output.

## 3-Channel Input Support

A dedicated test was performed using a 3-channel image.

Result:

```text
3-CHANNEL INPUT TEST: PASS
```

The pipeline successfully converted and processed the 3-channel input.

## Vesselness Experiment

A Frangi vesselness experiment was performed during development to evaluate a vessel-specific enhancement approach.

The experiment was evaluated using the ARCADE coronary vessel masks generated from the training annotations.

The best tested threshold was 0.0010.

Precision: 0.1376

Recall: 0.3133

Dice: 0.1912

The results showed that Frangi vesselness alone was not sufficiently selective for the target images.

Therefore, Frangi filtering was not used as the primary method in the final low-latency pipeline.

The final implementation instead uses a faster classical morphology-based enhancement approach.

## Coronary Vessel Masks

The ARCADE annotations were used during development to generate binary coronary vessel masks.

The 25 coronary vessel segment categories were combined into a single vessel mask.

The stenosis category was not included in the binary coronary vessel mask.

The generated masks were used for development and vesselness evaluation and are not required during normal image enhancement.

Generated masks are stored under:

```text
outputs/all_masks/
```

## Raw vs Enhanced Results

The project includes scripts for viewing sample images, generated masks, and processing results.

The enhanced images can be compared with the original ARCADE images to visually assess the improvement in coronary vessel visibility.

The processing output is currently saved under:

```text
outputs/final/
```

## How to Run

**1. Install Dependencies**

Create and activate a Python virtual environment if required.

Install the project dependencies:

```text
pip install -r requirements.txt
```

**2. Run the Image Processing Pipeline**

Run:

```text
python run_pipeline.py
```

The script processes the selected ARCADE sample images and saves the enhanced images in:

```text
outputs/final/
```

**3. Run the Latency Benchmark**

Run:

```text
python benchmark.py
```

The benchmark measures processing latency and FPS and saves the report to:

```text
outputs/latency_report.json
```

**4. Test 16-bit Input**

Run:

```text
python test_16bit.py
```

Expected result:

```text
16-BIT SUPPORT TEST: PASS
```

**5. Test 3-Channel Input**

Run:

```text
python test_3channel.py
```

Expected result:

```text
3-CHANNEL INPUT TEST: PASS
```

## Project Structure

```text
cardiology_image_processing/

├── dataset/
│   └── arcade/

├── src/
│   └── pipeline.py

├── outputs/
│   ├── all_masks/
│   ├── masks/
│   ├── final/
│   └── latency_report.json

├── benchmark.py
├── create_mask.py
├── evaluate_vesselness.py
├── generate_all_masks.py
├── inspect_annotations.py
├── inspect_dataset.py
├── inspect_frangi.py
├── inspect_images.py
├── run_pipeline.py
├── test_16bit.py
├── test_3channel.py
├── view_masks.py
├── view_results.py
├── view_samples.py
├── requirements.txt
└── README.md
```

## Main Files

`src/pipeline.py` --- Main coronary image enhancement pipeline

`run_pipeline.py` --- Runs the pipeline on sample images

`benchmark.py` --- Measures latency and FPS

`test_16bit.py` --- Tests 16-bit image support

`test_3channel.py` --- Tests 3-channel image support

`generate_all_masks.py` --- Generates coronary vessel masks from ARCADE annotations

`create_mask.py` --- Creates a vessel mask for an individual image

`evaluate_vesselness.py` --- Evaluates vesselness-based enhancement

`inspect_dataset.py` --- Inspects dataset properties

`inspect_annotations.py` --- Inspects ARCADE annotations

`inspect_images.py` --- Inspects image formats and intensity ranges

`inspect_frangi.py` --- Inspects Frangi vesselness responses

`view_samples.py` --- Displays sample dataset images

`view_masks.py` --- Displays generated vessel masks

`view_results.py` --- Displays processing results

`requirements.txt` --- Python dependencies

## Requirements

The project uses Python and the following main libraries:

- NumPy
- OpenCV
- scikit-image
- pycocotools
- Matplotlib

Install all required dependencies using:

```text
pip install -r requirements.txt
```

## Limitations

The current implementation is a classical image-processing solution and does not use a trained deep-learning segmentation model.

Some anatomical structures may still remain visible after processing because aggressive suppression could also remove coronary vessel structures.

The current pipeline is designed primarily for image enhancement and preprocessing rather than direct coronary artery segmentation.

Performance can vary depending on hardware, image resolution, image format, and system load.

The pipeline is not a clinical diagnostic tool and should not be used for medical diagnosis.

## Future Improvements

Possible future improvements include:

- More advanced vessel-specific filtering
- Adaptive suppression of ribs and other anatomical structures
- Improved handling of low-contrast vessels
- Temporal processing for cine sequences
- GPU acceleration where available
- Evaluation using additional image-quality metrics
- Development of a learned vessel segmentation model if computational resources permit

## Conclusion

A fast classical image-processing pipeline was developed to improve the visibility of coronary vessel structures in X-ray angiography images.

The pipeline combines intensity normalization, Gaussian denoising, CLAHE contrast enhancement, and morphological dark-structure enhancement.

The optimized implementation achieved:

- 11.35 ms/frame average latency
- 28.32 ms/frame maximum measured latency
- 88.08 FPS average
- Maximum latency requirement of 36 ms/frame: PASS

The implementation also passed dedicated tests for:

- 16-bit grayscale input
- 3-channel input

The final pipeline provides a lightweight CPU-based approach for coronary X-ray image enhancement while maintaining the required low processing latency.

## Disclaimer

This project is developed for technical and educational purposes.

The output is intended for image-processing research and preprocessing only and is not a medical diagnostic system.
