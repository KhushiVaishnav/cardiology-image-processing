import cv2
import numpy as np


def to_grayscale(image):
    """
    Convert grayscale or BGR image to grayscale.
    Supports uint8 and uint16 input.
    """

    if image is None:
        raise ValueError("Input image is None")

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image


def normalize_image(image):
    """
    Robust percentile normalization.

    Uses a spatially sampled image to reduce
    CPU cost while preserving contrast behavior.

    Supports uint8 and uint16 input.
    Returns uint8.
    """

    image_float = image.astype(
        np.float32
    )

    # Sample every 8th pixel.
    # This reduces percentile computation
    # substantially for 512x512 images.
    sample = image_float[::8, ::8]

    low = np.percentile(
        sample,
        1
    )

    high = np.percentile(
        sample,
        99
    )

    if high <= low:
        low = image_float.min()
        high = image_float.max()

    image_float = np.clip(
        image_float,
        low,
        high
    )

    image_float = (
        (image_float - low)
        / (high - low + 1e-6)
        * 255.0
    )

    return image_float.astype(
        np.uint8
    )


def denoise(image):
    """
    Lightweight Gaussian denoising.
    """

    return cv2.GaussianBlur(
        image,
        (3, 3),
        0
    )


def enhance_contrast(image):
    """
    Local contrast enhancement.
    """

    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8)
    )

    return clahe.apply(image)


def enhance_dark_structures(image):
    """
    Fast dark-structure enhancement.

    A single elliptical black-hat operation is used
    to detect dark vessel-like structures.
    """

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (11, 11)
    )

    response = cv2.morphologyEx(
        image,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    return response.astype(np.float32)


def normalize_response(response):
    """
    Fast robust normalization of a response map.
    """

    sample = response[::8, ::8]

    low = np.percentile(
        sample,
        50
    )

    high = np.percentile(
        sample,
        99
    )

    if high <= low:
        return np.zeros_like(
            response,
            dtype=np.float32
        )

    response = np.clip(
        response,
        low,
        high
    )

    response = (
        response - low
    ) / (
        high - low + 1e-6
    )

    return response.astype(
        np.float32
    )


def create_enhanced_image(image):
    """
    Fast coronary image enhancement pipeline.

    Supports:
    - uint8
    - uint16
    - grayscale
    - BGR

    Returns uint8.
    """

    # 1. Grayscale
    gray = to_grayscale(image)

    # 2. Robust normalization
    normalized = normalize_image(
        gray
    )

    # 3. Lightweight denoising
    denoised = denoise(
        normalized
    )

    # 4. Local contrast enhancement
    contrast = enhance_contrast(
        denoised
    )

    # 5. Dark vessel/structure response
    dark_response = enhance_dark_structures(
        contrast
    )

    dark_response = normalize_response(
        dark_response
    )

    # 6. Suppress very weak responses
    dark_response = np.maximum(
        dark_response - 0.20,
        0
    )

    dark_response = (
        dark_response
        / (dark_response.max() + 1e-6)
    )

    # 7. Preserve original enhanced image
    base = contrast.astype(
        np.float32
    )

    # 8. Soft enhancement
    strength = (
        0.45 * dark_response
    )

    result = (
        base
        * (1.0 - strength)
    )

    # 9. Clip
    result = np.clip(
        result,
        0,
        255
    )

    return result.astype(
        np.uint8
    )