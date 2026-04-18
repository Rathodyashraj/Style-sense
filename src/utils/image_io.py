"""
src/utils/image_io.py
─────────────────────
Low-level image loading and pre-processing helpers shared across all modules.

Design notes
------------
* All functions return BGR numpy arrays (OpenCV convention) OR convert
  explicitly when the caller requests RGB / PIL.
* A single ``load_image`` entry-point handles all supported formats and
  applies a consistent resize so downstream modules always receive the
  same spatial resolution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_image(
    path: str | Path,
    size: Tuple[int, int] = (224, 224),
    color_space: str = "BGR",
) -> np.ndarray:
    """
    Load an image from disk, resize it, and optionally convert its color space.

    Parameters
    ----------
    path : str | Path
        Path to the image file (JPEG, PNG, WEBP, …).
    size : (width, height)
        Target spatial resolution.  Defaults to 224 × 224.
    color_space : {"BGR", "RGB", "LAB", "GRAY"}
        Output color space.

    Returns
    -------
    np.ndarray
        uint8 array of shape (H, W, C) or (H, W) for grayscale.

    Raises
    ------
    FileNotFoundError
        If *path* does not point to an existing file.
    ValueError
        If the file cannot be decoded by OpenCV.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # cv2.IMREAD_COLOR always decodes to 3-channel BGR (ignores alpha)
    image: np.ndarray = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"OpenCV could not decode image: {path}")

    # Resize with INTER_AREA for downscaling (minimises aliasing)
    if image.shape[:2] != (size[1], size[0]):
        interp = cv2.INTER_AREA if image.shape[0] > size[1] else cv2.INTER_LINEAR
        image = cv2.resize(image, size, interpolation=interp)

    # ── Color space conversion ─────────────────────────────────────────────
    color_space = color_space.upper()
    if color_space == "BGR":
        return image
    if color_space == "RGB":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if color_space == "LAB":
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    if color_space == "GRAY":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    raise ValueError(f"Unsupported color_space '{color_space}'. "
                     "Choose from BGR | RGB | LAB | GRAY.")


def load_image_pil(path: str | Path, size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """
    Load an image as a PIL Image (RGB).  Used by the CLIP pre-processor.

    Parameters
    ----------
    path : str | Path
        Path to the image file.
    size : (width, height)
        Resize target.

    Returns
    -------
    PIL.Image.Image  — RGB mode, resized.
    """
    img = Image.open(path).convert("RGB")
    img = img.resize(size, Image.LANCZOS)
    return img


# ---------------------------------------------------------------------------
# Conversion utilities
# ---------------------------------------------------------------------------

def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR array to RGB in-place (view, not copy)."""
    return image[:, :, ::-1].copy()


def normalise_to_float(image: np.ndarray) -> np.ndarray:
    """
    Scale a uint8 image to [0.0, 1.0] float32.

    Parameters
    ----------
    image : np.ndarray  (uint8)

    Returns
    -------
    np.ndarray  (float32, same shape)
    """
    return (image.astype(np.float32) / 255.0)


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Zero out pixels where *mask* == 0 (background).

    Parameters
    ----------
    image : np.ndarray  shape (H, W, C) or (H, W)
    mask  : np.ndarray  shape (H, W), dtype uint8, values 0 or 255

    Returns
    -------
    np.ndarray  — image with background pixels set to 0.
    """
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    if image.ndim == 3:
        mask_3ch = mask[:, :, np.newaxis]          # broadcast over channels
        return (image * (mask_3ch > 0)).astype(image.dtype)
    return (image * (mask > 0)).astype(image.dtype)
