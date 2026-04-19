from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image



def load_image(
    path: str | Path,
    size: Tuple[int, int] = (224, 224),
    color_space: str = "BGR",
) -> np.ndarray:

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

    img = Image.open(path).convert("RGB")
    img = img.resize(size, Image.LANCZOS)
    return img




def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR array to RGB in-place (view, not copy)."""
    return image[:, :, ::-1].copy()


def normalise_to_float(image: np.ndarray) -> np.ndarray:

    return (image.astype(np.float32) / 255.0)


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:

    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    if image.ndim == 3:
        mask_3ch = mask[:, :, np.newaxis]          # broadcast over channels
        return (image * (mask_3ch > 0)).astype(image.dtype)
    return (image * (mask > 0)).astype(image.dtype)
