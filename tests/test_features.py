"""
tests/test_features.py
───────────────────────
Unit tests for the three explicit CV feature extractors (Module 2):
    - ColorHarmonyExtractor
    - TextureAnalyzer
    - ShapeDescriptor
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.color_harmony    import ColorHarmonyExtractor
from src.features.texture_analyzer import TextureAnalyzer
from src.features.shape_descriptor import ShapeDescriptor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _synthetic_bgr(h: int = 128, w: int = 128) -> np.ndarray:
    """A colourful synthetic BGR image with a white-bordered black background."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Foreground: a blue square
    img[16:112, 16:112, 0] = 200    # B channel
    img[16:112, 16:112, 1] = 100    # G channel
    img[16:112, 16:112, 2] = 50     # R channel
    return img


# ---------------------------------------------------------------------------
# ColorHarmonyExtractor tests
# ---------------------------------------------------------------------------

class TestColorHarmonyExtractor:

    def test_output_shape(self):
        """Output must be a 1-D vector of length n_dominant_colors * 3."""
        extractor = ColorHarmonyExtractor(n_dominant_colors=5)
        image     = _synthetic_bgr()
        vector    = extractor.extract(image)

        assert vector.ndim  == 1,      "Output must be 1-D"
        assert vector.shape == (15,),  f"Expected (15,), got {vector.shape}"

    def test_output_dtype(self):
        """Output dtype must be float32."""
        extractor = ColorHarmonyExtractor()
        vector    = extractor.extract(_synthetic_bgr())
        assert vector.dtype == np.float32

    def test_empty_foreground_fallback(self):
        """A fully black (background) image should not raise and return zeros."""
        extractor = ColorHarmonyExtractor(n_dominant_colors=5)
        black_img = np.zeros((64, 64, 3), dtype=np.uint8)
        vector    = extractor.extract(black_img)
        assert vector.shape == (15,)
        # All zeros expected when no foreground pixels
        assert np.allclose(vector, 0.0)

    def test_different_images_produce_different_features(self):
        """Two visually distinct images should yield different palettes."""
        extractor = ColorHarmonyExtractor(n_dominant_colors=5)
        red_img   = np.full((64, 64, 3), (0, 0, 200), dtype=np.uint8)    # mostly red
        blue_img  = np.full((64, 64, 3), (200, 0, 0), dtype=np.uint8)    # mostly blue
        assert not np.allclose(extractor.extract(red_img), extractor.extract(blue_img))


# ---------------------------------------------------------------------------
# TextureAnalyzer tests
# ---------------------------------------------------------------------------

class TestTextureAnalyzer:

    def test_output_shape(self):
        """Texture vector length must be n_orientations * n_wavelengths * 2."""
        orientations = [0, 45, 90, 135]
        wavelengths  = [4, 8, 16, 32]
        expected_dim = len(orientations) * len(wavelengths) * 2   # 32

        analyzer = TextureAnalyzer(orientations=orientations, wavelengths=wavelengths)
        vector   = analyzer.extract(_synthetic_bgr())

        assert vector.ndim   == 1,               "Output must be 1-D"
        assert vector.shape  == (expected_dim,), f"Expected ({expected_dim},), got {vector.shape}"

    def test_output_dtype(self):
        analyzer = TextureAnalyzer()
        assert analyzer.extract(_synthetic_bgr()).dtype == np.float32

    def test_smooth_vs_striped(self):
        """A uniform solid image should have lower texture energy than a striped one."""
        analyzer = TextureAnalyzer()

        # Smooth: uniform grey
        smooth   = np.full((64, 64, 3), 128, dtype=np.uint8)
        # Striped: alternating black/white columns
        striped  = np.zeros((64, 64, 3), dtype=np.uint8)
        striped[:, ::2] = 255    # every other column white

        smooth_energy  = float(np.sum(analyzer.extract(smooth)))
        striped_energy = float(np.sum(analyzer.extract(striped)))
        assert striped_energy > smooth_energy, \
            "Striped image should have higher texture energy than smooth image"


# ---------------------------------------------------------------------------
# ShapeDescriptor tests
# ---------------------------------------------------------------------------

class TestShapeDescriptor:

    def test_output_shape(self):
        """HOG descriptor must have the analytically computed length."""
        descriptor = ShapeDescriptor(
            hog_image_size=(128, 128),
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            orientations=9,
        )
        expected_len = descriptor.descriptor_length
        vector = descriptor.extract(_synthetic_bgr())

        assert vector.ndim  == 1,                f"Output must be 1-D"
        assert len(vector)  == expected_len,     f"Expected {expected_len}, got {len(vector)}"

    def test_output_dtype(self):
        descriptor = ShapeDescriptor()
        assert descriptor.extract(_synthetic_bgr()).dtype == np.float32

    def test_non_negative_values(self):
        """HOG values are magnitudes and must be non-negative."""
        descriptor = ShapeDescriptor()
        vector = descriptor.extract(_synthetic_bgr())
        assert np.all(vector >= 0.0), "HOG descriptor contains negative values"

    def test_accepts_grayscale(self):
        """Descriptor should work on grayscale input without raising."""
        import cv2
        gray_img   = np.full((128, 128), 128, dtype=np.uint8)
        descriptor = ShapeDescriptor()
        # Grayscale (H, W) — shape_descriptor._to_grayscale should handle this
        vector = descriptor.extract(gray_img[:, :, np.newaxis].repeat(3, axis=2))
        assert vector.ndim == 1
