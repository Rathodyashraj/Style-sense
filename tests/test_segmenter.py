"""
tests/test_segmenter.py
────────────────────────
Unit tests for the GrabCutSegmenter (Module 1).

Tests use synthetically generated images so no dataset is required.
"""

from __future__ import annotations

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.segmenter import GrabCutSegmenter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_image(h: int = 128, w: int = 128) -> np.ndarray:
    """Create a synthetic BGR image: white square on black background."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    margin = 20
    img[margin : h - margin, margin : w - margin] = 255   # white foreground
    return img


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGrabCutSegmenter:

    def test_output_shapes(self):
        """Segmented image and mask must have the same spatial dimensions as input."""
        segmenter = GrabCutSegmenter(grabcut_iterations=3, border_margin=5)
        image     = _make_synthetic_image(128, 128)

        segmented, mask = segmenter.segment(image)

        assert segmented.shape == image.shape,         "Segmented image shape mismatch"
        assert mask.shape      == image.shape[:2],     "Mask shape mismatch"

    def test_mask_is_binary(self):
        """Mask values must be exactly 0 or 255."""
        segmenter = GrabCutSegmenter(grabcut_iterations=3)
        image     = _make_synthetic_image(128, 128)

        _, mask = segmenter.segment(image)
        unique_values = set(np.unique(mask))

        assert unique_values.issubset({0, 255}), f"Non-binary mask values: {unique_values}"

    def test_foreground_is_not_empty(self):
        """A white-square image should yield a non-trivial foreground mask."""
        segmenter = GrabCutSegmenter(grabcut_iterations=5)
        image     = _make_synthetic_image(128, 128)

        _, mask = segmenter.segment(image)
        fg_ratio = (mask == 255).sum() / mask.size

        assert fg_ratio > 0.01, f"Foreground ratio too low: {fg_ratio:.3f}"

    def test_invalid_input_raises(self):
        """A non-3-channel input must raise ValueError."""
        segmenter = GrabCutSegmenter()
        gray_image = np.zeros((64, 64), dtype=np.uint8)

        with pytest.raises(ValueError):
            segmenter.segment(gray_image)

    def test_fallback_on_tiny_image(self):
        """A near-empty image should fall back gracefully without raising."""
        segmenter = GrabCutSegmenter(border_margin=20)
        tiny      = np.zeros((48, 48, 3), dtype=np.uint8)  # border_margin eats entire image
        segmented, mask = segmenter.segment(tiny)

        # Should not raise and should return arrays of correct shape
        assert segmented.shape == (48, 48, 3)
        assert mask.shape == (48, 48)
