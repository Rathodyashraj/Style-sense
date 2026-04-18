"""
src/features/shape_descriptor.py
──────────────────────────────────
Module 2 (Part C) — Shape / Silhouette Feature via HOG

Theory
------
The Histogram of Oriented Gradients (HOG) descriptor characterises an image's
local edge structure by:

1.  Computing image gradients  (Gx, Gy)  using a 1×3 / 3×1 derivative mask.
2.  Computing per-pixel gradient magnitude  M = √(Gx² + Gy²)
    and orientation  θ = arctan(Gy / Gx).
3.  Dividing the image into small spatial *cells* (e.g. 8×8 pixels).
    Within each cell, building a histogram of gradient orientations,
    weighted by magnitude.
4.  Grouping cells into overlapping *blocks* and L2-normalising each block
    to achieve illumination/contrast invariance.
5.  Concatenating all block descriptors into a single feature vector.

For garment classification, HOG captures:
  * Silhouette outline (the garment edge produces strong gradients)
  * Cut / drape lines (darts, seams, pleats)
  * Structural elements (collars, buttons, pockets)

Important: we resize images to a fixed ``hog_image_size`` before computing HOG
so the descriptor length is constant regardless of the original resolution.

References
----------
Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human
detection. CVPR 2005.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from skimage.feature import hog

from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# ShapeDescriptor
# ---------------------------------------------------------------------------

class ShapeDescriptor:
    """
    Computes a fixed-length HOG shape descriptor for a garment image.

    Parameters
    ----------
    hog_image_size    : (width, height)
        Image is resized to this resolution before HOG computation.
        Must match the values used during training and inference.
    pixels_per_cell   : (int, int)
        HOG cell size in pixels.
    cells_per_block   : (int, int)
        Number of cells in each normalisation block.
    orientations      : int
        Number of orientation histogram bins.
    """

    def __init__(
        self,
        hog_image_size:  Tuple[int, int] = (128, 128),
        pixels_per_cell: Tuple[int, int] = (8, 8),
        cells_per_block: Tuple[int, int] = (2, 2),
        orientations:    int             = 9,
    ) -> None:
        self.hog_image_size  = hog_image_size     # (W, H)
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.orientations    = orientations

        # Pre-compute the expected descriptor length so callers can verify
        self.descriptor_length = self._compute_descriptor_length()
        log.debug(
            "ShapeDescriptor initialised: HOG size=%s, descriptor_length=%d",
            hog_image_size, self.descriptor_length,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, segmented_image: np.ndarray) -> np.ndarray:
        """
        Compute the HOG descriptor for *segmented_image*.

        Parameters
        ----------
        segmented_image : np.ndarray  shape (H, W, 3) or (H, W) uint8
            Background-free garment image.

        Returns
        -------
        np.ndarray  shape (descriptor_length,)  float32
            L2-Hys normalised HOG feature vector.
        """
        # ── Resize to canonical HOG resolution ───────────────────────────────
        w, h = self.hog_image_size
        resized = cv2.resize(segmented_image, (w, h), interpolation=cv2.INTER_AREA)

        # ── Convert to grayscale ──────────────────────────────────────────────
        if resized.ndim == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized

        # ── Edge enhancement: apply Canny + blend with original ───────────────
        # This sharpens the silhouette, helping HOG capture the garment outline
        # even when the background removal leaves soft edges.
        gray = self._enhance_silhouette_edges(gray)

        # ── Compute HOG via scikit-image ──────────────────────────────────────
        # block_norm='L2-Hys' — the standard Dalal & Triggs normalisation.
        # feature_vector=True — flatten to 1-D automatically.
        hog_vector = hog(
            gray,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm="L2-Hys",
            visualize=False,
            feature_vector=True,
        )

        return hog_vector.astype(np.float32)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_descriptor_length(self) -> int:
        """
        Calculate the HOG descriptor dimensionality analytically.

        Formula
        -------
        n_cells_per_axis  = (image_dim / pixels_per_cell_dim)
        n_blocks_per_axis = n_cells_per_axis − (cells_per_block_dim − 1)
        total_dims        = ∏ n_blocks × ∏ cells_per_block × orientations
        """
        w, h     = self.hog_image_size
        cpb_r, cpb_c = self.cells_per_block
        ppc_r, ppc_c = self.pixels_per_cell

        n_cells_r = h // ppc_r
        n_cells_c = w // ppc_c
        n_blocks_r = n_cells_r - (cpb_r - 1)
        n_blocks_c = n_cells_c - (cpb_c - 1)

        descriptor_length = n_blocks_r * n_blocks_c * cpb_r * cpb_c * self.orientations
        return descriptor_length

    @staticmethod
    def _enhance_silhouette_edges(gray: np.ndarray) -> np.ndarray:
        """
        Blend a Canny edge map with the grayscale image to emphasise
        garment silhouette and internal structural lines.

        Parameters
        ----------
        gray : np.ndarray  uint8 grayscale

        Returns
        -------
        np.ndarray  uint8 — enhanced grayscale image.
        """
        # Canny with auto-thresholds based on the image's median intensity
        median_val = float(np.median(gray))
        sigma      = 0.33
        lower      = max(0,   int((1.0 - sigma) * median_val))
        upper      = min(255, int((1.0 + sigma) * median_val))

        edges = cv2.Canny(gray, lower, upper)

        # Blend: 70 % original + 30 % edges
        enhanced = cv2.addWeighted(gray, 0.70, edges, 0.30, 0)
        return enhanced
