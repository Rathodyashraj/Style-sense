"""
src/preprocessing/segmenter.py
────────────────────────────────
Module 1 — Pre-Processing & Graph-Based Segmentation

Algorithm
---------
OpenCV's ``grabCut`` implements the GrabCut algorithm (Rother et al., 2004):

1.  **Seed initialisation** — pixels inside a tight inner rectangle are
    marked as *probable foreground*; a fixed-width border region around the
    image edge is marked as *definite background*.

2.  **GMM fitting** — Two 5-component Gaussian Mixture Models (one for
    foreground, one for background) are fitted to the seeded pixels in the
    CIE L*a*b* colour space.  Each pixel is assigned to the GMM component
    with the highest likelihood, providing per-pixel colour probabilities.

3.  **Graph / Min-Cut** — The algorithm builds a pixel-level Markov Random
    Field whose energy encodes both colour likelihood (data term) and spatial
    smoothness (smoothness term).  The min-cut of this graph separates
    foreground from background.

4.  **EM iteration** — Steps 2–3 are alternated for ``grabcut_iterations``
    rounds until convergence.

5.  **Mask extraction** — Pixels labelled GC_FGD or GC_PR_FGD form the
    foreground binary mask returned to the caller.

References
----------
Rother, C., Kolmogorov, V., & Blake, A. (2004). GrabCut: Interactive
foreground extraction using iterated graph cuts. ACM SIGGRAPH.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# GrabCutSegmenter
# ---------------------------------------------------------------------------

class GrabCutSegmenter:
    """
    Isolates the garment from background using GrabCut + GMM.

    Parameters
    ----------
    grabcut_iterations : int
        Number of GrabCut EM iterations (more ≈ cleaner but slower).
    border_margin : int
        Width (pixels) of the image border treated as *definite background*.
        A larger margin is safer for images where the garment touches the edge.
    min_foreground_ratio : float
        If the foreground mask covers less than this fraction of the image
        area after segmentation, the result is considered a failure and the
        full image (no masking) is returned as a fallback.
    """

    def __init__(
        self,
        grabcut_iterations: int = 10,
        border_margin: int = 10,
        min_foreground_ratio: float = 0.05,
    ) -> None:
        self.grabcut_iterations   = grabcut_iterations
        self.border_margin        = border_margin
        self.min_foreground_ratio = min_foreground_ratio

    # ── Public API ────────────────────────────────────────────────────────────

    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run GrabCut on *image* and return the masked foreground.

        Parameters
        ----------
        image : np.ndarray  shape (H, W, 3) BGR uint8

        Returns
        -------
        segmented_image : np.ndarray  (H, W, 3) BGR uint8
            Original image with background pixels zeroed out.
        mask : np.ndarray  (H, W) uint8
            Binary mask: 255 = foreground, 0 = background.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected a 3-channel BGR image, got shape {image.shape}"
            )

        h, w = image.shape[:2]

        # ── Step 1 : Build the initialisation rectangle ───────────────────────
        # GrabCut requires a rectangle that tightly encloses the foreground.
        # We place it *border_margin* pixels inset from each edge.
        m = self.border_margin
        rect = (m, m, w - 2 * m, h - 2 * m)

        # ── Step 2 : Allocate GrabCut working arrays ──────────────────────────
        # bgd_model / fgd_model: internal GMM state arrays (must be float64,
        # shape (1, 65) each as required by OpenCV).
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        # Initialise the mask: OpenCV fills it based on the rectangle.
        gc_mask = np.zeros((h, w), dtype=np.uint8)

        # ── Step 3 : Run GrabCut ──────────────────────────────────────────────
        try:
            cv2.grabCut(
                image,
                gc_mask,
                rect,
                bgd_model,
                fgd_model,
                self.grabcut_iterations,
                cv2.GC_INIT_WITH_RECT,   # initialise via rectangle (first call)
            )
        except cv2.error as exc:
            log.warning("GrabCut failed ({err}). Returning original image.", err=str(exc))
            full_mask = np.full((h, w), 255, dtype=np.uint8)
            return image.copy(), full_mask

        # ── Step 4 : Convert GrabCut labels to binary mask ───────────────────
        # GrabCut uses four labels:
        #   GC_BGD    (0) — definite background
        #   GC_FGD    (1) — definite foreground
        #   GC_PR_BGD (2) — probable background
        #   GC_PR_FGD (3) — probable foreground
        # We treat both FGD and PR_FGD as foreground.
        foreground_mask = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
            np.uint8(255),
            np.uint8(0),
        )

        # ── Step 5 : Post-process — fill holes and remove specks ──────────────
        foreground_mask = self._clean_mask(foreground_mask)

        # ── Step 6 : Sanity check — fall back if foreground is too small ──────
        fg_ratio = foreground_mask.astype(bool).sum() / (h * w)
        if fg_ratio < self.min_foreground_ratio:
            log.warning(
                "Foreground ratio {r:.2%} is below threshold {t:.2%}. "
                "Returning unmasked image.",
                r=fg_ratio, t=self.min_foreground_ratio,
            )
            foreground_mask = np.full((h, w), 255, dtype=np.uint8)

        # ── Step 7 : Apply mask to image ──────────────────────────────────────
        mask_3ch = cv2.merge([foreground_mask, foreground_mask, foreground_mask])
        segmented = cv2.bitwise_and(image, mask_3ch)

        return segmented, foreground_mask

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _clean_mask(mask: np.ndarray) -> np.ndarray:
        """
        Morphological post-processing to fill small holes and remove noise.

        1. Closing  — fills gaps *within* the garment silhouette.
        2. Opening  — removes tiny isolated foreground specks.
        3. Largest connected component — keeps only the biggest foreground blob
           (handles cases where GrabCut leaks to background patches).
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # Close: dilation then erosion — joins broken edges
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        # Open:  erosion then dilation — removes tiny specks
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)

        # Keep only the largest connected component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        if num_labels <= 1:
            return mask    # no foreground at all — return as-is

        # stats row 0 is the background; find the largest non-background component
        areas = stats[1:, cv2.CC_STAT_AREA]      # skip background (index 0)
        largest_label = np.argmax(areas) + 1      # +1 because we sliced off index 0

        cleaned = np.where(labels == largest_label, np.uint8(255), np.uint8(0))
        return cleaned
