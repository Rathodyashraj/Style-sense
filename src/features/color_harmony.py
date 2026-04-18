from __future__ import annotations

import numpy as np
import cv2
from sklearn.cluster import KMeans

from src.utils.logger import get_logger

log = get_logger(__name__)



# ColorHarmonyExtractor
class ColorHarmonyExtractor:
    """
    Extracts a dominant-colour palette in CIE L*a*b* space using K-Means.

    Parameters
    ----------
    n_dominant_colors : int
        Number of colour clusters (K in K-Means). Default 5.
    kmeans_max_iter   : int
        Maximum EM iterations for K-Means convergence. Default 300.
    kmeans_n_init     : int
        Number of times K-Means is re-run with different centroid seeds.
        The best result (lowest inertia) is kept. Default 10.
    max_pixel_sample  : int
        Cap on the number of foreground pixels fed to K-Means.
        Pixels are drawn by uniform random sub-sampling. Default 50_000.
    """

    def __init__(
        self,
        n_dominant_colors: int = 5,
        kmeans_max_iter: int = 300,
        kmeans_n_init: int = 10,
        max_pixel_sample: int = 50_000,
    ) -> None:
        self.n_dominant_colors = n_dominant_colors
        self.kmeans_max_iter   = kmeans_max_iter
        self.kmeans_n_init     = kmeans_n_init
        self.max_pixel_sample  = max_pixel_sample

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, segmented_image: np.ndarray) -> np.ndarray:
        """
        Compute the dominant colour palette for a background-free garment image.

        Parameters
        ----------
        segmented_image : np.ndarray  shape (H, W, 3) BGR uint8
            Garment image with background pixels zeroed out (from GrabCut).

        Returns
        -------
        np.ndarray  shape (n_dominant_colors * 3,)  float32
            Flattened array of K cluster centres in CIE L*a*b* space.
            Layout: [L0, a0, b0, L1, a1, b1, ..., L_{K-1}, a_{K-1}, b_{K-1}]

        Notes
        -----
        If the foreground pixel count is less than *n_dominant_colors*, the
        function pads with the mean foreground colour repeated K times.
        """
        # ── Convert BGR → CIE L*a*b* ─────────────────────────────────────────
        # cv2.COLOR_BGR2LAB maps:
        #   L*  ∈ [0,   100]  — perceptual lightness
        #   a*  ∈ [-127, 127] — green (−) to red (+)
        #   b*  ∈ [-127, 127] — blue (−) to yellow (+)
        # OpenCV internally scales to [0,255] storage; we convert back to
        # canonical float ranges for interpretability.
        lab_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2LAB)
        lab_float = self._opencv_lab_to_canonical(lab_image)

        # ── Collect foreground pixels (exclude pure-black background) ─────────
        # A pixel is background if all three channels are exactly 0 in the
        # original BGR image.
        bg_mask = (
            (segmented_image[:, :, 0] == 0) &
            (segmented_image[:, :, 1] == 0) &
            (segmented_image[:, :, 2] == 0)
        )
        fg_pixels = lab_float[~bg_mask]   # shape (N_fg, 3)

        n_fg = len(fg_pixels)
        log.debug("Foreground pixel count: {n}", n=n_fg)

        # ── Handle degenerate case ────────────────────────────────────────────
        if n_fg < self.n_dominant_colors:
            log.warning(
                "Too few foreground pixels ({n}) for K={k} clusters. "
                "Returning mean colour repeated K times.",
                n=n_fg, k=self.n_dominant_colors,
            )
            if n_fg == 0:
                placeholder = np.zeros(self.n_dominant_colors * 3, dtype=np.float32)
                return placeholder
            mean_colour = fg_pixels.mean(axis=0)          # shape (3,)
            palette = np.tile(mean_colour, self.n_dominant_colors)
            return palette.astype(np.float32)

        # ── Sub-sample if pixel count exceeds threshold ───────────────────────
        if n_fg > self.max_pixel_sample:
            rng = np.random.default_rng(seed=42)
            idx = rng.choice(n_fg, size=self.max_pixel_sample, replace=False)
            fg_pixels = fg_pixels[idx]

        # ── K-Means clustering in L*a*b* space ───────────────────────────────
        kmeans = KMeans(
            n_clusters=self.n_dominant_colors,
            max_iter=self.kmeans_max_iter,
            n_init=self.kmeans_n_init,
            random_state=42,
        )
        kmeans.fit(fg_pixels)

        # Cluster centres in CIE L*a*b*; sort by cluster size descending
        centres   = kmeans.cluster_centers_          # shape (K, 3)
        counts    = np.bincount(kmeans.labels_, minlength=self.n_dominant_colors)
        sort_idx  = np.argsort(-counts)              # descending frequency
        palette   = centres[sort_idx]                # shape (K, 3)

        # ── Flatten to 1-D feature vector ────────────────────────────────────
        feature_vector = palette.flatten().astype(np.float32)
        return feature_vector

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _opencv_lab_to_canonical(lab_uint8: np.ndarray) -> np.ndarray:
        """
        Convert OpenCV's uint8 L*a*b* storage to canonical floating-point ranges.

        OpenCV stores CIE L*a*b* as:
            L_stored = L* * (255 / 100)
            a_stored = a* + 128
            b_stored = b* + 128

        We invert this to recover L* ∈ [0,100], a* / b* ∈ [-127, 127].
        """
        lab = lab_uint8.astype(np.float32)
        lab[:, :, 0] = lab[:, :, 0] * (100.0 / 255.0)   # L* → [0, 100]
        lab[:, :, 1] = lab[:, :, 1] - 128.0              # a* → [-128, 127]
        lab[:, :, 2] = lab[:, :, 2] - 128.0              # b* → [-128, 127]
        return lab
