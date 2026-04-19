from __future__ import annotations

import numpy as np
import cv2
from sklearn.cluster import KMeans

from src.utils.logger import get_logger

log = get_logger(__name__)



# ColorHarmonyExtractor
class ColorHarmonyExtractor:

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

    # Public API

    def extract(self, segmented_image: np.ndarray) -> np.ndarray:


        lab_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2LAB)
        lab_float = self._opencv_lab_to_canonical(lab_image)

        bg_mask = (
            (segmented_image[:, :, 0] == 0) &
            (segmented_image[:, :, 1] == 0) &
            (segmented_image[:, :, 2] == 0)
        )
        fg_pixels = lab_float[~bg_mask]   # shape (N_fg, 3)

        n_fg = len(fg_pixels)
        log.debug("Foreground pixel count: {n}", n=n_fg)

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

        # Sub-sample if pixel count exceeds threshold
        if n_fg > self.max_pixel_sample:
            rng = np.random.default_rng(seed=42)
            idx = rng.choice(n_fg, size=self.max_pixel_sample, replace=False)
            fg_pixels = fg_pixels[idx]

        # K-Means clustering in L*a*b* space
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

        # ── Flatten to 1-D feature vector 
        feature_vector = palette.flatten().astype(np.float32)
        return feature_vector

    #  Private helpers

    @staticmethod
    def _opencv_lab_to_canonical(lab_uint8: np.ndarray) -> np.ndarray:
        lab = lab_uint8.astype(np.float32)
        lab[:, :, 0] = lab[:, :, 0] * (100.0 / 255.0)   # L* → [0, 100]
        lab[:, :, 1] = lab[:, :, 1] - 128.0              # a* → [-128, 127]
        lab[:, :, 2] = lab[:, :, 2] - 128.0              # b* → [-128, 127]
        return lab
