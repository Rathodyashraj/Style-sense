from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from skimage.feature import hog

from src.utils.logger import get_logger

log = get_logger(__name__)



# ShapeDescriptor

class ShapeDescriptor:


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

    # Public API 

    def extract(self, segmented_image: np.ndarray) -> np.ndarray:

        # Resize to canonical HOG resolution 
        w, h = self.hog_image_size
        resized = cv2.resize(segmented_image, (w, h), interpolation=cv2.INTER_AREA)

        # Convert to grayscale 
        if resized.ndim == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized

        gray = self._enhance_silhouette_edges(gray)


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


    def _compute_descriptor_length(self) -> int:
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
        # Canny with auto-thresholds based on the image's median intensity
        median_val = float(np.median(gray))
        sigma      = 0.33
        lower      = max(0,   int((1.0 - sigma) * median_val))
        upper      = min(255, int((1.0 + sigma) * median_val))

        edges = cv2.Canny(gray, lower, upper)

        # Blend: 70 % original + 30 % edges
        enhanced = cv2.addWeighted(gray, 0.70, edges, 0.30, 0)
        return enhanced
