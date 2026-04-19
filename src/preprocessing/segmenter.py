from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)



# GrabCutSegmenter


class GrabCutSegmenter:

    def __init__(
        self,
        grabcut_iterations: int = 10,
        border_margin: int = 10,
        min_foreground_ratio: float = 0.05,
    ) -> None:
        self.grabcut_iterations   = grabcut_iterations
        self.border_margin        = border_margin
        self.min_foreground_ratio = min_foreground_ratio

    # Public API

    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected a 3-channel BGR image, got shape {image.shape}"
            )

        h, w = image.shape[:2]

        # Step 1 : Build the initialisation rectangle
        m = self.border_margin
        rect = (m, m, w - 2 * m, h - 2 * m)

        # Step 2 : Allocate GrabCut working arrays
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        # Initialise the mask: OpenCV fills it based on the rectangle.
        gc_mask = np.zeros((h, w), dtype=np.uint8)

        # Step 3 : Run GrabCut
        try:
            cv2.grabCut(
                image,
                gc_mask,
                rect,
                bgd_model,
                fgd_model,
                self.grabcut_iterations,
                cv2.GC_INIT_WITH_RECT,   
            )
        except cv2.error as exc:
            log.warning("GrabCut failed ({err}). Returning original image.", err=str(exc))
            full_mask = np.full((h, w), 255, dtype=np.uint8)
            return image.copy(), full_mask

        #  Step 4 : Convert GrabCut labels to binary mask
        foreground_mask = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
            np.uint8(255),
            np.uint8(0),
        )

        # Step 5 : Post-process — fill holes and remove specks 
        foreground_mask = self._clean_mask(foreground_mask)

        # Step 6 : Sanity check — fall back if foreground is too small
        fg_ratio = foreground_mask.astype(bool).sum() / (h * w)
        if fg_ratio < self.min_foreground_ratio:
            log.warning(
                "Foreground ratio {r:.2%} is below threshold {t:.2%}. "
                "Returning unmasked image.",
                r=fg_ratio, t=self.min_foreground_ratio,
            )
            foreground_mask = np.full((h, w), 255, dtype=np.uint8)

        # Step 7 : Apply mask to image
        mask_3ch = cv2.merge([foreground_mask, foreground_mask, foreground_mask])
        segmented = cv2.bitwise_and(image, mask_3ch)

        return segmented, foreground_mask

    # Private helpers

    @staticmethod
    def _clean_mask(mask: np.ndarray) -> np.ndarray:

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        if num_labels <= 1:
            return mask    

        areas = stats[1:, cv2.CC_STAT_AREA]      
        largest_label = np.argmax(areas) + 1      

        cleaned = np.where(labels == largest_label, np.uint8(255), np.uint8(0))
        return cleaned
