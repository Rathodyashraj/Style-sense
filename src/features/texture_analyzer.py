from __future__ import annotations

from typing import List

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


# TextureAnalyzer

class TextureAnalyzer:

    def __init__(
        self,
        orientations: List[float] = None,
        wavelengths:  List[float] = None,
        sigma_to_lambda_ratio: float = 0.56,
        gabor_aspect_ratio:    float = 0.5,
    ) -> None:
        self.orientations          = orientations or [0.0, 45.0, 90.0, 135.0]
        self.wavelengths           = wavelengths  or [4.0, 8.0, 16.0, 32.0]
        self.sigma_to_lambda_ratio = sigma_to_lambda_ratio
        self.gabor_aspect_ratio    = gabor_aspect_ratio

        # Pre-build the filter bank at construction time (filters are reused
        # for every image, so this amortises the kernel creation cost).
        self._filter_bank = self._build_filter_bank()

        expected_dims = len(self.orientations) * len(self.wavelengths) * 2
        log.debug(
            "TextureAnalyzer initialised: {n_filt} filters → {d}-dim feature vector.",
            n_filt=len(self._filter_bank),
            d=expected_dims,
        )

    # Public API

    def extract(self, segmented_image: np.ndarray) -> np.ndarray:

        gray = self._to_grayscale(segmented_image)

        # Mask: treat zero-valued (background) pixels separately so they do
        # not dilute foreground statistics.
        fg_mask = gray > 0

        features: List[float] = []

        for kernel in self._filter_bank:
            # convolve with the Gabor kernel; use CV_32F to preserve signs
            response = cv2.filter2D(gray, cv2.CV_32F, kernel)

            # Only consider foreground pixels
            fg_responses = response[fg_mask]

            if fg_responses.size == 0:
                # Degenerate: no foreground pixels
                features.extend([0.0, 0.0])
                continue

            mean = float(np.mean(np.abs(fg_responses)))    # rectified mean energy
            var  = float(np.var(fg_responses))             # irregularity measure
            features.extend([mean, var])

        return np.array(features, dtype=np.float32)

    # Private helpers 

    def _build_filter_bank(self) -> List[np.ndarray]:

        filter_bank: List[np.ndarray] = []

        for wavelength in self.wavelengths:
            sigma = self.sigma_to_lambda_ratio * wavelength

            # Kernel size: 6σ rounded up to the next odd integer
            ksize_val = int(np.ceil(6 * sigma))
            if ksize_val % 2 == 0:
                ksize_val += 1
            ksize = (ksize_val, ksize_val)

            for orientation_deg in self.orientations:
                theta_rad = np.deg2rad(orientation_deg)

               
                kernel = cv2.getGaborKernel(
                    ksize,
                    sigma,
                    theta_rad,
                    wavelength,
                    self.gabor_aspect_ratio,
                    psi=0.0,
                    ktype=cv2.CV_32F,
                )
                # Normalise kernel to zero sum (removes DC component)
                kernel -= kernel.mean()
                filter_bank.append(kernel)

        return filter_bank

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert to uint8 grayscale if needed."""
        if image.ndim == 2:
            return image.astype(np.uint8)
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        raise ValueError(f"Unsupported image shape: {image.shape}")
