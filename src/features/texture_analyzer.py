"""
src/features/texture_analyzer.py
─────────────────────────────────
Module 2 (Part B) — Texture Signature via Gabor Filter Bank

Theory
------
A Gabor filter is a sinusoidal plane wave modulated by a Gaussian envelope.
It is jointly localised in *both* spatial and frequency domains — analogous
to the human visual cortex's simple-cell receptive fields.

For texture analysis we build a *filter bank* by varying two parameters:

  θ  (orientation) — detects edges / ridges at a specific angle.
               Stripe fabric → strong response at the stripe's normal angle.
               Smooth fabric  → uniformly low response across all θ.

  λ  (wavelength) — controls the spatial scale of the pattern.
               Fine weave → strong response at small λ.
               Large print → strong response at large λ.

Per filter we compute:
  * mean response  — encodes average texture energy at this (θ, λ).
  * variance       — encodes texture irregularity / randomness.

Feature vector length = n_orientations × n_wavelengths × 2
                      = 4 × 4 × 2 = 32 dimensions (default).

References
----------
Manjunath, B. S., & Ma, W. Y. (1996).
Texture features for browsing and retrieval of image data.
IEEE Transactions on Pattern Analysis and Machine Intelligence.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# TextureAnalyzer
# ---------------------------------------------------------------------------

class TextureAnalyzer:
    """
    Builds a Gabor filter bank and extracts a texture signature vector.

    Parameters
    ----------
    orientations           : list of float
        Gabor orientations in *degrees* (converted to radians internally).
        Typical choice: [0, 45, 90, 135] — four cardinal directions.
    wavelengths            : list of float
        Gabor wavelengths (λ) in pixels.  Covers fine-to-coarse scales.
    sigma_to_lambda_ratio  : float
        σ = sigma_to_lambda_ratio × λ.  Controls envelope width relative to
        wavelength.  0.56 is a standard neurologically-inspired value.
    gabor_aspect_ratio     : float
        γ — spatial aspect ratio of the Gaussian envelope (γ=1 → isotropic).
        0.5 makes the filter elongated along the orientation axis.
    """

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

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, segmented_image: np.ndarray) -> np.ndarray:
        """
        Compute the Gabor texture signature for a garment image.

        Parameters
        ----------
        segmented_image : np.ndarray  shape (H, W, 3) or (H, W), uint8
            Background-free garment image.  If 3-channel, converted to
            grayscale internally before filtering.

        Returns
        -------
        np.ndarray  shape (n_orientations * n_wavelengths * 2,)  float32
            [mean_0, var_0, mean_1, var_1, ..., mean_N, var_N]
            where N = n_orientations × n_wavelengths.
        """
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

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_filter_bank(self) -> List[np.ndarray]:
        """
        Pre-compute one Gabor kernel per (orientation, wavelength) combination.

        Kernel size is chosen automatically as the smallest odd integer that
        fits 6σ (three standard deviations on each side).

        Returns
        -------
        list of np.ndarray  — list of 2-D float32 Gabor kernels.
        """
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

                # cv2.getGaborKernel arguments:
                #   ksize  — kernel size (must be odd)
                #   sigma  — std dev of the Gaussian envelope
                #   theta  — orientation of the normal to the parallel stripes (radians)
                #   lambd  — wavelength of the cosine factor
                #   gamma  — spatial aspect ratio
                #   psi    — phase offset (0 = even/symmetric filter)
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
