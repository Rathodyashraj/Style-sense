from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


# Rule result dataclass
@dataclass
class ColorHarmonyAnalysis:
    overall_score : float
    rule_scores   : Dict[str, float]
    dominant_rule : str
    palette_a     : np.ndarray
    palette_b     : np.ndarray
    hue_angle_a   : np.ndarray
    hue_angle_b   : np.ndarray

    def __str__(self) -> str:
        rule_str = "  ".join(f"{k}={v:.2f}" for k, v in self.rule_scores.items())
        return (
            f"ColorHarmonyAnalysis("
            f"overall={self.overall_score:.3f}, "
            f"dominant='{self.dominant_rule}', "
            f"rules=[{rule_str}])"
        )

# ColorRuleScorer
class ColorRuleScorer:

    RULE_NAMES: List[str] = [
        "complementary",
        "analogous",
        "triadic",
        "split_complementary",
        "monochromatic",
        "neutral",
    ]

    def __init__(
        self,
        n_dominant_colors:       int   = 5,
        complementary_tol:       float = 25.0,
        triadic_tol:             float = 25.0,
        split_comp_tol:          float = 20.0,
        analogous_max:           float = 30.0,
        monochromatic_max:       float = 15.0,
        neutral_chroma_threshold: float = 15.0,
        rule_weights:            Dict[str, float] | None = None,
    ) -> None:
        self.n_dominant_colors        = n_dominant_colors
        self.complementary_tol        = complementary_tol
        self.triadic_tol              = triadic_tol
        self.split_comp_tol           = split_comp_tol
        self.analogous_max            = analogous_max
        self.monochromatic_max        = monochromatic_max
        self.neutral_chroma_threshold = neutral_chroma_threshold

        # Normalise rule weights 
        if rule_weights is None:
            rule_weights = {
                "complementary":      1.5,
                "analogous":          1.5,
                "triadic":            1.0,
                "split_complementary": 1.0,
                "monochromatic":      1.2,
                "neutral":            0.8,
            }
        total_w = sum(rule_weights.values())
        self._weights: Dict[str, float] = {
            k: v / total_w for k, v in rule_weights.items()
        }

        log.debug(
            "ColorRuleScorer initialised | K=%d | weights=%s",
            n_dominant_colors, self._weights,
        )

    # Public API

    def analyse(
        self,
        palette_a: np.ndarray,
        palette_b: np.ndarray,
    ) -> ColorHarmonyAnalysis:

        # Normalise palette shapes to (K, 3)
        pal_a = self._ensure_palette_shape(palette_a)
        pal_b = self._ensure_palette_shape(palette_b)

        # Derive hue angles in [0, 360) from a* and b*
        hue_a = self._palette_to_hue_angles(pal_a)   # shape (K,)
        hue_b = self._palette_to_hue_angles(pal_b)   # shape (K,)

        # Compute chroma (saturation-like) for each colour
        chroma_a = self._palette_to_chroma(pal_a)     # shape (K,)
        chroma_b = self._palette_to_chroma(pal_b)     # shape (K,)

        # All-pairs angular distance matrix
        # delta[i, j] = minimum arc distance between hue_a[i] and hue_b[j]
        # in the circular [0, 360) space, range [0, 180].
        delta = self._hue_angle_distance_matrix(hue_a, hue_b)   # (K, K)

        # Evaluate each rule
        rule_scores: Dict[str, float] = {
            "complementary":       self._score_complementary(delta),
            "analogous":           self._score_analogous(delta),
            "triadic":             self._score_triadic(delta),
            "split_complementary": self._score_split_complementary(delta),
            "monochromatic":       self._score_monochromatic(delta, chroma_a, chroma_b),
            "neutral":             self._score_neutral(chroma_a, chroma_b),
        }

        # Weighted aggregate 
        overall = sum(
            self._weights[rule] * score
            for rule, score in rule_scores.items()
        )

        # Clamp to [0, 1] to guard against floating-point edge cases
        overall = float(np.clip(overall, 0.0, 1.0))

        dominant_rule = max(rule_scores, key=rule_scores.__getitem__)

        analysis = ColorHarmonyAnalysis(
            overall_score = overall,
            rule_scores   = rule_scores,
            dominant_rule = dominant_rule,
            palette_a     = pal_a,
            palette_b     = pal_b,
            hue_angle_a   = hue_a,
            hue_angle_b   = hue_b,
        )
        log.debug("ColorHarmonyAnalysis: %s", analysis)
        return analysis

    def analyse_from_vectors(
        self,
        explicit_a: np.ndarray,
        explicit_b: np.ndarray,
        color_dim: int = None,
    ) -> ColorHarmonyAnalysis:
        if color_dim is None:
            color_dim = self.n_dominant_colors * 3

        palette_a = explicit_a[:color_dim]
        palette_b = explicit_b[:color_dim]
        return self.analyse(palette_a, palette_b)

    # Individual rule scorers

    def _score_complementary(self, delta: np.ndarray) -> float:
        return self._max_pair_score(
            delta,
            target_angle=180.0,
            tolerance=self.complementary_tol,
        )

    def _score_analogous(self, delta: np.ndarray) -> float:
        # Soft weight: 1 at δ=0, 0 at δ=analogous_max
        weights = np.maximum(0.0, 1.0 - delta / self.analogous_max)
        # Mean over all K×K pairs
        return float(weights.mean())

    def _score_triadic(self, delta: np.ndarray) -> float:

        score_120 = self._max_pair_score(delta, target_angle=120.0, tolerance=self.triadic_tol)
        score_240 = self._max_pair_score(delta, target_angle=240.0, tolerance=self.triadic_tol)
        return float(max(score_120, score_240))

    def _score_split_complementary(self, delta: np.ndarray) -> float:
        score_150 = self._max_pair_score(delta, 150.0, self.split_comp_tol)
        score_210 = self._max_pair_score(delta, 210.0, self.split_comp_tol)
        return float(max(score_150, score_210))

    def _score_monochromatic(
        self,
        delta:    np.ndarray,
        chroma_a: np.ndarray,
        chroma_b: np.ndarray,
    ) -> float:

        hue_score = float(
            np.mean(np.maximum(0.0, 1.0 - delta / self.monochromatic_max))
        )

        # Penalise if both palettes are dominated by neutral/achromatic colours
        mean_chroma_a = float(chroma_a.mean())
        mean_chroma_b = float(chroma_b.mean())
        chroma_factor = min(1.0, (mean_chroma_a + mean_chroma_b) / (2 * self.neutral_chroma_threshold))

        return float(hue_score * chroma_factor)

    def _score_neutral(
        self,
        chroma_a: np.ndarray,
        chroma_b: np.ndarray,
    ) -> float:
        neutral_frac_a = float((chroma_a < self.neutral_chroma_threshold).mean())
        neutral_frac_b = float((chroma_b < self.neutral_chroma_threshold).mean())
        # Geometric mean — both must be neutral for a high score
        return float(np.sqrt(neutral_frac_a * neutral_frac_b))

    #Geometry helpers 

    @staticmethod
    def _palette_to_hue_angles(palette: np.ndarray) -> np.ndarray:

        a_star = palette[:, 1]
        b_star = palette[:, 2]
        angles_rad = np.arctan2(b_star, a_star)          # ∈ (-π, π]
        angles_deg = np.degrees(angles_rad) % 360.0       # wrap to [0, 360)
        return angles_deg

    @staticmethod
    def _palette_to_chroma(palette: np.ndarray) -> np.ndarray:
        return np.sqrt(palette[:, 1] ** 2 + palette[:, 2] ** 2)

    @staticmethod
    def _hue_angle_distance_matrix(
        hue_a: np.ndarray,
        hue_b: np.ndarray,
    ) -> np.ndarray:

        # Broadcasting: hue_a[:, None] - hue_b[None, :] → (K_a, K_b)
        raw_diff = np.abs(hue_a[:, np.newaxis] - hue_b[np.newaxis, :])
        return np.minimum(raw_diff, 360.0 - raw_diff)

    def _max_pair_score(
        self,
        delta:        np.ndarray,
        target_angle: float,
        tolerance:    float,
    ) -> float:
        deviation = np.abs(delta - target_angle)
        scores    = np.maximum(0.0, 1.0 - deviation / tolerance)
        return float(scores.max())

    # Reshape helpers 

    def _ensure_palette_shape(self, palette: np.ndarray) -> np.ndarray:

        if palette.ndim == 1:
            # Flat vector from ColorHarmonyExtractor.extract() → reshape to (K, 3)
            expected_flat = self.n_dominant_colors * 3
            if palette.shape[0] != expected_flat:
                raise ValueError(
                    f"Expected flat palette of length {expected_flat} "
                    f"(n_dominant_colors={self.n_dominant_colors} × 3), "
                    f"got {palette.shape[0]}."
                )
            return palette.reshape(self.n_dominant_colors, 3)

        if palette.ndim == 2 and palette.shape[1] == 3:
            return palette

        raise ValueError(
            f"palette must be shape (K*3,) or (K, 3), got {palette.shape}."
        )
