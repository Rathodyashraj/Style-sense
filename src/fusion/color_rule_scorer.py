"""
src/fusion/color_rule_scorer.py
────────────────────────────────
Deterministic Colour Harmony Rule Engine

Overview
--------
This module encodes classical colour theory as a set of *explicit, interpretable*
mathematical rules applied to the dominant-colour palettes extracted by
``ColorHarmonyExtractor``.  It provides a supplementary harmony signal that is
human-interpretable and does not depend on any trained model weights.

The rules are applied in CIE L*a*b* space — specifically using the **hue angle**
derived from the a* and b* axes, which defines a perceptually uniform colour
wheel.

Colour Harmony Rules Implemented
---------------------------------
┌──────────────────────────────┬──────────────────────────────────────────────┐
│ Rule                         │ Geometric definition (hue-angle δ)           │
├──────────────────────────────┼──────────────────────────────────────────────┤
│ Complementary                │ δ ≈ 180 ° ± tolerance                       │
│ Analogous                    │ δ ≤ 30 ° (adjacent hues)                    │
│ Triadic                      │ δ ≈ 120 ° or 240 ° ± tolerance              │
│ Split-complementary          │ δ ≈ 150 ° or 210 ° ± tolerance              │
│ Neutral / Achromatic         │ low chroma C* = √(a*² + b*²) for both items │
│ Monochromatic                │ same hue, varying lightness (δ ≤ 15 °)      │
└──────────────────────────────┴──────────────────────────────────────────────┘

Scoring approach
----------------
For two garments A and B, each characterised by K dominant colours, we compute
an ALL-PAIRS hue-angle distance matrix (K × K) and evaluate how many pairs
satisfy each rule within the configured tolerance window.  The fraction of
satisfying pairs drives a per-rule sub-score, which is then combined into a
single **colour harmony score** ∈ [0.0, 1.0].

This score is exposed to the ``HarmonyScorer`` and can be:
  1. Used as an additional feature for the SVM/MLP (by appending it to the
     pairwise feature vector at training time).
  2. Used as a soft regulariser to the final harmony percentage.
  3. Surfaced directly in the UI alongside the ML score for interpretability.

References
----------
Itten, J. (1961). The Art of Color. Reinhold Publishing.
Moon, P., & Spencer, D. E. (1944). Geometric formulation of classical color
harmony. Journal of the Optical Society of America.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Rule result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ColorHarmonyAnalysis:
    """
    Detailed breakdown of the colour harmony evaluation between two garments.

    Attributes
    ----------
    overall_score : float
        Weighted aggregate colour harmony score ∈ [0.0, 1.0].
    rule_scores   : dict  rule_name → float ∈ [0.0, 1.0]
        Per-rule satisfaction fractions.
    dominant_rule : str
        The rule with the highest individual score.
    palette_a     : np.ndarray  (K, 3)  CIE L*a*b* centres for garment A.
    palette_b     : np.ndarray  (K, 3)  CIE L*a*b* centres for garment B.
    hue_angle_a   : np.ndarray  (K,)    Hue angles in degrees for garment A.
    hue_angle_b   : np.ndarray  (K,)    Hue angles in degrees for garment B.
    """
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


# ---------------------------------------------------------------------------
# ColorRuleScorer
# ---------------------------------------------------------------------------

class ColorRuleScorer:
    """
    Evaluates perceptual colour harmony between two garments using
    classical colour theory rules applied to their dominant-colour palettes.

    Parameters
    ----------
    n_dominant_colors : int
        Number of dominant colours per garment (must match ColorHarmonyExtractor).
        Default 5.
    complementary_tol : float
        Angular tolerance (degrees) around 180° to accept as complementary.
        Default ±25°.
    triadic_tol       : float
        Angular tolerance around 120° / 240° for triadic rule.  Default ±25°.
    split_comp_tol    : float
        Angular tolerance around 150° / 210° for split-complementary.
        Default ±20°.
    analogous_max     : float
        Maximum hue-angle difference (°) to qualify as analogous.  Default 30°.
    monochromatic_max : float
        Maximum hue-angle difference (°) to qualify as monochromatic.  Default 15°.
    neutral_chroma_threshold : float
        Maximum C* (chroma) for a colour to be considered achromatic/neutral.
        Default 15.0  (C* ∈ [0, ~180] in CIE L*a*b*).
    rule_weights      : dict | None
        Weight of each rule when computing the overall score.  If None,
        equal weights are used.  Weights are normalised internally.
    """

    # Names of all implemented rules — order determines column order in output
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

        # ── Normalise rule weights ─────────────────────────────────────────────
        if rule_weights is None:
            # Default: complementary and analogous are the most commonly cited
            # and are weighted slightly higher in fashion literature.
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

    # ── Public API ────────────────────────────────────────────────────────────

    def analyse(
        self,
        palette_a: np.ndarray,
        palette_b: np.ndarray,
    ) -> ColorHarmonyAnalysis:
        """
        Analyse colour harmony between two garment palettes.

        Parameters
        ----------
        palette_a, palette_b : np.ndarray  shape (K,) or (K*3,)  float32
            Dominant colour palettes in CIE L*a*b*.  Accepts either the raw
            flattened feature vector from ``ColorHarmonyExtractor.extract()``
            (shape K×3) or a pre-reshaped (K, 3) array.

        Returns
        -------
        ColorHarmonyAnalysis
        """
        # ── Normalise palette shapes to (K, 3) ───────────────────────────────
        pal_a = self._ensure_palette_shape(palette_a)
        pal_b = self._ensure_palette_shape(palette_b)

        # ── Derive hue angles in [0, 360) from a* and b* ─────────────────────
        hue_a = self._palette_to_hue_angles(pal_a)   # shape (K,)
        hue_b = self._palette_to_hue_angles(pal_b)   # shape (K,)

        # ── Compute chroma (saturation-like) for each colour ──────────────────
        chroma_a = self._palette_to_chroma(pal_a)     # shape (K,)
        chroma_b = self._palette_to_chroma(pal_b)     # shape (K,)

        # ── All-pairs angular distance matrix ─────────────────────────────────
        # delta[i, j] = minimum arc distance between hue_a[i] and hue_b[j]
        # in the circular [0, 360) space, range [0, 180].
        delta = self._hue_angle_distance_matrix(hue_a, hue_b)   # (K, K)

        # ── Evaluate each rule ────────────────────────────────────────────────
        rule_scores: Dict[str, float] = {
            "complementary":       self._score_complementary(delta),
            "analogous":           self._score_analogous(delta),
            "triadic":             self._score_triadic(delta),
            "split_complementary": self._score_split_complementary(delta),
            "monochromatic":       self._score_monochromatic(delta, chroma_a, chroma_b),
            "neutral":             self._score_neutral(chroma_a, chroma_b),
        }

        # ── Weighted aggregate ─────────────────────────────────────────────────
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
        """
        Convenience wrapper: extract the colour sub-vector from full explicit
        feature vectors and run ``analyse()``.

        Parameters
        ----------
        explicit_a, explicit_b : np.ndarray  shape (D_explicit,)
            Full explicit feature vectors from ``ExplicitFeatureExtractor``.
        color_dim : int | None
            Length of the colour sub-vector at the *start* of the explicit
            vector.  Defaults to ``n_dominant_colors × 3``.

        Returns
        -------
        ColorHarmonyAnalysis
        """
        if color_dim is None:
            color_dim = self.n_dominant_colors * 3

        palette_a = explicit_a[:color_dim]
        palette_b = explicit_b[:color_dim]
        return self.analyse(palette_a, palette_b)

    # ── Individual rule scorers ───────────────────────────────────────────────

    def _score_complementary(self, delta: np.ndarray) -> float:
        """
        Complementary colours sit opposite each other on the colour wheel (δ ≈ 180°).

        A pair of garments scores high if *at least one* dominant colour from A
        is complementary to *at least one* dominant colour from B.

        We use a soft scoring window: the score falls off linearly from 1.0 at
        exactly 180° to 0.0 at (180 ± tolerance).
        """
        return self._max_pair_score(
            delta,
            target_angle=180.0,
            tolerance=self.complementary_tol,
        )

    def _score_analogous(self, delta: np.ndarray) -> float:
        """
        Analogous colours are adjacent on the wheel (δ ≤ 30°).

        High score when *most* dominant-colour pairs are closely spaced in hue.
        We score the *fraction* of pairs that satisfy δ ≤ analogous_max,
        weighted by their proximity (closer = higher).
        """
        # Soft weight: 1 at δ=0, 0 at δ=analogous_max
        weights = np.maximum(0.0, 1.0 - delta / self.analogous_max)
        # Mean over all K×K pairs
        return float(weights.mean())

    def _score_triadic(self, delta: np.ndarray) -> float:
        """
        Triadic: colours evenly spaced at 120° intervals (δ ≈ 120° or 240°).

        Score the best single pair that lies near 120° or 240°.  We take the
        max over both targets because the 240° arc is equivalent to -120°.
        """
        score_120 = self._max_pair_score(delta, target_angle=120.0, tolerance=self.triadic_tol)
        score_240 = self._max_pair_score(delta, target_angle=240.0, tolerance=self.triadic_tol)
        return float(max(score_120, score_240))

    def _score_split_complementary(self, delta: np.ndarray) -> float:
        """
        Split-complementary: one colour plus the two colours adjacent to its complement
        (δ ≈ 150° or 210°).

        Like triadic, we score the single best pair at either target angle.
        """
        score_150 = self._max_pair_score(delta, 150.0, self.split_comp_tol)
        score_210 = self._max_pair_score(delta, 210.0, self.split_comp_tol)
        return float(max(score_150, score_210))

    def _score_monochromatic(
        self,
        delta:    np.ndarray,
        chroma_a: np.ndarray,
        chroma_b: np.ndarray,
    ) -> float:
        """
        Monochromatic: same hue family, different lightness / chroma.

        Requires:
          1. δ ≤ monochromatic_max (very small hue difference).
          2. At least one garment has reasonable chroma (not purely neutral),
             so we aren't just scoring two achromatic garments as "monochromatic".

        Score is the fraction of pairs that satisfy the hue proximity condition,
        discounted if both garments are achromatic.
        """
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
        """
        Neutral / Achromatic harmony: both garments are dominated by colours
        with very low chroma (blacks, whites, greys, beiges).

        Score is the product of the fraction of neutral colours in each palette.
        """
        neutral_frac_a = float((chroma_a < self.neutral_chroma_threshold).mean())
        neutral_frac_b = float((chroma_b < self.neutral_chroma_threshold).mean())
        # Geometric mean — both must be neutral for a high score
        return float(np.sqrt(neutral_frac_a * neutral_frac_b))

    # ── Geometry helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _palette_to_hue_angles(palette: np.ndarray) -> np.ndarray:
        """
        Convert CIE L*a*b* colour centres to hue angles in degrees.

        Hue angle h* = atan2(b*, a*)  mapped to [0, 360).

        Parameters
        ----------
        palette : np.ndarray  shape (K, 3)  — [L*, a*, b*] per row

        Returns
        -------
        np.ndarray  shape (K,)  — hue angles in [0, 360)
        """
        a_star = palette[:, 1]
        b_star = palette[:, 2]
        angles_rad = np.arctan2(b_star, a_star)          # ∈ (-π, π]
        angles_deg = np.degrees(angles_rad) % 360.0       # wrap to [0, 360)
        return angles_deg

    @staticmethod
    def _palette_to_chroma(palette: np.ndarray) -> np.ndarray:
        """
        Compute CIE chroma  C* = √(a*² + b*²)  for each colour in the palette.

        Parameters
        ----------
        palette : np.ndarray  shape (K, 3)  — [L*, a*, b*] per row

        Returns
        -------
        np.ndarray  shape (K,)  — chroma values ≥ 0
        """
        return np.sqrt(palette[:, 1] ** 2 + palette[:, 2] ** 2)

    @staticmethod
    def _hue_angle_distance_matrix(
        hue_a: np.ndarray,
        hue_b: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the all-pairs *circular* arc distance between two sets of hue angles.

        The circular distance between angles α and β is:
            δ(α, β) = min(|α − β|, 360 − |α − β|)

        This is NOT the signed difference — it gives the shorter arc, ∈ [0, 180].

        Parameters
        ----------
        hue_a : np.ndarray  shape (K,)
        hue_b : np.ndarray  shape (K,)

        Returns
        -------
        np.ndarray  shape (K, K)  — distance[i, j] = arc distance between
                                   hue_a[i] and hue_b[j].
        """
        # Broadcasting: hue_a[:, None] - hue_b[None, :] → (K_a, K_b)
        raw_diff = np.abs(hue_a[:, np.newaxis] - hue_b[np.newaxis, :])
        return np.minimum(raw_diff, 360.0 - raw_diff)

    def _max_pair_score(
        self,
        delta:        np.ndarray,
        target_angle: float,
        tolerance:    float,
    ) -> float:
        """
        Compute the maximum soft-match score over all (i, j) pairs for a rule
        centred on *target_angle*.

        The soft score for a pair is:
            score(i, j) = max(0,  1 − |delta[i,j] − target_angle| / tolerance)

        This is a tent function: 1.0 at exactly *target_angle*, linearly
        decreasing to 0.0 at *target_angle ± tolerance*.

        We return the *maximum* over all pairs because the garment only needs
        *one* strongly-matching pair to satisfy the rule (e.g., one
        complementary accent colour is sufficient for complementary harmony).
        """
        deviation = np.abs(delta - target_angle)
        scores    = np.maximum(0.0, 1.0 - deviation / tolerance)
        return float(scores.max())

    # ── Reshape helpers ───────────────────────────────────────────────────────

    def _ensure_palette_shape(self, palette: np.ndarray) -> np.ndarray:
        """
        Accept either a flat (K×3,) vector from ``ColorHarmonyExtractor``
        or an already-shaped (K, 3) array, and return shape (K, 3).
        """
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
