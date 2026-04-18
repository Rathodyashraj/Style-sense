"""
tests/test_color_rules.py
──────────────────────────
Unit tests for the ColorRuleScorer (src/fusion/color_rule_scorer.py).

Tests verify:
    * Each individual rule fires correctly on hand-crafted palettes.
    * Overall score is in [0.0, 1.0].
    * Output dataclass fields are populated.
    * Edge cases (achromatic palettes, identical palettes).
    * analyse_from_vectors() extracts the palette sub-vector correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fusion.color_rule_scorer import ColorRuleScorer, ColorHarmonyAnalysis


# ---------------------------------------------------------------------------
# Palette factory helpers
# ---------------------------------------------------------------------------

def _make_palette_flat(
    hues_deg: list,
    chroma: float = 50.0,
    lightness: float = 60.0,
    n_colors: int = 5,
) -> np.ndarray:
    """
    Build a flat (K*3,) CIE L*a*b* palette from a list of hue angles.

    Fills remaining slots with copies of the first colour if fewer hues than K
    are supplied.
    """
    while len(hues_deg) < n_colors:
        hues_deg.append(hues_deg[0])

    centres = []
    for h in hues_deg[:n_colors]:
        theta = np.deg2rad(h)
        a_star = chroma * np.cos(theta)
        b_star = chroma * np.sin(theta)
        centres.append([lightness, a_star, b_star])

    return np.array(centres, dtype=np.float32).flatten()


def _make_neutral_palette(n_colors: int = 5, chroma: float = 5.0) -> np.ndarray:
    """Build a palette of near-achromatic greys (very low chroma)."""
    centres = []
    for i in range(n_colors):
        lightness = 20.0 + i * 15.0     # range 20–80
        centres.append([lightness, chroma * 0.1, chroma * 0.1])
    return np.array(centres, dtype=np.float32).flatten()


# ---------------------------------------------------------------------------
# Basic interface tests
# ---------------------------------------------------------------------------

class TestColorRuleScorerInterface:

    def test_analyse_returns_analysis_object(self):
        """analyse() must return a ColorHarmonyAnalysis instance."""
        scorer = ColorRuleScorer(n_dominant_colors=5)
        pal_a  = _make_palette_flat([0.0])
        pal_b  = _make_palette_flat([90.0])
        result = scorer.analyse(pal_a, pal_b)
        assert isinstance(result, ColorHarmonyAnalysis)

    def test_overall_score_in_range(self):
        """overall_score must always be ∈ [0.0, 1.0]."""
        scorer = ColorRuleScorer(n_dominant_colors=5)
        rng    = np.random.default_rng(7)
        for _ in range(20):
            hues_a = list(rng.uniform(0, 360, size=5))
            hues_b = list(rng.uniform(0, 360, size=5))
            pal_a  = _make_palette_flat(hues_a)
            pal_b  = _make_palette_flat(hues_b)
            result = scorer.analyse(pal_a, pal_b)
            assert 0.0 <= result.overall_score <= 1.0, (
                f"overall_score={result.overall_score} out of [0,1]"
            )

    def test_rule_scores_all_present(self):
        """rule_scores dict must contain all six rule keys."""
        scorer   = ColorRuleScorer(n_dominant_colors=5)
        pal      = _make_palette_flat([0.0])
        result   = scorer.analyse(pal, pal)
        expected = {
            "complementary", "analogous", "triadic",
            "split_complementary", "monochromatic", "neutral",
        }
        assert set(result.rule_scores.keys()) == expected

    def test_dominant_rule_is_a_known_rule(self):
        """dominant_rule must be one of the six named rules."""
        scorer = ColorRuleScorer(n_dominant_colors=5)
        pal_a  = _make_palette_flat([30.0])
        pal_b  = _make_palette_flat([210.0])
        result = scorer.analyse(pal_a, pal_b)
        assert result.dominant_rule in ColorRuleScorer.RULE_NAMES

    def test_output_palette_shapes(self):
        """palette_a and palette_b in the result must be (K, 3)."""
        K      = 5
        scorer = ColorRuleScorer(n_dominant_colors=K)
        pal_a  = _make_palette_flat([0.0], n_colors=K)
        pal_b  = _make_palette_flat([90.0], n_colors=K)
        result = scorer.analyse(pal_a, pal_b)
        assert result.palette_a.shape == (K, 3)
        assert result.palette_b.shape == (K, 3)

    def test_hue_angle_arrays_shape(self):
        """hue_angle_a and hue_angle_b must be 1-D arrays of length K."""
        K      = 5
        scorer = ColorRuleScorer(n_dominant_colors=K)
        pal    = _make_palette_flat([45.0], n_colors=K)
        result = scorer.analyse(pal, pal)
        assert result.hue_angle_a.shape == (K,)
        assert result.hue_angle_b.shape == (K,)

    def test_accepts_preshaped_palette(self):
        """analyse() must accept (K, 3) shaped arrays directly."""
        scorer = ColorRuleScorer(n_dominant_colors=5)
        pal_2d = np.zeros((5, 3), dtype=np.float32)
        pal_2d[:, 0] = 60.0    # all same L*
        result = scorer.analyse(pal_2d, pal_2d)
        assert isinstance(result, ColorHarmonyAnalysis)


# ---------------------------------------------------------------------------
# Complementary rule
# ---------------------------------------------------------------------------

class TestComplementaryRule:

    def test_complementary_fires_at_180_degrees(self):
        """
        A palette pair where one colour is exactly 180° from the other
        must yield a high complementary score (≥ 0.9).
        """
        scorer = ColorRuleScorer(n_dominant_colors=5, complementary_tol=25.0)
        pal_a  = _make_palette_flat([0.0])      # hue = 0°
        pal_b  = _make_palette_flat([180.0])    # hue = 180° — exact complement
        result = scorer.analyse(pal_a, pal_b)
        assert result.rule_scores["complementary"] >= 0.9, (
            f"Expected complementary ≥ 0.9, got {result.rule_scores['complementary']:.4f}"
        )

    def test_complementary_low_for_analogous_pair(self):
        """
        Two analogous hues (10° apart) should score low on complementary.
        """
        scorer = ColorRuleScorer(n_dominant_colors=5)
        pal_a  = _make_palette_flat([0.0])
        pal_b  = _make_palette_flat([10.0])
        result = scorer.analyse(pal_a, pal_b)
        assert result.rule_scores["complementary"] < 0.3, (
            f"Analogous pair should score low on complementary, "
            f"got {result.rule_scores['complementary']:.4f}"
        )


# ---------------------------------------------------------------------------
# Analogous rule
# ---------------------------------------------------------------------------

class TestAnalogousRule:

    def test_analogous_fires_for_nearby_hues(self):
        """
        A 20° hue separation is well within the analogous window (≤ 30°)
        and must yield a score close to 1.0.
        """
        scorer = ColorRuleScorer(n_dominant_colors=5, analogous_max=30.0)
        pal_a  = _make_palette_flat([0.0])
        pal_b  = _make_palette_flat([20.0])
        result = scorer.analyse(pal_a, pal_b)
        assert result.rule_scores["analogous"] >= 0.5, (
            f"Expected analogous ≥ 0.5, got {result.rule_scores['analogous']:.4f}"
        )

    def test_analogous_low_for_complementary_pair(self):
        """180° hue difference must score low on analogous."""
        scorer = ColorRuleScorer(n_dominant_colors=5)
        pal_a  = _make_palette_flat([0.0])
        pal_b  = _make_palette_flat([180.0])
        result = scorer.analyse(pal_a, pal_b)
        assert result.rule_scores["analogous"] < 0.2

    def test_identical_hues_maximum_analogous(self):
        """Same hue → δ=0 → maximum analogous score."""
        scorer = ColorRuleScorer(n_dominant_colors=5)
        pal    = _make_palette_flat([45.0])
        result = scorer.analyse(pal, pal)
        # Weight: 1 - 0/30 = 1.0 for all pairs → mean = 1.0
        assert result.rule_scores["analogous"] >= 0.99


# ---------------------------------------------------------------------------
# Triadic rule
# ---------------------------------------------------------------------------

class TestTriadicRule:

    def test_triadic_fires_at_120_degrees(self):
        """120° separation is the exact triadic angle → high score."""
        scorer = ColorRuleScorer(n_dominant_colors=5, triadic_tol=25.0)
        pal_a  = _make_palette_flat([0.0])
        pal_b  = _make_palette_flat([120.0])
        result = scorer.analyse(pal_a, pal_b)
        assert result.rule_scores["triadic"] >= 0.9

    def test_triadic_fires_at_240_degrees(self):
        """240° is the second triadic position → should also score high."""
        scorer = ColorRuleScorer(n_dominant_colors=5, triadic_tol=25.0)
        pal_a  = _make_palette_flat([0.0])
        pal_b  = _make_palette_flat([240.0])
        result = scorer.analyse(pal_a, pal_b)
        assert result.rule_scores["triadic"] >= 0.9


# ---------------------------------------------------------------------------
# Monochromatic rule
# ---------------------------------------------------------------------------

class TestMonochromaticRule:

    def test_monochromatic_fires_for_same_hue_different_lightness(self):
        """
        Two palettes with the same hue but different lightness levels represent
        monochromatic harmony and should score well.
        """
        scorer = ColorRuleScorer(n_dominant_colors=5, monochromatic_max=15.0)
        pal_a  = _make_palette_flat([30.0], chroma=50.0, lightness=30.0)
        pal_b  = _make_palette_flat([30.0], chroma=50.0, lightness=70.0)
        result = scorer.analyse(pal_a, pal_b)
        assert result.rule_scores["monochromatic"] >= 0.6

    def test_monochromatic_penalised_for_achromatic(self):
        """
        Two fully achromatic palettes should score low on monochromatic
        (they would score on 'neutral' instead).
        """
        scorer  = ColorRuleScorer(n_dominant_colors=5)
        pal_a   = _make_neutral_palette()
        pal_b   = _make_neutral_palette()
        result  = scorer.analyse(pal_a, pal_b)
        mono    = result.rule_scores["monochromatic"]
        neutral = result.rule_scores["neutral"]
        # neutral should outperform monochromatic for achromatic palettes
        assert neutral >= mono, (
            f"Expected neutral ({neutral:.3f}) ≥ monochromatic ({mono:.3f}) "
            "for achromatic palettes"
        )


# ---------------------------------------------------------------------------
# Neutral rule
# ---------------------------------------------------------------------------

class TestNeutralRule:

    def test_neutral_fires_for_low_chroma_palettes(self):
        """Achromatic / low-chroma palettes should score high on neutral."""
        scorer = ColorRuleScorer(n_dominant_colors=5, neutral_chroma_threshold=15.0)
        pal_a  = _make_neutral_palette(chroma=3.0)
        pal_b  = _make_neutral_palette(chroma=3.0)
        result = scorer.analyse(pal_a, pal_b)
        assert result.rule_scores["neutral"] >= 0.8

    def test_neutral_low_for_saturated_palettes(self):
        """Highly saturated palettes should score low on neutral."""
        scorer = ColorRuleScorer(n_dominant_colors=5, neutral_chroma_threshold=15.0)
        pal_a  = _make_palette_flat([0.0], chroma=80.0)
        pal_b  = _make_palette_flat([90.0], chroma=80.0)
        result = scorer.analyse(pal_a, pal_b)
        assert result.rule_scores["neutral"] < 0.2


# ---------------------------------------------------------------------------
# analyse_from_vectors() helper
# ---------------------------------------------------------------------------

class TestAnalyseFromVectors:

    def test_from_vectors_matches_direct_analyse(self):
        """
        analyse_from_vectors() must produce the same result as calling
        analyse() directly with the colour sub-vector.
        """
        K      = 5
        scorer = ColorRuleScorer(n_dominant_colors=K)
        color_dim = K * 3

        # Build a fake explicit vector: [color_features | other_features]
        rng       = np.random.default_rng(99)
        color_a   = rng.standard_normal(color_dim).astype(np.float32)
        color_b   = rng.standard_normal(color_dim).astype(np.float32)
        padding_a = rng.standard_normal(100).astype(np.float32)
        padding_b = rng.standard_normal(100).astype(np.float32)
        explicit_a = np.concatenate([color_a, padding_a])
        explicit_b = np.concatenate([color_b, padding_b])

        result_via_vectors = scorer.analyse_from_vectors(explicit_a, explicit_b)
        result_direct      = scorer.analyse(color_a, color_b)

        assert abs(result_via_vectors.overall_score - result_direct.overall_score) < 1e-5

    def test_wrong_palette_length_raises(self):
        """A flat palette vector of wrong length must raise ValueError."""
        scorer     = ColorRuleScorer(n_dominant_colors=5)   # expects 15-dim
        wrong_flat = np.zeros(12, dtype=np.float32)          # 4*3=12 ≠ 15
        with pytest.raises(ValueError):
            scorer.analyse(wrong_flat, wrong_flat)

    def test_wrong_palette_ndim_raises(self):
        """A 3-D palette array must raise ValueError."""
        scorer    = ColorRuleScorer(n_dominant_colors=5)
        bad_shape = np.zeros((5, 3, 1), dtype=np.float32)
        with pytest.raises(ValueError):
            scorer.analyse(bad_shape, bad_shape)


# ---------------------------------------------------------------------------
# Hue geometry helpers
# ---------------------------------------------------------------------------

class TestHueGeometry:

    def test_hue_angles_in_range(self):
        """All derived hue angles must be ∈ [0, 360)."""
        scorer = ColorRuleScorer(n_dominant_colors=5)
        pal    = _make_palette_flat([0, 90, 180, 270, 45])
        result = scorer.analyse(pal, pal)
        assert np.all(result.hue_angle_a >= 0.0)
        assert np.all(result.hue_angle_a < 360.0)

    def test_circular_distance_symmetric(self):
        """
        The circular arc distance between α and β should equal the distance
        between β and α (symmetry check via score equality).
        """
        scorer = ColorRuleScorer(n_dominant_colors=5)
        pal_a  = _make_palette_flat([30.0])
        pal_b  = _make_palette_flat([210.0])
        res_ab = scorer.analyse(pal_a, pal_b)
        res_ba = scorer.analyse(pal_b, pal_a)
        assert abs(res_ab.overall_score - res_ba.overall_score) < 1e-5, (
            "Colour harmony score should be symmetric"
        )
