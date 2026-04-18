"""
tests/test_fusion.py
─────────────────────
Unit tests for the PairwiseScorer (Module 4a).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fusion.pairwise_scorer import PairwiseScorer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _random_explicit(seed: int = 0, dim: int = 100) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(dim).astype(np.float32)


def _random_latent(seed: int = 0, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v   = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)   # L2-normalised


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPairwiseScorer:

    def test_single_pair_output_shape(self):
        """Single-pair pairwise feature must have the expected length."""
        scorer      = PairwiseScorer()
        exp_a       = _random_explicit(0, 100)
        exp_b       = _random_explicit(1, 100)
        lat_a       = _random_latent(0, 512)
        lat_b       = _random_latent(1, 512)

        pairwise = scorer.build_pairwise_feature(exp_a, exp_b, lat_a, lat_b)

        # Expected: 1 (eucl) + 1 (cos) + 100 (diff) + 100 (hadamard exp) + 512 (hadamard lat)
        expected_dim = 1 + 1 + 100 + 100 + 512
        assert pairwise.shape == (expected_dim,), \
            f"Expected dim {expected_dim}, got {pairwise.shape}"

    def test_single_pair_dtype(self):
        scorer   = PairwiseScorer()
        pairwise = scorer.build_pairwise_feature(
            _random_explicit(0), _random_explicit(1),
            _random_latent(0),   _random_latent(1),
        )
        assert pairwise.dtype == np.float32

    def test_cosine_similarity_range(self):
        """Cosine similarity component must be in [-1, 1]."""
        scorer  = PairwiseScorer()
        pairwise = scorer.build_pairwise_feature(
            _random_explicit(0), _random_explicit(1),
            _random_latent(0),   _random_latent(1),
        )
        cosine_sim = float(pairwise[1])    # index 1 = cosine similarity
        assert -1.0 <= cosine_sim <= 1.0, \
            f"Cosine similarity {cosine_sim} is out of range [-1, 1]"

    def test_identical_items_zero_diff(self):
        """When item_A == item_B, the signed diff sub-vector should be all zeros."""
        scorer = PairwiseScorer()
        exp    = _random_explicit(0, 100)
        lat    = _random_latent(0, 512)

        pairwise = scorer.build_pairwise_feature(exp, exp, lat, lat)
        diff_subvector = pairwise[2 : 2 + 100]   # signed diff starts at index 2
        assert np.allclose(diff_subvector, 0.0), \
            "Signed diff of identical vectors should be zero"

    def test_batch_matches_single(self):
        """Batch result for a single item must match the single-call result."""
        scorer = PairwiseScorer()
        exp_a  = _random_explicit(0, 50)
        exp_b  = _random_explicit(1, 50)
        lat_a  = _random_latent(0, 64)
        lat_b  = _random_latent(1, 64)

        single = scorer.build_pairwise_feature(exp_a, exp_b, lat_a, lat_b)
        batch  = scorer.build_pairwise_batch(
            exp_a[np.newaxis], exp_b[np.newaxis],
            lat_a[np.newaxis], lat_b[np.newaxis],
        )
        assert np.allclose(single, batch[0], atol=1e-5), \
            "Batch and single results diverge"

    def test_mismatched_shapes_raise(self):
        """Mismatched explicit vector shapes must raise ValueError."""
        scorer = PairwiseScorer()
        with pytest.raises(ValueError):
            scorer.build_pairwise_feature(
                np.zeros(50, dtype=np.float32),
                np.zeros(60, dtype=np.float32),   # different dim
                _random_latent(0), _random_latent(1),
            )

    def test_euclidean_distance_non_negative(self):
        """Euclidean distance (first component) must be >= 0."""
        scorer   = PairwiseScorer()
        pairwise = scorer.build_pairwise_feature(
            _random_explicit(0), _random_explicit(1),
            _random_latent(0),   _random_latent(1),
        )
        assert float(pairwise[0]) >= 0.0, "Euclidean distance must be non-negative"
