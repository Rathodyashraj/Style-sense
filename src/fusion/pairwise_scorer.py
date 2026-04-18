"""
src/fusion/pairwise_scorer.py
──────────────────────────────
Module 4 (Part A) — Compatibility Fusion & Pairwise Feature Construction

Responsibility
--------------
Given the explicit CV vector and CLIP latent vector for *two* garments (A and B),
this module computes a rich *pairwise feature vector* that captures their
relationship from multiple geometric perspectives.

Pairwise feature composition
-----------------------------
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. euclidean_dist   : scalar — L2 distance in explicit CV space            │
│  2. cosine_sim       : scalar — cosine similarity in CLIP latent space       │
│  3. explicit_diff    : D_exp  — element-wise signed difference (A − B)       │
│  4. explicit_hadamard: D_exp  — element-wise product A ⊙ B                   │
│  5. latent_diff      : D_lat  — element-wise signed difference of CLIP vecs  │
│  6. latent_hadamard  : D_lat  — element-wise product of CLIP vectors         │
└─────────────────────────────────────────────────────────────────────────────┘
Total pairwise feature length = 2 + 2 * D_exp + 2 * D_lat

Design rationale
----------------
* The **Euclidean distance** on explicit features encodes how visually
  different two garments are in colour, texture and silhouette — a pair
  that is too similar or too dissimilar in all three may be incompatible.

* The **cosine similarity** on CLIP embeddings captures semantic alignment:
  garments that share a style archetype (e.g. "casual streetwear") have
  nearby CLIP embeddings regardless of colour differences.

* The **element-wise diff** preserves directional information: the SVM/MLP
  can learn which dimensions drive incompatibility.

* The **Hadamard products** encode co-activation: large values indicate
  both items are strong in the same direction (agreement signal).

References
----------
Han, X. et al. (2017). Learning Fashion Compatibility with Bidirectional LSTMs.
Vasileva, M. I. et al. (2018). Learning Type-Aware Embeddings for Fashion Compatibility.
"""

from __future__ import annotations

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# PairwiseScorer
# ---------------------------------------------------------------------------

class PairwiseScorer:
    """
    Constructs pairwise feature vectors from per-item explicit + latent vectors.

    This class is stateless — all methods are deterministic functions of their
    inputs.  It is kept as a class (rather than free functions) to allow easy
    sub-classing if alternative fusion strategies are needed.
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def build_pairwise_feature(
        self,
        explicit_a: np.ndarray,
        explicit_b: np.ndarray,
        latent_a:   np.ndarray,
        latent_b:   np.ndarray,
    ) -> np.ndarray:
        """
        Construct a pairwise feature vector from two garments' feature vectors.

        Parameters
        ----------
        explicit_a, explicit_b : np.ndarray  shape (D_exp,)  float32
            Explicit CV feature vectors for garments A and B.
        latent_a,   latent_b   : np.ndarray  shape (D_lat,)  float32
            L2-normalised CLIP embedding vectors for garments A and B.

        Returns
        -------
        np.ndarray  shape (2 + 2*D_exp + 2*D_lat,)  float32
            Pairwise feature vector ready to feed into the SVM / MLP classifier.
        """
        # ── Validate shapes ───────────────────────────────────────────────────
        if explicit_a.shape != explicit_b.shape:
            raise ValueError(
                f"Explicit vector shape mismatch: {explicit_a.shape} vs {explicit_b.shape}"
            )
        if latent_a.shape != latent_b.shape:
            raise ValueError(
                f"Latent vector shape mismatch: {latent_a.shape} vs {latent_b.shape}"
            )

        # ── 1. Euclidean distance in explicit (CV) space ──────────────────────
        euclidean_dist = np.array(
            [self._euclidean_distance(explicit_a, explicit_b)],
            dtype=np.float32,
        )

        # ── 2. Cosine similarity in CLIP latent space ─────────────────────────
        # Both latent vectors are already L2-normalised (by LatentFeatureExtractor),
        # so cosine similarity reduces to the dot product.
        cosine_sim = np.array(
            [self._cosine_similarity(latent_a, latent_b)],
            dtype=np.float32,
        )

        # ── 3. Signed element-wise difference: A − B ──────────────────────────
        explicit_diff = (explicit_a - explicit_b).astype(np.float32)

        # ── 4. Hadamard product of explicit vectors: A ⊙ B ───────────────────
        explicit_hadamard = (explicit_a * explicit_b).astype(np.float32)

        # ── 5. Signed element-wise difference of latent vectors: A − B ────────
        latent_diff = (latent_a - latent_b).astype(np.float32)

        # ── 6. Hadamard product of latent vectors: A ⊙ B ─────────────────────
        latent_hadamard = (latent_a * latent_b).astype(np.float32)

        # ── Concatenate all components ────────────────────────────────────────
        pairwise = np.concatenate([
            euclidean_dist,      # 1 dim
            cosine_sim,          # 1 dim
            explicit_diff,       # D_exp dims
            explicit_hadamard,   # D_exp dims
            latent_diff,         # D_lat dims
            latent_hadamard,     # D_lat dims
        ])

        log.debug(
            "Pairwise feature built | "
            "eucl=%.4f  cos=%.4f  total_dim=%d",
            euclidean_dist[0], cosine_sim[0], pairwise.shape[0],
        )

        return pairwise

    def build_pairwise_batch(
        self,
        explicit_batch_a: np.ndarray,
        explicit_batch_b: np.ndarray,
        latent_batch_a:   np.ndarray,
        latent_batch_b:   np.ndarray,
    ) -> np.ndarray:
        """
        Vectorised batch construction of pairwise feature vectors.

        Parameters
        ----------
        explicit_batch_a, explicit_batch_b : np.ndarray  shape (N, D_exp)
        latent_batch_a,   latent_batch_b   : np.ndarray  shape (N, D_lat)

        Returns
        -------
        np.ndarray  shape (N, 2 + 2*D_exp + 2*D_lat)  float32
        """
        N = explicit_batch_a.shape[0]

        # ── Scalar metrics: shape (N, 1) ──────────────────────────────────────
        # Euclidean distance per row
        diff_exp = explicit_batch_a - explicit_batch_b          # (N, D_exp)
        euclidean_dists = np.sqrt((diff_exp ** 2).sum(axis=1, keepdims=True))  # (N,1)

        # Cosine similarity — vectors are already L2-normalised so this is a dot
        cosine_sims = (latent_batch_a * latent_batch_b).sum(axis=1, keepdims=True)  # (N,1)

        # ── Vector metrics ────────────────────────────────────────────────────
        explicit_hadamard = explicit_batch_a * explicit_batch_b  # (N, D_exp)
        latent_diff       = latent_batch_a   - latent_batch_b    # (N, D_lat)
        latent_hadamard   = latent_batch_a   * latent_batch_b    # (N, D_lat)

        # ── Concatenate along feature axis ────────────────────────────────────
        pairwise_batch = np.concatenate([
            euclidean_dists,      # (N, 1)
            cosine_sims,          # (N, 1)
            diff_exp,             # (N, D_exp)
            explicit_hadamard,    # (N, D_exp)
            latent_diff,          # (N, D_lat)
            latent_hadamard,      # (N, D_lat)
        ], axis=1).astype(np.float32)

        log.debug("Pairwise batch built: N=%d, feature_dim=%d", N, pairwise_batch.shape[1])
        return pairwise_batch

    # ── Static math helpers ───────────────────────────────────────────────────

    @staticmethod
    def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Compute L2 distance between two 1-D vectors."""
        return float(np.sqrt(np.sum((a - b) ** 2)))

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two 1-D vectors.
        Handles the zero-vector edge case by returning 0.0.
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
