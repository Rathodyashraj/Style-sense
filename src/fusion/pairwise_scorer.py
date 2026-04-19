from __future__ import annotations

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)



# PairwiseScorer


class PairwiseScorer:

    # Public API

    def build_pairwise_feature(
        self,
        explicit_a: np.ndarray,
        explicit_b: np.ndarray,
        latent_a:   np.ndarray,
        latent_b:   np.ndarray,
    ) -> np.ndarray:
        latent_diff = (latent_a - latent_b).astype(np.float32)
        if explicit_a.shape != explicit_b.shape:
            raise ValueError(
                f"Explicit vector shape mismatch: {explicit_a.shape} vs {explicit_b.shape}"
            )
        if latent_a.shape != latent_b.shape:
            raise ValueError(
                f"Latent vector shape mismatch: {latent_a.shape} vs {latent_b.shape}"
            )



        
        #1. Euclidean distance in explicit (CV) space
        euclidean_dist = np.array(
            [self._euclidean_distance(explicit_a, explicit_b)],
            dtype=np.float32,
        )

        #2. Cosine similarity in CLIP latent space
        cosine_sim = np.array(
            [self._cosine_similarity(latent_a, latent_b)],
            dtype=np.float32,
        )

        #3. Signed element-wise difference: A − B 
        explicit_diff = (explicit_a - explicit_b).astype(np.float32)

        #4. Hadamard product of explicit vectors: A ⊙ B 
        explicit_hadamard = (explicit_a * explicit_b).astype(np.float32)

        #5. Hadamard product of latent vectors: A ⊙ B
        latent_hadamard = (latent_a * latent_b).astype(np.float32)

        
        
        pairwise = np.concatenate([
            euclidean_dist,cosine_sim, explicit_diff, explicit_hadamard,latent_diff,latent_hadamard,     
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

        N = explicit_batch_a.shape[0]

        # Scalar metrics: shape (N, 1)
        # Euclidean distance per row
        diff_exp = explicit_batch_a - explicit_batch_b          # (N, D_exp)
        euclidean_dists = np.sqrt((diff_exp ** 2).sum(axis=1, keepdims=True))  # (N,1)

        # Cosine similarity — vectors are already L2-normalised so this is a dot
        cosine_sims = (latent_batch_a * latent_batch_b).sum(axis=1, keepdims=True)  # (N,1)

        # Vector metrics
        explicit_hadamard = explicit_batch_a * explicit_batch_b  # (N, D_exp)
        latent_diff       = latent_batch_a   - latent_batch_b    # (N, D_lat)
        latent_hadamard   = latent_batch_a   * latent_batch_b    # (N, D_lat)

        # Concatenate along feature axis
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

    # Static math helpers

    @staticmethod
    def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(np.sum((a - b) ** 2)))

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
