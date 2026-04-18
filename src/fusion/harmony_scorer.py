"""
src/fusion/harmony_scorer.py
──────────────────────────────
Top-level Harmony Scorer — the public-facing API for the entire pipeline.

This class is the single entry-point that end users, scripts, and the
inference CLI interact with.  It orchestrates all four modules:

    Module 1 → GrabCutSegmenter        (inside ExplicitFeatureExtractor)
    Module 2 → ExplicitFeatureExtractor
    Module 3 → LatentFeatureExtractor  (CLIP)
    Module 4 → PairwiseScorer + ColorRuleScorer + classifier

Fusion strategy
---------------
The final harmony percentage is a weighted blend of two signals:

    harmony = α × ml_score  +  (1 − α) × colour_rule_score

where:
    ml_score          — SVM / MLP probability ∈ [0, 1]
    colour_rule_score — deterministic colour harmony rule score ∈ [0, 1]
    α                 — ``ml_weight`` parameter (default 0.85)

This gives the ML model primary authority while letting classical colour
theory provide a stable, interpretable regularisation signal.

Usage
-----
    scorer = HarmonyScorer.from_config(cfg)
    scorer.load_model(cfg.paths.checkpoint_dir)

    result = scorer.score(image_a, image_b)
    print(f"Harmony : {result.harmony_percent:.1f}%")
    print(f"Verdict : {result.verdict}")
    print(f"Colour  : {result.color_analysis.dominant_rule}")

The ``HarmonyResult`` dataclass bundles the scalar percentage together with
all intermediate feature metrics so they can be surfaced in UIs or logs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np

from src.features.explicit_extractor  import ExplicitFeatureExtractor
from src.features.latent_extractor    import LatentFeatureExtractor
from src.fusion.pairwise_scorer       import PairwiseScorer
from src.fusion.color_rule_scorer     import ColorRuleScorer, ColorHarmonyAnalysis
from src.models.model_factory         import build_model, CompatibilityModel
from src.utils.logger                 import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class HarmonyResult:
    """
    Holds all outputs produced by a single (item_A, item_B) scoring call.

    Attributes
    ----------
    harmony_percent   : float
        Final compatibility score mapped to [0, 100].
        Blend of ML classifier score and colour rule score.
    verdict           : str
        Human-readable label ("Excellent Match", "Good Match", …).
    prob_compatible   : float
        Raw ML classifier probability ∈ [0, 1].
    color_rule_score  : float
        Deterministic colour harmony rule score ∈ [0, 1].
    color_analysis    : ColorHarmonyAnalysis
        Full breakdown of each colour harmony rule (complementary, analogous,
        triadic, split-complementary, monochromatic, neutral).
    euclidean_dist    : float
        L2 distance between the two explicit CV vectors.
    cosine_sim        : float
        Cosine similarity between the two CLIP embeddings ∈ [−1, 1].
    latency_ms        : float
        Wall-clock time for the full scoring call in milliseconds.
    """
    harmony_percent  : float
    verdict          : str
    prob_compatible  : float
    color_rule_score : float
    color_analysis   : ColorHarmonyAnalysis
    euclidean_dist   : float
    cosine_sim       : float
    latency_ms       : float = field(default=0.0)

    def __str__(self) -> str:
        return (
            f"HarmonyResult("
            f"harmony={self.harmony_percent:.1f}%, "
            f"verdict='{self.verdict}', "
            f"ml_prob={self.prob_compatible:.4f}, "
            f"color_rule={self.color_rule_score:.4f}, "
            f"dominant_color_rule='{self.color_analysis.dominant_rule}', "
            f"eucl={self.euclidean_dist:.4f}, "
            f"cos={self.cosine_sim:.4f}, "
            f"latency={self.latency_ms:.1f}ms)"
        )


# ---------------------------------------------------------------------------
# HarmonyScorer
# ---------------------------------------------------------------------------

class HarmonyScorer:
    """
    Orchestrates the full four-module pipeline and returns a harmony score.

    Parameters
    ----------
    explicit_extractor : ExplicitFeatureExtractor
    latent_extractor   : LatentFeatureExtractor
    pairwise_scorer    : PairwiseScorer
    color_rule_scorer  : ColorRuleScorer
    model              : CompatibilityModel  (SVM or MLP)
    ml_weight          : float
        Weight given to the ML classifier score in the final blend.
        (1 − ml_weight) is given to the colour rule score.
        Default 0.85 — ML is the primary signal.
    """

    # Thresholds for the verdict labels (applied to the final harmony_percent)
    _VERDICT_THRESHOLDS = [
        (85, "Excellent Match"),
        (70, "Good Match"),
        (50, "Moderate Match"),
        (30, "Poor Match"),
        (0,  "Incompatible"),
    ]

    def __init__(
        self,
        explicit_extractor : ExplicitFeatureExtractor,
        latent_extractor   : LatentFeatureExtractor,
        pairwise_scorer    : PairwiseScorer,
        color_rule_scorer  : ColorRuleScorer,
        model              : CompatibilityModel,
        ml_weight          : float = 0.85,
    ) -> None:
        self._explicit_extractor = explicit_extractor
        self._latent_extractor   = latent_extractor
        self._pairwise_scorer    = pairwise_scorer
        self._color_rule_scorer  = color_rule_scorer
        self._model              = model
        self._pairwise_scaler    = None   # loaded in load_model()

        # Clamp ml_weight to a sensible range so colour rules always contribute
        self._ml_weight    = float(np.clip(ml_weight, 0.5, 1.0))
        self._color_weight = 1.0 - self._ml_weight

        log.info(
            "HarmonyScorer ready | ml_weight=%.2f | color_weight=%.2f",
            self._ml_weight, self._color_weight,
        )

    # ── Factory constructor ───────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        cfg,
        scaler_path: Optional[str | Path] = None,
        ml_weight: float = 0.85,
    ) -> "HarmonyScorer":
        """
        Convenience factory: instantiate all sub-components from *cfg*.

        Parameters
        ----------
        cfg         : dot-accessible config namespace.
        scaler_path : optional path to a fitted StandardScaler for the
                      explicit feature vectors.
        ml_weight   : blend weight for the ML classifier score (default 0.85).

        Returns
        -------
        HarmonyScorer  (model not yet loaded — call ``.load_model()`` next)
        """
        explicit_extractor = ExplicitFeatureExtractor(cfg, scaler_path=scaler_path)

        latent_extractor = LatentFeatureExtractor(
            model_name = cfg.clip.model_name,
            device     = cfg.clip.device,
            batch_size = cfg.clip.batch_size,
        )

        pairwise_scorer = PairwiseScorer()

        # Colour rule scorer uses the same K as the colour harmony extractor
        color_rule_scorer = ColorRuleScorer(
            n_dominant_colors = cfg.color_harmony.n_dominant_colors,
        )

        model = build_model(cfg)

        log.info("HarmonyScorer assembled from config.")
        return cls(
            explicit_extractor,
            latent_extractor,
            pairwise_scorer,
            color_rule_scorer,
            model,
            ml_weight=ml_weight,
        )

    def load_model(self, checkpoint_dir: str | Path) -> "HarmonyScorer":
        """
        Load the best checkpoint from *checkpoint_dir*.

        Tries ``best_model.pt`` (MLP) then ``best_model.pkl`` (SVM).
        Also loads the pairwise feature scaler if present.
        """
        checkpoint_dir = Path(checkpoint_dir)
        pt_path  = checkpoint_dir / "best_model.pt"
        pkl_path = checkpoint_dir / "best_model.pkl"

        if pt_path.exists():
            self._model.load(pt_path)
        elif pkl_path.exists():
            self._model.load(pkl_path)
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {checkpoint_dir}. "
                "Run scripts/train.py first."
            )

        # Load pairwise scaler (optional — backward compatible)
        pairwise_scaler_path = checkpoint_dir / "pairwise_scaler.pkl"
        if pairwise_scaler_path.exists():
            self._pairwise_scaler = joblib.load(pairwise_scaler_path)
            log.info("Pairwise scaler loaded from {p}", p=pairwise_scaler_path)
        else:
            log.warning("No pairwise scaler found; proceeding without it.")

        return self

    # ── Single-pair scoring ───────────────────────────────────────────────────

    def score(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
    ) -> HarmonyResult:
        """
        Score the compatibility of two garment images.

        Parameters
        ----------
        image_a, image_b : np.ndarray  shape (H, W, 3) BGR uint8

        Returns
        -------
        HarmonyResult
        """
        t0 = time.perf_counter()

        # ── Module 2: Explicit CV features ────────────────────────────────────
        explicit_a = self._explicit_extractor.extract(image_a)
        explicit_b = self._explicit_extractor.extract(image_b)

        # ── Module 3: CLIP latent embeddings ──────────────────────────────────
        latent_a = self._latent_extractor.extract(image_a)
        latent_b = self._latent_extractor.extract(image_b)

        # ── Module 4a: Pairwise feature construction ──────────────────────────
        pairwise = self._pairwise_scorer.build_pairwise_feature(
            explicit_a, explicit_b, latent_a, latent_b
        )

        # ── Module 4a-ii: Apply pairwise scaler if available ──────────────────
        if self._pairwise_scaler is not None:
            pairwise = self._pairwise_scaler.transform(
                pairwise.reshape(1, -1)
            ).flatten().astype(np.float32)

        # ── Module 4b: ML classifier inference ────────────────────────────────
        prob_compatible = self._model.score_single_pair(pairwise)

        # ── Module 4c: Deterministic colour harmony rules ─────────────────────
        # Extract the colour sub-vector (first n_colors*3 dims of explicit vec)
        color_analysis = self._color_rule_scorer.analyse_from_vectors(
            explicit_a, explicit_b
        )
        color_rule_score = color_analysis.overall_score

        # ── Fuse ML probability with colour rule score ────────────────────────
        blended_score   = (self._ml_weight * prob_compatible
                           + self._color_weight * color_rule_score)
        harmony_percent = round(float(np.clip(blended_score * 100.0, 0.0, 100.0)), 2)
        verdict         = self._prob_to_verdict(blended_score)

        # ── Intermediate scalars for diagnostics ──────────────────────────────
        eucl = float(np.linalg.norm(explicit_a - explicit_b))
        cos  = float(np.dot(latent_a, latent_b))   # already L2-normalised

        latency_ms = (time.perf_counter() - t0) * 1000.0

        result = HarmonyResult(
            harmony_percent  = harmony_percent,
            verdict          = verdict,
            prob_compatible  = prob_compatible,
            color_rule_score = color_rule_score,
            color_analysis   = color_analysis,
            euclidean_dist   = eucl,
            cosine_sim       = cos,
            latency_ms       = latency_ms,
        )
        log.info("%s", result)
        return result

    # ── Batch scoring ─────────────────────────────────────────────────────────

    def score_batch(
        self,
        images_a: List[np.ndarray],
        images_b: List[np.ndarray],
    ) -> List[HarmonyResult]:
        """
        Score a list of (image_a, image_b) pairs in batch mode.

        Parameters
        ----------
        images_a, images_b : list of np.ndarray  (H, W, 3) BGR uint8

        Returns
        -------
        list of HarmonyResult
        """
        assert len(images_a) == len(images_b), "Batch sizes must match."
        N = len(images_a)

        # ── Batch explicit feature extraction ─────────────────────────────────
        explicit_a_batch = np.stack([self._explicit_extractor.extract(img) for img in images_a])
        explicit_b_batch = np.stack([self._explicit_extractor.extract(img) for img in images_b])

        # ── Batch CLIP feature extraction ─────────────────────────────────────
        latent_a_batch = self._latent_extractor.extract_batch(images_a)
        latent_b_batch = self._latent_extractor.extract_batch(images_b)

        # ── Batch pairwise feature construction ───────────────────────────────
        pairwise_batch = self._pairwise_scorer.build_pairwise_batch(
            explicit_a_batch, explicit_b_batch,
            latent_a_batch,   latent_b_batch,
        )

        # ── Apply pairwise scaler if available ────────────────────────────────
        if self._pairwise_scaler is not None:
            pairwise_batch = self._pairwise_scaler.transform(
                pairwise_batch
            ).astype(np.float32)

        # ── Batch ML inference ────────────────────────────────────────────────
        probs = self._model.predict_proba(pairwise_batch)[:, 1]   # P(compatible)

        # ── Assemble per-pair results ─────────────────────────────────────────
        results: List[HarmonyResult] = []
        for i in range(N):
            p   = float(probs[i])
            eucl = float(np.linalg.norm(explicit_a_batch[i] - explicit_b_batch[i]))
            cos  = float(np.dot(latent_a_batch[i], latent_b_batch[i]))

            # Colour rule analysis is cheap — run per-pair even in batch mode
            color_analysis   = self._color_rule_scorer.analyse_from_vectors(
                explicit_a_batch[i], explicit_b_batch[i]
            )
            color_rule_score = color_analysis.overall_score

            blended      = self._ml_weight * p + self._color_weight * color_rule_score
            harmony_pct  = round(float(np.clip(blended * 100.0, 0.0, 100.0)), 2)

            results.append(HarmonyResult(
                harmony_percent  = harmony_pct,
                verdict          = self._prob_to_verdict(blended),
                prob_compatible  = p,
                color_rule_score = color_rule_score,
                color_analysis   = color_analysis,
                euclidean_dist   = eucl,
                cosine_sim       = cos,
            ))

        return results

    # ── Internal helpers ──────────────────────────────────────────────────────

    @classmethod
    def _prob_to_verdict(cls, prob: float) -> str:
        """Map a blended probability in [0,1] to a human-readable verdict."""
        harmony_pct = prob * 100.0
        for threshold, label in cls._VERDICT_THRESHOLDS:
            if harmony_pct >= threshold:
                return label
        return "Incompatible"
