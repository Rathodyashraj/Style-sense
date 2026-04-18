"""
scripts/train.py
─────────────────
CLI script — Train the outfit compatibility classifier.

Pre-requisite
-------------
Run ``scripts/extract_features.py`` first so the HDF5 feature caches exist.

What this script does
---------------------
1. Load cached explicit and latent feature vectors for every training pair.
2. Build pairwise feature vectors using ``PairwiseScorer``.
3. Fit a ``StandardScaler`` on the explicit features (saved for inference).
4. Train the SVM or MLP classifier (selected via config).
5. Save the best model checkpoint to ``outputs/checkpoints/``.

Usage
-----
    python scripts/train.py [--config PATH] [--model svm|mlp]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.fusion.pairwise_scorer  import PairwiseScorer
from src.models.model_factory    import build_model
from src.utils.config_loader     import load_config
from src.utils.dataset_loader    import load_pairs
from src.utils.feature_cache     import FeatureCache
from src.utils.logger            import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data assembly helpers
# ---------------------------------------------------------------------------

def _load_pairwise_dataset(
    pairs_json: str | Path,
    cache: FeatureCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load cached feature vectors for every compatible/incompatible pair.

    Returns
    -------
    explicit_a, explicit_b : (N, D_exp)  explicit feature matrices
    latent_a,   latent_b   : (N, D_lat)  CLIP embedding matrices
    labels                 : (N,)        int array of 0/1 labels
    """
    pairs = load_pairs(pairs_json)

    explicit_a_list, explicit_b_list = [], []
    latent_a_list,   latent_b_list   = [], []
    labels_list: list = []
    skipped = 0

    for pair in tqdm(pairs, desc="Loading pair features from cache"):
        id_a, id_b = pair["item_1"], pair["item_2"]
        label      = int(pair["label"])

        exp_a = cache.load_explicit(id_a)
        exp_b = cache.load_explicit(id_b)
        lat_a = cache.load_latent(id_a)
        lat_b = cache.load_latent(id_b)

        # Skip pairs where any feature vector is missing
        if any(v is None for v in (exp_a, exp_b, lat_a, lat_b)):
            skipped += 1
            continue

        explicit_a_list.append(exp_a)
        explicit_b_list.append(exp_b)
        latent_a_list.append(lat_a)
        latent_b_list.append(lat_b)
        labels_list.append(label)

    if skipped:
        log.warning("Skipped {n} pairs with missing cached features.", n=skipped)

    return (
        np.stack(explicit_a_list),
        np.stack(explicit_b_list),
        np.stack(latent_a_list),
        np.stack(latent_b_list),
        np.array(labels_list, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the outfit compatibility classifier.")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--model",  default=None, choices=["svm", "mlp"],
                        help="Override config model type")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg.model.type = args.model

    checkpoint_dir = Path(cfg.paths.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load cached features for the training split ──────────────────
    log.info("Loading training pair features from HDF5 cache …")
    with FeatureCache(cfg.paths.feature_cache_dir, split="train") as cache:
        exp_a, exp_b, lat_a, lat_b, labels = _load_pairwise_dataset(
            cfg.paths.train_json, cache
        )

    log.info(
        "Training pairs loaded: {n} | positive={p} | negative={neg}",
        n=len(labels),
        p=(labels == 1).sum(),
        neg=(labels == 0).sum(),
    )

    # ── Step 2: Fit StandardScaler on the raw explicit features ──────────────
    # We fit on the concatenation of A and B explicit vectors so the scaler
    # sees the full marginal distribution of each feature dimension.
    all_explicit = np.vstack([exp_a, exp_b])
    scaler = StandardScaler()
    scaler.fit(all_explicit)

    # Normalise both sets
    exp_a_scaled = scaler.transform(exp_a).astype(np.float32)
    exp_b_scaled = scaler.transform(exp_b).astype(np.float32)

    # Save scaler for use during inference
    scaler_path = checkpoint_dir / "explicit_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    log.info("StandardScaler fitted and saved to {p}", p=scaler_path)

    # ── Step 3: Build pairwise feature matrix ─────────────────────────────────
    log.info("Building pairwise feature vectors …")
    scorer    = PairwiseScorer()
    X_pairwise = scorer.build_pairwise_batch(exp_a_scaled, exp_b_scaled, lat_a, lat_b)
    y          = labels.astype(np.float32)

    log.info("Pairwise feature matrix: shape=%s", X_pairwise.shape)

    # ── Step 3b: Fit StandardScaler on the *full* pairwise vector ─────────────
    # The pairwise vector concatenates heterogeneous blocks (scalars, diffs,
    # hadamards) at very different numeric scales.  Normalising the assembled
    # vector before the classifier significantly improves convergence.
    pairwise_scaler = StandardScaler()
    X_pairwise = pairwise_scaler.fit_transform(X_pairwise).astype(np.float32)

    pairwise_scaler_path = checkpoint_dir / "pairwise_scaler.pkl"
    joblib.dump(pairwise_scaler, pairwise_scaler_path)
    log.info("Pairwise StandardScaler fitted and saved to {p}", p=pairwise_scaler_path)

    # ── Step 4: Train / validation split ─────────────────────────────────────
    val_split   = cfg.training.val_split
    random_seed = cfg.training.random_seed

    X_train, X_val, y_train, y_val = train_test_split(
        X_pairwise, y,
        test_size=val_split,
        random_state=random_seed,
        stratify=y,
    )
    log.info(
        "Split: train=%d  val=%d", len(y_train), len(y_val)
    )

    # ── Step 5: Instantiate and train the model ───────────────────────────────
    model = build_model(cfg)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    # ── Step 6: Save model checkpoint ────────────────────────────────────────
    model_type = cfg.model.type.lower()
    if model_type == "mlp":
        ckpt_path = checkpoint_dir / "best_model.pt"
    else:
        ckpt_path = checkpoint_dir / "best_model.pkl"

    model.save(ckpt_path)
    log.info("Model saved to {p}", p=ckpt_path)
    log.info("Training complete.")


if __name__ == "__main__":
    main()
