"""
scripts/extract_features.py
─────────────────────────────
CLI script — Pre-compute and cache all explicit CV features and CLIP latent
embeddings for the entire Polyvore dataset.

This is a one-time (or on-demand) pre-processing step.  Running it before
`train.py` means the training loop never re-runs expensive feature extraction.

Usage
-----
    python scripts/extract_features.py [--config PATH] [--split SPLIT]

Arguments
---------
--config  : path to config YAML  (default: configs/config.yaml)
--split   : "train" | "val" | "test" | "all"  (default: "all")
--device  : "cuda" | "cpu"  (overrides config)
--dry_run : if set, process only the first 100 items per split (for testing)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Allow running from the project root without pip install ──────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm import tqdm

from src.features.explicit_extractor import ExplicitFeatureExtractor
from src.features.latent_extractor   import LatentFeatureExtractor
from src.utils.config_loader         import load_config
from src.utils.dataset_loader        import (
    PolyvoreCompatibilityDataset,
    build_item_image_map,
    load_pairs,
)
from src.utils.feature_cache         import FeatureCache
from src.utils.image_io              import load_image
from src.utils.logger                import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_unique_ids_for_split(pairs_json: Path) -> list:
    """Return the de-duplicated item IDs appearing in a pairs JSON file."""
    pairs = load_pairs(pairs_json)
    ids: set = set()
    for p in pairs:
        ids.add(p["item_1"])
        ids.add(p["item_2"])
    return sorted(ids)


# ---------------------------------------------------------------------------
# Main extraction routine for one split
# ---------------------------------------------------------------------------

def extract_split(
    split: str,
    cfg,
    explicit_extractor: ExplicitFeatureExtractor,
    latent_extractor:   LatentFeatureExtractor,
    dry_run: bool = False,
) -> None:
    """
    Extract and cache both explicit and latent features for all items in *split*.

    Parameters
    ----------
    split              : "train" | "val" | "test"
    cfg                : dot-accessible config namespace
    explicit_extractor : initialised ExplicitFeatureExtractor
    latent_extractor   : initialised LatentFeatureExtractor
    dry_run            : if True, only process the first 100 items
    """
    log.info("═══ Extracting features for split: {s} ═══", s=split)

    # ── Resolve the pairs JSON for this split ─────────────────────────────────
    split_json_map = {
        "train": cfg.paths.train_json,
        "val":   cfg.paths.val_json,
        "test":  cfg.paths.test_json,
    }
    pairs_json = Path(split_json_map[split])
    if not pairs_json.exists():
        log.warning("Pairs JSON not found for split '{s}': {p}. Skipping.", s=split, p=pairs_json)
        return

    item_ids    = _collect_unique_ids_for_split(pairs_json)
    item_map    = build_item_image_map(cfg.paths.images_dir)
    image_size  = tuple(cfg.dataset.image_size)

    # Filter to only IDs that have a corresponding image on disk
    item_ids = [iid for iid in item_ids if iid in item_map]
    log.info("Unique items with images for split '{s}': {n}", s=split, n=len(item_ids))

    if dry_run:
        item_ids = item_ids[:100]
        log.info("Dry-run mode: processing only {n} items.", n=len(item_ids))

    # ── Open the feature cache for this split ─────────────────────────────────
    with FeatureCache(cfg.paths.feature_cache_dir, split=split) as cache:

        # ── Pass 1: Explicit CV features ──────────────────────────────────────
        ids_missing_explicit = [
            iid for iid in item_ids if not cache.has_explicit(iid)
        ]
        log.info(
            "Explicit features to compute: {n}/{t}",
            n=len(ids_missing_explicit), t=len(item_ids),
        )

        for item_id in tqdm(ids_missing_explicit, desc=f"[{split}] Explicit CV"):
            image = load_image(item_map[item_id], size=image_size, color_space="BGR")
            try:
                explicit_vec = explicit_extractor.extract(image)
                cache.save_explicit(item_id, explicit_vec)
            except Exception as exc:
                log.warning("Explicit extraction failed for {id}: {e}", id=item_id, e=exc)

        # ── Pass 2: CLIP latent features (batched) ────────────────────────────
        ids_missing_latent = [
            iid for iid in item_ids if not cache.has_latent(iid)
        ]
        log.info(
            "Latent CLIP features to compute: {n}/{t}",
            n=len(ids_missing_latent), t=len(item_ids),
        )

        batch_size = cfg.clip.batch_size
        for start in tqdm(
            range(0, len(ids_missing_latent), batch_size),
            desc=f"[{split}] CLIP latent",
        ):
            batch_ids   = ids_missing_latent[start : start + batch_size]
            batch_images = [
                load_image(item_map[iid], size=image_size, color_space="BGR")
                for iid in batch_ids
            ]
            try:
                latent_batch = latent_extractor.extract_batch(batch_images)
                cache.save_latent_batch(dict(zip(batch_ids, latent_batch)))
            except Exception as exc:
                log.warning("CLIP extraction failed for batch starting {id}: {e}",
                            id=batch_ids[0], e=exc)

    log.info("Feature extraction complete for split: {s}", s=split)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute explicit CV + CLIP latent features for Polyvore items."
    )
    parser.add_argument("--config",  default=None,  help="Path to config YAML")
    parser.add_argument("--split",   default="all",
                        choices=["train", "val", "test", "all"])
    parser.add_argument("--device",  default=None,  help="Override device: cuda | cpu")
    parser.add_argument("--dry_run", action="store_true",
                        help="Process only 100 items per split for testing")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Override device if provided on CLI
    if args.device:
        cfg.clip.device = args.device

    # ── Initialise extractors once and reuse across splits ───────────────────
    log.info("Initialising feature extractors …")
    explicit_extractor = ExplicitFeatureExtractor(cfg, scaler_path=None)
    latent_extractor   = LatentFeatureExtractor(
        model_name = cfg.clip.model_name,
        device     = cfg.clip.device,
        batch_size = cfg.clip.batch_size,
    )

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    for split in splits:
        extract_split(split, cfg, explicit_extractor, latent_extractor, dry_run=args.dry_run)

    log.info("All feature extraction jobs finished.")


if __name__ == "__main__":
    main()
