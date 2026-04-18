"""
CLI script — Evaluate the trained compatibility classifier on the test split.

Metrics included
* Accuracy
* Precision, Recall, F1-score (macro and per-class)
* ROC-AUC
* Average Precision (AP)
* Confusion matrix
* Harmony score distribution (mean, std, min, max)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from tqdm import tqdm

from src.fusion.pairwise_scorer  import PairwiseScorer
from src.models.model_factory    import build_model
from src.utils.config_loader     import load_config
from src.utils.dataset_loader    import load_pairs
from src.utils.feature_cache     import FeatureCache
from src.utils.logger            import get_logger

log = get_logger(__name__)



# Helpers


def _load_pairwise_dataset(pairs_json, cache):
    """Mirror of train.py helper — load cached pairs from HDF5."""
    pairs = load_pairs(pairs_json)
    exp_a_list, exp_b_list, lat_a_list, lat_b_list, labels = [], [], [], [], []
    skipped = 0

    for pair in tqdm(pairs, desc="Loading cached features"):
        id_a, id_b = pair["item_1"], pair["item_2"]
        exp_a = cache.load_explicit(id_a)
        exp_b = cache.load_explicit(id_b)
        lat_a = cache.load_latent(id_a)
        lat_b = cache.load_latent(id_b)

        if any(v is None for v in (exp_a, exp_b, lat_a, lat_b)):
            skipped += 1
            continue

        exp_a_list.append(exp_a)
        exp_b_list.append(exp_b)
        lat_a_list.append(lat_a)
        lat_b_list.append(lat_b)
        labels.append(int(pair["label"]))

    if skipped:
        log.warning("Skipped {n} pairs with missing features.", n=skipped)

    return (
        np.stack(exp_a_list), np.stack(exp_b_list),
        np.stack(lat_a_list), np.stack(lat_b_list),
        np.array(labels, dtype=np.int32),
    )



# Main evaluation routine


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the compatibility scorer.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split",  default="test", choices=["test", "val"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    split = args.split

    checkpoint_dir = Path(cfg.paths.checkpoint_dir)
    results_dir    = Path(cfg.paths.results_dir)

    # Load scaler
    import joblib
    scaler_path = checkpoint_dir / "explicit_scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. Run scripts/train.py first."
        )
    scaler = joblib.load(scaler_path)
    log.info("Scaler loaded from {p}", p=scaler_path)

    # Load pairwise scaler
    pairwise_scaler_path = checkpoint_dir / "pairwise_scaler.pkl"
    pairwise_scaler = None
    if pairwise_scaler_path.exists():
        pairwise_scaler = joblib.load(pairwise_scaler_path)
        log.info("Pairwise scaler loaded from {p}", p=pairwise_scaler_path)
    else:
        log.warning("Pairwise scaler not found at {p}. Proceeding without it.", p=pairwise_scaler_path)

    # Load features for the evaluation split
    split_json_map = {"test": cfg.paths.test_json, "val": cfg.paths.val_json}
    with FeatureCache(cfg.paths.feature_cache_dir, split=split) as cache:
        exp_a, exp_b, lat_a, lat_b, labels = _load_pairwise_dataset(
            split_json_map[split], cache
        )

    log.info("Pairs loaded: {n}", n=len(labels))

    # Scale explicit features
    exp_a_scaled = scaler.transform(exp_a).astype(np.float32)
    exp_b_scaled = scaler.transform(exp_b).astype(np.float32)

    # Build pairwise features
    scorer_obj = PairwiseScorer()
    X = scorer_obj.build_pairwise_batch(exp_a_scaled, exp_b_scaled, lat_a, lat_b)

    # Apply pairwise scaler 
    if pairwise_scaler is not None:
        X = pairwise_scaler.transform(X).astype(np.float32)

    # Load model 
    model     = build_model(cfg)
    model_type = cfg.model.type.lower()
    ckpt_path  = checkpoint_dir / ("best_model.pt" if model_type == "mlp" else "best_model.pkl")
    model.load(ckpt_path)
    log.info("Model loaded from {p}", p=ckpt_path)

    # Inference 
    t0       = time.perf_counter()
    probs    = model.predict_proba(X)[:, 1]    # P(compatible) for each pair
    preds    = (probs >= 0.5).astype(np.int32)
    elapsed  = time.perf_counter() - t0
    harmony_scores = probs * 100.0

    # Compute metrics 
    acc       = accuracy_score(labels, preds)
    roc_auc   = roc_auc_score(labels, probs)
    avg_prec  = average_precision_score(labels, probs)
    cm        = confusion_matrix(labels, preds).tolist()
    cls_report = classification_report(labels, preds,
                                        target_names=["Incompatible", "Compatible"],
                                        output_dict=True)

    # Print summary 
    print("\n" + "═" * 60)
    print(f"  EVALUATION RESULTS — split: {split.upper()}")
    print("═" * 60)
    print(f"  Pairs evaluated   : {len(labels)}")
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  ROC-AUC           : {roc_auc:.4f}")
    print(f"  Average Precision : {avg_prec:.4f}")
    print(f"  Inference time    : {elapsed:.2f}s  ({elapsed/len(labels)*1000:.1f}ms/pair)")
    print()
    print("  Harmony Score Distribution")
    print(f"    mean  = {harmony_scores.mean():.1f}%")
    print(f"    std   = {harmony_scores.std():.1f}%")
    print(f"    min   = {harmony_scores.min():.1f}%")
    print(f"    max   = {harmony_scores.max():.1f}%")
    print()
    print("  Classification Report:")
    print(classification_report(labels, preds,
                                 target_names=["Incompatible", "Compatible"]))
    print("  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"    {cm[0]}")
    print(f"    {cm[1]}")
    print("═" * 60 + "\n")

    # Save JSON report
    timestamp   = time.strftime("%Y%m%d_%H%M%S")
    report_path = results_dir / f"eval_{split}_{timestamp}.json"

    report = {
        "split":              split,
        "n_pairs":            int(len(labels)),
        "accuracy":           float(acc),
        "roc_auc":            float(roc_auc),
        "average_precision":  float(avg_prec),
        "confusion_matrix":   cm,
        "classification_report": cls_report,
        "harmony_distribution": {
            "mean": float(harmony_scores.mean()),
            "std":  float(harmony_scores.std()),
            "min":  float(harmony_scores.min()),
            "max":  float(harmony_scores.max()),
        },
        "inference_total_sec": float(elapsed),
        "model_type": cfg.model.type,
    }

    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)

    log.info("Evaluation report saved to {p}", p=report_path)


if __name__ == "__main__":
    main()
