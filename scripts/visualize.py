"""
CLI script — Generate all visualisation plots and save them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Allow running from project root without pip install ──────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import json
import glob
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for CLI
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from src.utils.config_loader     import load_config
from src.utils.image_io          import load_image
from src.utils.logger            import get_logger

log = get_logger(__name__)



# Matplotlib style

def _apply_style() -> None:
    """Apply a clean, publication-friendly matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.family":      "DejaVu Sans",
        "font.size":        11,
        "savefig.dpi":      150,
        "savefig.bbox":     "tight",
    })


# Helper: convert L*a*b* palette vector → list of (R, G, B) tuples

def _lab_vec_to_rgb(vec: np.ndarray) -> list:
    """Convert a flat (K*3,) canonical L*a*b* vector to RGB tuples."""
    

    if vec.size % 3 != 0:
        # truncate safely to nearest multiple of 3
        valid_size = (vec.size // 3) * 3
        if valid_size == 0:
            raise ValueError(
                f"Cannot convert vector of size {vec.size} to LAB triplets."
            )
        vec = vec[:valid_size]

    centres = vec.reshape(-1, 3)

    rgbs = []
    for L, a, b in centres:
        L_cv = float(L) * (255.0 / 100.0)
        a_cv = float(a) + 128.0
        b_cv = float(b) + 128.0
        lab_px  = np.array([[[L_cv, a_cv, b_cv]]], dtype=np.float32)
        bgr_px  = cv2.cvtColor(lab_px.astype(np.uint8), cv2.COLOR_LAB2BGR)
        r, g, bc = bgr_px[0, 0, 2], bgr_px[0, 0, 1], bgr_px[0, 0, 0]
        rgbs.append((r / 255.0, g / 255.0, bc / 255.0))

    return rgbs


# Helper: per-block L2 normalisation (mirrors train.py)

def _normalise_subblocks(
    vectors: np.ndarray, color_dim: int, texture_dim: int
) -> np.ndarray:
    c_end = color_dim
    t_end = color_dim + texture_dim

    def l2(b: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(b, axis=1, keepdims=True)
        return b / np.where(n == 0, 1.0, n)

    return np.concatenate([
        l2(vectors[:, :c_end]),
        l2(vectors[:, c_end:t_end]),
        l2(vectors[:, t_end:]),
    ], axis=1).astype(np.float32)


# SET A — Pipeline visualisations

def run_pipeline_plots(cfg, item_a_path: str, item_b_path: str, out_dir: Path) -> None:
    """Generate all per-module CV feature visualisations."""
    from src.preprocessing.segmenter    import GrabCutSegmenter
    from src.features.color_harmony     import ColorHarmonyExtractor
    from src.features.texture_analyzer  import TextureAnalyzer
    from src.features.shape_descriptor  import ShapeDescriptor
    from src.features.explicit_extractor import ExplicitFeatureExtractor
    from src.fusion.color_rule_scorer   import ColorRuleScorer
    from skimage.feature                import hog as sk_hog

    log.info("Loading images for pipeline visualisation …")
    img_size = tuple(cfg.dataset.image_size)
    img_a = load_image(item_a_path, size=img_size)
    img_b = load_image(item_b_path, size=img_size)

    # Segmentation 
    log.info("Running GrabCut segmentation …")
    segmenter = GrabCutSegmenter(
        grabcut_iterations   = cfg.segmentation.grabcut_iterations,
        border_margin        = cfg.segmentation.border_margin,
        min_foreground_ratio = cfg.segmentation.min_foreground_ratio,
    )
    seg_a, mask_a = segmenter.segment(img_a)
    seg_b, mask_b = segmenter.segment(img_b)

    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    fig.suptitle("Module 1 — GrabCut Background Removal",
                 fontsize=14, fontweight="bold", y=1.01)

    for row, (orig, mask, seg, lbl) in enumerate([
        (img_a, mask_a, seg_a, f"Item A  ({Path(item_a_path).stem})"),
        (img_b, mask_b, seg_b, f"Item B  ({Path(item_b_path).stem})"),
    ]):
        fg = (mask > 0).sum() / mask.size
        axes[row, 0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
        axes[row, 0].set_title("Original",          fontsize=10)
        axes[row, 0].set_ylabel(lbl,                fontsize=10)
        axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=255)
        axes[row, 1].set_title("GrabCut Mask",      fontsize=10)
        axes[row, 1].text(0.02, 0.98, f"FG: {fg:.1%}",
                          transform=axes[row, 1].transAxes,
                          va="top", ha="left", fontsize=9, color="white",
                          fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))
        axes[row, 2].imshow(cv2.cvtColor(seg, cv2.COLOR_BGR2RGB))
        axes[row, 2].set_title("Segmented",         fontsize=10)

    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    p = out_dir / "01_segmentation.png"
    plt.savefig(p); plt.close(); log.info("Saved {p}", p=p)

    # Colour palettes
    log.info("Extracting colour palettes …")
    color_ext = ColorHarmonyExtractor(n_dominant_colors=5)
    vec_a = color_ext.extract(seg_a)
    vec_b = color_ext.extract(seg_b)

    # FIX: force vectors to expected palette size (5 colors × 3 = 15)
    vec_a = vec_a[:15] if vec_a.size >= 15 else np.pad(vec_a, (0, 15 - vec_a.size))
    vec_b = vec_b[:15] if vec_b.size >= 15 else np.pad(vec_b, (0, 15 - vec_b.size))
    sw_a      = _lab_vec_to_rgb(vec_a)
    sw_b      = _lab_vec_to_rgb(vec_b)

    fig, axes = plt.subplots(2, 6, figsize=(14, 5),
                              gridspec_kw={"width_ratios": [2, 1, 1, 1, 1, 1]})
    fig.suptitle("Module 2a — Dominant Colour Palettes (CIE L*a*b* → RGB)",
                 fontsize=13, fontweight="bold")

    for row, (img, swatches, lbl) in enumerate([
        (cv2.cvtColor(seg_a, cv2.COLOR_BGR2RGB), sw_a, "Item A"),
        (cv2.cvtColor(seg_b, cv2.COLOR_BGR2RGB), sw_b, "Item B"),
    ]):
        axes[row, 0].imshow(img)
        axes[row, 0].set_title("Segmented", fontsize=9)
        axes[row, 0].set_ylabel(lbl,        fontsize=10)
        axes[row, 0].axis("off")
        for col, rgb in enumerate(swatches[:5], start=1):
            axes[row, col].set_facecolor(rgb)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            tc  = "white" if lum < 0.5 else "black"
            hx  = "#{:02X}{:02X}{:02X}".format(
                int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
            axes[row, col].text(0.5, 0.5, hx, ha="center", va="center",
                                fontsize=8, color=tc, fontweight="bold",
                                transform=axes[row, col].transAxes)
            if row == 0:
                axes[row, col].set_title(f"#{col}", fontsize=9)

    plt.tight_layout()
    p = out_dir / "02_colour_palettes.png"
    plt.savefig(p); plt.close(); log.info("Saved {p}", p=p)

    # Colour rules radar 
    log.info("Computing colour rule scores …")
    rule_scorer = ColorRuleScorer(n_dominant_colors=5)
    analysis    = rule_scorer.analyse(vec_a, vec_b)

    rule_names  = list(analysis.rule_scores.keys())
    scores      = [analysis.rule_scores[r] for r in rule_names]
    n_rules     = len(rule_names)
    angles      = np.linspace(0, 2 * np.pi, n_rules, endpoint=False).tolist()
    angles     += angles[:1]
    scores_p    = scores + scores[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.suptitle(
        f"Module 4b — Colour Rule Breakdown\n"
        f"Overall: {analysis.overall_score:.2%}  |  Dominant: {analysis.dominant_rule}",
        fontsize=12, fontweight="bold",
    )
    for ring in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ax.plot(angles, [ring] * (n_rules + 1), color="lightgrey",
                linewidth=0.5, linestyle="--")
    ax.fill(angles, scores_p, alpha=0.25, color="steelblue")
    ax.plot(angles, scores_p, color="steelblue", linewidth=2)
    ax.scatter(angles[:-1], scores, s=80, color="steelblue", zorder=5)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [r.replace("_", "\n") for r in rule_names],
        fontsize=10, fontweight="bold",
    )
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=8, color="grey")
    ax.set_ylim(0, 1)
    ax.spines["polar"].set_visible(False)
    for angle, score in zip(angles[:-1], scores):
        ax.annotate(f"{score:.2f}", xy=(angle, score),
                    xytext=(angle, score + 0.09),
                    ha="center", va="center", fontsize=9,
                    color="steelblue", fontweight="bold")
    plt.tight_layout()
    p = out_dir / "03_colour_rules_radar.png"
    plt.savefig(p); plt.close(); log.info("Saved {p}", p=p)

    # Gabor responses
    log.info("Computing Gabor filter bank responses …")
    analyzer = TextureAnalyzer(
        orientations          = list(cfg.texture.orientations),
        wavelengths           = list(cfg.texture.wavelengths),
        sigma_to_lambda_ratio = cfg.texture.sigma_to_lambda_ratio,
        gabor_aspect_ratio    = cfg.texture.gabor_aspect_ratio,
    )
    gray_a = cv2.cvtColor(seg_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(seg_b, cv2.COLOR_BGR2GRAY)

    orientations_list = list(cfg.texture.orientations)
    wavelengths_list  = list(cfg.texture.wavelengths)
    n_orient = len(orientations_list)

    fig, axes = plt.subplots(4, 8, figsize=(18, 9))
    fig.suptitle("Module 2b — Gabor Filter Bank Responses (4 orientations × 4 wavelengths)",
                 fontsize=12, fontweight="bold")

    for idx, kernel in enumerate(analyzer._filter_bank):
        resp_a = cv2.filter2D(gray_a, cv2.CV_32F, kernel)
        resp_b = cv2.filter2D(gray_b, cv2.CV_32F, kernel)
        orient = orientations_list[idx % n_orient]
        wlen   = wavelengths_list[idx // n_orient]
        row_i  = idx // n_orient
        col_i  = (idx % n_orient) * 2
        axes[row_i, col_i].imshow(np.abs(resp_a), cmap="hot")
        axes[row_i, col_i].set_title(f"A  θ={orient}° λ={wlen}px", fontsize=7)
        axes[row_i, col_i].axis("off")
        axes[row_i, col_i + 1].imshow(np.abs(resp_b), cmap="hot")
        axes[row_i, col_i + 1].set_title(f"B  θ={orient}° λ={wlen}px", fontsize=7)
        axes[row_i, col_i + 1].axis("off")

    plt.tight_layout()
    p = out_dir / "04_gabor_responses.png"
    plt.savefig(p); plt.close(); log.info("Saved {p}", p=p)

    #HOG
    log.info("Computing HOG descriptors …")

    def hog_vis(bgr, hog_size):
        w, h    = hog_size
        resized = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        _, hog_img = sk_hog(
            gray,
            orientations    = cfg.shape.hog_orientations,
            pixels_per_cell = tuple(cfg.shape.hog_pixels_per_cell),
            cells_per_block = tuple(cfg.shape.hog_cells_per_block),
            block_norm      = "L2-Hys",
            visualize       = True,
        )
        return gray, hog_img

    hog_size = tuple(cfg.shape.hog_image_size)
    gr_a, hi_a = hog_vis(seg_a, hog_size)
    gr_b, hi_b = hog_vis(seg_b, hog_size)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(
        f"Module 2c — HOG Shape Descriptor  "
        f"({hog_size[0]}×{hog_size[1]}px, {cfg.shape.hog_orientations} bins)",
        fontsize=12, fontweight="bold",
    )
    axes[0, 0].imshow(gr_a, cmap="gray")
    axes[0, 0].set_title("Item A — segmented (gray)", fontsize=10)
    axes[0, 1].imshow(hi_a, cmap="magma")
    axes[0, 1].set_title("Item A — HOG gradient map", fontsize=10)
    axes[1, 0].imshow(gr_b, cmap="gray")
    axes[1, 0].set_title("Item B — segmented (gray)", fontsize=10)
    axes[1, 1].imshow(hi_b, cmap="magma")
    axes[1, 1].set_title("Item B — HOG gradient map", fontsize=10)
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    p = out_dir / "05_hog_shape.png"
    plt.savefig(p); plt.close(); log.info("Saved {p}", p=p)

    # Explicit feature comparison
    log.info("Extracting explicit feature vectors …")
    explicit_ext = ExplicitFeatureExtractor(cfg, scaler_path=None)
    exp_a_raw    = explicit_ext.extract(img_a)
    exp_b_raw    = explicit_ext.extract(img_b)
    dims         = explicit_ext.get_sub_vector_lengths()
    c_dim        = dims["color"]
    t_dim        = dims["texture"]
    s_dim        = dims["shape"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle("Module 2 — Explicit Feature Sub-Vectors (Item A vs Item B)",
                 fontsize=13, fontweight="bold")

    sections = [
        ("Color (L*a*b* palette)",     0,        c_dim),
        ("Texture (Gabor energy)",      c_dim,    c_dim + t_dim),
        ("Shape (HOG — first 128 dims)", c_dim + t_dim, c_dim + t_dim + min(128, s_dim)),
    ]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for ax, (title, start, end), color in zip(axes, sections, colors):
        x  = np.arange(end - start)
        va = exp_a_raw[start:end]
        vb = exp_b_raw[start:end]
        w  = 0.4
        ax.bar(x - w/2, va, width=w, alpha=0.75, label="Item A", color=color)
        ax.bar(x + w/2, vb, width=w, alpha=0.55, label="Item B", color="grey")
        ax.set_title(title,          fontsize=10, fontweight="bold")
        ax.set_xlabel("Feature dim", fontsize=9)
        ax.set_ylabel("Value",       fontsize=9)
        ax.legend(fontsize=9)
        dist = np.linalg.norm(va - vb)
        ax.text(0.98, 0.97, f"L2 dist = {dist:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                color="darkred",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                          ec="orange", alpha=0.8))

    plt.tight_layout()
    p = out_dir / "06_explicit_features.png"
    plt.savefig(p); plt.close(); log.info("Saved {p}", p=p)

    log.info("All pipeline plots saved to {d}", d=out_dir)


# SET B — Result visualisations

def run_results_plots(cfg, out_dir: Path) -> None:
    """Generate all model evaluation plots."""
    from sklearn.metrics import (
        confusion_matrix, roc_curve, auc,
        precision_recall_curve, average_precision_score,
        accuracy_score, roc_auc_score, classification_report,
    )
    from sklearn.calibration import calibration_curve
    from src.utils.dataset_loader   import load_pairs
    from src.utils.feature_cache    import FeatureCache
    from src.fusion.pairwise_scorer import PairwiseScorer
    from src.models.model_factory   import build_model

    checkpoint_dir = Path(cfg.paths.checkpoint_dir)
    results_dir    = Path(cfg.paths.results_dir)

    # Load scaler and normalisation metadata 
    scaler_path = checkpoint_dir / "explicit_scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. Run scripts/train.py first."
        )
    scaler = joblib.load(scaler_path)

    nm_path = checkpoint_dir / "norm_meta.pkl"
    if nm_path.exists():
        nm = joblib.load(nm_path)
        do_norm   = nm.get("normalize_subblocks", False)
        color_dim = nm.get("color_dim",   15)
        tex_dim   = nm.get("texture_dim", 32)
    else:
        do_norm, color_dim, tex_dim = False, 15, 32

    # Load test features from HDF5
    log.info("Loading test features from cache …")
    pairs  = load_pairs(cfg.paths.test_json)
    ea_l, eb_l, la_l, lb_l, labs = [], [], [], [], []
    skipped = 0

    with FeatureCache(cfg.paths.feature_cache_dir, split="test") as cache:
        for pair in tqdm(pairs, desc="Loading features"):
            id_a, id_b = pair["item_1"], pair["item_2"]
            ea = cache.load_explicit(id_a); eb = cache.load_explicit(id_b)
            la = cache.load_latent(id_a);   lb = cache.load_latent(id_b)
            if any(v is None for v in (ea, eb, la, lb)):
                skipped += 1; continue
            ea_l.append(ea); eb_l.append(eb)
            la_l.append(la); lb_l.append(lb)
            labs.append(int(pair["label"]))

    if skipped:
        log.warning("Skipped {n} pairs with missing cache entries.", n=skipped)

    exp_a  = np.stack(ea_l); exp_b = np.stack(eb_l)
    lat_a  = np.stack(la_l); lat_b = np.stack(lb_l)
    y_true = np.array(labs, dtype=np.int32)

    if do_norm:
        exp_a = _normalise_subblocks(exp_a, color_dim, tex_dim)
        exp_b = _normalise_subblocks(exp_b, color_dim, tex_dim)

    exp_a = scaler.transform(exp_a).astype(np.float32)
    exp_b = scaler.transform(exp_b).astype(np.float32)

    scorer_obj = PairwiseScorer()
    X = scorer_obj.build_pairwise_batch(exp_a, exp_b, lat_a, lat_b)

    model      = build_model(cfg)
    model_type = cfg.model.type.lower()
    ckpt       = checkpoint_dir / (
        "best_model.pt" if model_type == "mlp" else "best_model.pkl"
    )
    model.load(str(ckpt))
    log.info("Running inference on {n} test pairs …", n=len(y_true))
    y_prob         = model.predict_proba(X)[:, 1]
    y_pred         = (y_prob >= 0.5).astype(np.int32)
    harmony_scores = y_prob * 100.0

    # Precompute common metric
    acc  = accuracy_score(y_true, y_pred)
    roc  = roc_auc_score(y_true, y_prob)
    ap   = average_precision_score(y_true, y_prob)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fpr_, tpr_, thresh_roc = roc_curve(y_true, y_prob)
    prec_, rec_, thresh_pr = precision_recall_curve(y_true, y_prob)
    frac_pos_, mean_pred_  = calibration_curve(y_true, y_prob, n_bins=10)
    cr = classification_report(
        y_true, y_pred,
        target_names=["Incompatible", "Compatible"],
        output_dict=True,
    )

    # Plot 1: Confusion matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["Incompatible", "Compatible"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix — Test Split", fontsize=13, fontweight="bold", pad=14)
    total = len(y_true)
    ax.text(0.5, -0.18,
            f"TP={tp} ({tp/total:.1%})  FP={fp} ({fp/total:.1%})"
            f"   FN={fn} ({fn/total:.1%})  TN={tn} ({tn/total:.1%})",
            transform=ax.transAxes, ha="center", fontsize=9, color="dimgray",
            bbox=dict(boxstyle="round", fc="lightyellow", ec="#ccc", alpha=0.9))
    plt.tight_layout()
    p = out_dir / "r01_confusion_matrix.png"
    plt.savefig(p); plt.close(); log.info("Saved {p}", p=p)

    # Plot 2: ROC curve
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr_, tpr_, color="steelblue", lw=2.5,
            label=f"ROC  (AUC = {roc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.fill_between(fpr_, tpr_, alpha=0.08, color="steelblue")
    youden_idx = np.argmax(tpr_ - fpr_)
    ax.scatter([fpr_[youden_idx]], [tpr_[youden_idx]], s=120, zorder=5,
               color="orangered",
               label=f"Optimal threshold = {thresh_roc[youden_idx]:.3f}")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title("ROC Curve — Test Split", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    p = out_dir / "r02_roc_curve.png"
    plt.savefig(p); plt.close(); log.info("Saved {p}", p=p)

    # Plot 3: PR curve
    f1_scores = (2 * prec_ * rec_ /
                 np.where((prec_ + rec_) == 0, 1, prec_ + rec_))
    best_f1_idx = np.argmax(f1_scores[:-1])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.step(rec_, prec_, where="post", color="darkorange", lw=2.5,
            label=f"P-R curve  (AP = {ap:.4f})")
    ax.fill_between(rec_, prec_, step="post", alpha=0.08, color="darkorange")
    ax.axhline(y_true.mean(), color="grey", linestyle="--", lw=1,
               label=f"Random baseline ({y_true.mean():.2f})")
    ax.scatter([rec_[best_f1_idx]], [prec_[best_f1_idx]],
               s=120, zorder=5, color="darkgreen",
               label=f"Best F1 = {f1_scores[best_f1_idx]:.3f}")
    ax.set_xlabel("Recall", fontsize=11); ax.set_ylabel("Precision", fontsize=11)
    ax.set_xlim([0, 1]);   ax.set_ylim([0, 1.05])
    ax.set_title("Precision-Recall Curve — Test Split", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    p = out_dir / "r03_pr_curve.png"
    plt.savefig(p); plt.close(); log.info("Saved {p}", p=p)

    # Plot 4: Harmony score distributions
    compat_sc   = harmony_scores[y_true == 1]
    incompat_sc = harmony_scores[y_true == 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Harmony Score Distributions — Test Split",
                 fontsize=13, fontweight="bold")

    bins = np.linspace(0, 100, 41)
    axes[0].hist(incompat_sc, bins=bins, alpha=0.55, color="#E53935",
                 label=f"Incompatible (n={len(incompat_sc)})", density=True)
    axes[0].hist(compat_sc,   bins=bins, alpha=0.55, color="#1E88E5",
                 label=f"Compatible   (n={len(compat_sc)})",   density=True)
    axes[0].axvline(50, color="black", linestyle="--", lw=1.2, alpha=0.6)
    axes[0].set_xlabel("Harmony score (%)", fontsize=10)
    axes[0].set_ylabel("Density",           fontsize=10)
    axes[0].set_title("Score distributions", fontsize=10)
    axes[0].legend(fontsize=9)

    vp = axes[1].violinplot(
        [incompat_sc, compat_sc], positions=[1, 2],
        showmedians=True, showextrema=True,
    )
    vp["cmedians"].set_color("black")
    vp["bodies"][0].set_facecolor("#E53935"); vp["bodies"][0].set_alpha(0.6)
    vp["bodies"][1].set_facecolor("#1E88E5"); vp["bodies"][1].set_alpha(0.6)
    for part in ["cbars", "cmins", "cmaxes"]:
        vp[part].set_color("grey"); vp[part].set_linewidth(0.8)
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(["Incompatible", "Compatible"], fontsize=10)
    axes[1].set_ylabel("Harmony score (%)", fontsize=10)
    axes[1].set_title("Violin plot", fontsize=10)
    axes[1].axhline(50, color="black", linestyle="--", lw=1, alpha=0.5)
    axes[1].set_ylim(0, 105)
    for i, (sc, color) in enumerate([(incompat_sc, "#E53935"),
                                      (compat_sc,   "#1E88E5")], start=1):
        axes[1].text(i, np.median(sc) + 3, f"med={np.median(sc):.1f}%",
                     ha="center", fontsize=8.5, color=color, fontweight="bold")

    plt.tight_layout()
    p = out_dir / "r04_harmony_distributions.png"
    plt.savefig(p); plt.close(); log.info("Saved {p}", p=p)

    # Plot 5: Per-class metrics bar chart
    metrics = ["precision", "recall", "f1-score"]
    classes = ["Incompatible", "Compatible"]
    x = np.arange(len(metrics)); width = 0.32
    colors_cls = ["#E53935", "#1E88E5"]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Per-Class Metrics — Test Split", fontsize=13, fontweight="bold")
    for i, (cls, color) in enumerate(zip(classes, colors_cls)):
        vals = [cr[cls][m] for m in metrics]
        bars = ax.bar(x + (i - 0.5) * width, vals, width,
                      label=cls, color=color, alpha=0.80)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["Precision", "Recall", "F1-score"], fontsize=11)
    ax.set_ylabel("Score", fontsize=11); ax.set_ylim(0, 1.12)
    ax.axhline(acc, color="grey", linestyle="--", lw=1,
               label=f"Overall accuracy = {acc:.3f}")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    p = out_dir / "r05_per_class_metrics.png"
    plt.savefig(p); plt.close(); log.info("Saved {p}", p=p)

    # Plot 6: Calibration
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Calibration Analysis — Test Split",
                 fontsize=13, fontweight="bold")

    axes[0].plot(mean_pred_, frac_pos_, "s-", color="steelblue", lw=2, ms=7,
                 label="Model calibration")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
    axes[0].fill_between(mean_pred_, frac_pos_, mean_pred_,
                          alpha=0.1, color="steelblue")
    axes[0].set_xlabel("Mean predicted probability", fontsize=10)
    axes[0].set_ylabel("Fraction of positives",      fontsize=10)
    axes[0].set_title("Reliability diagram", fontsize=10)
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3, linestyle="--")
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)

    axes[1].hist(y_prob, bins=40, color="steelblue", alpha=0.7,
                 edgecolor="white", lw=0.3)
    axes[1].set_xlabel("Predicted P(compatible)", fontsize=10)
    axes[1].set_ylabel("Count",                   fontsize=10)
    axes[1].set_title("Predicted probability distribution", fontsize=10)
    axes[1].axvline(0.5, color="black", linestyle="--", lw=1.2,
                    label="Decision boundary")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    p = out_dir / "r06_calibration.png"
    plt.savefig(p); plt.close(); log.info("Saved {p}", p=p)

    # Plot 7: Summary dashboard
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Model Performance Dashboard — Test Split",
                 fontsize=15, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    metric_data = [
        ("Accuracy",      f"{acc:.1%}", "steelblue"),
        ("ROC-AUC",       f"{roc:.4f}", "darkorange"),
        ("Avg Precision", f"{ap:.4f}",  "green"),
    ]
    for i, (title, val, color) in enumerate(metric_data):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor("#f8f9fa")
        ax.text(0.5, 0.65, val,   ha="center", va="center", fontsize=26,
                fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(0.5, 0.20, title, ha="center", va="center", fontsize=11,
                color="grey", transform=ax.transAxes)
        ax.axis("off")

    ax_cm = fig.add_subplot(gs[1, 0])
    cm_d = np.array([[tn, fp], [fn, tp]])
    ax_cm.imshow(cm_d, cmap="Blues", aspect="auto")
    for r in range(2):
        for c in range(2):
            ax_cm.text(c, r, str(cm_d[r, c]), ha="center", va="center",
                       fontsize=14, fontweight="bold",
                       color="white" if cm_d[r, c] > cm_d.max() * 0.5 else "black")
    ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Pred:Inc.", "Pred:Com."], fontsize=8)
    ax_cm.set_yticklabels(["True:Inc.", "True:Com."], fontsize=8)
    ax_cm.set_title("Confusion matrix", fontsize=10)

    ax_roc = fig.add_subplot(gs[1, 1])
    ax_roc.plot(fpr_, tpr_, color="steelblue", lw=2)
    ax_roc.plot([0,1],[0,1],"k--",lw=0.8,alpha=0.5)
    ax_roc.fill_between(fpr_, tpr_, alpha=0.08, color="steelblue")
    ax_roc.set_title(f"ROC  (AUC={roc:.3f})", fontsize=10)
    ax_roc.set_xlabel("FPR", fontsize=8); ax_roc.set_ylabel("TPR", fontsize=8)

    ax_dist = fig.add_subplot(gs[1, 2])
    bins_ = np.linspace(0, 100, 31)
    ax_dist.hist(harmony_scores[y_true==0], bins=bins_, alpha=0.55,
                 color="#E53935", label="Incompatible", density=True)
    ax_dist.hist(harmony_scores[y_true==1], bins=bins_, alpha=0.55,
                 color="#1E88E5", label="Compatible",   density=True)
    ax_dist.axvline(50, color="black", linestyle="--", lw=1, alpha=0.6)
    ax_dist.set_title("Harmony score dist.", fontsize=10)
    ax_dist.set_xlabel("Score (%)", fontsize=8); ax_dist.legend(fontsize=8)

    ax_pr = fig.add_subplot(gs[2, 0])
    ax_pr.step(rec_, prec_, where="post", color="darkorange", lw=2)
    ax_pr.fill_between(rec_, prec_, step="post", alpha=0.08, color="darkorange")
    ax_pr.axhline(y_true.mean(), color="grey", linestyle="--", lw=0.8)
    ax_pr.set_title(f"P-R  (AP={ap:.3f})", fontsize=10)
    ax_pr.set_xlabel("Recall", fontsize=8); ax_pr.set_ylabel("Precision", fontsize=8)

    ax_cal = fig.add_subplot(gs[2, 1])
    ax_cal.plot(mean_pred_, frac_pos_, "s-", color="steelblue", lw=1.5, ms=5)
    ax_cal.plot([0,1],[0,1],"k--",lw=0.8,alpha=0.5)
    ax_cal.set_title("Calibration", fontsize=10)
    ax_cal.set_xlabel("Pred prob", fontsize=8); ax_cal.set_ylabel("Actual rate", fontsize=8)

    ax_f1 = fig.add_subplot(gs[2, 2])
    bars_f1 = ax_f1.bar(
        ["Incomp.", "Compat."],
        [cr["Incompatible"]["f1-score"], cr["Compatible"]["f1-score"]],
        color=["#E53935", "#1E88E5"], alpha=0.8, width=0.5,
    )
    for bar, v in zip(bars_f1, [cr["Incompatible"]["f1-score"],
                                  cr["Compatible"]["f1-score"]]):
        ax_f1.text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() + 0.01,
                   f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax_f1.set_ylim(0, 1.05); ax_f1.set_title("F1 by class", fontsize=10)
    ax_f1.axhline(acc, color="grey", linestyle="--", lw=0.8, alpha=0.7)

    plt.savefig(out_dir / "r07_dashboard.png")
    plt.close()
    log.info("Saved {p}", p=out_dir / "r07_dashboard.png")
    log.info("All result plots saved to {d}", d=out_dir)


# Entry-point

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pipeline and result visualisation plots."
    )
    parser.add_argument(
        "--mode", default="all",
        choices=["all", "pipeline", "results"],
        help="Which set of plots to generate (default: all)",
    )
    parser.add_argument(
        "--item1", default=None,
        help="Item ID or image path for pipeline plot Item A. "
             "If omitted, the first image in images_dir is used.",
    )
    parser.add_argument(
        "--item2", default=None,
        help="Item ID or image path for pipeline plot Item B. "
             "If omitted, the second image in images_dir is used.",
    )
    parser.add_argument(
        "--out_dir", default=None,
        help="Output directory for plots. "
             "Defaults to outputs/results/plots/",
    )
    parser.add_argument("--config", default=None, help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _apply_style()

    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg.paths.results_dir) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: {d}", d=out_dir)

    # Resolve item image paths for pipeline plot
    if args.mode in ("all", "pipeline"):
        images_dir = Path(cfg.paths.images_dir)
        all_images = sorted(images_dir.glob("*.jpg"))

        def _resolve(arg: str | None, fallback_idx: int) -> str:
            if arg is None:
                if not all_images:
                    raise FileNotFoundError(
                        f"No .jpg images found in {images_dir}. "
                        "Set --item1 / --item2 explicitly."
                    )
                return str(all_images[fallback_idx])
            p = Path(arg)
            if p.exists():
                return str(p)
            # Treat as item_id — look in images_dir
            candidate = images_dir / f"{arg}.jpg"
            if candidate.exists():
                return str(candidate)
            raise FileNotFoundError(
                f"Cannot find image for '{arg}'. "
                f"Tried: {arg}, {candidate}"
            )

        item_a = _resolve(args.item1, 0)
        item_b = _resolve(args.item2, 1)
        log.info("Pipeline plots — Item A: {a}", a=Path(item_a).name)
        log.info("Pipeline plots — Item B: {b}", b=Path(item_b).name)

        run_pipeline_plots(cfg, item_a, item_b, out_dir)

    if args.mode in ("all", "results"):
        run_results_plots(cfg, out_dir)

    log.info("Visualisation complete. All plots in: {d}", d=out_dir)


if __name__ == "__main__":
    main()
