"""
scripts/infer.py
─────────────────
CLI script — Score the compatibility between two garment images.

Usage
-----
    # Score a pair of images
    python scripts/infer.py --item1 path/to/shirt.jpg --item2 path/to/pants.jpg

    # Score from a JSON file containing a list of pairs
    python scripts/infer.py --pairs_json my_pairs.json --output results.json

    # Use a non-default config
    python scripts/infer.py --item1 a.jpg --item2 b.jpg --config configs/config.yaml

Pairs JSON format (for batch mode)
-----------------------------------
[
    {"item_1": "path/to/a.jpg", "item_2": "path/to/b.jpg"},
    ...
]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fusion.harmony_scorer   import HarmonyScorer
from src.utils.config_loader     import load_config
from src.utils.image_io          import load_image
from src.utils.logger            import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

_VERDICT_COLORS = {
    "Excellent Match": "\033[92m",   # green
    "Good Match":      "\033[96m",   # cyan
    "Moderate Match":  "\033[93m",   # yellow
    "Poor Match":      "\033[91m",   # red
    "Incompatible":    "\033[31m",   # dark red
}
_RESET = "\033[0m"


def _print_result(item1: str, item2: str, result) -> None:
    """Render a single HarmonyResult to stdout with ANSI colour."""
    color = _VERDICT_COLORS.get(result.verdict, "")
    bar_filled = int(result.harmony_percent / 2)   # 50-char bar
    bar = "█" * bar_filled + "░" * (50 - bar_filled)

    print()
    print("┌─ Outfit Compatibility Scorer ──────────────────────────────┐")
    print(f"│  Item A : {Path(item1).name:<49} │")
    print(f"│  Item B : {Path(item2).name:<49} │")
    print(f"│  {bar} │")
    print(f"│  Harmony  : {color}{result.harmony_percent:>6.1f}%{_RESET}  "
          f"│  Verdict: {color}{result.verdict:<18}{_RESET} │")
    print(f"│  ML prob  : {result.prob_compatible:.4f}          "
          f"│  Color rule: {result.color_rule_score:.4f}       │")
    print(f"│  CLIP cos : {result.cosine_sim:+.4f}          "
          f"│  Eucl dist: {result.euclidean_dist:.4f}         │")
    print(f"│  Dom.rule : {result.color_analysis.dominant_rule:<49} │")
    print(f"│  Latency  : {result.latency_ms:.1f} ms{'':<43} │")
    print("└─────────────────────────────────────────────────────────────┘")
    print()


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score outfit compatibility between two garment images."
    )
    # Single-pair mode
    parser.add_argument("--item1", default=None, help="Path to the first garment image")
    parser.add_argument("--item2", default=None, help="Path to the second garment image")
    # Batch mode
    parser.add_argument("--pairs_json", default=None,
                        help="JSON file with a list of {item_1, item_2} dicts")
    parser.add_argument("--output", default=None,
                        help="(batch mode) Path to save JSON results")
    # Common
    parser.add_argument("--config",  default=None, help="Path to config YAML")
    parser.add_argument("--device",  default=None, help="Override device: cuda | cpu")
    args = parser.parse_args()

    # ── Validate arguments ────────────────────────────────────────────────────
    if args.item1 is None and args.pairs_json is None:
        parser.error("Provide either --item1/--item2 for single mode "
                     "or --pairs_json for batch mode.")

    cfg = load_config(args.config)
    if args.device:
        cfg.clip.device = args.device

    # ── Build and load the scorer ─────────────────────────────────────────────
    scaler_path = Path(cfg.paths.checkpoint_dir) / "explicit_scaler.pkl"
    scorer = HarmonyScorer.from_config(cfg, scaler_path=scaler_path if scaler_path.exists() else None)
    scorer.load_model(cfg.paths.checkpoint_dir)

    image_size = tuple(cfg.dataset.image_size)

    # ─────────────────────────────────────────────────────────────────────────
    # Single-pair mode
    # ─────────────────────────────────────────────────────────────────────────
    if args.item1 is not None:
        if args.item2 is None:
            parser.error("--item2 is required when --item1 is provided.")

        image_a = load_image(args.item1, size=image_size)
        image_b = load_image(args.item2, size=image_size)

        result = scorer.score(image_a, image_b)
        _print_result(args.item1, args.item2, result)

    # ─────────────────────────────────────────────────────────────────────────
    # Batch mode
    # ─────────────────────────────────────────────────────────────────────────
    else:
        with open(args.pairs_json) as fh:
            pairs = json.load(fh)

        log.info("Running batch inference on {n} pairs …", n=len(pairs))
        output_records = []

        for pair in pairs:
            p1, p2 = pair["item_1"], pair["item_2"]
            try:
                img_a  = load_image(p1, size=image_size)
                img_b  = load_image(p2, size=image_size)
                result = scorer.score(img_a, img_b)
                _print_result(p1, p2, result)
                output_records.append({
                    "item_1":            p1,
                    "item_2":            p2,
                    "harmony_percent":   result.harmony_percent,
                    "verdict":           result.verdict,
                    "prob_compatible":   result.prob_compatible,
                    "color_rule_score":  result.color_rule_score,
                    "dominant_color_rule": result.color_analysis.dominant_rule,
                    "cosine_sim":        result.cosine_sim,
                    "euclidean_dist":    result.euclidean_dist,
                    "latency_ms":        result.latency_ms,
                })
            except Exception as exc:
                log.warning("Skipping pair ({p1}, {p2}): {e}", p1=p1, p2=p2, e=exc)

        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as fh:
                json.dump(output_records, fh, indent=2)
            log.info("Batch results saved to {p}", p=args.output)


if __name__ == "__main__":
    main()
