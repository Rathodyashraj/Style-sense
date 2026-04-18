"""
src/utils/__init__.py
──────────────────────
Shared utility sub-package.

Public exports
--------------
load_config                      — parse config.yaml → dot-accessible namespace.
get_logger                       — loguru logger bound to a module name.
load_image / load_image_pil      — consistent image loading helpers (BGR / PIL).
apply_mask / bgr_to_rgb          — image manipulation helpers.
FeatureCache                     — HDF5-backed per-item feature store.
PolyvoreCompatibilityDataset     — PyTorch Dataset for (itemA, itemB, label) pairs.
load_pairs / load_categories     — parse Polyvore JSON split files.
build_item_image_map             — scan images directory → item_id → Path dict.
"""

from src.utils.config_loader  import load_config
from src.utils.logger         import get_logger
from src.utils.image_io       import load_image, load_image_pil, apply_mask, bgr_to_rgb
from src.utils.feature_cache  import FeatureCache
from src.utils.dataset_loader import (
    PolyvoreCompatibilityDataset,
    load_pairs,
    load_categories,
    build_item_image_map,
)

__all__ = [
    "load_config",
    "get_logger",
    "load_image",
    "load_image_pil",
    "apply_mask",
    "bgr_to_rgb",
    "FeatureCache",
    "PolyvoreCompatibilityDataset",
    "load_pairs",
    "load_categories",
    "build_item_image_map",
]
