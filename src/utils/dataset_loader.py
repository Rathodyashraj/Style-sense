from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset

from src.utils.image_io import load_image, load_image_pil
from src.utils.logger import get_logger

log = get_logger(__name__)


def load_pairs(json_path: str | Path) -> List[Dict]:

    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Pairs file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as fh:
        pairs: List[Dict] = json.load(fh)

    log.info("Loaded {n} pairs from {p}", n=len(pairs), p=json_path.name)
    return pairs


def load_categories(json_path: str | Path) -> Dict[str, str]:

    json_path = Path(json_path)
    if not json_path.exists():
        log.warning("Category file not found: {p}. "
                    "Category-level analysis will be unavailable.", p=json_path)
        return {}

    with open(json_path, "r", encoding="utf-8") as fh:
        categories: Dict[str, str] = json.load(fh)

    log.info("Loaded {n} item→category entries.", n=len(categories))
    return categories




def build_item_image_map(images_dir: str | Path) -> Dict[str, Path]:

    images_dir = Path(images_dir)
    _EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
    item_map: Dict[str, Path] = {}

    for fp in images_dir.iterdir():
        if fp.suffix.lower() in _EXTENSIONS:
            item_map[fp.stem] = fp

    log.info("Found {n} images in {d}.", n=len(item_map), d=images_dir)
    return item_map




class PolyvoreCompatibilityDataset(Dataset):


    def __init__(
        self,
        pairs_json: str | Path,
        images_dir: str | Path,
        image_size: Tuple[int, int] = (224, 224),
        categories: Optional[Dict[str, str]] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        self.image_size = image_size
        self.categories = categories or {}

        self.pairs: List[Dict] = load_pairs(pairs_json)
        if max_samples:
            self.pairs = self.pairs[:max_samples]

        self.item_map: Dict[str, Path] = build_item_image_map(images_dir)

        # Filter out pairs where either image is missing
        before = len(self.pairs)
        self.pairs = [
            p for p in self.pairs
            if p["item_1"] in self.item_map and p["item_2"] in self.item_map
        ]
        after = len(self.pairs)
        if before != after:
            log.warning(
                "Dropped {n} pairs with missing images ({before} → {after}).",
                n=before - after, before=before, after=after,
            )

        # Label distribution
        n_pos = sum(1 for p in self.pairs if p["label"] == 1)
        n_neg = len(self.pairs) - n_pos
        log.info("Dataset ready — {tot} pairs | {p} positive | {n} negative",
                 tot=len(self.pairs), p=n_pos, n=n_neg)



    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict:
        pair = self.pairs[index]
        id_a, id_b = pair["item_1"], pair["item_2"]
        label: int = int(pair["label"])

        image_a = load_image(self.item_map[id_a], size=self.image_size, color_space="BGR")
        image_b = load_image(self.item_map[id_b], size=self.image_size, color_space="BGR")

        return {
            "image_a":  image_a,
            "image_b":  image_b,
            "label":    label,
            "item_ids": (id_a, id_b),
        }


    def get_labels(self) -> np.ndarray:
        """Return all labels as a numpy array (useful for stratified splits)."""
        return np.array([p["label"] for p in self.pairs], dtype=np.int32)

    def get_all_item_ids(self) -> List[str]:
        """Return the de-duplicated list of every item ID present in the split."""
        ids: set = set()
        for p in self.pairs:
            ids.add(p["item_1"])
            ids.add(p["item_2"])
        return sorted(ids)
