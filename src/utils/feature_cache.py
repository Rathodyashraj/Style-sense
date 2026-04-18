"""
src/utils/feature_cache.py
──────────────────────────
HDF5-backed feature cache.

Why HDF5?
---------
Feature extraction (especially CLIP) is expensive.  Rather than re-running the
encoder every training epoch, we serialise each item's feature vector to disk
exactly once.  Subsequent runs retrieve vectors in O(1) via item_id key.

The cache stores two HDF5 groups:
    /explicit/   — deterministic CV feature vectors (float32)
    /latent/     — CLIP embedding vectors            (float32)

Both groups use the item_id string as the dataset key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# FeatureCache
# ---------------------------------------------------------------------------

class FeatureCache:
    """
    Read/write interface for persisted feature vectors.

    Parameters
    ----------
    cache_dir : str | Path
        Directory where HDF5 files are stored.
    split : str
        Dataset split name (e.g. "train", "val", "test").
        Determines the HDF5 filename: ``<cache_dir>/<split>_features.h5``.
    read_only : bool
        Open in read-only mode (e.g., during inference).
    """

    _EXPLICIT_GROUP = "explicit"
    _LATENT_GROUP   = "latent"

    def __init__(
        self,
        cache_dir: str | Path,
        split: str = "train",
        read_only: bool = False,
    ) -> None:
        self._path = Path(cache_dir) / f"{split}_features.h5"
        self._mode = "r" if read_only else "a"   # 'a' = read/write, create if absent
        self._file: Optional[h5py.File] = None

    # ── Context-manager protocol ──────────────────────────────────────────────

    def __enter__(self) -> "FeatureCache":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def open(self) -> None:
        """Open (or create) the HDF5 file."""
        self._file = h5py.File(self._path, self._mode)
        # Ensure the two top-level groups always exist
        if self._mode != "r":
            for group in (self._EXPLICIT_GROUP, self._LATENT_GROUP):
                if group not in self._file:
                    self._file.create_group(group)
        log.debug("Feature cache opened: {p} (mode={m})", p=self._path, m=self._mode)

    def close(self) -> None:
        """Flush and close the HDF5 file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    # ── Write helpers ─────────────────────────────────────────────────────────

    def save_explicit(self, item_id: str, vector: np.ndarray) -> None:
        """Persist an explicit CV feature vector for *item_id*."""
        self._assert_open()
        grp = self._file[self._EXPLICIT_GROUP]
        if item_id in grp:
            del grp[item_id]           # overwrite silently
        grp.create_dataset(item_id, data=vector.astype(np.float32), compression="gzip")

    def save_latent(self, item_id: str, vector: np.ndarray) -> None:
        """Persist a CLIP latent embedding for *item_id*."""
        self._assert_open()
        grp = self._file[self._LATENT_GROUP]
        if item_id in grp:
            del grp[item_id]
        grp.create_dataset(item_id, data=vector.astype(np.float32), compression="gzip")

    def save_explicit_batch(self, batch: Dict[str, np.ndarray]) -> None:
        """Bulk-save explicit features from a ``{item_id: vector}`` dict."""
        for item_id, vector in batch.items():
            self.save_explicit(item_id, vector)

    def save_latent_batch(self, batch: Dict[str, np.ndarray]) -> None:
        """Bulk-save CLIP embeddings from a ``{item_id: vector}`` dict."""
        for item_id, vector in batch.items():
            self.save_latent(item_id, vector)

    # ── Read helpers ──────────────────────────────────────────────────────────

    def load_explicit(self, item_id: str) -> Optional[np.ndarray]:
        """
        Retrieve the explicit feature vector for *item_id*.
        Returns ``None`` if the item is not cached.
        """
        self._assert_open()
        grp = self._file[self._EXPLICIT_GROUP]
        if item_id not in grp:
            return None
        return grp[item_id][:]     # [:] materialises the HDF5 dataset as ndarray

    def load_latent(self, item_id: str) -> Optional[np.ndarray]:
        """
        Retrieve the CLIP embedding for *item_id*.
        Returns ``None`` if the item is not cached.
        """
        self._assert_open()
        grp = self._file[self._LATENT_GROUP]
        if item_id not in grp:
            return None
        return grp[item_id][:]

    def has_explicit(self, item_id: str) -> bool:
        """Check whether an explicit feature vector exists in the cache."""
        self._assert_open()
        return item_id in self._file[self._EXPLICIT_GROUP]

    def has_latent(self, item_id: str) -> bool:
        """Check whether a CLIP embedding exists in the cache."""
        self._assert_open()
        return item_id in self._file[self._LATENT_GROUP]

    def cached_explicit_ids(self) -> list:
        """Return a list of all item IDs with cached explicit features."""
        self._assert_open()
        return list(self._file[self._EXPLICIT_GROUP].keys())

    def cached_latent_ids(self) -> list:
        """Return a list of all item IDs with cached CLIP embeddings."""
        self._assert_open()
        return list(self._file[self._LATENT_GROUP].keys())

    # ── Internal ─────────────────────────────────────────────────────────────

    def _assert_open(self) -> None:
        if self._file is None:
            raise RuntimeError(
                "FeatureCache is not open. "
                "Use it as a context manager: `with FeatureCache(...) as cache:`"
            )

    def __repr__(self) -> str:
        return f"FeatureCache(path={self._path}, mode={self._mode})"
