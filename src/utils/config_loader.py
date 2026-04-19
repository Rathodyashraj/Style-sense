from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class _DotDict(dict):
    """A dict subclass whose values are accessible as attributes (recursive)."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'") from None
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key) from None


def _dict_to_dotdict(mapping: dict) -> _DotDict:
    """Recursively convert nested plain dicts to _DotDict."""
    result = _DotDict()
    for key, value in mapping.items():
        if isinstance(value, dict):
            result[key] = _dict_to_dotdict(value)
        else:
            result[key] = value
    return result

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def load_config(config_path: str | Path | None = None) -> _DotDict:

    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Please copy configs/config.yaml and adjust your dataset paths."
        )

    with open(path, "r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh)

    config = _dict_to_dotdict(raw)

    for sub_path in (
        config.paths.feature_cache_dir,
        config.paths.checkpoint_dir,
        config.paths.results_dir,
    ):
        os.makedirs(sub_path, exist_ok=True)

    return config
