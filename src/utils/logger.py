"""
src/utils/logger.py
───────────────────
Centralised logging setup using *loguru*.

Usage (in any module):
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Processing item {item_id}", item_id="abc123")
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


# ---------------------------------------------------------------------------
# Module-level initialisation — runs once when first imported
# ---------------------------------------------------------------------------

# Remove the default loguru sink so we control formatting entirely.
logger.remove()

# ── Human-readable console sink ───────────────────────────────────────────────
logger.add(
    sys.stderr,
    level="INFO",
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    ),
    colorize=True,
    enqueue=True,          # thread-safe async logging
)

# ── Persistent file sink (rotates at 50 MB, keeps 10 compressed files) ───────
_LOG_DIR = Path(__file__).resolve().parents[2] / "outputs" / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.add(
    _LOG_DIR / "outfit_scorer_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
    rotation="50 MB",
    retention=10,
    compression="zip",
    enqueue=True,
)


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def get_logger(name: str):
    """
    Return a loguru logger bound to *name* (typically ``__name__``).

    The returned object is the global *logger* with the module name bound,
    so all log records carry the originating module automatically.
    """
    return logger.bind(name=name)
