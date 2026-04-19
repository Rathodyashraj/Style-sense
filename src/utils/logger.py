
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger




# Remove the default loguru sink so we control formatting entirely.
logger.remove()

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



def get_logger(name: str):
    return logger.bind(name=name)
