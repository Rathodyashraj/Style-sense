# tests/conftest.py
# ─────────────────
# Shared pytest fixtures and configuration.
# Kept minimal — each test module imports only what it needs.

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the project root is on sys.path for all test modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def synthetic_bgr_image():
    """
    A 128×128 BGR test image: coloured foreground square on black background.
    Reused across multiple test modules via this session-scoped fixture.
    """
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[16:112, 16:112, 0] = 180   # Blue channel
    img[16:112, 16:112, 1] = 90    # Green channel
    img[16:112, 16:112, 2] = 45    # Red channel
    return img


@pytest.fixture(scope="session")
def random_explicit_vector():
    rng = np.random.default_rng(0)
    return rng.random(2963).astype(np.float32)  # typical explicit dim


@pytest.fixture(scope="session")
def random_latent_vector():
    rng = np.random.default_rng(0)
    v   = rng.random(512).astype(np.float32)
    return v / np.linalg.norm(v)
