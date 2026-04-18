"""
src/__init__.py
───────────────
Top-level package for the Graph-Based Stylistic Compatibility and Harmony Scorer.

Sub-packages
------------
preprocessing  — GrabCut garment segmentation (Module 1)
features       — Explicit CV + CLIP latent feature extraction (Modules 2 & 3)
fusion         — Pairwise scoring and harmony rules engine (Module 4)
models         — SVM / MLP compatibility classifiers
utils          — Dataset I/O, logging, caching, configuration helpers
"""

__version__ = "1.0.0"
__author__  = "Outfit Compatibility Scorer"
