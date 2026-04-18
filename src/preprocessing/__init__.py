"""
src/preprocessing/__init__.py
──────────────────────────────
Garment segmentation sub-package (Module 1).

Public exports
--------------
GrabCutSegmenter — isolates garment foreground using GrabCut + GMM.
"""

from src.preprocessing.segmenter import GrabCutSegmenter

__all__ = ["GrabCutSegmenter"]
