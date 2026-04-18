"""
src/models/__init__.py
───────────────────────
Compatibility classifier sub-package.

Public exports
--------------
SVMCompatibilityClassifier  — sklearn Pipeline (StandardScaler → PCA → SVC).
MLPCompatibilityClassifier  — PyTorch MLP with BatchNorm, Dropout, early stopping.
build_model                 — factory function: returns the correct classifier
                              based on ``cfg.model.type``.
CompatibilityModel          — Union type alias for type hints.
"""

from src.models.svm_classifier import SVMCompatibilityClassifier
from src.models.mlp_classifier import MLPCompatibilityClassifier
from src.models.model_factory  import build_model, CompatibilityModel

__all__ = [
    "SVMCompatibilityClassifier",
    "MLPCompatibilityClassifier",
    "build_model",
    "CompatibilityModel",
]
