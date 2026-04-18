"""
src/features/__init__.py
─────────────────────────
Feature extraction sub-package (Modules 2 & 3).

Public exports
--------------
ColorHarmonyExtractor   — dominant colour palette in CIE L*a*b* (Module 2a).
TextureAnalyzer         — Gabor filter bank texture signature   (Module 2b).
ShapeDescriptor         — HOG silhouette descriptor             (Module 2c).
ExplicitFeatureExtractor— wires all three CV extractors together (Module 2).
LatentFeatureExtractor  — CLIP / Fashion-CLIP embeddings        (Module 3).
"""

from src.features.color_harmony       import ColorHarmonyExtractor
from src.features.texture_analyzer    import TextureAnalyzer
from src.features.shape_descriptor    import ShapeDescriptor
from src.features.explicit_extractor  import ExplicitFeatureExtractor
from src.features.latent_extractor    import LatentFeatureExtractor

__all__ = [
    "ColorHarmonyExtractor",
    "TextureAnalyzer",
    "ShapeDescriptor",
    "ExplicitFeatureExtractor",
    "LatentFeatureExtractor",
]
