

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import joblib

from src.preprocessing.segmenter      import GrabCutSegmenter
from src.features.color_harmony       import ColorHarmonyExtractor
from src.features.texture_analyzer    import TextureAnalyzer
from src.features.shape_descriptor    import ShapeDescriptor
from src.utils.logger                 import get_logger

log = get_logger(__name__)


# ExplicitFeatureExtractor

class ExplicitFeatureExtractor:

    def __init__(self, cfg, scaler_path: Optional[str | Path] = None) -> None:
        seg_cfg = cfg.segmentation
        col_cfg = cfg.color_harmony
        tex_cfg = cfg.texture
        shp_cfg = cfg.shape

        # ── Sub-module instantiation ──────────────────────────────────────────
        self.segmenter = GrabCutSegmenter(
            grabcut_iterations   = seg_cfg.grabcut_iterations,
            border_margin        = seg_cfg.border_margin,
            min_foreground_ratio = seg_cfg.min_foreground_ratio,
        )
        self.color_extractor = ColorHarmonyExtractor(
            n_dominant_colors = col_cfg.n_dominant_colors,
            kmeans_max_iter   = col_cfg.kmeans_max_iter,
            kmeans_n_init     = col_cfg.kmeans_n_init,
            max_pixel_sample  = col_cfg.max_pixel_sample,
        )
        self.texture_extractor = TextureAnalyzer(
            orientations          = list(tex_cfg.orientations),
            wavelengths           = list(tex_cfg.wavelengths),
            sigma_to_lambda_ratio = tex_cfg.sigma_to_lambda_ratio,
            gabor_aspect_ratio    = tex_cfg.gabor_aspect_ratio,
        )
        self.shape_extractor = ShapeDescriptor(
            hog_image_size  = tuple(shp_cfg.hog_image_size),
            pixels_per_cell = tuple(shp_cfg.hog_pixels_per_cell),
            cells_per_block = tuple(shp_cfg.hog_cells_per_block),
            orientations    = shp_cfg.hog_orientations,
        )


        self.scaler = None
        if scaler_path is not None:
            scaler_path = Path(scaler_path)
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                log.info("Loaded StandardScaler from {p}", p=scaler_path)
            else:
                log.warning(
                    "Scaler path {p} not found. Running without normalisation.",
                    p=scaler_path,
                )

    # Public API

    def extract(self, image: np.ndarray) -> np.ndarray:

        # Step 1 : GrabCut segmentation
        segmented, _mask = self.segmenter.segment(image)

        # Step 2a : Colour palette
        color_vec = self.color_extractor.extract(segmented)
        log.debug("color_vec  dim=%d", color_vec.shape[0])

        # Step 2b : Texture signature
        texture_vec = self.texture_extractor.extract(segmented)
        log.debug("texture_vec dim=%d", texture_vec.shape[0])

        # Step 2c : Shape / HOG
        shape_vec = self.shape_extractor.extract(segmented)
        log.debug("shape_vec   dim=%d", shape_vec.shape[0])

        # Concatenate into single explicit vector
        explicit = np.concatenate([color_vec, texture_vec, shape_vec]).astype(np.float32)

        if self.scaler is not None:
            explicit = self.scaler.transform(explicit.reshape(1, -1)).flatten()
            explicit = explicit.astype(np.float32)

        return explicit

    def get_sub_vector_lengths(self) -> dict:

        n_colors  = self.color_extractor.n_dominant_colors
        color_dim = n_colors * 3

        n_orientations = len(self.texture_extractor.orientations)
        n_wavelengths  = len(self.texture_extractor.wavelengths)
        texture_dim    = n_orientations * n_wavelengths * 2

        shape_dim = self.shape_extractor.descriptor_length

        return {
            "color":   color_dim,
            "texture": texture_dim,
            "shape":   shape_dim,
            "total":   color_dim + texture_dim + shape_dim,
        }
