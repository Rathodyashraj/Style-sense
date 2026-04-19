from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.utils.logger import get_logger

log = get_logger(__name__)

_BROKEN_MODELS = {"patrickjohncyh/fashion-clip"}


# LatentFeatureExtractor
class LatentFeatureExtractor:


    def __init__(
        self,
        model_name: str = "patrickjohncyh/fashion-clip",
        device:     str = "cuda",
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

        # Resolve device
        if device == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        self.device = torch.device(device)

        # Load model + processor
        log.info("Loading CLIP model: {m}", m=model_name)
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._model     = CLIPModel.from_pretrained(model_name).to(self.device)
        self._model.eval()

        # Choose extraction strategy
        # For known-broken checkpoints use the manual vision-encoder path always.
        # For all other models, try get_image_features() and fall back if needed.
        self._use_manual = (model_name in _BROKEN_MODELS) or self._probe_needs_manual()

        # Infer embedding dimensionality from the model config
        self.embedding_dim: int = self._model.config.projection_dim
        log.info(
            "CLIP model ready | device={d} | embedding_dim={e} | manual_path={m}",
            d=self.device,
            e=self.embedding_dim,
            m=self._use_manual,
        )

    # Public API

    @torch.no_grad()
    def extract(self, image: np.ndarray) -> np.ndarray:

        return self.extract_pil([self._bgr_to_pil(image)])[0]

    @torch.no_grad()
    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:

        return self.extract_pil([self._bgr_to_pil(img) for img in images])

    @torch.no_grad()
    def extract_pil(self, pil_images: List[Image.Image]) -> np.ndarray:

        all_embeddings: List[np.ndarray] = []

        for start in range(0, len(pil_images), self.batch_size):
            batch = pil_images[start : start + self.batch_size]

            # CLIPProcessor: resize → centre-crop → normalise → tensor
            inputs = self._processor(
                images=batch,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            # Extract features
            if self._use_manual:
                feats = self._manual_image_features(inputs["pixel_values"])
            else:
                feats = self._model.get_image_features(**inputs)

            # L2-normalise
            feats = F.normalize(feats, dim=-1)
            all_embeddings.append(feats.cpu().float().numpy())

        return np.concatenate(all_embeddings, axis=0)   # (N, D)

    # Private: manual vision-encoder + projection path 

    def _manual_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:

        # Step 1: vision encoder forward pass
        vision_out = self._model.vision_model(
            pixel_values=pixel_values,
            return_dict=True,
        )

        # Use pooler_output (post-pool CLS) if available; else fall back to
        # the raw CLS token from last_hidden_state[:, 0, :].
        if (
            hasattr(vision_out, "pooler_output")
            and vision_out.pooler_output is not None
        ):
            pooled = vision_out.pooler_output          # (B, hidden_size)
        else:
            pooled = vision_out.last_hidden_state[:, 0, :]

        # Step 2: project into shared CLIP embedding space
        projected = self._model.visual_projection(pooled)  # (B, projection_dim)
        return projected

    def _probe_needs_manual(self) -> bool:

        import warnings

        # Use a properly-sized probe — same size the processor will produce
        probe_pil = Image.fromarray(
            np.full((224, 224, 3), 128, dtype=np.uint8), mode="RGB"
        )
        probe_inputs = self._processor(
            images=[probe_pil], return_tensors="pt"
        ).to(self.device)

        try:
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _ = self._model.get_image_features(**probe_inputs)
            return False   # get_image_features() works fine

        except (AttributeError, Exception) as exc:
            log.warning(
                "get_image_features() failed for '{m}' ({e}). "
                "Using manual vision-encoder path for all batches.",
                m=self.model_name,
                e=str(exc),
            )
            return True

    # Static helpers

    @staticmethod
    def _bgr_to_pil(bgr_image: np.ndarray) -> Image.Image:
        """Convert an OpenCV BGR uint8 array to a PIL RGB Image."""
        rgb = bgr_image[:, :, ::-1].copy()
        return Image.fromarray(rgb.astype(np.uint8))
