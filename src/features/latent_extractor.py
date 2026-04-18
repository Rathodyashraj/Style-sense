"""
src/features/latent_extractor.py
──────────────────────────────────
Module 3 — Latent Feature Extraction via CLIP / Fashion-CLIP

Overview
--------
We use the image encoder of a pre-trained vision-language model to obtain
a dense embedding vector that captures the *semantic vibe* of a garment —
style, formality, occasion, aesthetic — which deterministic CV features
cannot easily encode.

Model choice
------------
*  ``patrickjohncyh/fashion-clip``  (default)
   A CLIP model fine-tuned on ~800k product images + captions from the
   Farfetch fashion e-commerce platform.

*  ``openai/clip-vit-base-patch32``
   The original OpenAI CLIP ViT-B/32.

Both are loaded via the ``transformers`` library (HuggingFace hub).

Compatibility note — 'BaseModelOutputWithPooling' has no attribute 'norm'
--------------------------------------------------------------------------
``patrickjohncyh/fashion-clip`` stores its vision encoder weights in a way
that is incompatible with the ``get_image_features()`` helper in newer
versions of ``transformers``.  Specifically, the helper tries to call a
post-layernorm (``.norm``) on the pooler output, but this checkpoint does
not expose that attribute.

The solution is to bypass ``get_image_features()`` entirely and call the
vision encoder + projection head directly.  We always use this manual path
for ``fashion-clip`` (detected by model name prefix), and fall through to
``get_image_features()`` only for standard ``openai/clip-*`` checkpoints.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.utils.logger import get_logger

log = get_logger(__name__)

# Models that are known to break get_image_features() — always use manual path
_BROKEN_MODELS = {"patrickjohncyh/fashion-clip"}


# ---------------------------------------------------------------------------
# LatentFeatureExtractor
# ---------------------------------------------------------------------------

class LatentFeatureExtractor:
    """
    Wraps a CLIP (or Fashion-CLIP) image encoder for feature extraction.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    device     : str
        "cuda" or "cpu".  Falls back to CPU automatically if CUDA is absent.
    batch_size : int
        Number of images processed per forward pass.
    """

    def __init__(
        self,
        model_name: str = "patrickjohncyh/fashion-clip",
        device:     str = "cuda",
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

        # ── Resolve device ─────────────────────────────────────────────────────
        if device == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        self.device = torch.device(device)

        # ── Load model + processor ────────────────────────────────────────────
        log.info("Loading CLIP model: {m}", m=model_name)
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._model     = CLIPModel.from_pretrained(model_name).to(self.device)
        self._model.eval()

        # ── Choose extraction strategy ────────────────────────────────────────
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

    # ── Public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Compute the L2-normalised CLIP embedding for a single BGR image.

        Parameters
        ----------
        image : np.ndarray  shape (H, W, 3) BGR uint8

        Returns
        -------
        np.ndarray  shape (embedding_dim,)  float32
        """
        return self.extract_pil([self._bgr_to_pil(image)])[0]

    @torch.no_grad()
    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Compute L2-normalised CLIP embeddings for a list of BGR images.

        Parameters
        ----------
        images : list of np.ndarray  each (H, W, 3) BGR uint8

        Returns
        -------
        np.ndarray  shape (N, embedding_dim)  float32
        """
        return self.extract_pil([self._bgr_to_pil(img) for img in images])

    @torch.no_grad()
    def extract_pil(self, pil_images: List[Image.Image]) -> np.ndarray:
        """
        Compute L2-normalised CLIP embeddings from pre-loaded PIL images.

        Parameters
        ----------
        pil_images : list of PIL.Image.Image  (RGB)

        Returns
        -------
        np.ndarray  shape (N, embedding_dim)  float32
        """
        all_embeddings: List[np.ndarray] = []

        for start in range(0, len(pil_images), self.batch_size):
            batch = pil_images[start : start + self.batch_size]

            # CLIPProcessor: resize → centre-crop → normalise → tensor
            inputs = self._processor(
                images=batch,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            # ── Extract features via the correct path ─────────────────────────
            if self._use_manual:
                feats = self._manual_image_features(inputs["pixel_values"])
            else:
                feats = self._model.get_image_features(**inputs)

            # L2-normalise: cosine similarity = dot product on unit vectors
            feats = F.normalize(feats, dim=-1)
            all_embeddings.append(feats.cpu().float().numpy())

        return np.concatenate(all_embeddings, axis=0)   # (N, D)

    # ── Private: manual vision-encoder + projection path ─────────────────────

    def _manual_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract image features by calling the vision encoder and projection
        head directly, bypassing the broken ``get_image_features()`` helper.

        This avoids the ``'BaseModelOutputWithPooling' has no attribute 'norm'``
        error that occurs with patrickjohncyh/fashion-clip on newer transformers.

        Steps
        -----
        1. Run pixel_values through the vision encoder → pooled CLS token.
        2. Apply the visual projection Linear layer → shared embedding space.
        3. Return raw projected tensor (L2-norm applied by caller).
        """
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
        """
        Run a real-size 224×224 probe through ``get_image_features()`` to
        detect whether it raises AttributeError for this model/transformers
        version combination.

        Using a proper 224×224 image (not 3×3) is critical — small images
        can accidentally avoid the broken code path.

        Returns True if the manual path is required.
        """
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

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _bgr_to_pil(bgr_image: np.ndarray) -> Image.Image:
        """Convert an OpenCV BGR uint8 array to a PIL RGB Image."""
        rgb = bgr_image[:, :, ::-1].copy()
        return Image.fromarray(rgb.astype(np.uint8))
