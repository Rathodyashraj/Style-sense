from __future__ import annotations

from typing import Union

from src.models.svm_classifier import SVMCompatibilityClassifier
from src.models.mlp_classifier import MLPCompatibilityClassifier
from src.utils.logger          import get_logger

log = get_logger(__name__)

# Union type alias for type hints downstream
CompatibilityModel = Union[SVMCompatibilityClassifier, MLPCompatibilityClassifier]


def build_model(cfg) -> CompatibilityModel:

    model_type = cfg.model.type.lower()
    log.info("Building model of type: '{t}'", t=model_type)

    if model_type == "svm":
        svm_cfg = cfg.model.svm
        return SVMCompatibilityClassifier(
            kernel=svm_cfg.kernel,
            C=float(svm_cfg.C),
            gamma=svm_cfg.gamma,
        )

    if model_type == "mlp":
        mlp_cfg = cfg.model.mlp
        device  = cfg.clip.device          
        return MLPCompatibilityClassifier(
            hidden_dims             = list(mlp_cfg.hidden_dims),
            dropout                 = float(mlp_cfg.dropout),
            learning_rate           = float(mlp_cfg.learning_rate),
            batch_size              = int(mlp_cfg.batch_size),
            epochs                  = int(mlp_cfg.epochs),
            early_stopping_patience = int(mlp_cfg.early_stopping_patience),
            weight_decay            = float(mlp_cfg.weight_decay),
            device                  = device,
        )

    raise ValueError(
        f"Unknown model type '{model_type}'. "
        "Valid options: 'svm', 'mlp'. "
        "Update configs/config.yaml → model.type."
    )
