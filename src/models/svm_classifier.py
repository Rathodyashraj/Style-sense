from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.utils.logger import get_logger

log = get_logger(__name__)


# SVMCompatibilityClassifier

class SVMCompatibilityClassifier:

    def __init__(
        self,
        kernel:           str   = "rbf",
        C:                float = 1.0,
        gamma:            str   = "scale",
        pca_n_components: Optional[int] = 256,
    ) -> None:
        self.kernel           = kernel
        self.C                = C
        self.gamma            = gamma
        self.pca_n_components = pca_n_components

        self._pipeline: Optional[Pipeline] = None
        self._is_fitted: bool              = False

        log.info(
            "SVMCompatibilityClassifier created | kernel=%s C=%.2f gamma=%s pca=%s",
            kernel, C, gamma, pca_n_components,
        )

    # Training

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   Optional[np.ndarray] = None,
        y_val:   Optional[np.ndarray] = None,
    ) -> "SVMCompatibilityClassifier":

        steps = [("scaler", StandardScaler())]

        if self.pca_n_components is not None:
            # Clamp PCA components to the feature dimensionality
            n_comp = min(self.pca_n_components, X_train.shape[1])
            steps.append(("pca", PCA(n_components=n_comp, random_state=42)))

        steps.append((
            "svm",
            SVC(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma,
                probability=True,      # enables predict_proba via Platt scaling
                class_weight="balanced",
                random_state=42,
            ),
        ))

        self._pipeline = Pipeline(steps)

        log.info("Training SVM on %d samples, %d features …", *X_train.shape)
        self._pipeline.fit(X_train, y_train)
        self._is_fitted = True
        log.info("SVM training complete.")

        # Validation report
        train_acc = (self._pipeline.predict(X_train) == y_train).mean()
        log.info("Train accuracy: {:.4f}", train_acc)

        if X_val is not None and y_val is not None:
            val_acc = (self._pipeline.predict(X_val) == y_val).mean()
            log.info("Validation accuracy: {:.4f}", val_acc)

        return self

    # Inference

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        self._assert_fitted()
        return self._pipeline.predict_proba(X).astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard labels (0 or 1) for each sample."""
        self._assert_fitted()
        return self._pipeline.predict(X)

    def score_single_pair(self, pairwise_feature: np.ndarray) -> float:

        self._assert_fitted()
        x = pairwise_feature.reshape(1, -1)
        prob_compatible = float(self._pipeline.predict_proba(x)[0, 1])
        return prob_compatible

    # Persistence

    def save(self, path: str | Path) -> None:
        """Serialise the fitted pipeline to disk using joblib."""
        self._assert_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path)
        log.info("SVM pipeline saved to {p}", p=path)

    def load(self, path: str | Path) -> "SVMCompatibilityClassifier":
        """Load a previously saved pipeline from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"SVM checkpoint not found: {path}")
        self._pipeline = joblib.load(path)
        self._is_fitted = True
        log.info("SVM pipeline loaded from {p}", p=path)
        return self

    # Internal

    def _assert_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "The classifier has not been fitted yet. "
                "Call .fit() or .load() before inference."
            )
