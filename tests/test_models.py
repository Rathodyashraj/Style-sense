
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.mlp_classifier import MLPCompatibilityClassifier
from src.models.svm_classifier import SVMCompatibilityClassifier




def _make_synthetic_dataset(
    n_samples: int = 200, n_features: int = 64, seed: int = 42
):
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y   = rng.integers(0, 2, size=n_samples).astype(np.float32)
    return X, y


class TestMLPClassifier:

    def test_fit_predict_proba(self):
        """MLP must fit without error and return probabilities in [0,1]."""
        X, y = _make_synthetic_dataset(200, 64)
        model = MLPCompatibilityClassifier(
            hidden_dims=[32, 16], dropout=0.0, epochs=3,
            batch_size=32, device="cpu",
        )
        model.fit(X, y)
        probs = model.predict_proba(X)

        assert probs.shape == (200, 2), f"Expected (200,2), got {probs.shape}"
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0), "Probabilities out of [0,1]"
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5), "Probabilities must sum to 1"

    def test_predict_binary(self):
        """Hard predictions must be 0 or 1."""
        X, y = _make_synthetic_dataset(100, 32)
        model = MLPCompatibilityClassifier(hidden_dims=[16], epochs=2, device="cpu")
        model.fit(X, y)
        preds = model.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_score_single_pair(self):
        """score_single_pair must return a float in [0,1]."""
        X, y = _make_synthetic_dataset(100, 32)
        model = MLPCompatibilityClassifier(hidden_dims=[16], epochs=2, device="cpu")
        model.fit(X, y)
        score = model.score_single_pair(X[0])
        assert 0.0 <= score <= 1.0

    def test_save_load_roundtrip(self):
        """Loaded model must produce identical predictions to original."""
        X, y = _make_synthetic_dataset(100, 32)
        model = MLPCompatibilityClassifier(hidden_dims=[16], epochs=2, device="cpu")
        model.fit(X, y)
        probs_before = model.predict_proba(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "mlp.pt"
            model.save(ckpt)

            model2 = MLPCompatibilityClassifier(device="cpu")
            model2.load(ckpt)
            probs_after = model2.predict_proba(X)

        assert np.allclose(probs_before, probs_after, atol=1e-5), \
            "Loaded model produces different predictions"

    def test_unfitted_raises(self):
        """Calling predict on an unfitted model must raise RuntimeError."""
        model = MLPCompatibilityClassifier(device="cpu")
        with pytest.raises(RuntimeError):
            model.predict(np.zeros((5, 32), dtype=np.float32))



class TestSVMClassifier:

    def test_fit_predict_proba(self):
        """SVM must fit without error and return probabilities in [0,1]."""
        X, y = _make_synthetic_dataset(200, 32)
        model = SVMCompatibilityClassifier(kernel="rbf", C=1.0, pca_n_components=16)
        model.fit(X, y.astype(np.int32))
        probs = model.predict_proba(X)

        assert probs.shape == (200, 2)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    def test_save_load_roundtrip(self):
        """SVM roundtrip via joblib must preserve predictions."""
        X, y = _make_synthetic_dataset(100, 32)
        model = SVMCompatibilityClassifier(pca_n_components=16)
        model.fit(X, y.astype(np.int32))
        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "svm.pkl"
            model.save(ckpt)
            model2 = SVMCompatibilityClassifier()
            model2.load(ckpt)
            preds_after = model2.predict(X)

        assert np.array_equal(preds_before, preds_after)

    def test_unfitted_raises(self):
        model = SVMCompatibilityClassifier()
        with pytest.raises(RuntimeError):
            model.predict(np.zeros((5, 32), dtype=np.float32))
