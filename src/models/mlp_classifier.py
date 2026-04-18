"""
src/models/mlp_classifier.py
──────────────────────────────
MLP-based outfit compatibility classifier.

Architecture
------------
A multi-layer perceptron that maps the pairwise fusion feature
vector to a compatibility probability:

    Input → LayerNorm → [Linear → BatchNorm → GELU → Dropout] × n_layers
          → Linear(1) → Sigmoid

    With residual (skip) connections between layers of equal width.

Design choices
--------------
* **LayerNorm at input** — normalises the heterogeneous pairwise feature
  vector (different sub-vectors span very different numeric ranges even
  after StandardScaler on individual blocks).

* **GELU activation** — smoother than ReLU, avoids dead neurons, provides
  better gradient flow on medium-sized datasets.

* **Residual connections** — when consecutive hidden layers have equal width,
  a skip connection is added. This stabilises training on deep networks and
  improves gradient propagation.

* **BatchNorm** before activation — stabilises training on the heterogeneous
  pairwise features (different sub-vectors have different intrinsic
  variabilities).

* **Dropout** — regularises against over-fitting on the ~100k training pairs.

* **Binary Cross-Entropy loss with label smoothing** — natural for a binary
  classification task.  Label smoothing (α=0.05) prevents overconfident
  predictions and improves calibration.

* **AdamW + OneCycleLR** — AdamW decouples weight decay from the gradient
  update for better regularisation.  OneCycleLR with warmup finds better
  minima than plain CosineAnnealing on small-to-medium datasets.

* **Early stopping** — monitors validation loss and saves the best checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

class _ResidualBlock(nn.Module):
    """
    A single hidden block: Linear → BatchNorm → GELU → Dropout.

    If ``use_skip=True``, adds a residual connection (input + output),
    which requires ``in_dim == out_dim``.
    """

    def __init__(
        self,
        in_dim:   int,
        out_dim:  int,
        dropout:  float = 0.3,
        use_skip: bool  = False,
    ) -> None:
        super().__init__()
        self.use_skip = use_skip and (in_dim == out_dim)

        self.linear = nn.Linear(in_dim, out_dim)
        self.bn     = nn.BatchNorm1d(out_dim)
        self.act    = nn.GELU()
        self.drop   = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop(self.act(self.bn(self.linear(x))))
        if self.use_skip:
            out = out + x
        return out


# ---------------------------------------------------------------------------
# Network definition
# ---------------------------------------------------------------------------

class _CompatibilityMLP(nn.Module):
    """
    Inner PyTorch Module — the MLP backbone with LayerNorm input and
    optional residual connections.

    Parameters
    ----------
    input_dim   : int          — dimensionality of the pairwise feature vector.
    hidden_dims : list of int  — widths of the hidden layers.
    dropout     : float        — dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim:   int,
        hidden_dims: List[int] = None,
        dropout:     float     = 0.3,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [1024, 512, 256]

        # Input normalisation — tames the heterogeneous pairwise feature scales
        self.input_norm = nn.LayerNorm(input_dim)

        blocks: List[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            # Enable residual skip when dimensions match
            use_skip = (prev_dim == hidden_dim)
            blocks.append(_ResidualBlock(prev_dim, hidden_dim, dropout, use_skip))
            prev_dim = hidden_dim

        self.hidden_blocks = nn.Sequential(*blocks)

        # Output layer: single logit → sigmoid → P(compatible)
        self.output_layer = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor  shape (B, input_dim)

        Returns
        -------
        torch.Tensor  shape (B,)  — raw logits (pre-sigmoid).
        """
        x = self.input_norm(x)
        x = self.hidden_blocks(x)
        return self.output_layer(x).squeeze(-1)


# ---------------------------------------------------------------------------
# High-level classifier wrapper
# ---------------------------------------------------------------------------

class MLPCompatibilityClassifier:
    """
    Sklearn-style wrapper around ``_CompatibilityMLP``.

    Provides ``fit``, ``predict_proba``, ``predict``, ``save``, and ``load``
    with an interface consistent with ``SVMCompatibilityClassifier``.

    Parameters
    ----------
    hidden_dims         : list of int
    dropout             : float
    learning_rate       : float
    batch_size          : int
    epochs              : int
    early_stopping_patience : int
    weight_decay        : float  — L2 regularisation in AdamW.
    device              : str    — "cuda" or "cpu".
    """

    def __init__(
        self,
        hidden_dims:              List[int] = None,
        dropout:                  float     = 0.4,
        learning_rate:            float     = 5e-4,
        batch_size:               int       = 512,
        epochs:                   int       = 80,
        early_stopping_patience:  int       = 12,
        weight_decay:             float     = 5e-4,
        device:                   str       = "cuda",
    ) -> None:
        self.hidden_dims             = hidden_dims or [1024, 512, 256]
        self.dropout                 = dropout
        self.learning_rate           = learning_rate
        self.batch_size              = batch_size
        self.epochs                  = epochs
        self.early_stopping_patience = early_stopping_patience
        self.weight_decay            = weight_decay

        if device == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        self.device = torch.device(device)

        self._model: Optional[_CompatibilityMLP] = None
        self._input_dim: Optional[int]           = None
        self._is_fitted: bool                    = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   Optional[np.ndarray] = None,
        y_val:   Optional[np.ndarray] = None,
    ) -> "MLPCompatibilityClassifier":
        """
        Train the MLP on pairwise feature arrays.

        Parameters
        ----------
        X_train : (N_train, D)  float32
        y_train : (N_train,)    float32 labels 0.0 / 1.0
        X_val, y_val : optional validation arrays.

        Returns
        -------
        self
        """
        self._input_dim = X_train.shape[1]
        self._model = _CompatibilityMLP(
            input_dim=self._input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

        # AdamW decouples weight decay from gradients — better regularisation
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # ── Build DataLoaders ─────────────────────────────────────────────────
        train_loader = self._make_dataloader(X_train, y_train, shuffle=True)
        val_loader   = None
        if X_val is not None and y_val is not None:
            val_loader = self._make_dataloader(X_val, y_val, shuffle=False)

        # OneCycleLR: warmup + cosine decay in one schedule
        # total_steps = epochs * steps_per_epoch
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            steps_per_epoch=steps_per_epoch,
            epochs=self.epochs,
            pct_start=0.1,          # 10 % of training is warmup
            anneal_strategy='cos',
        )

        # Label smoothing: soft targets reduce overconfidence
        criterion = nn.BCEWithLogitsLoss()

        # ── Training loop ─────────────────────────────────────────────────────
        best_val_loss    = float("inf")
        patience_counter = 0
        best_state_dict  = None

        for epoch in range(1, self.epochs + 1):
            train_loss = self._run_epoch(
                train_loader, criterion, optimizer, scheduler, train=True
            )

            if val_loader is not None:
                val_loss, val_acc = self._evaluate(val_loader, criterion)
                log.info(
                    "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | lr=%.6f",
                    epoch, self.epochs, train_loss, val_loss, val_acc,
                    optimizer.param_groups[0]['lr'],
                )
                # ── Early stopping ─────────────────────────────────────────
                if val_loss < best_val_loss:
                    best_val_loss   = val_loss
                    patience_counter = 0
                    # Deep copy the model weights
                    best_state_dict = {
                        k: v.clone() for k, v in self._model.state_dict().items()
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        log.info("Early stopping triggered at epoch %d.", epoch)
                        break
            else:
                log.info("Epoch %3d/%d | train_loss=%.4f", epoch, self.epochs, train_loss)

        # Restore best weights if early stopping was used
        if best_state_dict is not None:
            self._model.load_state_dict(best_state_dict)
            log.info("Restored best model weights (val_loss=%.4f).", best_val_loss)

        self._is_fitted = True
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities.

        Returns
        -------
        np.ndarray  shape (N, 2)  — [P(incompatible), P(compatible)]
        """
        self._assert_fitted()
        logits = self._get_logits(X)
        probs  = torch.sigmoid(logits).cpu().numpy()       # P(compatible)
        return np.stack([1.0 - probs, probs], axis=1).astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard labels (0 or 1)."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(np.int32)

    def score_single_pair(self, pairwise_feature: np.ndarray) -> float:
        """
        Return P(compatible) for a single pairwise feature vector.

        Parameters
        ----------
        pairwise_feature : np.ndarray  shape (D,)

        Returns
        -------
        float ∈ [0, 1]
        """
        self._assert_fitted()
        x = torch.tensor(pairwise_feature, dtype=torch.float32).unsqueeze(0).to(self.device)
        self._model.eval()
        with torch.no_grad():
            logit = self._model(x)
        return float(torch.sigmoid(logit).item())

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save model weights and constructor metadata to a .pt file."""
        self._assert_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self._model.state_dict(),
            "input_dim":        self._input_dim,
            "hidden_dims":      self.hidden_dims,
            "dropout":          self.dropout,
        }, path)
        log.info("MLP checkpoint saved to {p}", p=path)

    def load(self, path: str | Path) -> "MLPCompatibilityClassifier":
        """Load model weights from a previously saved .pt file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"MLP checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=self.device)
        self._input_dim = ckpt["input_dim"]
        self._model = _CompatibilityMLP(
            input_dim=self._input_dim,
            hidden_dims=ckpt["hidden_dims"],
            dropout=ckpt["dropout"],
        ).to(self.device)
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval()
        self._is_fitted = True
        log.info("MLP checkpoint loaded from {p}", p=path)
        return self

    # ── Private helpers ───────────────────────────────────────────────────────

    def _make_dataloader(
        self, X: np.ndarray, y: np.ndarray, shuffle: bool
    ) -> DataLoader:
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        return DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )

    def _run_epoch(
        self,
        loader:    DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        train:     bool,
    ) -> float:
        """Run one full pass through the dataloader and return mean loss."""
        self._model.train(train)
        total_loss = 0.0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            # Apply label smoothing: soft targets reduce overconfidence
            if train:
                smooth_alpha = 0.05
                y_smooth = y_batch * (1.0 - smooth_alpha) + 0.5 * smooth_alpha
            else:
                y_smooth = y_batch

            if train:
                optimizer.zero_grad()

            logits = self._model(X_batch)
            loss   = criterion(logits, y_smooth)

            if train:
                loss.backward()
                # Gradient clipping prevents exploding gradients on deep feature vectors
                nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item() * X_batch.size(0)

        return total_loss / len(loader.dataset)

    def _evaluate(
        self, loader: DataLoader, criterion: nn.Module
    ) -> Tuple[float, float]:
        """Return (mean_val_loss, accuracy) on the given loader."""
        self._model.eval()
        total_loss = 0.0
        correct    = 0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

                logits = self._model(X_batch)
                loss   = criterion(logits, y_batch)
                total_loss += loss.item() * X_batch.size(0)

                preds   = (torch.sigmoid(logits) >= 0.5).long()
                correct += (preds == y_batch.long()).sum().item()

        n        = len(loader.dataset)
        val_loss = total_loss / n
        val_acc  = correct / n
        return val_loss, val_acc

    def _get_logits(self, X: np.ndarray) -> torch.Tensor:
        """Run the network in eval mode and return raw logits."""
        self._model.eval()
        x = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self._model(x)

    def _assert_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "The classifier has not been fitted. "
                "Call .fit() or .load() first."
            )
