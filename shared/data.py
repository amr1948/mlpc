"""
Data loading for Strategy A and Strategy B.

Both strategies produce a splits.npz file with the same schema (plus
extras for B). This module gives notebooks one canonical entry point:

    >>> import sys; sys.path.append('../../_shared')
    >>> from data import load_splits
    >>> d = load_splits('A')
    >>> d.X_train.shape
    (100166, 960)
    >>> print(d.summary())

Conventions:
- Arrays inside SplitData are exactly the contents of splits.npz (no copies).
- SplitData is frozen: do not mutate. If you need a derived array, make
  it locally in your notebook.
- Strategy B's source_train / source_val / source_test arrays tag each
  segment as 'real' or 'aug'. For Strategy A these are None.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Path resolution — same convention as the rest of the project
# ---------------------------------------------------------------------------

def find_root() -> Path:
    """Walk up from cwd looking for the project root signature
    (config.json + data/)."""
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / 'config.json').is_file() and (p / 'data').is_dir():
            return p
    raise RuntimeError(
        "Project root not found above cwd. Expected a directory "
        "containing both 'config.json' and a 'data/' folder."
    )


def load_config() -> dict:
    """Load the project config.json from project root."""
    with open(find_root() / 'config.json') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SplitData:
    """Frozen container for the contents of a strategy's splits.npz."""

    strategy: str                              # 'A' or 'B'

    # Schema-parity arrays (present in both A and B)
    X_train:        np.ndarray                 # (N_tr, D)  float32, scaled
    X_val:          np.ndarray                 # (N_va, D)  float32, scaled
    X_test:         np.ndarray                 # (N_te, D)  float32, scaled
    Y_train:        np.ndarray                 # (N_tr, C)  int8
    Y_val:          np.ndarray                 # (N_va, C)  int8
    Y_test:         np.ndarray                 # (N_te, C)  int8
    class_names:    np.ndarray                 # (C,)       <U
    feature_names:  np.ndarray                 # (56,)      <U  feature group names
    scaler_mean:    np.ndarray                 # (D,)       float64
    scaler_scale:   np.ndarray                 # (D,)       float64
    train_idx:      np.ndarray                 # (N_tr,)    int64
    val_idx:        np.ndarray                 # (N_va,)    int64
    test_idx:       np.ndarray                 # (N_te,)    int64

    # Strategy-B-only extras. None for A.
    source_train:   Optional[np.ndarray] = None    # (N_tr,) <U4 'real'/'aug'
    source_val:     Optional[np.ndarray] = None    # (N_va,) all 'real' in B
    source_test:    Optional[np.ndarray] = None    # (N_te,) all 'real' in B
    n_real_total:   Optional[int] = None
    n_aug_total:    Optional[int] = None

    # ---- convenience properties ----
    @property
    def D(self) -> int:
        return self.X_train.shape[1]

    @property
    def C(self) -> int:
        return self.Y_train.shape[1]

    @property
    def N_train(self) -> int:
        return self.X_train.shape[0]

    @property
    def N_val(self) -> int:
        return self.X_val.shape[0]

    @property
    def N_test(self) -> int:
        return self.X_test.shape[0]

    def summary(self) -> str:
        lines = [
            f"Strategy {self.strategy}",
            f"  X_train: {self.X_train.shape}  Y_train: {self.Y_train.shape}",
            f"  X_val:   {self.X_val.shape}  Y_val:   {self.Y_val.shape}",
            f"  X_test:  {self.X_test.shape}  Y_test:  {self.Y_test.shape}",
            f"  D = {self.D} features, C = {self.C} classes",
        ]
        if self.strategy == 'B' and self.source_train is not None:
            n_aug_tr = int((self.source_train == 'aug').sum())
            n_real_tr = self.N_train - n_aug_tr
            lines.append(
                f"  train composition: {n_real_tr:,} real + {n_aug_tr:,} aug "
                f"= {self.N_train:,} ({n_aug_tr/n_real_tr:+.1%} of real)"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_splits(strategy: str) -> SplitData:
    """
    Load the prepared splits for a given strategy.

    Parameters
    ----------
    strategy : {'A', 'B'}
        'A' → strategy_A_baseline/artifacts/splits.npz  (real only)
        'B' → strategy_B_with_aug/artifacts/splits.npz  (real + aug in train)

    Returns
    -------
    SplitData
    """
    if strategy not in ('A', 'B'):
        raise ValueError(f"strategy must be 'A' or 'B', got {strategy!r}")

    cfg = load_config()
    P = cfg['paths']
    root = find_root()

    key = 'strategy_a_artifacts' if strategy == 'A' else 'strategy_b_artifacts'
    path = root / P[key] / 'splits.npz'

    if not path.is_file():
        raise FileNotFoundError(
            f"splits.npz not found for strategy {strategy}: {path}\n"
            f"Run the data preparation notebook for strategy {strategy} first."
        )

    with np.load(path, allow_pickle=True) as f:
        kwargs = dict(
            strategy=strategy,
            X_train=f['X_train'],
            X_val=f['X_val'],
            X_test=f['X_test'],
            Y_train=f['Y_train'],
            Y_val=f['Y_val'],
            Y_test=f['Y_test'],
            class_names=f['class_names'],
            feature_names=f['feature_names'],
            scaler_mean=f['scaler_mean'],
            scaler_scale=f['scaler_scale'],
            train_idx=f['train_idx'],
            val_idx=f['val_idx'],
            test_idx=f['test_idx'],
        )
        if 'source_train' in f.files:
            kwargs.update(
                source_train=f['source_train'],
                source_val=f['source_val'],
                source_test=f['source_test'],
                n_real_total=int(f['n_real_total']),
                n_aug_total=int(f['n_aug_total']),
            )

    return SplitData(**kwargs)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def unscale(X_scaled: np.ndarray,
            scaler_mean: np.ndarray,
            scaler_scale: np.ndarray) -> np.ndarray:
    """Invert StandardScaler using the saved mean/scale arrays."""
    return X_scaled * scaler_scale + scaler_mean


# ---------------------------------------------------------------------------
# PyTorch wrapper — lazy import so the lib works without torch installed
# ---------------------------------------------------------------------------

def TorchSplitsDataset(X: np.ndarray, Y: np.ndarray):
    """
    Build a minimal torch Dataset wrapping (X, Y).

    Targets are float32 (suitable for BCEWithLogitsLoss). Convert to long
    in your training loop if you need int labels.
    """
    try:
        import torch
        from torch.utils.data import Dataset
    except ImportError as e:
        raise ImportError(
            "torch is not installed. Install with `pip install torch` to use "
            "TorchSplitsDataset, or stick to numpy-based classifiers."
        ) from e

    class _SplitsDataset(Dataset):
        def __init__(self, X, Y):
            self.X = torch.as_tensor(X, dtype=torch.float32)
            self.Y = torch.as_tensor(Y, dtype=torch.float32)

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, i):
            return self.X[i], self.Y[i]

    return _SplitsDataset(X, Y)
