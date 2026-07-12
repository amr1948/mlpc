"""
Imbalance interventions: class weights, sample weights, focal loss.

The MLPC 2026 SED task has 47x imbalance between the most common
(footsteps, ~12% positive rate) and rarest (light_switch, ~0.27%) classes.
No single intervention is universally best — they compose. Useful
starting combinations:

  (1) class_weight='balanced' (sklearn) or pos_weight=...
      (BCEWithLogitsLoss). Quick lift, almost free.

  (2) WeightedRandomSampler with sample_weights.
      For NN training. Oversamples rare-class-positive segments at
      training time without altering label semantics.

  (3) Focal Loss (gamma=2).
      Down-weights easy negatives. Particularly useful for the rarest
      classes where most negatives are easy.

  (4) Per-class threshold tuning on val. See eval.py.
      Often the single highest-impact intervention. Costs nothing at
      training time.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Class weights — for loss weighting (sklearn class_weight, torch pos_weight)
# ---------------------------------------------------------------------------

def compute_class_weights(
    Y_train: np.ndarray,
    kind: str = 'balanced',
    clip: tuple[float, float] | None = (0.5, 30.0),
) -> np.ndarray:
    """
    Compute per-class positive-class weights from training labels.

    Parameters
    ----------
    Y_train : (N, C) int multi-label matrix
    kind : str
        'balanced'          n_neg / n_pos              (sklearn-style)
        'inverse_freq'      1 / pos_rate               (same as 'balanced' mod constant)
        'sqrt_inverse_freq' 1 / sqrt(pos_rate)         (gentler — good default)
        'none'              all ones
    clip : (min, max) or None
        Clamp weights to avoid extremes. light_switch at 0.27% positive
        rate would otherwise get a 'balanced' weight of ~370, which can
        destabilize training. Default (0.5, 30.0) is a reasonable range.
        Set to None to disable.

    Returns
    -------
    weights : (C,) float32
        sklearn:  use to construct class_weight dict (per-class)
        torch:    pass directly to BCEWithLogitsLoss(pos_weight=...)
    """
    C = Y_train.shape[1]
    pos = Y_train.sum(axis=0).astype(np.float64)
    neg = Y_train.shape[0] - pos

    if kind == 'balanced':
        w = neg / np.maximum(pos, 1)
    elif kind == 'inverse_freq':
        freq = pos / Y_train.shape[0]
        w = 1.0 / np.maximum(freq, 1e-6)
    elif kind == 'sqrt_inverse_freq':
        freq = pos / Y_train.shape[0]
        w = 1.0 / np.sqrt(np.maximum(freq, 1e-6))
    elif kind == 'none':
        w = np.ones(C)
    else:
        raise ValueError(
            f"Unknown kind {kind!r}; expected one of "
            f"'balanced', 'inverse_freq', 'sqrt_inverse_freq', 'none'"
        )

    if clip is not None:
        w = np.clip(w, clip[0], clip[1])

    return w.astype(np.float32)


# ---------------------------------------------------------------------------
# Sample weights — for WeightedRandomSampler / minibatching
# ---------------------------------------------------------------------------

def compute_sample_weights(
    Y_train: np.ndarray,
    class_weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Per-segment sample weight for torch.utils.data.WeightedRandomSampler.

    Multi-label aware: each segment can have multiple positive classes.
    We weight each segment by the maximum class weight among its active
    classes. Silent segments (no active class) get the minimum class
    weight rather than zero (so background segments aren't excluded).

    Parameters
    ----------
    Y_train : (N, C) int
    class_weights : (C,) float or None
        If None, derived from Y_train with kind='sqrt_inverse_freq'.

    Returns
    -------
    weights : (N,) float32  — pass to WeightedRandomSampler(weights=...)
    """
    if class_weights is None:
        class_weights = compute_class_weights(Y_train, kind='sqrt_inverse_freq')

    weighted = Y_train * class_weights[None, :]            # (N, C)
    max_per_segment = weighted.max(axis=1)                 # (N,)

    silent = (Y_train.sum(axis=1) == 0)
    min_w = class_weights.min()
    max_per_segment = np.where(silent, min_w, max_per_segment)

    return max_per_segment.astype(np.float32)


# ---------------------------------------------------------------------------
# Focal loss — torch only
# ---------------------------------------------------------------------------

def make_focal_loss(
    gamma: float = 2.0,
    pos_weight: np.ndarray | None = None,
):
    """
    Returns a torch nn.Module implementing multi-label focal loss.

        FL(p, y) = -[ α y (1-p)^γ log p  +  (1-α)(1-y) p^γ log(1-p) ]

    γ = 0 reduces to weighted BCE. γ = 2 (default) down-weights
    examples where the prediction is already confident-correct,
    focusing gradient on hard examples.

    Parameters
    ----------
    gamma : float
        Focusing parameter. Higher values down-weight easy examples more.
    pos_weight : (C,) array or None
        Per-class positive weight (α). Compute via compute_class_weights()
        if rare classes need explicit boosting in addition to focal.
    """
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
    except ImportError as e:
        raise ImportError(
            "torch required for focal loss. `pip install torch`."
        ) from e

    class FocalLoss(nn.Module):
        def __init__(self, gamma, pos_weight):
            super().__init__()
            self.gamma = gamma
            if pos_weight is not None:
                self.register_buffer(
                    'pos_weight',
                    torch.as_tensor(pos_weight, dtype=torch.float32),
                )
            else:
                self.pos_weight = None

        def forward(self, logits, targets):
            # logits: (N, C); targets: (N, C) in {0, 1}
            p = torch.sigmoid(logits)
            ce_pos = -targets * F.logsigmoid(logits)             # -y log p
            ce_neg = -(1 - targets) * F.logsigmoid(-logits)      # -(1-y) log(1-p)
            mod_pos = (1 - p).pow(self.gamma)
            mod_neg = p.pow(self.gamma)

            loss_pos = mod_pos * ce_pos
            loss_neg = mod_neg * ce_neg

            if self.pos_weight is not None:
                loss_pos = loss_pos * self.pos_weight[None, :]

            return (loss_pos + loss_neg).mean()

    return FocalLoss(gamma, pos_weight)
