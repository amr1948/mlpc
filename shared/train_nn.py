"""
Shared training loop for multi-label NN classifiers (CNN, LSTM, AST).

Design choices:
- Early-stop on val mean-AP (threshold-free).
- min_epochs / patience / max_epochs default to the values that worked
  for MLP v2 in 07_train_mlp.ipynb (min=15, patience=7, max=60).
- Grad clipping + skip-non-finite-loss-batch are on by default. They
  fix the MPS BatchNorm NaN bug observed in MLP v1 (terminal_claude.md
  section 6) and cost nothing on stable backends.
- Best state_dict is held in memory, not on disk -- final notebook saves
  it once at the end.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Device pick
# ---------------------------------------------------------------------------
def auto_device():
    """CUDA > MPS > CPU. MPS is fine with grad clipping (see train_one_run)."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Inference -- returns probs and targets as numpy
# ---------------------------------------------------------------------------
def eval_probs(model, loader, device):
    """Run model on a loader, return (probs (N,C), Y (N,C)) numpy arrays."""
    import torch
    model.eval()
    probs_all, y_all = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs_all.append(torch.sigmoid(logits).detach().cpu().numpy())
            y_all.append(y.numpy())
    probs = np.concatenate(probs_all, axis=0).astype(np.float32)
    Y     = np.concatenate(y_all,     axis=0).astype(np.int8)
    return probs, Y


# ---------------------------------------------------------------------------
# Threshold-free metric -- mean Average Precision
# ---------------------------------------------------------------------------
def mean_ap(probs: np.ndarray, Y: np.ndarray) -> float:
    """Unweighted mean of per-class Average Precision.

    Classes with zero positives in `Y` are skipped (they have undefined AP).
    """
    from sklearn.metrics import average_precision_score
    C = Y.shape[1]
    aps = []
    for c in range(C):
        if Y[:, c].sum() == 0:
            continue
        aps.append(average_precision_score(Y[:, c], probs[:, c]))
    return float(np.mean(aps)) if aps else 0.0


# ---------------------------------------------------------------------------
# Training history container
# ---------------------------------------------------------------------------
@dataclass
class TrainHistory:
    train_loss:  list = field(default_factory=list)
    val_mAP:     list = field(default_factory=list)
    val_f1_05:   list = field(default_factory=list)
    skipped:     list = field(default_factory=list)
    best_epoch:  int  = -1
    best_val_mAP: float = -1.0

    def summary(self) -> str:
        return (f"best epoch={self.best_epoch}, "
                f"best val mAP={self.best_val_mAP:.4f}, "
                f"final val F1@0.5={self.val_f1_05[-1]:.4f}, "
                f"final train loss={self.train_loss[-1]:.4f}")


# ---------------------------------------------------------------------------
# One-config training run
# ---------------------------------------------------------------------------
def train_one_run(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    *,
    scheduler=None,
    device=None,
    max_epochs: int = 60,
    min_epochs: int = 15,
    patience:   int = 7,
    grad_clip:  float = 1.0,
    max_loss:   float = 1000.0,
    log_every:  int = 1,
    log_fn:     Callable = print,
) -> tuple[dict, TrainHistory]:
    """Train until val mAP stops improving. Return (best_state_dict, history).

    The model is mutated in place; the caller can load `best_state_dict` back
    into it after training, or use the returned state_dict directly.
    """
    import torch
    from eval import macro_f1

    if device is None:
        device = auto_device()
    model.to(device)

    hist = TrainHistory()
    best_state: Optional[dict] = None
    epochs_since_best = 0

    for ep in range(1, max_epochs + 1):
        # ---- train one epoch ------------------------------------------------
        model.train()
        running_loss, n_steps, skipped = 0.0, 0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)

            # Skip the batch entirely if loss is non-finite or unreasonably
            # large. Apple MPS BatchNorm reductions occasionally emit huge
            # values; the grad-clip below would protect training even
            # without this check, but the running_loss reporting would get
            # poisoned by the outlier. See terminal_claude.md section 6.
            if not torch.isfinite(loss) or loss.item() > max_loss:
                skipped += 1
                continue

            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running_loss += float(loss.item())
            n_steps += 1

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / max(n_steps, 1)

        # ---- val ------------------------------------------------------------
        probs, Y = eval_probs(model, val_loader, device)
        mAP = mean_ap(probs, Y)
        f1_05 = macro_f1(Y, (probs >= 0.5).astype(np.int8))

        hist.train_loss.append(train_loss)
        hist.val_mAP.append(mAP)
        hist.val_f1_05.append(f1_05)
        hist.skipped.append(skipped)

        if ep % log_every == 0:
            log_fn(f"  ep {ep:2d}  tr_loss={train_loss:.4f}  "
                   f"val_mAP={mAP:.4f}  val_F1@0.5={f1_05:.4f}  "
                   f"skipped={skipped}")

        # ---- early stop ----------------------------------------------------
        if mAP > hist.best_val_mAP:
            hist.best_val_mAP = mAP
            hist.best_epoch = ep
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if ep >= min_epochs and epochs_since_best >= patience:
                log_fn(f"  early stop at epoch {ep} (best={hist.best_epoch})")
                break

    return (best_state or model.state_dict()), hist


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def seed_all(seed: int = 42) -> None:
    """Seed numpy, python random, and torch (CUDA + MPS deterministic flags)."""
    import random, os
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # MPS does not currently support full determinism on reduction ops; we
    # accept this and rely on the grad-clip + skip-NaN safety net.
