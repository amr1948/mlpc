"""
Multi-label evaluation: thresholding, per-class metrics, reports.

Functions are stateless and return pandas DataFrames so notebooks can
freely concatenate, save, plot, or pipe into LaTeX.

Default conventions
-------------------
- Probabilities `probs`     : (N, C) float in [0, 1]
- Binary predictions Y_pred : (N, C) int8 / bool
- Targets         Y_true    : (N, C) int8
- Class ordering           : alphabetical, matching SplitData.class_names
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Thresholding
# ---------------------------------------------------------------------------

def predict_with_threshold(
    probs: np.ndarray,                         # (N, C)
    thresholds: float | np.ndarray = 0.5,      # scalar or (C,)
) -> np.ndarray:
    """Apply per-class (or scalar) threshold to convert probs to {0, 1}."""
    if np.isscalar(thresholds):
        return (probs >= thresholds).astype(np.int8)
    thresholds = np.asarray(thresholds).reshape(1, -1)
    return (probs >= thresholds).astype(np.int8)


def tune_thresholds_on_val(
    val_probs:   np.ndarray,                   # (N_val, C)
    Y_val:       np.ndarray,                   # (N_val, C)
    grid:        np.ndarray | None = None,
    class_names: list | None = None,
    verbose:     bool = False,
) -> pd.DataFrame:
    """
    Sweep thresholds per class on val, pick the F1-optimal one per class.

    Returns
    -------
    DataFrame indexed by class with columns:
      threshold, val_f1, val_precision, val_recall, val_support

    Notes
    -----
    Per-class threshold tuning is often the single highest-impact lever
    in multi-label SED. The default 0.5 systematically under-predicts
    rare classes whose priors are far below 0.5. Tuning lets each class
    find its own operating point on the P-R curve.
    """
    if grid is None:
        grid = np.arange(0.05, 0.96, 0.01)

    N, C = val_probs.shape
    rows = []
    for c in range(C):
        cls = class_names[c] if class_names is not None else f"class_{c}"
        y_true = Y_val[:, c]
        p = val_probs[:, c]
        support = int(y_true.sum())

        if support == 0:
            rows.append({
                'class': cls, 'threshold': 0.5,
                'val_f1': 0.0, 'val_precision': 0.0,
                'val_recall': 0.0, 'val_support': 0,
            })
            continue

        best_f1, best_th = -1.0, 0.5
        for th in grid:
            y_pred = (p >= th).astype(np.int8)
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            if tp + fp == 0 or tp + fn == 0:
                f1 = 0.0
            else:
                prec = tp / (tp + fp)
                rec  = tp / (tp + fn)
                f1 = 2 * prec * rec / max(prec + rec, 1e-12)
            if f1 > best_f1:
                best_f1, best_th = f1, th

        # Compute precision/recall at the chosen threshold for the report
        y_pred = (p >= best_th).astype(np.int8)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)

        rows.append({
            'class':         cls,
            'threshold':     round(float(best_th), 3),
            'val_f1':        round(float(best_f1), 4),
            'val_precision': round(float(prec), 4),
            'val_recall':    round(float(rec), 4),
            'val_support':   support,
        })
        if verbose:
            print(f"  {cls:<30s}  th={best_th:.2f}  f1={best_f1:.4f}  support={support}")

    return pd.DataFrame(rows).set_index('class')


# ---------------------------------------------------------------------------
# Classification report
# ---------------------------------------------------------------------------

def classification_report(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    class_names: list | None = None,
) -> pd.DataFrame:
    """
    Per-class precision/recall/F1/support + macro/micro/weighted averages.

    Returns
    -------
    DataFrame indexed by class name (plus 'macro avg', 'micro avg',
    'weighted avg') with columns: precision, recall, f1, support.
    """
    N, C = Y_true.shape
    if class_names is None:
        class_names = [f"class_{i}" for i in range(C)]

    rows = []
    for c in range(C):
        y_t = Y_true[:, c]; y_p = Y_pred[:, c]
        tp = int(((y_p == 1) & (y_t == 1)).sum())
        fp = int(((y_p == 1) & (y_t == 0)).sum())
        fn = int(((y_p == 0) & (y_t == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-12)
        rows.append({
            'class':     class_names[c],
            'precision': round(float(prec), 4),
            'recall':    round(float(rec), 4),
            'f1':        round(float(f1), 4),
            'support':   int(y_t.sum()),
        })

    df = pd.DataFrame(rows).set_index('class')

    # Macro avg (unweighted mean over classes)
    df.loc['macro avg'] = [
        round(float(df['precision'].mean()), 4),
        round(float(df['recall'].mean()),    4),
        round(float(df['f1'].mean()),        4),
        int(df['support'].sum()),
    ]

    # Micro avg (treat all class predictions as one big binary problem)
    tp_m = int(((Y_pred == 1) & (Y_true == 1)).sum())
    fp_m = int(((Y_pred == 1) & (Y_true == 0)).sum())
    fn_m = int(((Y_pred == 0) & (Y_true == 1)).sum())
    pr = tp_m / max(tp_m + fp_m, 1)
    re = tp_m / max(tp_m + fn_m, 1)
    f1 = 2 * pr * re / max(pr + re, 1e-12)
    df.loc['micro avg'] = [round(float(pr), 4), round(float(re), 4),
                           round(float(f1), 4), int(Y_true.sum())]

    # Weighted avg (per-class metrics weighted by support)
    support = df.loc[class_names, 'support'].values.astype(float)
    if support.sum() > 0:
        w = support / support.sum()
        df.loc['weighted avg'] = [
            round(float((df.loc[class_names, 'precision'].values * w).sum()), 4),
            round(float((df.loc[class_names, 'recall'].values * w).sum()),    4),
            round(float((df.loc[class_names, 'f1'].values * w).sum()),        4),
            int(support.sum()),
        ]

    return df


# ---------------------------------------------------------------------------
# Strategy comparison
# ---------------------------------------------------------------------------

def compare_strategies(
    report_A: pd.DataFrame,
    report_B: pd.DataFrame,
    metric: str = 'f1',
) -> pd.DataFrame:
    """
    Side-by-side per-class metric comparison between Strategy A and B.

    Both reports must be DataFrames produced by classification_report()
    with matching class ordering. Class rows are kept; aggregate rows
    (macro/micro/weighted avg) are also included for one-shot comparison.

    Returns
    -------
    DataFrame with columns A_<metric>, B_<metric>, delta.
    """
    return pd.DataFrame({
        f'A_{metric}': report_A[metric],
        f'B_{metric}': report_B[metric],
        'delta':       (report_B[metric] - report_A[metric]).round(4),
    })


# ---------------------------------------------------------------------------
# Quick scalar metrics — for training loops
# ---------------------------------------------------------------------------

def macro_f1(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """Unweighted mean of per-class F1 — the headline metric."""
    C = Y_true.shape[1]
    f1s = []
    for c in range(C):
        y_t = Y_true[:, c]; y_p = Y_pred[:, c]
        tp = ((y_p == 1) & (y_t == 1)).sum()
        fp = ((y_p == 1) & (y_t == 0)).sum()
        fn = ((y_p == 0) & (y_t == 1)).sum()
        if tp == 0 and (fp + fn) == 0:
            f1s.append(0.0)
        elif tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
        else:
            prec = tp / (tp + fp)
            rec  = tp / (tp + fn)
            f1s.append(2 * prec * rec / max(prec + rec, 1e-12))
    return float(np.mean(f1s))


def micro_f1(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """Pool all classes' TP/FP/FN, then compute F1. Dominated by common classes."""
    tp = ((Y_pred == 1) & (Y_true == 1)).sum()
    fp = ((Y_pred == 1) & (Y_true == 0)).sum()
    fn = ((Y_pred == 0) & (Y_true == 1)).sum()
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec  = tp / (tp + fn)
    return float(2 * prec * rec / max(prec + rec, 1e-12))
