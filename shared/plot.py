"""
Plotting utilities for multi-label evaluation and training.

All functions return (fig, ax) tuples so notebooks can save / annotate
further. Style: minimal, paper-ready, no seaborn dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Per-class F1 — single strategy
# ---------------------------------------------------------------------------

def plot_per_class_f1(
    report_df:   pd.DataFrame,
    class_names: list | None = None,
    sort_by:     str | None = 'f1',
    title:       str = "Per-class F1",
    figsize:     tuple = (12, 5),
    color:       str = 'steelblue',
):
    """
    Horizontal bar plot of per-class F1.

    Parameters
    ----------
    report_df : DataFrame from eval.classification_report()
    class_names : optional list to filter & order; if None, all class rows
                  are used (aggregate rows like 'macro avg' are dropped).
    sort_by : 'f1' (default), 'support', or None for original order.
    """
    AGG_ROWS = {'macro avg', 'micro avg', 'weighted avg'}

    if class_names is None:
        df = report_df.drop(index=[r for r in AGG_ROWS if r in report_df.index]).copy()
    else:
        df = report_df.loc[class_names].copy()

    if sort_by:
        df = df.sort_values(sort_by, ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(df.index, df['f1'], color=color)
    ax.set_xlabel("F1")
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row['f1'] + 0.01, i,
                f"{row['f1']:.3f}  (n={int(row['support'])})",
                va='center', fontsize=9)

    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Strategy comparison
# ---------------------------------------------------------------------------

def plot_per_class_comparison(
    compare_df:  pd.DataFrame,      # output of eval.compare_strategies()
    class_names: list,
    metric:      str = 'f1',
    title:       str = "Strategy A vs Strategy B",
    figsize:     tuple = (12, 6),
):
    """Grouped bar plot of A and B side by side, per class."""
    df = compare_df.loc[class_names].copy().sort_values(f'A_{metric}', ascending=True)

    y = np.arange(len(df))
    h = 0.4
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(y - h / 2, df[f'A_{metric}'], height=h, label='Strategy A',
            color='steelblue')
    ax.barh(y + h / 2, df[f'B_{metric}'], height=h, label='Strategy B (aug)',
            color='coral')
    ax.set_yticks(y)
    ax.set_yticklabels(df.index)
    ax.set_xlabel(metric.upper())
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Confusion matrices (per class, since this is multi-label)
# ---------------------------------------------------------------------------

def plot_confusion_matrices(
    Y_true:      np.ndarray,
    Y_pred:      np.ndarray,
    class_names: list,
    ncols:       int = 5,
    figsize_per: tuple = (2.5, 2.2),
    title:       str | None = None,
):
    """
    One 2x2 binary CM per class, in a grid.

    For multi-label problems the standard CxC confusion matrix is
    meaningless (one segment can be many classes). Instead show
    per-class TN/FP/FN/TP.
    """
    C = Y_true.shape[1]
    nrows = (C + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
    )
    axes = np.array(axes).reshape(nrows, ncols)

    for c in range(C):
        ax = axes[c // ncols, c % ncols]
        y_t = Y_true[:, c]
        y_p = Y_pred[:, c]
        tn = int(((y_p == 0) & (y_t == 0)).sum())
        fp = int(((y_p == 1) & (y_t == 0)).sum())
        fn = int(((y_p == 0) & (y_t == 1)).sum())
        tp = int(((y_p == 1) & (y_t == 1)).sum())
        cm = np.array([[tn, fp], [fn, tp]])

        ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['pred 0', 'pred 1'], fontsize=8)
        ax.set_yticklabels(['true 0', 'true 1'], fontsize=8)
        ax.set_title(class_names[c], fontsize=9)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}",
                        ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black',
                        fontsize=8)

    for k in range(C, nrows * ncols):
        axes[k // ncols, k % ncols].axis('off')

    if title:
        fig.suptitle(title, y=1.0)
    plt.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Loss / metric curves for NN training
# ---------------------------------------------------------------------------

def plot_loss_curves(
    history: dict,
    title: str = "Training",
    figsize: tuple = (10, 4),
):
    """
    Plot loss + val metric over epochs.

    Expected history keys (any subset):
      'train_loss', 'val_loss', 'val_f1', 'val_macro_f1'
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='train')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='val')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_title('Loss')

    metric_key = next((k for k in ('val_macro_f1', 'val_f1', 'val_micro_f1') if k in history), None)
    if metric_key is not None:
        axes[1].plot(history[metric_key], color='green')
        axes[1].set_xlabel('epoch')
        axes[1].set_ylabel(metric_key)
        axes[1].grid(alpha=0.3)
        axes[1].set_title(f'Val {metric_key.replace("val_", "")}')
    else:
        axes[1].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    return fig, axes
