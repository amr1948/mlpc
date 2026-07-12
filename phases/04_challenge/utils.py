"""Shared helpers for the MLPC 2026 Challenge phase.

Extends the baseline notebook's helpers with:
  - probability inference (predict_proba support for any classifier)
  - per-class threshold application
  - median-filter post-processing on per-segment binary or probability arrays

Importing the official scoring functions from baseline/evaluate.py guarantees
that our internal evaluation uses the same metric as the graders.
"""
from __future__ import annotations

import os
import sys
import glob
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parent
PATH_TRAIN    = ROOT / "data" / "train"
PATH_VAL      = ROOT / "data" / "validation"
PATH_TEST     = ROOT / "data" / "test"
PATH_BASELINE = ROOT / "baseline"
PATH_ART      = ROOT / "artifacts"
PATH_SUB      = ROOT / "submissions"

# Make the baseline's evaluate.py importable so we use the identical scorer
sys.path.insert(0, str(PATH_BASELINE))
from evaluate import (                                       # noqa: E402
    aggregate_ground_truth_annotations,
    build_segment_frame_from_intervals,
    calculate_f1_score,
)

FEATURE_NAMES = [
    "zcr_mean", "zcr_std", "zcr_min", "zcr_max",
    "melspect_mean", "melspect_std", "melspect_min", "melspect_max",
    "mfcc_mean", "mfcc_std", "mfcc_min", "mfcc_max",
    "mfcc_d_mean", "mfcc_d_std", "mfcc_d_min", "mfcc_d_max",
    "mfcc_d2_mean", "mfcc_d2_std", "mfcc_d2_min", "mfcc_d2_max",
    "flux_mean", "flux_std", "flux_min", "flux_max",
    "flatness_mean", "flatness_std", "flatness_min", "flatness_max",
    "centroid_mean", "centroid_std", "centroid_min", "centroid_max",
    "bandwidth_mean", "bandwidth_std", "bandwidth_min", "bandwidth_max",
    "contrast_mean", "contrast_std", "contrast_min", "contrast_max",
    "rolloff_low_mean", "rolloff_low_std", "rolloff_low_min", "rolloff_low_max",
    "rolloff_high_mean", "rolloff_high_std", "rolloff_high_min", "rolloff_high_max",
    "energy_mean", "energy_std", "energy_min", "energy_max",
    "power_mean", "power_std", "power_min", "power_max",
]

CLASS_NAMES = [
    "bell_ringing", "coffee_machine", "cutlery_dishes", "door_open_close",
    "footsteps", "keyboard_typing", "keychain", "light_switch", "microwave",
    "phone_ringing", "running_water", "toilet_flushing", "vacuum_cleaner",
    "wardrobe_drawer_open_close", "window_open_close",
]

SEGMENT_LENGTH = 1.0    # 1-s window
HOP_SIZE       = 0.5    # 50 % overlap (every other segment retained at inference)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def build_feature_matrix(data: dict) -> np.ndarray:
    """Concatenate the 56 feature arrays in one .npz into (N_segments, 960)."""
    arrays = []
    for name in FEATURE_NAMES:
        feat = data[name]
        if feat.ndim == 1:
            feat = feat[:, np.newaxis]
        arrays.append(feat.astype(np.float32))
    return np.concatenate(arrays, axis=1)


def get_segment_labels(data: dict) -> np.ndarray:
    """[T, C, A] -> [T, C] via overlap>0 then strict majority across annotators."""
    annotations = data["annotations"]
    binary = (annotations > 0).astype(int)
    votes  = binary.sum(axis=2)
    return (votes > (binary.shape[2] // 2)).astype(int)


def load_all_segments(file_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Stack X and Y from a list of .npz files."""
    X_list, Y_list = [], []
    for fp in file_list:
        data = dict(np.load(fp, allow_pickle=True))
        X_list.append(build_feature_matrix(data))
        Y_list.append(get_segment_labels(data))
    return np.vstack(X_list), np.vstack(Y_list)


def split_validation(seed: int = 42) -> Tuple[List[str], List[str]]:
    """Reproduce the baseline's seed=42 split of the official validation set
    into a local validation set (model selection) and a non-hidden test set
    (final estimate before the hidden submission)."""
    val_files = sorted(glob.glob(str(PATH_VAL / "audio_features" / "*.npz")))
    rng = np.random.default_rng(seed=seed)
    shuffled = rng.permutation(val_files).tolist()
    n = len(shuffled) // 2
    return shuffled[:n], shuffled[n:]


# ---------------------------------------------------------------------------
# Inference: probabilities + threshold-aware binarisation
# ---------------------------------------------------------------------------
def predict_proba_multiout(classifier, X: np.ndarray) -> np.ndarray:
    """Multi-label predict_proba -> (N, C) of P(class=1).
    Works for sklearn MultiOutputClassifier wrapping any binary classifier
    that exposes predict_proba. Robust to a per-class estimator that saw
    only one class during training (returns 0 for the absent class)."""
    if not hasattr(classifier, "estimators_"):
        # Native multi-label model (e.g. ExtraTreesClassifier)
        proba = classifier.predict_proba(X)
        if isinstance(proba, list):
            cols = []
            for k, pk in enumerate(proba):
                classes = list(classifier.classes_[k]) if isinstance(classifier.classes_, list) else list(classifier.classes_)
                if pk.shape[1] == 1:
                    cols.append(np.full(len(X), 1.0 if classes[0] == 1 else 0.0))
                else:
                    cols.append(pk[:, classes.index(1)])
            return np.column_stack(cols)
        return proba

    # MultiOutputClassifier: list of per-class estimators
    cols = []
    for est in classifier.estimators_:
        classes = list(est.classes_)
        pk = est.predict_proba(X)
        if pk.shape[1] == 1:
            cols.append(np.full(len(X), 1.0 if classes[0] == 1 else 0.0))
        else:
            cols.append(pk[:, classes.index(1)])
    return np.column_stack(cols)


def tune_per_class_thresholds(
    probs_val: np.ndarray,
    y_val: np.ndarray,
    grid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """For each class sweep tau on the validation probabilities and keep the
    F1-optimal one. Returns (C,) float array."""
    if grid is None:
        grid = np.arange(0.05, 0.96, 0.02)
    C = probs_val.shape[1]
    thresholds = np.full(C, 0.5)
    for c in range(C):
        y_t = y_val[:, c]
        p   = probs_val[:, c]
        best_f1, best_t = -1.0, 0.5
        if y_t.sum() == 0:
            thresholds[c] = 0.5
            continue
        for t in grid:
            y_p = (p >= t).astype(int)
            tp = int(((y_p == 1) & (y_t == 1)).sum())
            fp = int(((y_p == 1) & (y_t == 0)).sum())
            fn = int(((y_p == 0) & (y_t == 1)).sum())
            if tp + fp == 0 or tp + fn == 0:
                continue
            prec = tp / (tp + fp)
            rec  = tp / (tp + fn)
            f1 = 2 * prec * rec / max(prec + rec, 1e-12)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = float(best_t)
    return thresholds


# ---------------------------------------------------------------------------
# Per-file SED inference (whole-second segments only)
# ---------------------------------------------------------------------------
def run_sed_inference_proba(
    filepath: str,
    classifier,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return per-class probabilities for the whole-second segments of one file.

    Returns
    -------
    probs : (N_segments, C) float
    start_times : (N_segments,) float (in seconds)
    filename : str (e.g. '000001.wav')
    """
    data = dict(np.load(filepath, allow_pickle=True))
    start_times_all = data["start_time"]
    X_all = build_feature_matrix(data)
    proba_all = predict_proba_multiout(classifier, X_all)
    whole_second_mask = np.isclose(start_times_all % 1.0, 0.0)
    probs       = proba_all[whole_second_mask]
    start_times = start_times_all[whole_second_mask]
    fname = os.path.basename(filepath).replace(".npz", ".wav")
    return probs, start_times, fname


def median_filter_binary(preds: np.ndarray, window: int) -> np.ndarray:
    """Median filter (per class along the time axis) of a binary (N,C) array."""
    if window <= 1:
        return preds
    pad = window // 2
    padded = np.pad(preds, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(preds)
    for t in range(preds.shape[0]):
        out[t] = np.median(padded[t:t + window], axis=0)
    return out.astype(preds.dtype)


def predictions_to_intervals(
    predictions: np.ndarray,
    start_times: np.ndarray,
    filename: str,
) -> List[Dict]:
    """Binary (N,C) predictions per whole-second segment -> onset/offset rows."""
    rows = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_preds = predictions[:, cls_idx]
        in_event, onset = False, None
        for t, pred in zip(start_times, cls_preds):
            if pred == 1 and not in_event:
                onset, in_event = float(t), True
            elif pred == 0 and in_event:
                rows.append({"filename": filename, "annotation": cls_name,
                             "onset": onset, "offset": float(t)})
                in_event = False
        if in_event:
            rows.append({"filename": filename, "annotation": cls_name,
                         "onset": onset,
                         "offset": float(start_times[-1]) + SEGMENT_LENGTH})
    return rows


def generate_predictions(
    file_list: List[str],
    classifier,
    thresholds: Optional[np.ndarray] = None,
    median_window: int = 1,
    use_probs: bool = True,
) -> pd.DataFrame:
    """Run per-file SED inference, optionally threshold + median-filter, and
    return one DataFrame of all detected (filename, class, onset, offset) rows.

    Parameters
    ----------
    classifier      : trained multi-label classifier exposing predict_proba
                      (or .predict if `use_probs=False`).
    thresholds      : (C,) array of per-class taus, or None for 0.5.
    median_window   : odd window length for the median filter; 1 = off.
    use_probs       : if False, fall back to classifier.predict (binary).
    """
    if thresholds is None:
        thresholds = np.full(len(CLASS_NAMES), 0.5)

    rows = []
    for fp in file_list:
        if use_probs:
            probs, ts, fname = run_sed_inference_proba(fp, classifier)
            preds = (probs >= thresholds[np.newaxis, :]).astype(int)
        else:
            data = dict(np.load(fp, allow_pickle=True))
            start_times_all = data["start_time"]
            X_all = build_feature_matrix(data)
            pred_all = classifier.predict(X_all)
            mask = np.isclose(start_times_all % 1.0, 0.0)
            preds = pred_all[mask].astype(int)
            ts    = start_times_all[mask]
            fname = os.path.basename(fp).replace(".npz", ".wav")
        if median_window > 1:
            preds = median_filter_binary(preds, median_window)
        rows.extend(predictions_to_intervals(preds, ts, fname))
    if not rows:
        return pd.DataFrame(columns=["filename", "annotation", "onset", "offset"])
    return pd.DataFrame(rows)


def evaluate_split(
    pred_df: pd.DataFrame,
    file_list: List[str],
    ann_df: pd.DataFrame,
) -> Tuple[float, pd.DataFrame]:
    """Official segment-based macro-F1 on a given split, via baseline/evaluate.py."""
    split_filenames = {os.path.basename(f).replace(".npz", ".wav") for f in file_list}
    ann_split = ann_df[ann_df["filename"].isin(split_filenames)].copy()
    gt        = aggregate_ground_truth_annotations(ann_split)
    gt_seg    = build_segment_frame_from_intervals(gt,      name="ground_truth")
    pred_seg  = build_segment_frame_from_intervals(pred_df, name="predictions")
    if len(pred_seg) > 0:
        pred_seg = pred_seg[pred_seg.index.get_level_values("filename").isin(split_filenames)]
    return calculate_f1_score(gt_seg, pred_seg)
