#!/usr/bin/env python3
"""Generate the hidden-test prediction CSV with the locked best system
(HistGB + per-class thresholds + best median window)."""
import os, sys, time, glob, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

PHASE = Path("/Users/amrabdullah/AI_2026ss/mlpc/phases/04_challenge")
sys.path.insert(0, str(PHASE))
import utils

ART = PHASE / "artifacts"
SUB = PHASE / "submissions"; SUB.mkdir(exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


log("=== Final predictions ===")
hgb = joblib.load(ART / "section2_histgb_best.joblib")
thr = np.load(ART / "section2_thresholds.npy")
with open(ART / "section3_summary.json") as f:
    best_w = int(json.load(f)["best_window"])
log(f"HistGB + thresholds + median window={best_w} loaded")

hidden_files = sorted(glob.glob(str(utils.PATH_TEST / "audio_features" / "*.npz")))
log(f"hidden test files: {len(hidden_files)}")

t0 = time.time()
pred_hidden = utils.generate_predictions(hidden_files, hgb, thresholds=thr,
                                          median_window=best_w, use_probs=True)
log(f"generated {len(pred_hidden)} intervals in {time.time()-t0:.0f}s")

# format checks
assert all(c in pred_hidden.columns for c in ["filename", "annotation", "onset", "offset"])
assert (pred_hidden["offset"] > pred_hidden["onset"]).all()
assert pred_hidden["annotation"].isin(utils.CLASS_NAMES).all()

OUT = SUB / "predictions_hidden_test.csv"
pred_hidden.to_csv(OUT, index=False)
log(f"saved -> {OUT}")

# distribution vs baseline
baseline_csv = PHASE / "baseline" / "predictions_hidden_test.csv"
if baseline_csv.is_file():
    base_df = pd.read_csv(baseline_csv)
    by_cls = pd.DataFrame({
        "baseline_events": base_df.groupby("annotation").size(),
        "ours_events":     pred_hidden.groupby("annotation").size(),
    }).reindex(utils.CLASS_NAMES).fillna(0).astype(int)
    by_cls["delta"] = by_cls["ours_events"] - by_cls["baseline_events"]
    by_cls.to_csv(ART / "final_class_event_counts.csv")
    log(f"baseline submission rows: {len(base_df)} | ours: {len(pred_hidden)}  "
        f"(delta {len(pred_hidden) - len(base_df):+d})")
log("=== Final predictions DONE ===")
