#!/usr/bin/env python3
"""Section 3: median-filter post-processing sweep over window size."""
import os, sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

PHASE = Path("/Users/amrabdullah/AI_2026ss/mlpc/phases/04_challenge")
sys.path.insert(0, str(PHASE))
import utils

ART = PHASE / "artifacts"
FIG = PHASE / "report" / "figures"; FIG.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


log("=== Section 3 run ===")
hgb  = joblib.load(ART / "section2_histgb_best.joblib")
thr  = np.load(ART / "section2_thresholds.npy")
our_val_files, our_nht_files = utils.split_validation(seed=42)
ann_df_val = pd.read_csv(utils.PATH_VAL / "annotations.csv")

# -- window sweep on local validation ----------------------------------------
WINDOWS = [1, 3, 5, 7, 9]
rows = []
per_class_at_w = {}
for w in WINDOWS:
    t0 = time.time()
    pred = utils.generate_predictions(our_val_files, hgb, thresholds=thr,
                                       median_window=w, use_probs=True)
    f1, per = utils.evaluate_split(pred, our_val_files, ann_df_val)
    rows.append({"window": int(w), "val_macro_f1": float(f1),
                 "n_intervals": int(len(pred)),
                 "seconds": round(time.time() - t0, 1)})
    per_class_at_w[w] = per[["annotation", "f1"]].set_index("annotation")["f1"]
    log(f"  window={w}  ->  val macro-F1 = {f1:.4f}  ({len(pred)} intervals)  "
        f"({time.time()-t0:.0f}s)")

sweep_df = pd.DataFrame(rows)
sweep_df.to_csv(ART / "section3_window_sweep.csv", index=False)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(sweep_df["window"], sweep_df["val_macro_f1"], marker="o", lw=2)
ax.set_xticks(WINDOWS)
ax.set_xlabel("Median filter window (segments, 1 s each)")
ax.set_ylabel("Macro-F1 (local validation)")
ax.set_title("Median-filter window sweep — local validation")
ax.grid(alpha=0.3)
for _, row in sweep_df.iterrows():
    ax.annotate(f"{row['val_macro_f1']:.3f}",
                (row["window"], row["val_macro_f1"]),
                textcoords="offset points", xytext=(0, 8), ha="center")
plt.tight_layout()
fig.savefig(FIG / "section3_window_sweep.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# -- per-class delta ----------------------------------------------------------
best_w = int(sweep_df.sort_values("val_macro_f1", ascending=False).iloc[0]["window"])
log(f"best window: {best_w}")

baseline_per = per_class_at_w[1]
best_per     = per_class_at_w[best_w]
cmp = pd.DataFrame({
    "class": utils.CLASS_NAMES,
    "f1_w1":     baseline_per.reindex(utils.CLASS_NAMES).values,
    f"f1_w{best_w}": best_per.reindex(utils.CLASS_NAMES).values})
cmp["delta"] = cmp[f"f1_w{best_w}"] - cmp["f1_w1"]
cmp.sort_values("delta", ascending=False).to_csv(ART / "section3_per_class_delta.csv", index=False)

fig, ax = plt.subplots(figsize=(13, 4.5))
order_by_delta = cmp.sort_values("delta", ascending=False)
ax.bar(np.arange(len(order_by_delta)), order_by_delta["delta"].values, color="steelblue")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(np.arange(len(order_by_delta)))
ax.set_xticklabels(order_by_delta["class"].values, rotation=45, ha="right")
ax.set_ylabel(f"F1 delta (w={best_w} - w=1)")
ax.set_title(f"Per-class effect of median filter (window {best_w}) on local validation")
plt.tight_layout()
fig.savefig(FIG / "section3_per_class_delta.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# -- locked single-shot non-hidden test eval ---------------------------------
pred_nht_pp = utils.generate_predictions(our_nht_files, hgb, thresholds=thr,
                                          median_window=best_w, use_probs=True)
f1_nht_pp, _ = utils.evaluate_split(pred_nht_pp, our_nht_files, ann_df_val)
pred_nht_nopp = utils.generate_predictions(our_nht_files, hgb, thresholds=thr,
                                            median_window=1, use_probs=True)
f1_nht_nopp, _ = utils.evaluate_split(pred_nht_nopp, our_nht_files, ann_df_val)

log(f"non-hidden test no postproc:    macro-F1 = {f1_nht_nopp:.4f}")
log(f"non-hidden test median w={best_w}:  macro-F1 = {f1_nht_pp:.4f}  "
    f"(delta = {f1_nht_pp - f1_nht_nopp:+.4f})")

summary = {
    "best_window": int(best_w),
    "window_sweep": sweep_df.to_dict("records"),
    "nht_macro_f1_no_postproc":   float(f1_nht_nopp),
    "nht_macro_f1_with_postproc": float(f1_nht_pp),
    "delta_nht":                  float(f1_nht_pp - f1_nht_nopp),
}
with open(ART / "section3_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
log("=== Section 3 DONE ===")
