#!/usr/bin/env python3
"""Error analysis for Section 2(d): pick three illustrative test files and
plot per-class timeline (GT boxes + framewise probabilities)."""
import os, sys, time, glob, json
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


log("=== Error analysis ===")
hgb = joblib.load(ART / "section2_histgb_best.joblib")
thr = np.load(ART / "section2_thresholds.npy")
with open(ART / "section3_summary.json") as f:
    best_w = int(json.load(f)["best_window"])
log(f"loaded HistGB + thresholds; median window = {best_w}")

our_val_files, our_nht_files = utils.split_validation(seed=42)
ann_df_val = pd.read_csv(utils.PATH_VAL / "annotations.csv")

# generate predictions on local val
pred_val = utils.generate_predictions(our_val_files, hgb, thresholds=thr,
                                       median_window=best_w, use_probs=True)
log(f"local-val intervals: {len(pred_val)}")

# Build per-second indicator frames
gt = utils.aggregate_ground_truth_annotations(
        ann_df_val[ann_df_val["filename"].isin(
            {os.path.basename(f).replace(".npz", ".wav") for f in our_val_files})])
gt_seg   = utils.build_segment_frame_from_intervals(gt,       name="ground_truth")
pred_seg = utils.build_segment_frame_from_intervals(pred_val, name="predictions")


def file_arrays(fname):
    has_g = fname in gt_seg.index.get_level_values("filename")
    has_p = fname in pred_seg.index.get_level_values("filename")
    if not has_g and not has_p: return None
    g = gt_seg.xs(fname, level="filename") if has_g else None
    p = pred_seg.xs(fname, level="filename") if has_p else None
    seconds = sorted(set(
        (g.index.tolist() if g is not None else []) +
        (p.index.tolist() if p is not None else [])
    ))
    classes = utils.CLASS_NAMES
    Y_t = np.zeros((len(seconds), len(classes)), dtype=int)
    Y_p = np.zeros_like(Y_t)
    si = {s: i for i, s in enumerate(seconds)}
    ci = {c: i for i, c in enumerate(classes)}
    if g is not None:
        for sec, row in g.iterrows():
            for cls in classes:
                if cls in row.index and row[cls] == 1: Y_t[si[sec], ci[cls]] = 1
    if p is not None:
        for sec, row in p.iterrows():
            for cls in classes:
                if cls in row.index and row[cls] == 1: Y_p[si[sec], ci[cls]] = 1
    return seconds, Y_t, Y_p


# Rank by active-class F1
rank_rows = []
for fp in our_val_files:
    fname = os.path.basename(fp).replace(".npz", ".wav")
    out = file_arrays(fname)
    if out is None: continue
    secs, Y_t, Y_p = out
    if Y_t.sum() == 0: continue
    gt_act, pred_act = (Y_t.sum(0) > 0), (Y_p.sum(0) > 0)
    active = gt_act | pred_act
    if active.sum() == 0: continue
    tp = ((Y_t == 1) & (Y_p == 1)).sum(0)[active]
    fp = ((Y_t == 0) & (Y_p == 1)).sum(0)[active]
    fn = ((Y_t == 1) & (Y_p == 0)).sum(0)[active]
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0)
    rec  = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0)
    f1 = np.where(prec + rec > 0,
                  2 * prec * rec / np.maximum(prec + rec, 1e-12), 0).mean()
    rank_rows.append({"file": fname, "n_sec": len(secs),
                      "n_active_gt": int(gt_act.sum()),
                      "n_active_pred": int(pred_act.sum()),
                      "n_active": int(active.sum()),
                      "predicts_wrong": int((pred_act & ~gt_act).any()),
                      "macro_f1_active": float(f1)})
rank = pd.DataFrame(rank_rows).sort_values("macro_f1_active", ascending=False)
log(f"ranked {len(rank)} files")

# Pick three
clean_cand = rank[(rank["n_active"].between(3, 5)) & (rank["n_sec"] >= 15)]
FILE_CLEAN = clean_cand.iloc[0]["file"] if len(clean_cand) else rank.iloc[0]["file"]
mixed_cand = rank[(rank["n_active"].between(4, 6)) &
                   (rank["macro_f1_active"].between(0.35, 0.65))]
FILE_MIXED = mixed_cand.iloc[len(mixed_cand)//2]["file"] if len(mixed_cand) \
              else rank.iloc[len(rank)//2]["file"]
fail_cand = rank[(rank["predicts_wrong"] == 1) & (rank["n_sec"] >= 12)]
FILE_FAIL = fail_cand.iloc[-1]["file"] if len(fail_cand) else rank.iloc[-1]["file"]

for name, fid in [("CLEAN", FILE_CLEAN), ("MIXED", FILE_MIXED), ("FAILURE", FILE_FAIL)]:
    r = rank[rank["file"] == fid].iloc[0]
    log(f"{name}: {fid}  {r['n_sec']}s, {r['n_active']} active, "
        f"F1={r['macro_f1_active']:.3f}")

with open(ART / "section2d_picks.json", "w") as f:
    json.dump({"clean": FILE_CLEAN, "mixed": FILE_MIXED, "failure": FILE_FAIL,
               "rank_top5":    rank.head().to_dict("records"),
               "rank_bottom5": rank.tail().to_dict("records")}, f, indent=2)

# Plot per-class timeline (slide-matching style)
_PALETTE = plt.get_cmap("tab20").colors
_BG = "#fafafa"


def _runs_of_ones(x):
    x = np.asarray(x).astype(int)
    if x.size == 0 or x.sum() == 0: return []
    diffs = np.diff(np.concatenate([[0], x, [0]]))
    return list(zip(np.where(diffs == 1)[0].tolist(),
                    (np.where(diffs == -1)[0] - 1).tolist()))


def file_probs_and_gt(fname):
    fpath = utils.PATH_VAL / "audio_features" / fname.replace(".wav", ".npz")
    if not fpath.is_file(): return None
    data = dict(np.load(fpath, allow_pickle=True))
    ts_all = data["start_time"]
    mask = np.isclose(ts_all % 1.0, 0.0)
    X = utils.build_feature_matrix(data)[mask]
    Y = utils.get_segment_labels(data)[mask]
    P = utils.predict_proba_multiout(hgb, X)
    return ts_all[mask], Y, P


def plot_timeline(fname, out_png, prob_floor=0.3):
    out = file_probs_and_gt(fname)
    if out is None: return
    seg_starts, y_t, p = out
    seg_centers = seg_starts + 0.5
    duration = float(seg_starts.max() + 1.0)
    active = [c for c in range(len(utils.CLASS_NAMES))
              if y_t[:, c].any() or (p[:, c] >= prob_floor).any()]
    if not active: active = list(range(len(utils.CLASS_NAMES)))
    n = len(active)
    fig, axes = plt.subplots(n, 1, figsize=(14, max(3.0, 0.95 * n + 1.4)),
                             sharex=True, gridspec_kw={"hspace": 0.0})
    if n == 1: axes = [axes]
    fig.patch.set_facecolor(_BG)
    for i, c in enumerate(active):
        ax = axes[i]; ax.set_facecolor(_BG)
        color = _PALETTE[c % len(_PALETTE)]
        for s, e in _runs_of_ones(y_t[:, c]):
            ax.axvspan(seg_starts[s], seg_starts[e] + 1.0,
                       ymin=0.04, ymax=0.96,
                       facecolor=color, alpha=0.22, linewidth=0)
        ax.plot(seg_centers, p[:, c], color=color, lw=1.7)
        ax.axhline(thr[c], color="#999", lw=0.6, ls="--", alpha=0.6)
        ax.axhline(0, color="#aaa", lw=0.5, alpha=0.5)
        ax.set_ylim(-0.05, 1.05); ax.set_yticks([])
        ax.set_xlim(0, duration)
        ax.set_ylabel(utils.CLASS_NAMES[c], rotation=0, ha="right", va="center",
                      fontsize=11, color="#333")
        for sp in ("left", "right"): ax.spines[sp].set_visible(False)
        ax.spines["top"].set_visible(i > 0); ax.spines["top"].set_color("#ddd")
        ax.spines["top"].set_linewidth(0.6)
        ax.spines["bottom"].set_visible(i == n - 1); ax.spines["bottom"].set_color("#bbb")
        ax.tick_params(axis="x", colors="#555", length=4)
        ax.grid(axis="x", alpha=0.15, color="#888", linewidth=0.5)
    axes[-1].set_xlabel("Time (s)", fontsize=11, color="#333")
    fig.suptitle(
        f"{fname} | local validation\n"
        "Transparent boxes: ground truth | Curves: HistGB framewise probability | "
        "dashed: per-class threshold\n"
        "Per-class event timeline",
        fontsize=12, fontweight="bold", y=1.00)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)


plot_timeline(FILE_CLEAN, FIG / "err_clean.png")
plot_timeline(FILE_MIXED, FIG / "err_mixed.png")
plot_timeline(FILE_FAIL,  FIG / "err_failure.png")
log("saved -> figures/err_{clean,mixed,failure}.png")
log("=== Error analysis DONE ===")
