#!/usr/bin/env python3
"""Section 2 heavy compute: DT replica + HistGB hyperparameter grid +
per-class threshold tuning + non-hidden test eval. Saves artifacts
incrementally so any partial progress is preserved across restarts.
Trimmed grid (4 configs) for tractable runtime; still varies the two
documented hyperparameters."""
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

ART = PHASE / "artifacts"; ART.mkdir(exist_ok=True)
FIG = PHASE / "report" / "figures"; FIG.mkdir(parents=True, exist_ok=True)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_histgb(max_iter=200, max_leaf_nodes=31, learning_rate=0.1):
    return MultiOutputClassifier(
        HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        ),
        n_jobs=-1,
    )


# -- data ---------------------------------------------------------------------
log("=== Section 2 run ===")
train_files = sorted(glob.glob(str(utils.PATH_TRAIN / "audio_features" / "*.npz")))
our_val_files, our_nht_files = utils.split_validation(seed=42)
log(f"train={len(train_files)} val={len(our_val_files)} nht={len(our_nht_files)}")

cached_train = ART / "_train_50k.npz"
if cached_train.is_file():
    log(f"loading cached training matrix from {cached_train.name}")
    z = np.load(cached_train)
    X_train, Y_train = z["X"], z["Y"]
else:
    log("loading all training segments (one-time, will be cached) ...")
    t0 = time.time()
    X_train, Y_train = utils.load_all_segments(train_files)
    log(f"  loaded {X_train.shape}  in {time.time()-t0:.0f}s")
    rng = np.random.default_rng(seed=42)
    if X_train.shape[0] > 50_000:
        idx = rng.choice(X_train.shape[0], size=50_000, replace=False)
        X_train, Y_train = X_train[idx], Y_train[idx]
        log(f"  subsampled to {X_train.shape[0]}")
    np.savez(cached_train, X=X_train, Y=Y_train)
    log(f"  cached -> {cached_train.name}")

ann_df_val = pd.read_csv(utils.PATH_VAL / "annotations.csv")

# -- 1) DT baseline -----------------------------------------------------------
dt_path = ART / "section2_dt_baseline.joblib"
if dt_path.is_file():
    log("loading cached DT baseline")
    dt = joblib.load(dt_path)
else:
    log("training DT baseline ...")
    t0 = time.time()
    dt = MultiOutputClassifier(
        DecisionTreeClassifier(max_depth=20, max_features="sqrt", random_state=42),
        n_jobs=-1)
    dt.fit(X_train, Y_train)
    log(f"  DT fit: {time.time()-t0:.0f}s")
    joblib.dump(dt, dt_path)

pred_dt_val = utils.generate_predictions(our_val_files, dt, use_probs=False)
f1_dt_val, per_dt_val = utils.evaluate_split(pred_dt_val, our_val_files, ann_df_val)
log(f"DT local-val macro-F1 = {f1_dt_val:.4f}")

# -- 2) HistGB hyperparameter grid (4 configs: max_iter x max_leaf_nodes) ----
GRID_PATH = ART / "section2_hp_grid.csv"
if GRID_PATH.is_file():
    log("loading cached HP grid")
    hp_df = pd.read_csv(GRID_PATH)
else:
    GRID = [(mi, ml) for mi in [200, 500] for ml in [31, 63]]
    hp_rows = []
    for mi, ml in GRID:
        log(f"HistGB max_iter={mi} max_leaf_nodes={ml} ...")
        t0 = time.time()
        m  = make_histgb(max_iter=mi, max_leaf_nodes=ml)
        m.fit(X_train, Y_train)
        fit_t = time.time() - t0
        pred = utils.generate_predictions(our_val_files, m, use_probs=False)
        f1, _ = utils.evaluate_split(pred, our_val_files, ann_df_val)
        hp_rows.append({"max_iter": mi, "max_leaf_nodes": ml,
                        "val_macro_f1": float(f1),
                        "fit_seconds": round(fit_t, 1)})
        log(f"  -> val macro-F1 = {f1:.4f}  ({fit_t:.0f}s)")
        pd.DataFrame(hp_rows).to_csv(GRID_PATH, index=False)
    hp_df = pd.DataFrame(hp_rows)

# plot heatmap
piv = hp_df.pivot(index="max_iter", columns="max_leaf_nodes", values="val_macro_f1")
fig, ax = plt.subplots(figsize=(6.5, 4))
im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
ax.set_yticks(range(len(piv.index)));   ax.set_yticklabels(piv.index)
ax.set_xlabel("max_leaf_nodes"); ax.set_ylabel("max_iter")
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                color="white" if v < piv.values.mean() else "black")
ax.set_title("HistGB hyperparameter sweep — local-val macro-F1")
plt.colorbar(im, ax=ax, fraction=0.04)
plt.tight_layout()
fig.savefig(FIG / "hp_grid_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log(f"saved -> figures/hp_grid_heatmap.png")

# -- 3) Refit best + threshold tuning ----------------------------------------
best_path = ART / "section2_histgb_best.joblib"
thr_path  = ART / "section2_thresholds.npy"

best_row = hp_df.sort_values("val_macro_f1", ascending=False).iloc[0]
best_cfg = {"max_iter": int(best_row["max_iter"]),
            "max_leaf_nodes": int(best_row["max_leaf_nodes"])}
log(f"best grid config: {best_cfg}  val macro-F1 (no thr) = {best_row['val_macro_f1']:.4f}")

if best_path.is_file() and thr_path.is_file():
    log("loading cached best HistGB + thresholds")
    hgb_best = joblib.load(best_path)
    thr      = np.load(thr_path)
else:
    log("refitting best config ...")
    t0 = time.time()
    hgb_best = make_histgb(**best_cfg)
    hgb_best.fit(X_train, Y_train)
    log(f"  refit: {time.time()-t0:.0f}s")
    joblib.dump(hgb_best, best_path)

    log("collecting val probabilities for threshold tuning ...")
    probs_all, labels_all = [], []
    for fp in our_val_files:
        data = dict(np.load(fp, allow_pickle=True))
        ts = data["start_time"]
        mask = np.isclose(ts % 1.0, 0.0)
        X = utils.build_feature_matrix(data)[mask]
        Y = utils.get_segment_labels(data)[mask]
        P = utils.predict_proba_multiout(hgb_best, X)
        probs_all.append(P); labels_all.append(Y)
    P_val = np.vstack(probs_all)
    Y_val = np.vstack(labels_all)
    log(f"  val probability matrix: {P_val.shape}")

    thr = utils.tune_per_class_thresholds(P_val, Y_val,
                                          grid=np.arange(0.05, 0.96, 0.02))
    np.save(thr_path, thr)
    log(f"  per-class thresholds saved; min={thr.min():.2f}  max={thr.max():.2f}")

# tuned-threshold val score
pred_tuned_val = utils.generate_predictions(our_val_files, hgb_best,
                                             thresholds=thr, use_probs=True)
f1_tuned_val, per_tuned_val = utils.evaluate_split(pred_tuned_val,
                                                    our_val_files, ann_df_val)
log(f"HistGB best+tuned local-val macro-F1 = {f1_tuned_val:.4f}")

# -- 4) Non-hidden test evaluation (single shot) ------------------------------
log("evaluating on non-hidden test set ...")
pred_dt_nht = utils.generate_predictions(our_nht_files, dt, use_probs=False)
f1_dt_nht, per_dt_nht = utils.evaluate_split(pred_dt_nht, our_nht_files, ann_df_val)
pred_hgb_nht = utils.generate_predictions(our_nht_files, hgb_best,
                                          thresholds=thr, use_probs=True)
f1_hgb_nht, per_hgb_nht = utils.evaluate_split(pred_hgb_nht, our_nht_files, ann_df_val)
log(f"DT  nht macro-F1 = {f1_dt_nht:.4f}")
log(f"HistGB+thr nht macro-F1 = {f1_hgb_nht:.4f}  (delta = {f1_hgb_nht - f1_dt_nht:+.4f})")

# per-class plot
order = utils.CLASS_NAMES
cmp = per_dt_nht[["annotation", "f1"]].rename(columns={"f1": "DT_f1"}).merge(
      per_hgb_nht[["annotation", "f1"]].rename(columns={"f1": "HistGB_f1"}),
      on="annotation")
cmp["delta"] = cmp["HistGB_f1"] - cmp["DT_f1"]
cmp.to_csv(ART / "section2_per_class_nht.csv", index=False)
dt_f1  = cmp.set_index("annotation").loc[order, "DT_f1"].values
hgb_f1 = cmp.set_index("annotation").loc[order, "HistGB_f1"].values

fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(order)); w = 0.4
ax.bar(x - w/2, dt_f1,  w, label=f"DT (macro {f1_dt_nht:.3f})", color="coral")
ax.bar(x + w/2, hgb_f1, w, label=f"HistGB+thr (macro {f1_hgb_nht:.3f})", color="steelblue")
ax.set_xticks(x); ax.set_xticklabels(order, rotation=45, ha="right")
ax.set_ylabel("F1"); ax.set_ylim(0, 1.05)
ax.set_title("Per-class F1 on non-hidden test — DT vs HistGB (per-class thresholds)")
ax.legend(); plt.tight_layout()
fig.savefig(FIG / "section2_perclass.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# -- summary JSON -------------------------------------------------------------
summary = {
    "best_config": best_cfg,
    "baseline_DT_val_macro_f1":    float(f1_dt_val),
    "HistGB_best_val_macro_f1_no_thresh": float(best_row["val_macro_f1"]),
    "HistGB_best_val_macro_f1_tuned":     float(f1_tuned_val),
    "baseline_DT_nht_macro_f1":           float(f1_dt_nht),
    "HistGB_best_nht_macro_f1_tuned":     float(f1_hgb_nht),
    "thresholds": thr.tolist(),
    "class_names": list(utils.CLASS_NAMES),
}
with open(ART / "section2_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
log(f"saved -> artifacts/section2_summary.json")
log("=== Section 2 DONE ===")
