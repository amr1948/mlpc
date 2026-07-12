import numpy as np
from pathlib import Path

ROOT = Path("/Users/amrabdullah/AI_2026ss/mlpc")
z = np.load(ROOT / "strategy_A_baseline/artifacts/splits.npz", allow_pickle=True)

print("Contents of Strategy A splits.npz:")
for k in z.files:
    a = z[k]
    print(f"  {k:24s} shape={getattr(a,'shape','-')}  dtype={a.dtype}")

# Is the training feature matrix already standardized? (biggest 960-col float array = train)
cand = [(k, z[k]) for k in z.files
        if z[k].ndim == 2 and z[k].dtype.kind == 'f' and z[k].shape[1] == 960]
if cand:
    k, X = max(cand, key=lambda kv: kv[1].shape[0])
    print(f"\nLikely train features: '{k}'  shape={X.shape}")
    print(f"  mean of per-col means: {X.mean(0).mean():+.3f}  (~0 if standardized)")
    print(f"  mean of per-col stds : {X.std(0).mean():.3f}  (~1 if standardized)")

# Any scaler params bundled in the file?
scaler_keys = [k for k in z.files
               if any(s in k.lower() for s in ['scaler', 'mean_', 'scale_', 'std_'])]
print(f"\nScaler params inside file: {scaler_keys or 'none found'}")