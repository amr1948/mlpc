"""
On-disk mel-spectrogram cache for spectrogram-input models (CNN, LSTM, AST).

Why this module exists
----------------------
The existing pipeline aggregates 14 feature groups into 56 keys via
mean/std/min/max, producing 960-D vectors per 1 s segment. That aggregation
destroys time and frequency structure -- which is exactly what CNN / LSTM /
AST need. This module computes per-segment log-mel-spectrograms with the
SAME windowing as features.py + 01_data_preparation, so segment indices in
this cache map 1:1 to splits.npz.

Usage
-----
>>> import sys; sys.path.append('../../_shared')
>>> from melspec import build_cache, load_cache, MelSpecDataset

Build once (writes ~4.3 GB at data/melspecs.npy):
>>> from melspec import build_cache
>>> build_cache()

Load read-only memmap in notebooks/models:
>>> cache, meta = load_cache()
>>> cache.shape           # (168239, 128, 100)
>>> meta['file_ids'][:3]  # ['000001', '000002', '000003']

PyTorch dataset for training (one global segment index per item):
>>> ds = MelSpecDataset(splits.train_idx, splits.Y_train)
>>> x, y = ds[0]          # x: (1, 128, 100) float32, y: (15,) float32

Notes
-----
- Cache is float16 to fit in ~4.3 GB. Loss vs float32 is negligible for
  log-mel values in the [0, ~12] range we observe.
- The cache does NOT apply per-feature normalization (z-score). Each model
  applies its own (e.g. AudioSet stats for AST, BatchNorm for CNN/LSTM).
- Cache is built from data/audio (NOT data/audio_aug). Strategy A only.
- The librosa params here MUST match features.py; if features.py ever
  changes, rebuild the cache or set SR / N_FFT / HOP / N_MELS to match.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from data import find_root, load_config


# ---------------------------------------------------------------------------
# Librosa params -- kept in sync with features.py and config.json
# ---------------------------------------------------------------------------
SR              = 16000
N_FFT           = 512
HOP_LENGTH      = 160
N_MELS          = 128
FMIN            = 50
FMAX            = None
MEL_NORM        = "slaney"
LOG_SCALE       = 1e5     # features.py applies log1p(1e5 * mel)

# Segmentation params -- match aggregate_features_over_time + 01_data_preparation
SEG_WIN_S       = 1.0
SEG_HOP_S       = 0.5
SEG_WIN_FRAMES  = int(SEG_WIN_S * SR / HOP_LENGTH)   # 100
SEG_HOP_FRAMES  = int(SEG_HOP_S * SR / HOP_LENGTH)   # 50


# ---------------------------------------------------------------------------
# Per-file extraction
# ---------------------------------------------------------------------------
def extract_logmel(wav: np.ndarray) -> np.ndarray:
    """Compute log-mel-spec for a full audio file.

    Returns (n_mels, n_frames) float32.
    """
    import librosa
    spec = np.abs(librosa.stft(
        wav, n_fft=N_FFT, hop_length=HOP_LENGTH, center=False
    ))
    mel = librosa.feature.melspectrogram(
        S=spec, sr=SR, n_fft=N_FFT, n_mels=N_MELS,
        fmin=FMIN, fmax=FMAX, norm=MEL_NORM
    )
    return np.log1p(LOG_SCALE * mel).astype(np.float32)


def segment_logmel(log_mel: np.ndarray, expected_T: int) -> np.ndarray:
    """Slice a file-level log-mel into per-segment patches.

    Parameters
    ----------
    log_mel : (n_mels, n_frames) float32
    expected_T : int
        Number of segments this file should produce, from features.py.

    Returns
    -------
    (T, n_mels, SEG_WIN_FRAMES) float32, with T == expected_T.
    Pads with zeros if librosa produced fewer frames than expected (rare
    edge case at end of file).
    """
    n_frames = log_mel.shape[1]
    max_T = max(0, (n_frames - SEG_WIN_FRAMES) // SEG_HOP_FRAMES + 1)
    T = min(max_T, expected_T)

    segs = np.zeros((expected_T, N_MELS, SEG_WIN_FRAMES), dtype=np.float32)
    for t in range(T):
        s = t * SEG_HOP_FRAMES
        segs[t] = log_mel[:, s:s + SEG_WIN_FRAMES]
    return segs


# ---------------------------------------------------------------------------
# Cache build
# ---------------------------------------------------------------------------
def _default_cache_path() -> Path:
    return find_root() / "data" / "melspecs.npy"


def _default_meta_path() -> Path:
    return find_root() / "data" / "melspecs_meta.npz"


def build_cache(
    cache_path: Optional[Path] = None,
    meta_path:  Optional[Path] = None,
    dtype: type = np.float16,
    progress: bool = True,
) -> None:
    """Extract log-mel-specs from data/audio and write a memmap cache.

    Runtime: ~12-15 minutes on a modern Mac for 3,656 files.
    Disk: ~4.3 GB for float16, ~8.6 GB for float32.
    """
    import librosa

    root      = find_root()
    cfg       = load_config()
    audio_dir = root / cfg["paths"]["orig_audio"]
    feat_dir  = root / cfg["paths"]["orig_npz"]

    cache_path = Path(cache_path) if cache_path else _default_cache_path()
    meta_path  = Path(meta_path)  if meta_path  else _default_meta_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Canonical file order -- MUST match sorted(FEATURES_DIR.glob('*.npz'))
    # used by 01_data_preparation. Otherwise global segment indices in
    # splits.npz would point to the wrong rows.
    npz_files = sorted(feat_dir.glob("*.npz"))
    n_files   = len(npz_files)
    if n_files == 0:
        raise RuntimeError(f"No npz files found under {feat_dir}")

    # First pass: per-file segment count, also = row count in features.py
    print(f"Pass 1/2: counting segments across {n_files} files ...")
    counts = np.zeros(n_files, dtype=np.int32)
    file_ids = []
    for i, p in enumerate(npz_files):
        with np.load(p, allow_pickle=True) as f:
            counts[i] = int(f["melspect_mean"].shape[0])
        file_ids.append(p.stem)

    total = int(counts.sum())
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    print(f"  -> {total:,} segments total "
          f"(expected from splits.npz: 168,239)")

    # Allocate memmap up-front
    print(f"Pass 2/2: extracting log-mel and writing memmap ({dtype.__name__}) ...")
    cache = np.lib.format.open_memmap(
        cache_path, mode="w+", dtype=dtype,
        shape=(total, N_MELS, SEG_WIN_FRAMES),
    )

    iterator = enumerate(npz_files)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=n_files,
                            desc="mel-spec", unit="file")
        except ImportError:
            pass

    errors = []
    for i, npz_path in iterator:
        file_id  = npz_path.stem
        wav_path = audio_dir / f"{file_id}.wav"
        T_exp    = int(counts[i])
        try:
            wav, _ = librosa.load(wav_path, sr=SR)
            log_mel = extract_logmel(wav)
            segs    = segment_logmel(log_mel, T_exp)
            cache[offsets[i]:offsets[i] + T_exp] = segs.astype(dtype)
        except Exception as e:
            errors.append((file_id, str(e)))
            cache[offsets[i]:offsets[i] + T_exp] = 0
    cache.flush()

    np.savez(meta_path,
             file_ids=np.array(file_ids, dtype="<U16"),
             n_segments=counts,
             offsets=offsets,
             total_segments=np.int64(total),
             sr=np.int32(SR),
             n_mels=np.int32(N_MELS),
             hop_length=np.int32(HOP_LENGTH),
             seg_win_frames=np.int32(SEG_WIN_FRAMES))

    size_gb = cache_path.stat().st_size / 1e9
    print(f"Done. Cache: {cache_path} ({size_gb:.2f} GB)")
    print(f"Meta:  {meta_path}")
    if errors:
        print(f"WARNING: {len(errors)} file(s) failed (zero-filled): "
              f"{errors[:3]}{' ...' if len(errors) > 3 else ''}")


# ---------------------------------------------------------------------------
# Read API
# ---------------------------------------------------------------------------
def load_cache(
    cache_path: Optional[Path] = None,
    meta_path:  Optional[Path] = None,
) -> Tuple[np.ndarray, dict]:
    """Return (memmap, meta_dict). Memmap is read-only -- do not write."""
    cache_path = Path(cache_path) if cache_path else _default_cache_path()
    meta_path  = Path(meta_path)  if meta_path  else _default_meta_path()

    if not cache_path.is_file():
        raise FileNotFoundError(
            f"mel-spec cache not found: {cache_path}\n"
            f"Build it first with `from melspec import build_cache; build_cache()`."
        )

    cache = np.load(cache_path, mmap_mode="r")
    with np.load(meta_path) as f:
        meta = {k: f[k] for k in f.files}
    return cache, meta


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
try:
    import torch as _torch
    from torch.utils.data import Dataset as _TorchDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    _TorchDataset = object  # type: ignore


class MelSpecDataset(_TorchDataset):
    """Torch Dataset that lazily reads mel-spec patches from the memmap cache.

    Parameters
    ----------
    idx : (N,) int64
        Global segment indices into the cache. Typically splits.train_idx etc.
    Y : (N, C) array_like
        Multi-label targets aligned 1:1 with idx.
    cache_path : Optional[Path]
        Override location of the memmap cache.
    transform : callable, optional
        Applied to each (1, n_mels, n_frames) tensor. Use for SpecAugment etc.
    normalize_global : bool
        If True, apply a single global z-score using the dataset-level
        mean/std stored in melspecs_meta.npz (set by `compute_global_stats`).
        If False, return the raw log-mel value.

    Top-level class so DataLoader workers can pickle it (num_workers > 0).
    """

    def __init__(self, idx, Y, cache_path=None, transform=None,
                 normalize_global: bool = True):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "torch is not installed. `pip install torch` to use MelSpecDataset."
            )
        cache_path = Path(cache_path) if cache_path else _default_cache_path()
        meta_path  = _default_meta_path()
        with np.load(meta_path) as f:
            stats_mean = float(f["global_mean"]) if "global_mean" in f.files else None
            stats_std  = float(f["global_std"])  if "global_std"  in f.files else None

        if normalize_global and (stats_mean is None or stats_std is None):
            raise RuntimeError(
                "Global stats not present in melspecs_meta.npz. Run "
                "`compute_global_stats()` first, or pass normalize_global=False."
            )

        self.idx        = np.asarray(idx, dtype=np.int64)
        self.Y          = _torch.as_tensor(np.asarray(Y), dtype=_torch.float32)
        self.cache_path = cache_path
        self.normalize  = normalize_global
        self.mean       = stats_mean
        self.std        = stats_std
        self.transform  = transform
        self._cache = None         # opened lazily per worker

    def _ensure_cache(self):
        if self._cache is None:
            self._cache = np.load(self.cache_path, mmap_mode="r")
        return self._cache

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        cache = self._ensure_cache()
        g = int(self.idx[i])
        mel = np.array(cache[g], dtype=np.float32)         # (n_mels, n_frames)
        if self.normalize:
            mel = (mel - self.mean) / self.std
        x = _torch.from_numpy(mel).unsqueeze(0)            # (1, n_mels, n_frames)
        if self.transform is not None:
            x = self.transform(x)
        return x, self.Y[i]


# ---------------------------------------------------------------------------
# Global stats helper (run once after build_cache)
# ---------------------------------------------------------------------------
def compute_global_stats(
    cache_path: Optional[Path] = None,
    meta_path:  Optional[Path] = None,
    sample_size: int = 20000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Estimate global mean / std from a random sample of cache rows.

    Reads `sample_size` random segments from the cache, computes the
    elementwise mean/std, and writes them back into meta_path under keys
    `global_mean` / `global_std`. These are then used by
    MelSpecDataset(..., normalize_global=True).
    """
    cache_path = Path(cache_path) if cache_path else _default_cache_path()
    meta_path  = Path(meta_path)  if meta_path  else _default_meta_path()
    cache = np.load(cache_path, mmap_mode="r")

    rng = np.random.default_rng(seed)
    n_total = cache.shape[0]
    sample_size = min(sample_size, n_total)
    rows = rng.choice(n_total, size=sample_size, replace=False)
    rows.sort()                                      # sequential read = faster
    chunk = np.array(cache[rows], dtype=np.float32)  # (S, n_mels, n_frames)

    mean = float(chunk.mean())
    std  = float(chunk.std())

    # Re-save meta with stats added
    with np.load(meta_path) as f:
        meta = {k: f[k] for k in f.files}
    meta["global_mean"] = np.float32(mean)
    meta["global_std"]  = np.float32(std)
    np.savez(meta_path, **meta)

    print(f"Global stats (over {sample_size:,} segments): "
          f"mean={mean:.4f}, std={std:.4f} -> saved to {meta_path}")
    return mean, std


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="Run build_cache() then compute_global_stats()")
    ap.add_argument("--stats-only", action="store_true",
                    help="Compute / refresh global stats only")
    args = ap.parse_args()

    if args.rebuild:
        build_cache()
        compute_global_stats()
    elif args.stats_only:
        compute_global_stats()
    else:
        print("Usage: python -m melspec --rebuild  (or --stats-only)")
