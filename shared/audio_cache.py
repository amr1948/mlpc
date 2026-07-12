"""
Per-file 16 kHz audio cache for models that need raw waveforms (AST).

Why a separate cache (not the mel-spec one)
-------------------------------------------
AST's pretrained feature extractor expects raw 16 kHz audio and applies its
own mel-spec + normalization internally (matched to AudioSet stats). Trying
to reuse our log1p(1e5 * mel) cache would require un-doing our normalization
AND converting between two different log conventions -- error-prone.
Cleaner to give AST raw audio and let its tokenizer do its thing.

Layout
------
data/audio_16k.bin           int16 memmap, shape (n_samples_total,)
                             concatenation of all files at 16 kHz mono
data/audio_16k_meta.npz      file_ids, file_offsets (start sample of each file),
                             file_lengths, segment_starts (global segment idx ->
                             starting sample in the cache).

Disk: ~2.7 GB for the full dataset. Build time: 10-15 minutes once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from data import find_root, load_config


# AST and our mel-spec pipeline both want 16 kHz mono audio.
SR             = 16000
SEG_WIN_S      = 1.0
SEG_HOP_S      = 0.5
SEG_WIN_SAMPLES = int(SEG_WIN_S * SR)   # 16000
SEG_HOP_SAMPLES = int(SEG_HOP_S * SR)   # 8000


def _default_cache_path() -> Path:
    return find_root() / "data" / "audio_16k.bin"


def _default_meta_path() -> Path:
    return find_root() / "data" / "audio_16k_meta.npz"


def build_audio_cache(
    cache_path: Optional[Path] = None,
    meta_path:  Optional[Path] = None,
    progress: bool = True,
) -> None:
    """Resample all wav files to 16 kHz mono int16 and concatenate to one file."""
    import librosa
    import soundfile as sf

    root      = find_root()
    cfg       = load_config()
    audio_dir = root / cfg["paths"]["orig_audio"]
    feat_dir  = root / cfg["paths"]["orig_npz"]

    cache_path = Path(cache_path) if cache_path else _default_cache_path()
    meta_path  = Path(meta_path)  if meta_path  else _default_meta_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Canonical sorted order, same as features.py
    npz_files = sorted(feat_dir.glob("*.npz"))
    n_files = len(npz_files)
    if n_files == 0:
        raise RuntimeError(f"No npz files under {feat_dir}")

    # Pass 1: compute lengths (load+resample on the fly; cheaper to overlap).
    # But we need to know total samples BEFORE allocating the memmap. To avoid
    # a double load, we use librosa.get_duration on the wav header first.
    print(f"Pass 1/2: measuring audio lengths across {n_files} files ...")
    file_ids, file_lengths = [], []
    iterator = npz_files
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(npz_files, desc="measure", unit="file")
        except ImportError:
            pass
    for p in iterator:
        fid = p.stem
        wav_path = audio_dir / f"{fid}.wav"
        # Header read -> fast
        info = sf.info(wav_path)
        # Length in 16 kHz samples after resample
        L = int(round(info.frames * SR / info.samplerate))
        file_ids.append(fid)
        file_lengths.append(L)
    file_lengths = np.array(file_lengths, dtype=np.int64)
    file_offsets = np.concatenate([[0], np.cumsum(file_lengths)[:-1]]).astype(np.int64)
    total_samples = int(file_lengths.sum())
    print(f"  -> total samples at 16 kHz: {total_samples:,} (~{total_samples/SR/3600:.2f} h)")

    # Allocate memmap
    cache = np.lib.format.open_memmap(
        cache_path.with_suffix(".npy"), mode="w+",
        dtype=np.int16, shape=(total_samples,),
    )

    # Pass 2: load + resample + write into memmap
    print(f"Pass 2/2: resampling and writing memmap ...")
    iterator = enumerate(npz_files)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=n_files, desc="resample", unit="file")
        except ImportError:
            pass

    errors = []
    for i, p in iterator:
        fid = p.stem
        wav_path = audio_dir / f"{fid}.wav"
        try:
            wav, _ = librosa.load(wav_path, sr=SR, mono=True)
            # float [-1, 1] -> int16
            wav_i16 = np.clip(wav * 32768.0, -32768, 32767).astype(np.int16)
            cache[file_offsets[i]:file_offsets[i] + file_lengths[i]] = (
                wav_i16[:file_lengths[i]]
            )
        except Exception as e:
            errors.append((fid, str(e)))
            cache[file_offsets[i]:file_offsets[i] + file_lengths[i]] = 0
    cache.flush()

    # Build segment_starts: for each global segment idx, the start sample.
    # This depends on per-file segment counts (taken from features.py output).
    seg_counts = np.zeros(n_files, dtype=np.int32)
    for i, p in enumerate(npz_files):
        with np.load(p, allow_pickle=True) as f:
            seg_counts[i] = int(f["melspect_mean"].shape[0])
    total_segments = int(seg_counts.sum())

    segment_starts = np.empty(total_segments, dtype=np.int64)
    seg_cursor = 0
    for i in range(n_files):
        for t in range(seg_counts[i]):
            segment_starts[seg_cursor] = file_offsets[i] + t * SEG_HOP_SAMPLES
            seg_cursor += 1
    assert seg_cursor == total_segments

    np.savez(meta_path,
             file_ids=np.array(file_ids, dtype="<U16"),
             file_offsets=file_offsets,
             file_lengths=file_lengths,
             seg_counts=seg_counts,
             segment_starts=segment_starts,
             total_segments=np.int64(total_segments),
             total_samples=np.int64(total_samples),
             sr=np.int32(SR),
             seg_win_samples=np.int32(SEG_WIN_SAMPLES))

    final_npy = cache_path.with_suffix(".npy")
    size_gb = final_npy.stat().st_size / 1e9
    print(f"Done. Cache: {final_npy} ({size_gb:.2f} GB)")
    print(f"Meta:  {meta_path}  ({total_segments:,} segment starts)")
    if errors:
        print(f"WARNING: {len(errors)} file(s) failed (zero-filled).")


# ---------------------------------------------------------------------------
# Read API
# ---------------------------------------------------------------------------
def load_audio_cache(
    cache_path: Optional[Path] = None,
    meta_path:  Optional[Path] = None,
) -> Tuple[np.ndarray, dict]:
    """Return (memmap (int16), meta_dict)."""
    cache_path = Path(cache_path) if cache_path else _default_cache_path()
    npy_path = cache_path.with_suffix(".npy") if cache_path.suffix != ".npy" else cache_path
    meta_path  = Path(meta_path)  if meta_path  else _default_meta_path()
    if not npy_path.is_file():
        raise FileNotFoundError(
            f"audio cache not found: {npy_path}\n"
            f"Build it first with `from audio_cache import build_audio_cache; "
            f"build_audio_cache()`."
        )
    cache = np.load(npy_path, mmap_mode="r")
    with np.load(meta_path) as f:
        meta = {k: f[k] for k in f.files}
    return cache, meta


def get_segment_audio(global_idx: int,
                      cache: np.ndarray,
                      segment_starts: np.ndarray) -> np.ndarray:
    """Return the 1-s waveform (16000 samples, float32 in [-1, 1]) for a
    global segment index. Pads with zeros at file-end."""
    start = int(segment_starts[global_idx])
    end = start + SEG_WIN_SAMPLES
    seg = np.asarray(cache[start:end], dtype=np.int16)
    if len(seg) < SEG_WIN_SAMPLES:                       # end of file
        seg = np.concatenate([seg, np.zeros(SEG_WIN_SAMPLES - len(seg),
                                            dtype=np.int16)])
    return seg.astype(np.float32) / 32768.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    if args.build:
        build_audio_cache()
    else:
        print("Usage: python -m audio_cache --build")
