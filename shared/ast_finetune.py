"""
Audio Spectrogram Transformer (AST) fine-tuning wrapper.

Uses HuggingFace's `MIT/ast-finetuned-audioset-10-10-0.4593`:
- Pretrained on AudioSet (527 classes; covers most of our 15 domestic-audio
  classes -- bell_ringing, microwave, vacuum_cleaner, etc.).
- Expects 16 kHz mono audio, internally padded/truncated to 10 s and
  converted to a 1024 x 128 log-mel spectrogram with AudioSet normalization.

Our segments are 1 s. AST will pad with silence to 10 s. This wastes ~90%
of the model's positional capacity but lets us reuse the pretrained
weights -- which is the entire point of using AST.

Two fine-tuning regimes
-----------------------
- `freeze_backbone=True`  : train classifier head only. Fast (~minutes
                            per epoch), low risk of overfit. Default.
- `freeze_backbone=False` : full fine-tune. Slow (~30+ min per epoch),
                            high VRAM, often better F1. Use only after
                            head-only confirms a useful signal.

Hardware
--------
GPU strongly recommended. Head-only on Apple MPS: ~15-25 min per epoch.
Full fine-tune on MPS would be ~2 h+ per epoch and might OOM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


AST_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"


# ---------------------------------------------------------------------------
# Model + feature extractor
# ---------------------------------------------------------------------------
def load_ast(num_labels: int = 15,
             freeze_backbone: bool = True,
             model_id: str = AST_MODEL_ID,
             cache_dir: Optional[Path] = None):
    """Return (model, feature_extractor).

    The classifier head is replaced with a fresh `nn.Linear(hidden, num_labels)`
    so we can do multi-label binary cross-entropy directly on the logits.
    """
    import torch
    from torch import nn
    from transformers import (
        ASTConfig, ASTForAudioClassification, ASTFeatureExtractor
    )

    kw = {"cache_dir": str(cache_dir)} if cache_dir else {}
    feature_extractor = ASTFeatureExtractor.from_pretrained(model_id, **kw)
    model = ASTForAudioClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,   # discard pretrained 527-way head
        **kw,
    )
    # Re-init the head deterministically (HF does it stochastically)
    torch.manual_seed(42)
    hidden = model.config.hidden_size
    model.classifier = nn.Linear(hidden, num_labels)

    if freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("classifier"):
                p.requires_grad_(False)

    return model, feature_extractor


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Dataset: pulls raw audio from the 16 kHz cache, applies AST's extractor
# ---------------------------------------------------------------------------
try:
    import torch as _torch
    from torch.utils.data import Dataset as _TorchDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    _TorchDataset = object  # type: ignore


class ASTAudioDataset(_TorchDataset):
    """Streams 1 s waveforms from the 16 kHz cache and runs AST's feature
    extractor per-item.

    Used for embedding extraction (the linear-probing path) and would also
    work for end-to-end fine-tuning if you have the compute budget for it.
    """

    def __init__(self, idx, Y, feature_extractor,
                 audio_cache_path: Optional[Path] = None,
                 meta_path:        Optional[Path] = None):
        if not _TORCH_AVAILABLE:
            raise ImportError("torch required for ASTAudioDataset.")

        from audio_cache import _default_cache_path, _default_meta_path

        audio_cache_path = Path(audio_cache_path) if audio_cache_path else _default_cache_path()
        npy_path = audio_cache_path.with_suffix(".npy")
        meta_path = Path(meta_path) if meta_path else _default_meta_path()

        with np.load(meta_path) as f:
            self.segment_starts = np.array(f["segment_starts"])
            self.seg_win_samples = int(f["seg_win_samples"])
            self.sr = int(f["sr"])

        self.idx = np.asarray(idx, dtype=np.int64)
        self.Y = _torch.as_tensor(np.asarray(Y), dtype=_torch.float32)
        self.feature_extractor = feature_extractor
        self.cache_path = npy_path
        self._cache = None

    def _ensure_cache(self):
        if self._cache is None:
            self._cache = np.load(self.cache_path, mmap_mode="r")
        return self._cache

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        cache = self._ensure_cache()
        g = int(self.idx[i])
        s = int(self.segment_starts[g])
        e = s + self.seg_win_samples
        wav = np.asarray(cache[s:e], dtype=np.int16)
        if len(wav) < self.seg_win_samples:
            wav = np.concatenate([wav, np.zeros(self.seg_win_samples - len(wav),
                                                dtype=np.int16)])
        wav_f = wav.astype(np.float32) / 32768.0           # (16000,) in [-1, 1]

        # Returns a dict {'input_values': (1, T, n_mels)}; squeeze batch dim
        feats = self.feature_extractor(
            wav_f, sampling_rate=self.sr, return_tensors="pt"
        )
        x = feats["input_values"].squeeze(0)               # (T, n_mels)
        return x, self.Y[i]


# ---------------------------------------------------------------------------
# Embedding extraction (linear-probing pipeline)
# ---------------------------------------------------------------------------
def extract_embeddings(
    model, feature_extractor, idx, Y,
    *,
    audio_cache_path: Optional[Path] = None,
    batch_size: int = 32,
    device=None,
    progress: bool = True,
    num_workers: int = 0,
):
    """Forward-pass AST on each segment and mean-pool the encoder's token
    sequence into a 768-D embedding.

    Returns
    -------
    embeddings : (N, 768) float32
    Y_out      : (N, C)   int8   (same as input, returned for convenience)
    """
    import torch
    from torch.utils.data import DataLoader

    if device is None:
        device = (torch.device("cuda") if torch.cuda.is_available()
                  else torch.device("mps")
                  if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                  else torch.device("cpu"))

    model = model.to(device).eval()
    ds = ASTAudioDataset(idx, Y, feature_extractor,
                         audio_cache_path=audio_cache_path)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, drop_last=False)

    iterator = dl
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(dl, total=len(dl), desc="AST embed",
                            unit="batch")
        except ImportError:
            pass

    embs = []
    Ys = []
    with torch.no_grad():
        for x, y in iterator:
            x = x.to(device, non_blocking=True)
            out = model.audio_spectrogram_transformer(input_values=x)
            # out.last_hidden_state: (B, n_tokens, 768); mean over tokens
            e = out.last_hidden_state.mean(dim=1)              # (B, 768)
            embs.append(e.cpu().numpy().astype(np.float32))
            Ys.append(y.numpy().astype(np.int8))
    return np.concatenate(embs, axis=0), np.concatenate(Ys, axis=0)


def save_embeddings(path: Path, embeddings: np.ndarray, Y: np.ndarray,
                    idx: np.ndarray, split_name: str) -> None:
    """Save embeddings to a single .npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path,
             embeddings=embeddings.astype(np.float32),
             Y=Y.astype(np.int8),
             idx=np.asarray(idx, dtype=np.int64),
             split_name=split_name)


def load_embeddings(path: Path):
    """Load embeddings produced by save_embeddings."""
    with np.load(path, allow_pickle=True) as f:
        return {
            "embeddings": f["embeddings"],
            "Y":          f["Y"],
            "idx":        f["idx"],
            "split_name": str(f["split_name"]),
        }
