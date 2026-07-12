#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import librosa
import numpy as np
from pathlib import Path

import tqdm


# =========================
# Basic I/O
# =========================

def get_waveform(file_path, sr=16000):
    wav, _ = librosa.load(file_path, sr=sr)
    return wav


def frame_times(n_frames, hop_length, sr):
    """
    Returns timestamp (seconds) for each frame center.
    """
    return np.arange(n_frames) * hop_length / sr


# =========================
# Time-domain features
# =========================

def get_time_domain_features(
        wav,
        sr=16000,
        win_length=512,  # 32 ms
        hop_length=160,  # 10 ms
):
    zcr = librosa.feature.zero_crossing_rate(
        wav,
        frame_length=win_length,
        hop_length=hop_length,
        center=False
    )

    zcr = zcr.T
    times = frame_times(zcr.shape[0], hop_length, sr)

    return {
        'zcr': zcr,
        'zcr_times': times
    }


# =========================
# Frequency-domain features
# =========================

def get_frequency_domain_features(
    wav,
    sr=16000,
    win_length=512, # 32 ms
    hop_length=160, # 10 ms
    n_mels=128,
    n_mfccs=32,
    fmin=50,
    fmax=None
):
    spect = np.abs(
        librosa.stft(
            wav,
            n_fft=win_length,
            hop_length=hop_length,
            center=False
        )
    )

    melspect = librosa.feature.melspectrogram(
        S=spect,
        sr=sr,
        n_fft=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        norm='slaney'
    )

    mfcc = librosa.feature.mfcc(S=np.log(melspect + 1e-5), n_mfcc=n_mfccs)
    mfcc_d = librosa.feature.delta(mfcc)
    mfcc_d2 = librosa.feature.delta(mfcc, order=2)

    melspect = np.log1p(1e5 * melspect)

    flatness = librosa.feature.spectral_flatness(S=spect)
    centroid = librosa.feature.spectral_centroid(S=spect, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(S=spect, centroid=centroid)
    contrast = librosa.feature.spectral_contrast(S=spect)
    rolloff_low = librosa.feature.spectral_rolloff(S=spect, sr=sr, roll_percent=0.1)
    rolloff_high = librosa.feature.spectral_rolloff(S=spect, sr=sr, roll_percent=0.9)

    flux = np.sqrt(np.sum((spect[:, 1:] - spect[:, :-1])**2, axis=0))
    flux = np.concatenate(([flux[0]], flux))

    energy = spect.sum(axis=0)
    power = (spect**2).sum(axis=0)

    n_frames = spect.shape[1]
    times = frame_times(n_frames, hop_length, sr)

    return {
        # 'spectrogram': spect.T,
        'melspect': melspect.T,
        'mfcc': mfcc.T,
        'mfcc_d': mfcc_d.T,
        'mfcc_d2': mfcc_d2.T,
        'flux': flux[:, None],
        'flatness': flatness.T,
        'centroid': centroid.T,
        'bandwidth': bandwidth.T,
        'contrast': contrast.T,
        'rolloff_low': rolloff_low.T,
        'rolloff_high': rolloff_high.T,
        'energy': energy[:, None],
        'power': power[:, None],
        'freq_times': times
    }

def aggregate_features_over_time(
    features,
    window_size=1.0,
    hop_size=0.5
):
    """
    Aggregate frame-based features into segment-level statistics.

    Parameters
    ----------
    features : dict
        Output from compute_all_features()

    window_size : float
        Window duration in seconds

    hop_size : float
        Hop between windows in seconds

    Returns
    -------
    dict
        {
            feature_mean: [n_windows, dim],
            feature_std:  [n_windows, dim],
            feature_min:  [n_windows, dim],
            feature_max:  [n_windows, dim]
        }
    """

    aggregated = {}

    # identify feature groups by corresponding *_times
    time_keys = [k for k in features if k.endswith('_times')]

    for time_key in time_keys:
        base = time_key[:-6]   # remove "_times"
        times = features[time_key]

        # map naming convention
        if base == 'freq':
            candidate_features = [
                'melspect', 'mfcc', 'mfcc_d', 'mfcc_d2',
                'flux', 'flatness', 'centroid', 'bandwidth',
                'contrast', 'rolloff_low', 'rolloff_high',
                'energy', 'power'
            ]
        elif base == 'zcr':
            candidate_features = ['zcr']
        elif base == 'embedding':
            candidate_features = ['embedding']
        else:
            continue

        for feat_name in candidate_features:
            if feat_name not in features:
                continue

            x = features[feat_name]

            # ensure 2D
            if x.ndim == 1:
                x = x[:, None]

            n_frames = len(times)

            starts = np.arange(
                0,
                times[-1] - window_size + 1e-8,
                hop_size
            )

            means = []
            stds = []
            mins = []
            maxs = []

            for start in starts:
                end = start + window_size

                idx = np.where((times >= start) & (times < end))[0]

                # drop incomplete windows
                if len(idx) == 0:
                    continue

                # require full coverage
                if times[idx[-1]] < end - (times[1] - times[0]):
                    continue

                seg = x[idx]

                means.append(seg.mean(axis=0))
                stds.append(seg.std(axis=0))
                mins.append(seg.min(axis=0))
                maxs.append(seg.max(axis=0))

            if len(means) == 0:
                continue

            aggregated[f"{feat_name}_mean"] = np.stack(means)
            aggregated[f"{feat_name}_std"] = np.stack(stds)
            aggregated[f"{feat_name}_min"] = np.stack(mins)
            aggregated[f"{feat_name}_max"] = np.stack(maxs)
    aggregated["start_time"] = starts
    aggregated["end_time"] = starts + window_size
    return aggregated


# # =========================
# # Deep embedding features
# # =========================
# from panns_inference import AudioTagging
# import torch

# # =========================
# # Load PANNs once globally
# # =========================

# audio_model = AudioTagging(checkpoint_path=None)
if False:  # disabled — requires panns_inference
    def get_embedding_features(
        wav,
        sr=16000,
        window_size=1.0,
        hop_size=0.5
    ):
        """
        PANNs deep embeddings:
        - 1 second windows
        - 0.5 second hop

        Output:
        embedding: [N, 2048]
        """

        if sr != 16000:
            raise ValueError("PANNs expects sr=16000")

        win_samples = int(window_size * sr)
        hop_samples = int(hop_size * sr)

        starts = np.arange(
            0,
            len(wav) - win_samples + 1,
            hop_samples
        )

        embeddings = []

        for start in starts:
            segment = wav[start:start + win_samples]

            segment = segment[None, :]  # batch dimension

            _, emb = audio_model.inference(segment)

            embeddings.append(emb[0])

        embeddings = np.stack(embeddings)

        times = starts / sr

        return {
            'embedding': embeddings
        }

# =========================
# Single file extraction
# =========================

def compute_all_features(file_path, sr=16000):
    wav = get_waveform(file_path, sr=sr)

    features = {}

    features.update(
        aggregate_features_over_time(get_time_domain_features(wav, sr))
    )
    features.update(
        aggregate_features_over_time(get_frequency_domain_features(wav, sr))
    )
    #features.update(
    #    get_embedding_features(wav, sr)
    #)
    # features["embedding"] = features["embedding"][:features["power_max"].shape[0]]  # ensure same length as other features
    assert all([f.shape[0] == features['power_max'].shape[0] for f in features.values()])

    return features


# =========================
# Batch processing
# =========================

def process_batch(source_files, output_files, sr=16000):
    if len(source_files) != len(output_files):
        raise ValueError("source_files and output_files must have same length")

    for src, out in tqdm.tqdm(zip(source_files, output_files), desc="Converting audio to features", total=len(source_files)):

        features = compute_all_features(src, sr=sr)

        Path(out).parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(out, **features)