import numpy as np
from typing import List
from .layer2_models import AudioFeatures


FEATURE_NAMES: List[str] = (
    [f"mfcc_mean_{i}" for i in range(13)]
    + [f"mfcc_std_{i}" for i in range(13)]
    + ["pitch_mean", "pitch_std", "pitch_variance"]
    + ["energy_mean", "energy_std"]
    + ["zero_crossing_rate"]
    + ["spectral_centroid_mean", "spectral_rolloff_mean"]
    + ["pause_count", "pause_duration_mean"]
    + ["speech_rate", "total_speech_duration", "total_pause_duration"]
)


def flatten_features(f: AudioFeatures) -> np.ndarray:
    parts = (
        list(f.mfcc_mean)
        + list(f.mfcc_std)
        + [f.pitch_mean, f.pitch_std, f.pitch_variance]
        + [f.energy_mean, f.energy_std]
        + [f.zero_crossing_rate]
        + [f.spectral_centroid_mean, f.spectral_rolloff_mean]
        + [float(f.pause_count), f.pause_duration_mean]
        + [f.speech_rate, f.total_speech_duration, f.total_pause_duration]
    )
    return np.asarray(parts, dtype=np.float64)
