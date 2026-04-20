import numpy as np
from src.layer2_models import AudioFeatures
from src.feature_vector import flatten_features, FEATURE_NAMES


def _make_features():
    return AudioFeatures(
        mfcc_mean=[0.1] * 13, mfcc_std=[0.2] * 13,
        pitch_mean=150.0, pitch_std=10.0, pitch_variance=100.0,
        energy_mean=0.05, energy_std=0.01,
        zero_crossing_rate=0.12,
        spectral_centroid_mean=2000.0, spectral_rolloff_mean=4000.0,
        pause_count=5, pause_duration_mean=0.3,
        speech_rate=4.2, total_speech_duration=20.0, total_pause_duration=1.5,
    )


def test_flatten_features_produces_39_element_vector():
    v = flatten_features(_make_features())
    assert isinstance(v, np.ndarray)
    assert v.shape == (39,)


def test_feature_names_match_vector_length():
    assert len(FEATURE_NAMES) == 39
    assert "pitch_mean" in FEATURE_NAMES
    assert "mfcc_mean_0" in FEATURE_NAMES
    assert "mfcc_std_12" in FEATURE_NAMES
