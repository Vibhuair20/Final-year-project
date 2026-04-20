import numpy as np
from src.detector import Layer3Detector
from src.text_classifier import TextClassifier
from src.acoustic_classifier import AcousticClassifier
from src.fusion import LateFusion
from src.layer2_models import AudioFeatures, Transcript, ProcessedAudio
from src.layer3_constants import CLASS_LABELS, N_CLASSES


class _FakeEncoder:
    def encode(self, texts, show_progress_bar=False, convert_to_numpy=True):
        rng = np.random.RandomState(42)
        return rng.randn(len(texts), 16).astype(np.float32)


def _feat(seed: int):
    rng = np.random.RandomState(seed)
    return AudioFeatures(
        mfcc_mean=rng.randn(13).tolist(), mfcc_std=rng.rand(13).tolist(),
        pitch_mean=150.0, pitch_std=10.0, pitch_variance=100.0,
        energy_mean=0.05, energy_std=0.01, zero_crossing_rate=0.12,
        spectral_centroid_mean=2000.0, spectral_rolloff_mean=4000.0,
        pause_count=5, pause_duration_mean=0.3,
        speech_rate=4.2, total_speech_duration=20.0, total_pause_duration=1.5,
    )


def _train_fake_detector():
    text_clf = TextClassifier(encoder=_FakeEncoder())
    ac_clf = AcousticClassifier()
    texts = [f"fake transcript {i}" for i in range(80)]
    feats = [_feat(i) for i in range(80)]
    labels = [CLASS_LABELS[i % N_CLASSES] for i in range(80)]
    text_clf.train(texts, labels)
    ac_clf.train(feats, labels)

    text_probs = [text_clf.predict_proba(t) for t in texts]
    ac_probs = [ac_clf.predict_proba(f) for f in feats]
    fusion = LateFusion()
    fusion.fit(text_probs, ac_probs, labels)

    vocab = ["verify", "account", "wire", "urgent", "hello"]
    coefs = np.array([3.0, 2.0, 4.0, 3.5, -2.0])
    return Layer3Detector(
        text_classifier=text_clf,
        acoustic_classifier=ac_clf,
        fusion=fusion,
        text_vocabulary=vocab,
        text_class_coefficients=coefs,
    )


def test_predict_returns_detection_result():
    detector = _train_fake_detector()
    pa = ProcessedAudio(
        file_id="test_001",
        audio_path="/dev/null",
        features=_feat(999),
        transcript=Transcript(
            full_text="verify your account wire urgent",
            segments=[], word_count=5, language="en-US",
        ),
        processing_timestamp="2026-04-20T00:00:00",
    )
    result = detector.predict(pa)
    assert result.file_id == "test_001"
    assert result.label in CLASS_LABELS
    assert 0.0 <= result.confidence <= 1.0
    assert set(result.class_probabilities.keys()) == set(CLASS_LABELS)
    assert "text" in result.per_modality
    assert "acoustic" in result.per_modality
    assert "fused" in result.per_modality
    assert len(result.top_keywords) <= 5
    assert len(result.top_acoustic_drivers) <= 5
