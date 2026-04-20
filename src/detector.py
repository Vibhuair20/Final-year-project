from typing import List, Optional
import numpy as np
from .text_classifier import TextClassifier
from .acoustic_classifier import AcousticClassifier
from .fusion import LateFusion
from .layer2_models import ProcessedAudio
from .layer3_models import DetectionResult
from .explainability import top_acoustic_drivers, top_keywords_from_tfidf


class Layer3Detector:
    def __init__(
        self,
        text_classifier: TextClassifier,
        acoustic_classifier: AcousticClassifier,
        fusion: LateFusion,
        text_vocabulary: Optional[List[str]] = None,
        text_class_coefficients: Optional[np.ndarray] = None,
    ):
        self.text_classifier = text_classifier
        self.acoustic_classifier = acoustic_classifier
        self.fusion = fusion
        self.text_vocabulary = text_vocabulary or []
        self.text_class_coefficients = text_class_coefficients

    def predict(self, processed_audio: ProcessedAudio) -> DetectionResult:
        text = processed_audio.transcript.full_text
        features = processed_audio.features

        text_probs = self.text_classifier.predict_proba(text)
        acoustic_probs = self.acoustic_classifier.predict_proba(features)
        fused_probs = self.fusion.predict_proba(text_probs, acoustic_probs)

        label = max(fused_probs, key=fused_probs.get)
        confidence = fused_probs[label]

        keywords: List[str] = []
        if self.text_vocabulary and self.text_class_coefficients is not None:
            keywords = top_keywords_from_tfidf(
                text, self.text_vocabulary, self.text_class_coefficients, k=5,
            )

        drivers: List[str] = []
        try:
            importances = self.acoustic_classifier.feature_importances()
            drivers = top_acoustic_drivers(features, importances, k=5)
        except RuntimeError:
            drivers = []

        return DetectionResult(
            file_id=processed_audio.file_id,
            label=label,
            confidence=confidence,
            class_probabilities=fused_probs,
            per_modality={
                "text": text_probs,
                "acoustic": acoustic_probs,
                "fused": fused_probs,
            },
            top_keywords=keywords,
            top_acoustic_drivers=drivers,
        )
