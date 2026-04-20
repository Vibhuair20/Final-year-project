"""Load a fully-trained Layer3Detector from models/ and models/text_vocab.json."""

import json
from pathlib import Path
import numpy as np
from .detector import Layer3Detector
from .text_classifier import TextClassifier
from .acoustic_classifier import AcousticClassifier
from .fusion import LateFusion
from .layer3_constants import IDX_TO_LABEL


def load_trained_detector(models_dir: str = "models") -> Layer3Detector:
    mdir = Path(models_dir)
    text_clf = TextClassifier.load(str(mdir / "text_classifier.joblib"))
    ac_clf = AcousticClassifier.load(str(mdir / "acoustic_classifier.joblib"))
    fusion = LateFusion.load(str(mdir / "fusion_classifier.joblib"))

    vocab_path = mdir / "text_vocab.json"
    vocabulary = []
    coefs = None
    if vocab_path.exists():
        payload = json.loads(vocab_path.read_text())
        vocabulary = payload["vocabulary"]
        classes = payload["classes"]
        coef_matrix = np.array(payload["coef"])
        phishing_row = None
        for i, cls_idx in enumerate(classes):
            if IDX_TO_LABEL[int(cls_idx)] == "phishing":
                phishing_row = i
                break
        coefs = coef_matrix[phishing_row] if phishing_row is not None else coef_matrix.mean(axis=0)

    return Layer3Detector(
        text_classifier=text_clf,
        acoustic_classifier=ac_clf,
        fusion=fusion,
        text_vocabulary=vocabulary,
        text_class_coefficients=coefs,
    )
