"""Evaluate Layer 3 on the test split. Produces:
  - reports/classification_report.txt
  - reports/confusion_matrix.png
  - reports/ablation.json  (text-only / acoustic-only / fused macro-F1)
"""

import sys
import os
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.text_classifier import TextClassifier
from src.acoustic_classifier import AcousticClassifier
from src.fusion import LateFusion
from src.layer3_constants import CLASS_LABELS
from training.load_processed import load_split


TEST_DIR = Path("data/tele_antifraud/processed/test")
REPORT_DIR = Path("reports")


def _argmax_label(probs):
    return max(probs, key=probs.get)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    texts, features, labels = load_split(TEST_DIR)
    print(f"Loaded {len(labels)} test samples")

    text_clf = TextClassifier.load("models/text_classifier.joblib")
    ac_clf = AcousticClassifier.load("models/acoustic_classifier.joblib")
    fusion = LateFusion.load("models/fusion_classifier.joblib")

    text_probs = [text_clf.predict_proba(t) for t in tqdm(texts, desc="text")]
    ac_probs = [ac_clf.predict_proba(f) for f in tqdm(features, desc="acoustic")]
    fused_probs = [fusion.predict_proba(t, a) for t, a in zip(text_probs, ac_probs)]

    text_pred = [_argmax_label(p) for p in text_probs]
    ac_pred = [_argmax_label(p) for p in ac_probs]
    fused_pred = [_argmax_label(p) for p in fused_probs]

    ablation = {
        "text_only_macro_f1": float(f1_score(labels, text_pred, average="macro", labels=CLASS_LABELS, zero_division=0)),
        "acoustic_only_macro_f1": float(f1_score(labels, ac_pred, average="macro", labels=CLASS_LABELS, zero_division=0)),
        "fused_macro_f1": float(f1_score(labels, fused_pred, average="macro", labels=CLASS_LABELS, zero_division=0)),
    }
    (REPORT_DIR / "ablation.json").write_text(json.dumps(ablation, indent=2))
    print("Ablation:", json.dumps(ablation, indent=2))

    report = classification_report(labels, fused_pred, labels=CLASS_LABELS, zero_division=0)
    (REPORT_DIR / "classification_report.txt").write_text(report)
    print(report)

    cm = confusion_matrix(labels, fused_pred, labels=CLASS_LABELS)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_LABELS)))
    ax.set_yticks(range(len(CLASS_LABELS)))
    ax.set_xticklabels(CLASS_LABELS, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Layer 3 Confusion Matrix (Fused)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=120)
    print(f"Wrote {REPORT_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
