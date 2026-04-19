# Layer 3: Multimodal Fraud Detection — Design

**Goal:** Classify a `ProcessedAudio` (L1 metadata + L2 acoustic features + transcript) into one of 8 categories: `legitimate`, `phishing`, `banking`, `investment`, `kidnapping`, `lottery`, `customer_service`, `identity_theft`.

**Dataset:** TeleAntiFraud-28k (audio + text, 7 fraud categories + legitimate).

**Approach:** Late fusion of two independent classifiers — text (sentence embeddings + Logistic Regression) and acoustic (XGBoost on `AudioFeatures`). A learned meta-classifier combines their probabilities.

## Output contract

```python
@dataclass
class DetectionResult:
    file_id: str
    label: str                         # one of 8 classes
    confidence: float                  # 0.0–1.0
    class_probabilities: Dict[str, float]   # all 8 classes
    per_modality: Dict[str, Dict[str, float]]  # {"text": {...}, "acoustic": {...}, "fused": {...}}
    top_keywords: List[str]            # top-5 tokens driving text prediction
    top_acoustic_drivers: List[str]    # top-5 acoustic features driving prediction
    version: str = "3.0"
    # + to_dict / to_json / save_to_file
```

## File structure

```
src/
  detector.py              # Layer3Detector — loads models, predicts
  layer3_models.py         # DetectionResult dataclass
  text_classifier.py       # TextClassifier (embed + LogReg)
  acoustic_classifier.py   # AcousticClassifier (XGBoost)
  fusion.py                # LateFusion meta-classifier
  explainability.py        # top-k keyword + feature-importance helpers
training/
  download_dataset.py      # TeleAntiFraud-28k fetch + cache
  prepare_data.py          # run L1+L2 on dataset audio, produce ProcessedAudio JSONs
  train_text.py            # embed transcripts, fit LogReg, save model
  train_acoustic.py        # fit XGBoost on AudioFeatures, save model
  train_fusion.py          # fit meta-classifier on validation split
  evaluate.py              # confusion matrix, per-class F1, ablation (text/acoustic/fused)
models/
  text_classifier.joblib
  acoustic_classifier.joblib
  fusion_classifier.joblib
  sentence_encoder/        # cached all-MiniLM-L6-v2
notebooks/
  train_colab.ipynb        # Colab-ready training notebook
layer3_main.py             # CLI: python layer3_main.py --input call.wav
pipeline.py                # MODIFY: call Layer3Detector after Layer 2
```

## Data pipeline

1. `download_dataset.py` — pull TeleAntiFraud-28k, split 70/15/15 (train/val/test), stratified by class.
2. `prepare_data.py` — run existing `Layer2Processor` over each audio file, cache `ProcessedAudio` JSONs to `data/tele_antifraud/processed/`. This is the slow step (ASR); cache so it runs once.

## Training pipeline (Colab free tier)

- **Text:** encode transcripts with `sentence-transformers/all-MiniLM-L6-v2` (one-time, ~10min on T4), fit `LogisticRegression(multi_class='multinomial', class_weight='balanced')`.
- **Acoustic:** fit `XGBClassifier(objective='multi:softprob')` on the flattened feature vector from `SignalProcessor` — 13 MFCC means + 13 MFCC stds + pitch (mean/std/var) + energy (mean/std) + ZCR + spectral centroid + spectral rolloff + pause count/mean + speech rate + total speech/pause duration = **39 features**. Use `sample_weight` for class balance.
- **Fusion:** for each val-set example, form `[text_probs || acoustic_probs]` (16 features), fit `LogisticRegression` — learns when to trust which modality per class.
- Save all three models to `models/` via `joblib`.

## Inference integration

`pipeline.py` becomes: L1 → L2 → L3. `Layer3Detector.predict(processed_audio)` returns a `DetectionResult`. Combined output JSON gets a new `layer3` key. `layer3_main.py` mirrors `layer2_main.py` for standalone use.

## Evaluation

`evaluate.py` produces, on the held-out test split:
- Per-class precision/recall/F1 + macro-F1
- Confusion matrix (saved as PNG)
- **Ablation table**: text-only accuracy, acoustic-only accuracy, fused accuracy — this is the capstone defense slide.

## Explainability (light)

- **Keywords:** for a predicted class, multiply the input's TF-IDF vector by LogReg coefficients, take top-5 positive contributors.
- **Acoustic drivers:** XGBoost's `feature_importances_` filtered to features present/high in this sample.

## Out of scope

- Real-time streaming, VoIP ingestion, database, HTTP API (flagged as future in `config.json`)
- Fine-tuning transformers (use embeddings only)
- Deep neural fusion (stick with late fusion + meta-classifier)
- Cross-lingual support (English only, per existing Vosk model)

## Acceptance criteria

- [ ] `python layer3_main.py --input data/input/call.wav` prints a `DetectionResult` JSON
- [ ] `python pipeline.py ...` emits combined JSON with a `layer3` section
- [ ] `python training/evaluate.py` prints ablation table + saves confusion matrix PNG
- [ ] Macro-F1 ≥ 0.70 on the test split (baseline target; adjust after first run)
- [ ] README updated with Layer 3 usage + training instructions
