#!/usr/bin/env python3

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.audio_processor import AudioProcessor
from src.layer2_processor import Layer2Processor
from src.detector_loader import load_trained_detector


def process_full_pipeline(audio_path: str):
    print("\n" + "=" * 60)
    print("🎯 FULL PIPELINE: Layer 1 + Layer 2 + Layer 3")
    print("=" * 60)

    print("\n📍 LAYER 1: Data Acquisition")
    print("-" * 60)
    l1 = AudioProcessor().process_file(audio_path)
    if l1.status != "ready_for_processing":
        print(f"❌ Layer 1 failed: {l1.error_message}")
        return
    print(f"✅ Metadata: duration={l1.metadata.duration_seconds}s, sr={l1.metadata.sample_rate_hz}Hz, ch={l1.metadata.channels}")
    l1_out = f"data/output/{l1.file_id}.json"
    os.makedirs("data/output", exist_ok=True)
    l1.save_to_file(l1_out)
    print(f"   Saved: {l1_out}")

    print("\n📍 LAYER 2: Signal & Text Processing")
    print("-" * 60)
    l2 = Layer2Processor().process(audio_path, file_id=l1.file_id)
    print(f"✅ Pitch {l2.features.pitch_mean:.1f}Hz | Rate {l2.features.speech_rate:.2f} syl/s | Pauses {l2.features.pause_count}")
    print(f"✅ Transcript: {l2.transcript.word_count} words — {l2.transcript.full_text[:80]}")
    l2_out = f"data/layer2_output/{l2.file_id}.json"
    os.makedirs("data/layer2_output", exist_ok=True)
    l2.save_to_file(l2_out)
    print(f"   Saved: {l2_out}")

    print("\n📍 LAYER 3: Fraud Detection")
    print("-" * 60)
    try:
        detector = load_trained_detector()
        l3 = detector.predict(l2)
        print(f"✅ Verdict: {l3.label}  (confidence {l3.confidence:.2%})")
        if l3.top_keywords:
            print(f"   Keywords: {', '.join(l3.top_keywords)}")
        if l3.top_acoustic_drivers:
            print(f"   Acoustic drivers: {', '.join(l3.top_acoustic_drivers)}")
        l3_out = f"data/layer3_output/{l1.file_id}.json"
        os.makedirs("data/layer3_output", exist_ok=True)
        l3.save_to_file(l3_out)
        print(f"   Saved: {l3_out}")
        l3_dict = l3.to_dict()
    except FileNotFoundError as e:
        print(f"⚠️  Layer 3 models not found — skipping. ({e})")
        print("   Train with: python training/train_text.py && python training/train_acoustic.py && python training/train_fusion.py")
        l3_dict = None

    combined = {"layer1": l1.to_dict(), "layer2": l2.to_dict(), "layer3": l3_dict}
    combined_path = f"data/combined/{l1.file_id}_combined.json"
    os.makedirs("data/combined", exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n📦 Combined: {combined_path}")

    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]
    if not os.path.exists(audio_file):
        print(f"❌ File not found: {audio_file}")
        sys.exit(1)

    try:
        process_full_pipeline(audio_file)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
