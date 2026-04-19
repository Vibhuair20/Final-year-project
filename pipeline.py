#!/usr/bin/env python3

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.audio_processor import AudioProcessor
from src.layer2_processor import Layer2Processor


def process_full_pipeline(audio_path: str):
    print("\n" + "=" * 60)
    print("🎯 FULL PIPELINE: Layer 1 + Layer 2")
    print("=" * 60)
    
    print("\n📍 LAYER 1: Data Acquisition")
    print("-" * 60)
    layer1_processor = AudioProcessor()
    layer1_result = layer1_processor.process_file(audio_path)
    
    if layer1_result.status != "ready_for_processing":
        print(f"❌ Layer 1 failed: {layer1_result.error_message}")
        return
    
    print(f"✅ Metadata extracted")
    print(f"   Duration: {layer1_result.metadata.duration_seconds}s")
    print(f"   Sample Rate: {layer1_result.metadata.sample_rate_hz}Hz")
    print(f"   Channels: {layer1_result.metadata.channels}")
    
    layer1_output = f"data/output/{layer1_result.file_id}.json"
    layer1_result.save_to_file(layer1_output)
    print(f"   Saved: {layer1_output}")
    
    print("\n📍 LAYER 2: Signal & Text Processing")
    print("-" * 60)
    layer2_processor = Layer2Processor()
    layer2_result = layer2_processor.process(audio_path, file_id=layer1_result.file_id)
    
    print(f"✅ Features extracted")
    print(f"   Pitch: {layer2_result.features.pitch_mean:.2f}Hz")
    print(f"   Speech Rate: {layer2_result.features.speech_rate:.2f} syl/s")
    print(f"   Pauses: {layer2_result.features.pause_count}")
    
    print(f"✅ Transcript generated")
    print(f"   Words: {layer2_result.transcript.word_count}")
    print(f"   Text: {layer2_result.transcript.full_text[:80]}...")
    
    layer2_output = f"data/layer2_output/{layer2_result.file_id}.json"
    layer2_result.save_to_file(layer2_output)
    print(f"   Saved: {layer2_output}")
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  Layer 1: {layer1_output}")
    print(f"  Layer 2: {layer2_output}")
    
    combined_output = {
        'layer1': layer1_result.to_dict(),
        'layer2': layer2_result.to_dict()
    }
    
    combined_path = f"data/combined/{layer1_result.file_id}_combined.json"
    os.makedirs("data/combined", exist_ok=True)
    with open(combined_path, 'w') as f:
        json.dump(combined_output, f, indent=2)
    
    print(f"  Combined: {combined_path}")


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
