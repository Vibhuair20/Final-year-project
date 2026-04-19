#!/usr/bin/env python3

import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.layer2_processor import Layer2Processor


def main():
    parser = argparse.ArgumentParser(description="Layer 2: Signal & Text Processing")
    parser.add_argument('--input', type=str, required=True, help='Path to audio file')
    parser.add_argument('--output', type=str, default='data/layer2_output', help='Output directory')
    parser.add_argument('--model', type=str, default='models/vosk-model-small-en-us-0.15', help='Vosk model path')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)
    
    print(f"\n🎯 Processing audio file: {args.input}")
    print("=" * 60)
    
    try:
        processor = Layer2Processor(vosk_model_path=args.model)
        result = processor.process(args.input)
        
        os.makedirs(args.output, exist_ok=True)
        output_path = os.path.join(args.output, f"{result.file_id}.json")
        result.save_to_file(output_path)
        
        print(f"\n✅ Processing complete!")
        print(f"\n📊 Feature Summary:")
        print(f"   Pitch Mean: {result.features.pitch_mean:.2f} Hz")
        print(f"   Energy Mean: {result.features.energy_mean:.4f}")
        print(f"   Speech Rate: {result.features.speech_rate:.2f} syllables/sec")
        print(f"   Pause Count: {result.features.pause_count}")
        print(f"\n📝 Transcript Summary:")
        print(f"   Word Count: {result.transcript.word_count}")
        print(f"   Text: {result.transcript.full_text[:100]}...")
        print(f"\n💾 Saved to: {output_path}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
