#!/usr/bin/env python3
"""Run Layer 1 + Layer 2 + Layer 3 on a single audio file, print DetectionResult JSON.

Usage: python layer3_main.py --input data/input/call.wav
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.audio_processor import AudioProcessor
from src.layer2_processor import Layer2Processor
from src.detector_loader import load_trained_detector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to 16kHz mono 16-bit WAV")
    ap.add_argument("--output", default=None, help="optional path to write DetectionResult JSON")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"File not found: {args.input}")
        sys.exit(1)

    l1 = AudioProcessor().process_file(args.input)
    if l1.status != "ready_for_processing":
        print(f"Layer 1 failed: {l1.error_message}")
        sys.exit(1)

    l2 = Layer2Processor().process(args.input, file_id=l1.file_id)
    detector = load_trained_detector()
    result = detector.predict(l2)

    print(result.to_json())
    if args.output:
        result.save_to_file(args.output)
        print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
