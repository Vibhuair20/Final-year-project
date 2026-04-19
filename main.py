#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path

from src.audio_processor import AudioProcessor


def load_config(config_path: str = "config/config.json") -> dict:
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  Config file not found: {config_path}")
        print("Using default configuration")
        return {}
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing config file: {e}")
        sys.exit(1)


def process_single_file(file_path: str, output_dir: str, config: dict):
    print(f"\n🎯 Processing single file: {file_path}")
    print("=" * 60)
    
    processor = AudioProcessor(config)
    result = processor.process_file(file_path)
    
    if result.status == "ready_for_processing":
        print(f"\n✅ Processing successful!")
        print(f"\n📋 Metadata:")
        print(json.dumps(result.metadata.to_dict(), indent=2))
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{result.file_id}.json")
            result.save_to_file(output_path)
            print(f"\n💾 Saved to: {output_path}")
    else:
        print(f"\n❌ Processing failed: {result.error_message}")
        sys.exit(1)


def process_batch(input_dir: str, output_dir: str, config: dict):
    print(f"\n🎯 Batch processing directory: {input_dir}")
    
    processor = AudioProcessor(config)
    results = processor.process_batch(input_dir, output_dir)
    
    if results['total'] == 0:
        print("\n⚠️  No audio files found in the specified directory")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Voice Phishing Detection - Data Acquisition Layer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input data/input/call_001.wav
  python main.py --batch data/input/
  python main.py --input call.wav --output results/
  python main.py --input call.wav --config custom_config.json
        """
    )
    
    parser.add_argument('--input', type=str, help='Path to a single audio file to process')
    parser.add_argument('--batch', type=str, help='Directory containing audio files for batch processing')
    parser.add_argument('--output', type=str, default='data/output', help='Output directory for JSON files (default: data/output)')
    parser.add_argument('--config', type=str, default='config/config.json', help='Path to configuration file (default: config/config.json)')
    parser.add_argument('--version', action='version', version='Data Acquisition Layer v1.0.0')
    
    args = parser.parse_args()
    
    if not args.input and not args.batch:
        parser.print_help()
        print("\n❌ Error: Either --input or --batch must be specified")
        sys.exit(1)
    
    if args.input and args.batch:
        print("❌ Error: Cannot use both --input and --batch simultaneously")
        sys.exit(1)
    
    config = load_config(args.config)
    
    if args.input:
        process_single_file(args.input, args.output, config)
    elif args.batch:
        process_batch(args.batch, args.output, config)


if __name__ == "__main__":
    main()
