#!/usr/bin/env python3
"""
Voice Phishing Detection System - Layer 1 & 2 Results Demonstration
This script provides a clear visualization of the results from both layers.
"""

import json
import sys
from pathlib import Path


def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_section(title):
    """Print a formatted section"""
    print(f"\n--- {title} ---")


def demonstrate_layer1(layer1_data):
    """Demonstrate Layer 1: Data Acquisition results"""
    print_header("LAYER 1: DATA ACQUISITION LAYER")
    
    print("📋 PURPOSE:")
    print("  Layer 1 validates and ingests audio files, extracting basic metadata")
    print("  to ensure the audio is properly formatted for further processing.\n")
    
    print("🎯 WHAT IT DOES:")
    print("  ✓ Validates audio file format (WAV, 16kHz, mono, 16-bit)")
    print("  ✓ Extracts technical metadata (duration, sample rate, channels)")
    print("  ✓ Generates unique file ID and checksum for data integrity")
    print("  ✓ Prepares audio for downstream processing\n")
    
    metadata = layer1_data.get('metadata', {})
    
    print_section("Layer 1 Results")
    print(f"  File ID:          {layer1_data.get('file_id', 'N/A')}")
    print(f"  Original File:    {metadata.get('original_filename', 'N/A')}")
    print(f"  File Size:        {metadata.get('file_size_bytes', 0):,} bytes")
    print(f"  Duration:         {metadata.get('duration_seconds', 0):.2f} seconds")
    print(f"  Sample Rate:      {metadata.get('sample_rate_hz', 0):,} Hz")
    print(f"  Channels:         {metadata.get('channels', 0)} (mono)")
    print(f"  Bit Depth:        {metadata.get('bit_depth', 0)} bits")
    print(f"  MD5 Checksum:     {metadata.get('checksum_md5', 'N/A')}")
    print(f"  Ingestion Time:   {metadata.get('ingestion_timestamp', 'N/A')}")
    print(f"  Status:           {layer1_data.get('status', 'N/A')}")
    
    print("\n💡 SIGNIFICANCE:")
    print("  • Ensures data quality before expensive processing")
    print("  • Provides audit trail with checksums and timestamps")
    print("  • Validates technical requirements for speech processing")
    print("  • Foundation for reproducible research")


def demonstrate_layer2(layer2_data):
    """Demonstrate Layer 2: Signal & Text Processing results"""
    print_header("LAYER 2: SIGNAL & TEXT PROCESSING LAYER")
    
    print("📋 PURPOSE:")
    print("  Layer 2 extracts acoustic features and transcribes speech to enable")
    print("  detection of voice phishing patterns through audio analysis.\n")
    
    print("🎯 WHAT IT DOES:")
    print("  ✓ Extracts acoustic features (MFCCs, pitch, energy, spectral features)")
    print("  ✓ Performs speech-to-text transcription with timestamps")
    print("  ✓ Analyzes speech patterns (pauses, speech rate, prosody)")
    print("  ✓ Generates feature vectors for machine learning\n")
    
    features = layer2_data.get('features', {})
    transcript = layer2_data.get('transcript', {})
    
    print_section("Audio Features Extracted")
    
    # MFCC Features
    mfcc_mean = features.get('mfcc_mean', [])
    print(f"\n  MFCCs (Mel-Frequency Cepstral Coefficients):")
    print(f"    Mean values: {len(mfcc_mean)} coefficients")
    print(f"    First 3: [{mfcc_mean[0]:.2f}, {mfcc_mean[1]:.2f}, {mfcc_mean[2]:.2f}]")
    print(f"    → Captures voice timbre and speaker characteristics")
    
    # Pitch Features
    print(f"\n  Pitch Analysis:")
    print(f"    Mean Pitch:      {features.get('pitch_mean', 0):.2f} Hz")
    print(f"    Pitch Std Dev:   {features.get('pitch_std', 0):.2f} Hz")
    print(f"    Pitch Variance:  {features.get('pitch_variance', 0):.2f}")
    print(f"    → Detects stress, emotion, and deception indicators")
    
    # Energy Features
    print(f"\n  Energy Analysis:")
    print(f"    Mean Energy:     {features.get('energy_mean', 0):.4f}")
    print(f"    Energy Std Dev:  {features.get('energy_std', 0):.4f}")
    print(f"    → Measures voice intensity and confidence")
    
    # Spectral Features
    print(f"\n  Spectral Features:")
    print(f"    Spectral Centroid: {features.get('spectral_centroid_mean', 0):.2f} Hz")
    print(f"    Spectral Rolloff:  {features.get('spectral_rolloff_mean', 0):.2f} Hz")
    print(f"    Zero Crossing Rate: {features.get('zero_crossing_rate', 0):.4f}")
    print(f"    → Analyzes voice quality and frequency distribution")
    
    # Speech Pattern Analysis
    print(f"\n  Speech Pattern Analysis:")
    print(f"    Speech Rate:      {features.get('speech_rate', 0):.2f} words/second")
    print(f"    Total Speech:     {features.get('total_speech_duration', 0):.2f} seconds")
    print(f"    Total Pauses:     {features.get('total_pause_duration', 0):.2f} seconds")
    print(f"    Pause Count:      {features.get('pause_count', 0)}")
    print(f"    Avg Pause Length: {features.get('pause_duration_mean', 0):.2f} seconds")
    print(f"    → Identifies hesitation and scripted speech patterns")
    
    print_section("Transcription Results")
    
    full_text = transcript.get('full_text', '')
    segments = transcript.get('segments', [])
    word_count = transcript.get('word_count', 0)
    
    print(f"\n  Language:         {transcript.get('language', 'N/A')}")
    print(f"  Word Count:       {word_count}")
    print(f"  Segments:         {len(segments)}")
    
    print(f"\n  Full Transcript:")
    print(f"  \"{full_text}\"")
    
    # Show sample segments with timing
    print(f"\n  Sample Word Timings (first 5 words):")
    for i, seg in enumerate(segments[:5]):
        print(f"    [{seg['start_time']:.2f}s - {seg['end_time']:.2f}s] "
              f"\"{seg['text']}\" (confidence: {seg['confidence']:.2%})")
    
    print("\n💡 SIGNIFICANCE:")
    print("  • MFCCs enable speaker verification and voice analysis")
    print("  • Pitch/energy patterns reveal emotional state and stress")
    print("  • Speech rate and pauses detect scripted/rehearsed speech")
    print("  • Transcription enables keyword and phrase detection")
    print("  • Combined features feed into ML models for phishing detection")


def demonstrate_combined(combined_data):
    """Show how layers work together"""
    print_header("COMBINED PIPELINE: LAYER 1 + LAYER 2")
    
    print("🔄 PIPELINE FLOW:")
    print("  1. Layer 1: Validates and ingests audio file")
    print("  2. Layer 2: Processes validated audio for features + transcript")
    print("  3. Output: Comprehensive dataset ready for ML analysis\n")
    
    layer1 = combined_data.get('layer1', {})
    layer2 = combined_data.get('layer2', {})
    
    print_section("Data Flow Summary")
    
    print(f"\n  Input Audio:")
    print(f"    File: {layer1.get('metadata', {}).get('original_filename', 'N/A')}")
    print(f"    Duration: {layer1.get('metadata', {}).get('duration_seconds', 0):.2f}s")
    
    print(f"\n  Layer 1 Output:")
    print(f"    ✓ Validated format and quality")
    print(f"    ✓ Generated file ID: {layer1.get('file_id', 'N/A')}")
    print(f"    ✓ Status: {layer1.get('status', 'N/A')}")
    
    print(f"\n  Layer 2 Output:")
    print(f"    ✓ Extracted {len(layer2.get('features', {}).get('mfcc_mean', []))} MFCC coefficients")
    print(f"    ✓ Analyzed pitch, energy, and spectral features")
    print(f"    ✓ Transcribed {layer2.get('transcript', {}).get('word_count', 0)} words")
    print(f"    ✓ Detected {layer2.get('features', {}).get('pause_count', 0)} pauses")
    
    print(f"\n  Ready for Next Stage:")
    print(f"    → Feature vectors ready for ML model")
    print(f"    → Transcript ready for NLP analysis")
    print(f"    → Combined multimodal analysis possible")


def main():
    """Main demonstration function"""
    if len(sys.argv) < 2:
        print("Usage: python demo_results.py <path_to_combined_json>")
        print("\nExample:")
        print("  python demo_results.py data/combined/test2_16k_20260201_040104_combined.json")
        sys.exit(1)
    
    json_path = Path(sys.argv[1])
    
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)
    
    # Load the combined results
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Check if it's a combined file or individual layer
    if 'layer1' in data and 'layer2' in data:
        # Combined file
        demonstrate_layer1(data['layer1'])
        demonstrate_layer2(data['layer2'])
        demonstrate_combined(data)
    elif 'metadata' in data and 'features' not in data:
        # Layer 1 only
        demonstrate_layer1(data)
    elif 'features' in data and 'transcript' in data:
        # Layer 2 only
        demonstrate_layer2(data)
    else:
        print("Error: Unrecognized JSON format")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("  END OF DEMONSTRATION")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
