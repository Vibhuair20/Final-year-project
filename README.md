# Voice Phishing Detection System

Multi-layer voice phishing detection system.

## Quick Start

```bash
source venv/bin/activate
python pipeline.py data/input/yourfile_16k.wav
```

## Layer 1: Data Acquisition

```bash
python main.py --input data/input/call.wav
```

Output: Metadata (duration, sample rate, channels, checksum)

## Layer 2: Signal & Text Processing

```bash
python layer2_main.py --input data/input/call.wav
```

Output: Audio features + transcript

## Full Pipeline

```bash
python pipeline.py data/input/call.wav
```

Output: Combined Layer 1 + Layer 2 results

## Resample Audio

```bash
python resample.py input.wav output_16k.wav
```

## Requirements

- Audio must be 16kHz, mono, 16-bit WAV
- Use resample.py if needed
# Final-year-project
