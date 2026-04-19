#!/bin/bash

echo "🚀 Setting up Layer 2: Signal & Text Processing"
echo "=" | head -c 60
echo ""

echo "📦 Installing Python dependencies..."
source venv/bin/activate
pip install -r requirements_layer2.txt

echo ""
echo "📥 Downloading Vosk speech recognition model..."
mkdir -p models
cd models

if [ ! -d "vosk-model-small-en-us-0.15" ]; then
    echo "Downloading vosk-model-small-en-us-0.15 (40MB)..."
    curl -L -O https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
    unzip -q vosk-model-small-en-us-0.15.zip
    rm vosk-model-small-en-us-0.15.zip
    echo "✅ Model downloaded and extracted"
else
    echo "✅ Model already exists"
fi

cd ..

echo ""
echo "✅ Layer 2 setup complete!"
echo ""
echo "To test:"
echo "  python layer2_main.py --input data/input/test.wav"
