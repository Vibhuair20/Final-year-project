#!/usr/bin/env python3

import sys
import os
import wave
import array

def resample_to_16khz(input_path, output_path):
    with wave.open(input_path, 'rb') as inp:
        params = inp.getparams()
        frames = inp.readframes(params.nframes)
        
        if params.sampwidth == 2:
            samples = array.array('h', frames)
        else:
            print(f"❌ Unsupported sample width: {params.sampwidth}")
            return False
        
        original_rate = params.framerate
        target_rate = 16000
        
        if original_rate == target_rate:
            print(f"✅ Already 16kHz, copying file...")
            with wave.open(output_path, 'wb') as out:
                out.setparams(params)
                out.writeframes(frames)
            return True
        
        ratio = original_rate / target_rate
        new_length = int(len(samples) / ratio)
        resampled = array.array('h', [0] * new_length)
        
        for i in range(new_length):
            old_index = int(i * ratio)
            if old_index < len(samples):
                resampled[i] = samples[old_index]
        
        with wave.open(output_path, 'wb') as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(16000)
            out.writeframes(resampled.tobytes())
        
        print(f"✅ Resampled from {original_rate}Hz to 16000Hz")
        return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python resample.py <input.wav> <output.wav>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    if resample_to_16khz(input_file, output_file):
        print(f"💾 Saved to: {output_file}")
    else:
        sys.exit(1)
