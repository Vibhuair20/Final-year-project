import os
import json
import wave
from vosk import Model, KaldiRecognizer
from typing import List
from .layer2_models import Transcript, TranscriptSegment


class SpeechRecognizer:
    def __init__(self, model_path: str = "models/vosk-model-small-en-us-0.15"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Vosk model not found at {model_path}")
        self.model = Model(model_path)
        self.sample_rate = 16000
    
    def transcribe(self, audio_path: str) -> Transcript:
        wf = wave.open(audio_path, "rb")
        
        if wf.getnchannels() != 1:
            raise ValueError("Audio must be mono channel")
        if wf.getsampwidth() != 2:
            raise ValueError("Audio must be 16-bit")
        if wf.getframerate() != self.sample_rate:
            raise ValueError(f"Audio must be {self.sample_rate}Hz sample rate")
        
        rec = KaldiRecognizer(self.model, self.sample_rate)
        rec.SetWords(True)
        
        segments = []
        full_text_parts = []
        
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if 'result' in result:
                    for word_info in result['result']:
                        segment = TranscriptSegment(
                            text=word_info['word'],
                            start_time=word_info['start'],
                            end_time=word_info['end'],
                            confidence=word_info.get('conf', 1.0)
                        )
                        segments.append(segment)
                        full_text_parts.append(word_info['word'])
        
        final_result = json.loads(rec.FinalResult())
        if 'text' in final_result and final_result['text']:
            full_text_parts.append(final_result['text'])
        
        full_text = ' '.join(full_text_parts)
        word_count = len(full_text.split())
        
        return Transcript(
            full_text=full_text,
            segments=segments,
            word_count=word_count,
            language='en-US'
        )
    
    def transcribe_with_fallback(self, audio_path: str) -> Transcript:
        try:
            return self.transcribe(audio_path)
        except Exception as e:
            return Transcript(
                full_text=f"[Transcription failed: {str(e)}]",
                segments=[],
                word_count=0,
                language='en-US'
            )
