import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from .signal_processor import SignalProcessor
from .speech_recognizer import SpeechRecognizer
from .layer2_models import ProcessedAudio


class Layer2Processor:
    def __init__(self, vosk_model_path: str = "models/vosk-model-small-en-us-0.15"):
        self.signal_processor = SignalProcessor()
        self.speech_recognizer = SpeechRecognizer(vosk_model_path)
    
    def process(self, audio_path: str, file_id: Optional[str] = None) -> ProcessedAudio:
        if file_id is None:
            file_id = self._generate_file_id(audio_path)
        
        print(f"🎵 Extracting audio features...")
        features = self.signal_processor.extract_features(audio_path)
        
        print(f"🗣️  Transcribing speech...")
        transcript = self.speech_recognizer.transcribe_with_fallback(audio_path)
        
        timestamp = datetime.now().astimezone().isoformat()
        
        return ProcessedAudio(
            file_id=file_id,
            audio_path=os.path.abspath(audio_path),
            features=features,
            transcript=transcript,
            processing_timestamp=timestamp
        )
    
    def _generate_file_id(self, file_path: str) -> str:
        filename = Path(file_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{filename}_layer2_{timestamp}"
