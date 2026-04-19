from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import json
import numpy as np


@dataclass
class AudioFeatures:
    mfcc_mean: List[float]
    mfcc_std: List[float]
    pitch_mean: float
    pitch_std: float
    pitch_variance: float
    energy_mean: float
    energy_std: float
    zero_crossing_rate: float
    spectral_centroid_mean: float
    spectral_rolloff_mean: float
    pause_count: int
    pause_duration_mean: float
    speech_rate: float
    total_speech_duration: float
    total_pause_duration: float
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()
            elif isinstance(value, (np.float32, np.float64)):
                data[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                data[key] = int(value)
        return data
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class TranscriptSegment:
    text: str
    start_time: float
    end_time: float
    confidence: float


@dataclass
class Transcript:
    full_text: str
    segments: List[TranscriptSegment]
    word_count: int
    language: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'full_text': self.full_text,
            'segments': [
                {
                    'text': seg.text,
                    'start_time': seg.start_time,
                    'end_time': seg.end_time,
                    'confidence': seg.confidence
                }
                for seg in self.segments
            ],
            'word_count': self.word_count,
            'language': self.language
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ProcessedAudio:
    file_id: str
    audio_path: str
    features: AudioFeatures
    transcript: Transcript
    processing_timestamp: str
    version: str = "2.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'file_id': self.file_id,
            'audio_path': self.audio_path,
            'features': self.features.to_dict(),
            'transcript': self.transcript.to_dict(),
            'processing_timestamp': self.processing_timestamp,
            'version': self.version
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    def save_to_file(self, output_path: str) -> None:
        with open(output_path, 'w') as f:
            f.write(self.to_json())
