from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any
import json


@dataclass
class AudioMetadata:
    original_filename: str
    file_format: str
    file_size_bytes: int
    duration_seconds: float
    sample_rate_hz: int
    channels: int
    bit_depth: Optional[int]
    ingestion_timestamp: str
    checksum_md5: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ProcessingResult:
    file_id: str
    metadata: AudioMetadata
    audio_path: str
    status: str
    version: str = "1.0"
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "file_id": self.file_id,
            "metadata": self.metadata.to_dict(),
            "audio_path": self.audio_path,
            "status": self.status,
            "version": self.version
        }
        if self.error_message:
            result["error_message"] = self.error_message
        return result
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    def save_to_file(self, output_path: str) -> None:
        with open(output_path, 'w') as f:
            f.write(self.to_json())


@dataclass
class ValidationResult:
    is_valid: bool
    error_message: Optional[str] = None
    warnings: Optional[list] = None
    
    def __bool__(self):
        return self.is_valid
