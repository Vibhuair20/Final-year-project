__version__ = "1.0.0"
__author__ = "Voice Phishing Detection Team"

from .audio_processor import AudioProcessor
from .metadata_extractor import MetadataExtractor
from .validator import AudioValidator
from .models import AudioMetadata, ProcessingResult

__all__ = [
    "AudioProcessor",
    "MetadataExtractor",
    "AudioValidator",
    "AudioMetadata",
    "ProcessingResult",
]
