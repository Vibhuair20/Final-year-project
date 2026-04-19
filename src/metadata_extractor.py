import os
import hashlib
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional
from mutagen import File as MutagenFile
from mutagen.mp3 import MP3
from mutagen.wave import WAVE
from .models import AudioMetadata


class MetadataExtractor:
    def __init__(self, timezone: str = "+05:30"):
        self.timezone = timezone
    
    def extract(self, file_path: str) -> AudioMetadata:
        file_size = os.path.getsize(file_path)
        filename = os.path.basename(file_path)
        file_format = Path(file_path).suffix.lower().replace('.', '')
        
        if file_format == 'wav':
            metadata_dict = self._extract_wav(file_path)
        elif file_format == 'mp3':
            metadata_dict = self._extract_mp3(file_path)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
        
        timestamp = datetime.now().astimezone().isoformat()
        checksum = self._calculate_checksum(file_path)
        
        return AudioMetadata(
            original_filename=filename,
            file_format=file_format,
            file_size_bytes=file_size,
            duration_seconds=metadata_dict['duration'],
            sample_rate_hz=metadata_dict['sample_rate'],
            channels=metadata_dict['channels'],
            bit_depth=metadata_dict.get('bit_depth'),
            ingestion_timestamp=timestamp,
            checksum_md5=checksum
        )
    
    def _extract_wav(self, file_path: str) -> dict:
        with wave.open(file_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            
            duration = frames / float(rate)
            bit_depth = sample_width * 8
            
            return {
                'duration': round(duration, 2),
                'sample_rate': rate,
                'channels': channels,
                'bit_depth': bit_depth
            }
    
    def _extract_mp3(self, file_path: str) -> dict:
        try:
            audio = MP3(file_path)
            duration = audio.info.length
            sample_rate = audio.info.sample_rate
            channels = audio.info.channels
            
            return {
                'duration': round(duration, 2),
                'sample_rate': sample_rate,
                'channels': channels,
                'bit_depth': 16
            }
        except Exception as e:
            audio = MutagenFile(file_path)
            if audio and audio.info:
                return {
                    'duration': round(audio.info.length, 2),
                    'sample_rate': getattr(audio.info, 'sample_rate', 44100),
                    'channels': getattr(audio.info, 'channels', 2),
                    'bit_depth': 16
                }
            raise e
    
    def _calculate_checksum(self, file_path: str, chunk_size: int = 8192) -> str:
        md5_hash = hashlib.md5()
        
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                md5_hash.update(chunk)
        
        return md5_hash.hexdigest()
    
    def extract_extended_metadata(self, file_path: str) -> dict:
        try:
            audio = MutagenFile(file_path)
            if audio:
                return {
                    'bitrate': getattr(audio.info, 'bitrate', None),
                    'codec': type(audio).__name__,
                    'tags': dict(audio.tags) if audio.tags else {}
                }
            return {}
        except Exception as e:
            return {'error': str(e)}
