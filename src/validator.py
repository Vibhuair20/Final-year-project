import os
from pathlib import Path
from typing import Set
from .models import ValidationResult


class AudioValidator:
    SUPPORTED_FORMATS: Set[str] = {'.wav', '.mp3'}
    MAX_FILE_SIZE_MB: int = 500
    
    def __init__(self, max_file_size_mb: int = 500):
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
    
    def validate_file(self, file_path: str) -> ValidationResult:
        warnings = []
        
        if not os.path.exists(file_path):
            return ValidationResult(
                is_valid=False,
                error_message=f"File not found: {file_path}"
            )
        
        if not os.path.isfile(file_path):
            return ValidationResult(
                is_valid=False,
                error_message=f"Path is not a file: {file_path}"
            )
        
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.SUPPORTED_FORMATS:
            return ValidationResult(
                is_valid=False,
                error_message=f"Unsupported format: {file_ext}. Supported: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return ValidationResult(
                is_valid=False,
                error_message="File is empty (0 bytes)"
            )
        
        if file_size > self.max_file_size_bytes:
            return ValidationResult(
                is_valid=False,
                error_message=f"File too large: {file_size / (1024*1024):.2f} MB (max: {self.max_file_size_bytes / (1024*1024):.2f} MB)"
            )
        
        if file_size < 1024:
            warnings.append(f"File is very small ({file_size} bytes), may not contain valid audio")
        
        if not os.access(file_path, os.R_OK):
            return ValidationResult(
                is_valid=False,
                error_message="File is not readable (permission denied)"
            )
        
        return ValidationResult(
            is_valid=True,
            warnings=warnings if warnings else None
        )
    
    def validate_batch(self, file_paths: list) -> dict:
        valid_files = []
        invalid_files = []
        
        for file_path in file_paths:
            result = self.validate_file(file_path)
            if result.is_valid:
                valid_files.append(file_path)
            else:
                invalid_files.append({
                    'path': file_path,
                    'error': result.error_message
                })
        
        return {
            'valid': valid_files,
            'invalid': invalid_files
        }
