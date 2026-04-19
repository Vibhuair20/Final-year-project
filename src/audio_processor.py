import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from .validator import AudioValidator
from .metadata_extractor import MetadataExtractor
from .models import ProcessingResult, ValidationResult


class AudioProcessor:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.validator = AudioValidator(
            max_file_size_mb=self.config.get('max_file_size_mb', 500)
        )
        self.metadata_extractor = MetadataExtractor(
            timezone=self.config.get('timezone', '+05:30')
        )
    
    def process_file(self, file_path: str) -> ProcessingResult:
        validation_result = self.validator.validate_file(file_path)
        
        if not validation_result.is_valid:
            return ProcessingResult(
                file_id=self._generate_file_id(file_path),
                metadata=None,
                audio_path=file_path,
                status="validation_failed",
                error_message=validation_result.error_message
            )
        
        try:
            metadata = self.metadata_extractor.extract(file_path)
            file_id = self._generate_file_id(file_path)
            
            result = ProcessingResult(
                file_id=file_id,
                metadata=metadata,
                audio_path=os.path.abspath(file_path),
                status="ready_for_processing"
            )
            
            if validation_result.warnings:
                print(f"⚠️  Warnings for {file_path}:")
                for warning in validation_result.warnings:
                    print(f"   - {warning}")
            
            return result
            
        except Exception as e:
            return ProcessingResult(
                file_id=self._generate_file_id(file_path),
                metadata=None,
                audio_path=file_path,
                status="processing_failed",
                error_message=f"Error during processing: {str(e)}"
            )
    
    def process_batch(self, input_dir: str, output_dir: Optional[str] = None) -> dict:
        audio_files = self._find_audio_files(input_dir)
        
        if not audio_files:
            return {
                'total': 0,
                'successful': 0,
                'failed': 0,
                'results': []
            }
        
        results = []
        successful = 0
        failed = 0
        
        print(f"\n📁 Found {len(audio_files)} audio file(s) in {input_dir}")
        print("=" * 60)
        
        for file_path in audio_files:
            print(f"\n🎵 Processing: {os.path.basename(file_path)}")
            
            result = self.process_file(file_path)
            results.append(result)
            
            if result.status == "ready_for_processing":
                successful += 1
                print(f"✅ Success - Duration: {result.metadata.duration_seconds}s, "
                      f"Sample Rate: {result.metadata.sample_rate_hz}Hz")
                
                if output_dir:
                    self._save_result(result, output_dir)
            else:
                failed += 1
                print(f"❌ Failed - {result.error_message}")
        
        print("\n" + "=" * 60)
        print(f"📊 Summary: {successful} successful, {failed} failed")
        
        return {
            'total': len(audio_files),
            'successful': successful,
            'failed': failed,
            'results': results
        }
    
    def _generate_file_id(self, file_path: str) -> str:
        filename = Path(file_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{filename}_{timestamp}"
    
    def _find_audio_files(self, directory: str) -> List[str]:
        audio_files = []
        supported_extensions = {'.wav', '.mp3'}
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if Path(file).suffix.lower() in supported_extensions:
                    audio_files.append(os.path.join(root, file))
        
        return sorted(audio_files)
    
    def _save_result(self, result: ProcessingResult, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{result.file_id}.json")
        result.save_to_file(output_path)
        print(f"💾 Saved metadata to: {output_path}")
