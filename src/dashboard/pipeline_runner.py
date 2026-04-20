from dataclasses import dataclass, field
from typing import Optional, Any

from src.audio_processor import AudioProcessor
from src.layer2_processor import Layer2Processor
from src.detector_loader import load_trained_detector


@dataclass
class PipelineResult:
    layer1: Optional[Any] = None
    layer2: Optional[Any] = None
    layer3: Optional[Any] = None
    error: Optional[str] = None


def run_pipeline(audio_path: str) -> PipelineResult:
    l1 = AudioProcessor().process_file(audio_path)
    if l1.status != "ready_for_processing":
        return PipelineResult(layer1=l1, error=l1.error_message or "Layer 1 failed")

    l2 = Layer2Processor().process(audio_path, file_id=l1.file_id)

    try:
        detector = load_trained_detector()
        l3 = detector.predict(l2)
    except FileNotFoundError:
        return PipelineResult(layer1=l1, layer2=l2, error="models_not_trained")

    return PipelineResult(layer1=l1, layer2=l2, layer3=l3)
