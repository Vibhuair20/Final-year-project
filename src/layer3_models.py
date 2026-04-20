from dataclasses import dataclass, field
from typing import Dict, List, Any
import json


@dataclass
class DetectionResult:
    file_id: str
    label: str
    confidence: float
    class_probabilities: Dict[str, float]
    per_modality: Dict[str, Dict[str, float]]
    top_keywords: List[str] = field(default_factory=list)
    top_acoustic_drivers: List[str] = field(default_factory=list)
    version: str = "3.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_id": self.file_id,
            "label": self.label,
            "confidence": self.confidence,
            "class_probabilities": self.class_probabilities,
            "per_modality": self.per_modality,
            "top_keywords": self.top_keywords,
            "top_acoustic_drivers": self.top_acoustic_drivers,
            "version": self.version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def save_to_file(self, output_path: str) -> None:
        with open(output_path, "w") as f:
            f.write(self.to_json())
