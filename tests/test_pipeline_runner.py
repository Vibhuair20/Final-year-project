import pytest
from unittest.mock import MagicMock, patch
from src.dashboard.pipeline_runner import run_pipeline, PipelineResult


def _mock_l1(file_id="call_001"):
    l1 = MagicMock()
    l1.status = "ready_for_processing"
    l1.file_id = file_id
    l1.error_message = None
    l1.metadata = MagicMock(duration_seconds=10.0, sample_rate_hz=16000, channels=1)
    return l1


def _mock_l2(file_id="call_001"):
    l2 = MagicMock()
    l2.file_id = file_id
    l2.features = MagicMock()
    l2.transcript = MagicMock(full_text="hello verify your account", word_count=5, segments=[])
    return l2


def _mock_l3():
    l3 = MagicMock()
    l3.label = "phishing"
    l3.confidence = 0.87
    l3.class_probabilities = {}
    l3.per_modality = {}
    l3.top_keywords = ["verify", "account"]
    l3.top_acoustic_drivers = ["speech_rate"]
    return l3


def test_happy_path_returns_all_layers():
    with patch("src.dashboard.pipeline_runner.AudioProcessor") as MockL1Cls, \
         patch("src.dashboard.pipeline_runner.Layer2Processor") as MockL2Cls, \
         patch("src.dashboard.pipeline_runner.load_trained_detector") as mock_loader:

        MockL1Cls.return_value.process_file.return_value = _mock_l1()
        MockL2Cls.return_value.process.return_value = _mock_l2()
        mock_loader.return_value.predict.return_value = _mock_l3()

        result = run_pipeline("/fake/call.wav")
        assert isinstance(result, PipelineResult)
        assert result.error is None
        assert result.layer1 is not None
        assert result.layer2 is not None
        assert result.layer3 is not None


def test_layer1_fail_returns_error():
    with patch("src.dashboard.pipeline_runner.AudioProcessor") as MockL1Cls:
        l1 = _mock_l1()
        l1.status = "error"
        l1.error_message = "unsupported format"
        MockL1Cls.return_value.process_file.return_value = l1

        result = run_pipeline("/fake/call.wav")
        assert result.error is not None
        assert "unsupported format" in result.error


def test_models_not_trained_returns_typed_error():
    with patch("src.dashboard.pipeline_runner.AudioProcessor") as MockL1Cls, \
         patch("src.dashboard.pipeline_runner.Layer2Processor") as MockL2Cls, \
         patch("src.dashboard.pipeline_runner.load_trained_detector") as mock_loader:

        MockL1Cls.return_value.process_file.return_value = _mock_l1()
        MockL2Cls.return_value.process.return_value = _mock_l2()
        mock_loader.side_effect = FileNotFoundError("models/text_classifier.joblib")

        result = run_pipeline("/fake/call.wav")
        assert result.error == "models_not_trained"
        assert result.layer3 is None
