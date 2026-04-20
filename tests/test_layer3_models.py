import json
from src.layer3_models import DetectionResult


def test_detection_result_to_dict_has_all_fields():
    result = DetectionResult(
        file_id="call_001",
        label="phishing",
        confidence=0.87,
        class_probabilities={"legitimate": 0.05, "phishing": 0.87, "banking": 0.03,
                             "investment": 0.01, "kidnapping": 0.01, "lottery": 0.01,
                             "customer_service": 0.01, "identity_theft": 0.01},
        per_modality={
            "text": {"legitimate": 0.1, "phishing": 0.85, "banking": 0.01, "investment": 0.01,
                     "kidnapping": 0.01, "lottery": 0.01, "customer_service": 0.01, "identity_theft": 0.0},
            "acoustic": {"legitimate": 0.0, "phishing": 0.9, "banking": 0.05, "investment": 0.01,
                         "kidnapping": 0.01, "lottery": 0.01, "customer_service": 0.01, "identity_theft": 0.01},
            "fused": {"legitimate": 0.05, "phishing": 0.87, "banking": 0.03, "investment": 0.01,
                      "kidnapping": 0.01, "lottery": 0.01, "customer_service": 0.01, "identity_theft": 0.01},
        },
        top_keywords=["verify", "account", "urgent", "wire", "transfer"],
        top_acoustic_drivers=["speech_rate", "pitch_variance", "pause_count", "energy_mean", "mfcc_3"],
    )
    d = result.to_dict()
    assert d["file_id"] == "call_001"
    assert d["label"] == "phishing"
    assert d["version"] == "3.0"
    assert len(d["top_keywords"]) == 5
    assert "fused" in d["per_modality"]


def test_detection_result_json_roundtrip(tmp_path):
    result = DetectionResult(
        file_id="x", label="legitimate", confidence=0.99,
        class_probabilities={k: 0.125 for k in
            ["legitimate","phishing","banking","investment","kidnapping","lottery","customer_service","identity_theft"]},
        per_modality={"text": {}, "acoustic": {}, "fused": {}},
        top_keywords=[], top_acoustic_drivers=[],
    )
    out = tmp_path / "r.json"
    result.save_to_file(str(out))
    loaded = json.loads(out.read_text())
    assert loaded["label"] == "legitimate"
    assert loaded["confidence"] == 0.99
