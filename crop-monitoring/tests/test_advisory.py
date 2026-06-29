"""
tests/test_advisory.py — unit tests for irrigation advisory engine
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from advisory.engine import IrrigationAdvisoryEngine, FieldAdvisory


MOCK_CFG = {
    "advisory": {
        "thresholds": { "stress_mild": 0.4, "stress_severe": 0.7 },
        "irrigation_rules": {
            "wheat": {
                "vegetative": { "stress_mild": 30, "stress_severe": 50 },
                "tillering":  { "stress_mild": 35, "stress_severe": 55 },
            },
            "rice": {
                "transplanting": { "stress_mild": 50, "stress_severe": 80 },
            },
        },
    }
}


@pytest.fixture
def engine(tmp_path):
    import yaml
    cfg_path = tmp_path / "pipeline_config.yaml"
    cfg_path.write_text(yaml.dump(MOCK_CFG))
    return IrrigationAdvisoryEngine(config_path=str(cfg_path))


def make_maps(h=10, w=10, crop_label=1, stage_label=2, stress_label=1):
    crop_map    = np.full((h, w), crop_label,   dtype=np.int64)
    stage_map   = np.full((h, w), stage_label,  dtype=np.int64)
    stress_map  = np.full((h, w), stress_label, dtype=np.int64)
    stress_probs = np.zeros((h, w, 3), dtype=np.float32)
    stress_probs[..., stress_label] = 1.0
    return crop_map, stage_map, stress_map, stress_probs


def test_advisory_wheat_mild_stress(engine):
    crop, stage, stress, probs = make_maps(crop_label=1, stage_label=2, stress_label=1)
    adv = engine.advise("field_001", crop, stage, stress, probs)

    assert isinstance(adv, FieldAdvisory)
    assert adv.field_id == "field_001"
    assert adv.dominant_crop == "wheat"
    assert adv.stress_level == "mild_stress"
    assert adv.urgency == "low"
    assert adv.recommended_irrigation_mm > 0


def test_advisory_no_stress(engine):
    crop, stage, stress, probs = make_maps(stress_label=0)
    adv = engine.advise("field_002", crop, stage, stress, probs)
    assert adv.stress_level == "no_stress"
    assert adv.recommended_irrigation_mm == 0.0
    assert adv.urgency == "none"


def test_advisory_to_dict(engine):
    crop, stage, stress, probs = make_maps()
    adv = engine.advise("field_003", crop, stage, stress, probs)
    d = adv.to_dict()
    assert "field_id" in d
    assert "recommended_irrigation_mm" in d
    assert isinstance(d["notes"], list)
