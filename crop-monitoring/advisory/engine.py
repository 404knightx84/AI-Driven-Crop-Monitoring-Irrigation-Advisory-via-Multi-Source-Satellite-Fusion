"""
advisory/engine.py
Rule-based irrigation advisory engine.
Combines crop type, phenological stage, and moisture stress level
to produce per-field irrigation recommendations in mm/day.

Logic:
  1. Look up crop + stage in the YAML config thresholds.
  2. Map stress level (0/1/2) to no-action / mild / urgent advisory.
  3. Compute field-aggregate statistics from pixel maps.
  4. Return structured advisory record per field.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from datetime import datetime, timezone

import numpy as np
import yaml
from loguru import logger


CROP_NAMES  = ["background", "wheat", "rice", "maize",
               "sugarcane", "cotton", "soybean", "other"]
STAGE_NAMES = ["pre_sowing", "germination", "vegetative",
               "reproductive", "maturation", "harvest_fallow"]
STRESS_NAMES = ["no_stress", "mild_stress", "severe_stress"]


@dataclass
class FieldAdvisory:
    field_id: str
    timestamp: str
    dominant_crop: str
    phenology_stage: str
    stress_level: str
    stress_fraction: float          # fraction of pixels with stress
    recommended_irrigation_mm: float
    urgency: str                    # none / low / high
    notes: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class IrrigationAdvisoryEngine:
    def __init__(self, config_path: str = "configs/pipeline_config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.rules      = cfg["advisory"]["irrigation_rules"]
        self.thresholds = cfg["advisory"]["thresholds"]

    def _dominant(self, label_map: np.ndarray, names: List[str]) -> str:
        """Return the most frequent non-background label name."""
        counts = np.bincount(label_map.ravel(), minlength=len(names))
        counts[0] = 0   # exclude background
        idx = int(counts.argmax())
        return names[idx] if counts[idx] > 0 else "unknown"

    def _stress_fraction(self, stress_map: np.ndarray) -> float:
        """Fraction of pixels with mild or severe stress."""
        return float((stress_map > 0).mean())

    def _irrigation_mm(
        self, crop: str, stage: str, stress: str, stress_prob: float
    ) -> float:
        """
        Look up irrigation amount from rule table.
        Returns 0 if crop/stage not found or no stress.
        """
        if stress == "no_stress":
            return 0.0
        crop_rules = self.rules.get(crop, {})
        stage_rules = crop_rules.get(stage, {})
        key = "stress_mild" if stress == "mild_stress" else "stress_severe"
        base_mm = stage_rules.get(key, 0.0)
        # Scale by stress confidence (stress_prob = prob of that class)
        return round(base_mm * stress_prob, 1)

    def advise(
        self,
        field_id: str,
        crop_map: np.ndarray,      # (H, W) int
        stage_map: np.ndarray,     # (H, W) int
        stress_map: np.ndarray,    # (H, W) int
        stress_probs: np.ndarray,  # (H, W, 3) float
    ) -> FieldAdvisory:
        """Generate an irrigation advisory for a single field."""
        crop  = self._dominant(crop_map,   CROP_NAMES)
        stage = self._dominant(stage_map,  STAGE_NAMES)

        # Dominant stress: pick most common non-zero level
        stress_counts = np.bincount(stress_map.ravel(), minlength=3)
        if stress_counts[2] / stress_map.size > self.thresholds["stress_severe"]:
            stress = "severe_stress"
        elif stress_counts[1] / stress_map.size > self.thresholds["stress_mild"]:
            stress = "mild_stress"
        else:
            stress = "no_stress"

        stress_frac = self._stress_fraction(stress_map)
        stress_conf = float(stress_probs[..., stress_map.ravel().argmax()].mean()) \
            if stress != "no_stress" else 0.0

        irrigation_mm = self._irrigation_mm(crop, stage, stress, stress_conf)

        urgency = "none"
        if stress == "mild_stress":
            urgency = "low"
        elif stress == "severe_stress":
            urgency = "high"

        notes = []
        if crop == "unknown":
            notes.append("Crop type could not be determined — verify field boundaries.")
        if stage in ("maturation", "harvest_fallow"):
            notes.append("Crop at maturation or post-harvest — irrigation may not be beneficial.")
        if irrigation_mm == 0 and stress != "no_stress":
            notes.append(f"No irrigation rule defined for {crop}/{stage} — consult agronomist.")

        return FieldAdvisory(
            field_id=field_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            dominant_crop=crop,
            phenology_stage=stage,
            stress_level=stress,
            stress_fraction=round(stress_frac, 3),
            recommended_irrigation_mm=irrigation_mm,
            urgency=urgency,
            notes=notes,
        )

    def advise_batch(
        self, fields: Dict[str, Dict]
    ) -> List[FieldAdvisory]:
        """
        fields: { field_id: { crop_map, stage_map, stress_map, stress_probs } }
        """
        advisories = []
        for fid, data in fields.items():
            try:
                adv = self.advise(fid, **data)
                advisories.append(adv)
                logger.info(f"[Advisory] {fid}: {adv.stress_level} | {adv.recommended_irrigation_mm}mm | urgency={adv.urgency}")
            except Exception as e:
                logger.error(f"[Advisory] Failed for {fid}: {e}")
        return advisories
