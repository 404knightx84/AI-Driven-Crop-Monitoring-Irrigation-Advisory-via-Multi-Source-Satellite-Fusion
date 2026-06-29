"""
advisory/api.py
FastAPI application exposing:
  GET  /advisory/{field_id}   — latest advisory for a field
  GET  /advisory              — all latest advisories
  GET  /fields                — registered field list
  POST /run                   — trigger a pipeline cycle (dev/testing)
"""

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from loguru import logger

from advisory.engine import FieldAdvisory, IrrigationAdvisoryEngine

app = FastAPI(
    title="Crop Monitoring & Irrigation Advisory API",
    description="Real-time irrigation recommendations from multi-source satellite fusion.",
    version="1.0.0",
)

# In-memory store — replace with DB in production
_advisories: dict[str, FieldAdvisory] = {}
_engine = IrrigationAdvisoryEngine()


# ── Schemas ──────────────────────────────────────────────────────────────────

class AdvisoryResponse(BaseModel):
    field_id: str
    timestamp: str
    dominant_crop: str
    phenology_stage: str
    stress_level: str
    stress_fraction: float
    recommended_irrigation_mm: float
    urgency: str
    notes: List[str]


class FieldInfo(BaseModel):
    field_id: str
    last_updated: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/fields", response_model=List[FieldInfo])
def list_fields():
    return [
        FieldInfo(field_id=fid, last_updated=adv.timestamp)
        for fid, adv in _advisories.items()
    ]


@app.get("/advisory", response_model=List[AdvisoryResponse])
def list_advisories():
    return [AdvisoryResponse(**adv.to_dict()) for adv in _advisories.values()]


@app.get("/advisory/{field_id}", response_model=AdvisoryResponse)
def get_advisory(field_id: str):
    if field_id not in _advisories:
        raise HTTPException(status_code=404, detail=f"No advisory found for field '{field_id}'")
    return AdvisoryResponse(**_advisories[field_id].to_dict())


@app.post("/run")
def trigger_run(background_tasks: BackgroundTasks):
    """Trigger one pipeline cycle in the background (for testing)."""
    background_tasks.add_task(_run_pipeline_cycle)
    return {"message": "Pipeline cycle triggered."}


def _run_pipeline_cycle():
    """Called in background task — runs full ingest → infer → advise cycle."""
    try:
        from pipeline.orchestrator import run_cycle
        run_cycle()
        logger.info("[API] Background pipeline cycle complete.")
    except Exception as e:
        logger.error(f"[API] Pipeline cycle failed: {e}")


def update_advisory(advisory: FieldAdvisory):
    """Called by the pipeline to push a new advisory into the API store."""
    _advisories[advisory.field_id] = advisory
