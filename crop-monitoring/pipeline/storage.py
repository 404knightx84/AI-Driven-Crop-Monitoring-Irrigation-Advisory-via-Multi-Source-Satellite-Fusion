"""
pipeline/storage.py
Writes advisory records and per-field statistics to
InfluxDB (time-series metrics) and PostgreSQL (advisory history).
"""

import os
from datetime import datetime, timezone
from typing import List, Dict, Any

from loguru import logger

INFLUX_URL    = os.getenv("INFLUX_URL",    "http://localhost:8086")
INFLUX_TOKEN  = os.getenv("INFLUX_TOKEN",  "")
INFLUX_ORG    = os.getenv("INFLUX_ORG",    "cropmon")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "crop_metrics")
DATABASE_URL  = os.getenv("DATABASE_URL",  "postgresql://cropmon:cropmon@localhost:5432/cropmon")


# ── InfluxDB ──────────────────────────────────────────────────────────────────

def _influx_client():
    from influxdb_client import InfluxDBClient
    return InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)


def write_advisory_influx(advisory_dict: Dict[str, Any]):
    """Write a FieldAdvisory as an InfluxDB point."""
    from influxdb_client import Point
    from influxdb_client.client.write_api import SYNCHRONOUS
    client    = _influx_client()
    write_api = client.write_api(write_options=SYNCHRONOUS)
    p = (
        Point("advisory")
        .tag("field_id",       advisory_dict["field_id"])
        .tag("dominant_crop",  advisory_dict["dominant_crop"])
        .tag("phenology_stage", advisory_dict["phenology_stage"])
        .tag("stress_level",   advisory_dict["stress_level"])
        .tag("urgency",        advisory_dict["urgency"])
        .field("stress_fraction",           advisory_dict["stress_fraction"])
        .field("recommended_irrigation_mm", advisory_dict["recommended_irrigation_mm"])
    )
    write_api.write(bucket=INFLUX_BUCKET, record=p)
    client.close()
    logger.debug(f"[Storage] InfluxDB write: {advisory_dict['field_id']}")


def write_advisories_influx(advisories: List[Dict[str, Any]]):
    for adv in advisories:
        try:
            write_advisory_influx(adv)
        except Exception as e:
            logger.error(f"[Storage] InfluxDB error for {adv.get('field_id')}: {e}")


# ── PostgreSQL ────────────────────────────────────────────────────────────────

def _pg_engine():
    from sqlalchemy import create_engine
    return create_engine(DATABASE_URL)


def _ensure_tables(engine):
    from sqlalchemy import text
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS advisory_history (
                id                          SERIAL PRIMARY KEY,
                field_id                    TEXT NOT NULL,
                timestamp                   TIMESTAMPTZ NOT NULL,
                dominant_crop               TEXT,
                phenology_stage             TEXT,
                stress_level                TEXT,
                stress_fraction             FLOAT,
                recommended_irrigation_mm   FLOAT,
                urgency                     TEXT,
                notes                       TEXT
            )
        """))
        conn.commit()


def write_advisory_pg(advisory_dict: Dict[str, Any]):
    from sqlalchemy import text
    engine = _pg_engine()
    _ensure_tables(engine)
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO advisory_history
                (field_id, timestamp, dominant_crop, phenology_stage,
                 stress_level, stress_fraction, recommended_irrigation_mm, urgency, notes)
            VALUES
                (:field_id, :timestamp, :dominant_crop, :phenology_stage,
                 :stress_level, :stress_fraction, :recommended_irrigation_mm, :urgency, :notes)
        """), {
            "field_id":                   advisory_dict["field_id"],
            "timestamp":                  advisory_dict["timestamp"],
            "dominant_crop":              advisory_dict["dominant_crop"],
            "phenology_stage":            advisory_dict["phenology_stage"],
            "stress_level":               advisory_dict["stress_level"],
            "stress_fraction":            advisory_dict["stress_fraction"],
            "recommended_irrigation_mm":  advisory_dict["recommended_irrigation_mm"],
            "urgency":                    advisory_dict["urgency"],
            "notes":                      "; ".join(advisory_dict.get("notes", [])),
        })
        conn.commit()
    logger.debug(f"[Storage] PostgreSQL write: {advisory_dict['field_id']}")


def write_advisories_pg(advisories: List[Dict[str, Any]]):
    for adv in advisories:
        try:
            write_advisory_pg(adv)
        except Exception as e:
            logger.error(f"[Storage] PostgreSQL error for {adv.get('field_id')}: {e}")


# ── Unified write ─────────────────────────────────────────────────────────────

def persist_advisories(advisories: List[Dict[str, Any]]):
    """Write to both InfluxDB (metrics) and PostgreSQL (history)."""
    write_advisories_influx(advisories)
    write_advisories_pg(advisories)
    logger.info(f"[Storage] Persisted {len(advisories)} advisories.")
