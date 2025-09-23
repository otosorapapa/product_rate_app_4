"""Utilities for surfacing labour-law related updates inside the dashboard."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

_DATA_DIR = Path(__file__).resolve().parent / "data" / "external"
_DATA_FILE = _DATA_DIR / "legal_updates_sample.json"


@dataclass
class LegalUpdate:
    """A single legal or regulatory update relevant for labour cost planning."""

    category: str
    region: str
    effective_from: date
    value: float
    unit: str
    source: str
    url: Optional[str] = None
    last_updated: Optional[date] = None
    notes: Optional[str] = None


def _normalise_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Return a normalised :class:`~pandas.DataFrame` for downstream use."""

    if not records:
        columns = [
            "category",
            "region",
            "effective_from",
            "value",
            "unit",
            "source",
            "url",
            "last_updated",
            "notes",
        ]
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(records)
    for col in ("effective_from", "last_updated"):
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], errors="coerce").dt.date
    numeric_cols = ["value"]
    for col in numeric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


@lru_cache(maxsize=1)
def _load_updates_cached(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return _normalise_records([])
    with path.open("r", encoding="utf-8") as fp:
        try:
            payload = json.load(fp)
        except json.JSONDecodeError:
            payload = []
    if isinstance(payload, dict):
        payload = [payload]
    return _normalise_records(payload)


def fetch_labor_standards_updates(source: Optional[Path] = None) -> pd.DataFrame:
    """Load labour-law related updates (minimum wage, insurance rates, etc.)."""

    target = source if source is not None else _DATA_FILE
    df = _load_updates_cached(str(target))
    # Return a copy so callers can mutate without affecting the cache entry.
    return df.copy()


def compute_average_hourly_wage(params: Dict[str, float], results: Dict[str, float]) -> float:
    """Approximate the average hourly wage implied by the current parameters."""

    annual_minutes = results.get("annual_minutes", 0.0)
    if annual_minutes <= 0:
        return 0.0
    labor_cost = float(params.get("labor_cost", 0.0))
    return labor_cost / annual_minutes * 60.0


def build_compliance_alerts(
    params: Dict[str, float],
    results: Dict[str, float],
    updates: pd.DataFrame,
    *,
    preferred_regions: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Create structured alert payloads for the Streamlit UI."""

    if updates is None or updates.empty:
        return []

    alerts: List[Dict[str, Any]] = []
    preferred_regions = list(preferred_regions or [])

    min_wage_df = updates[updates["category"] == "最低賃金"].copy()
    if not min_wage_df.empty:
        if preferred_regions:
            min_wage_df = min_wage_df.sort_values(
                by=["region", "effective_from"], ascending=[True, False]
            )
            preferred_df = min_wage_df[min_wage_df["region"].isin(preferred_regions)]
            others_df = min_wage_df[~min_wage_df["region"].isin(preferred_regions)]
            min_wage_df = pd.concat([preferred_df, others_df], ignore_index=True)
            min_wage_df = min_wage_df.drop_duplicates("region")
        latest_min = min_wage_df.sort_values(
            by=["effective_from", "last_updated"], ascending=False
        ).iloc[0]
        hourly_wage = compute_average_hourly_wage(params, results)
        target_value = float(latest_min.get("value", 0.0) or 0.0)
        gap = hourly_wage - target_value
        severity = "warning" if gap < 0 else "info"
        alerts.append(
            {
                "category": "最低賃金",
                "region": latest_min.get("region", ""),
                "value": target_value,
                "unit": latest_min.get("unit", ""),
                "effective_from": latest_min.get("effective_from"),
                "source": latest_min.get("source", ""),
                "url": latest_min.get("url"),
                "notes": latest_min.get("notes"),
                "severity": severity,
                "current_hourly_wage": hourly_wage,
                "gap": gap,
            }
        )

    social_df = updates[updates["category"] == "社会保険料率"].copy()
    if not social_df.empty:
        social_df = social_df.sort_values(
            by=["region", "effective_from"], ascending=[True, False]
        ).drop_duplicates("region")
        for _, row in social_df.iterrows():
            alerts.append(
                {
                    "category": "社会保険料率",
                    "region": row.get("region", ""),
                    "value": float(row.get("value", 0.0) or 0.0),
                    "unit": row.get("unit", ""),
                    "effective_from": row.get("effective_from"),
                    "source": row.get("source", ""),
                    "url": row.get("url"),
                    "notes": row.get("notes"),
                    "severity": "info",
                }
            )

    return alerts


__all__ = [
    "LegalUpdate",
    "fetch_labor_standards_updates",
    "compute_average_hourly_wage",
    "build_compliance_alerts",
]
