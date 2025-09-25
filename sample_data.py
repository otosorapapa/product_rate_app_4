"""Utilities for loading bundled sample datasets and priming session state."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import streamlit as st

from rate_utils import (
    infer_category_from_name,
    infer_major_customer,
    parse_hyochin,
    parse_products,
    read_excel_safely,
)
from standard_rate_core import DEFAULT_PARAMS, sanitize_params


@dataclass
class SamplePayload:
    """Parsed objects derived from the bundled sample workbook."""

    products: pd.DataFrame
    calc_params: Dict[str, float]
    sr_params: Dict[str, float]


_SAMPLE_PATH = Path(__file__).resolve().parent / "data" / "sample.xlsx"


@st.cache_data(show_spinner=False)
def _read_sample_bytes(path: Path) -> bytes:
    """Return the raw bytes for the bundled sample workbook."""

    if not path.exists():
        raise FileNotFoundError(f"サンプルデータが見つかりません: {path}")
    return path.read_bytes()


@st.cache_data(show_spinner=False)
def load_sample_workbook(path: Path | None = None) -> Optional[pd.ExcelFile]:
    """Return an :class:`~pandas.ExcelFile` for the bundled sample workbook."""

    workbook_path = path or _SAMPLE_PATH
    try:
        raw = _read_sample_bytes(workbook_path)
    except FileNotFoundError:
        return None
    return read_excel_safely(BytesIO(raw))


def _enrich_sample_products(df: pd.DataFrame) -> pd.DataFrame:
    """Attach derived helper columns to the sample product master."""

    if df.empty:
        return df
    enriched = df.copy()
    if "category" not in enriched.columns or enriched["category"].isna().all():
        enriched["category"] = enriched.get("product_name", pd.Series(dtype=str)).apply(
            infer_category_from_name
        )
    if "major_customer" not in enriched.columns or enriched["major_customer"].isna().all():
        enriched["major_customer"] = [
            infer_major_customer(no, name)
            for no, name in zip(
                enriched.get("product_no"),
                enriched.get("product_name"),
            )
        ]
    return enriched


def load_sample_payload(path: Path | None = None) -> Optional[SamplePayload]:
    """Parse the bundled sample workbook into reusable objects."""

    workbook = load_sample_workbook(path)
    if workbook is None:
        return None

    calc_params, sr_params, _ = parse_hyochin(workbook)
    products, _ = parse_products(workbook, sheet_name="R6.12")
    products = _enrich_sample_products(products)
    return SamplePayload(products=products, calc_params=calc_params, sr_params=sr_params)


def ensure_sample_session_state(*, notice_key: str = "sample_notice") -> bool:
    """Populate :mod:`streamlit` session state with the bundled sample data if empty.

    Parameters
    ----------
    notice_key:
        Session state key toggled to ``True`` when the sample was loaded during the
        current request. Callers can pop this key to surface a contextual message.

    Returns
    -------
    bool
        ``True`` if the sample dataset was loaded into the session, ``False`` otherwise.
    """

    df_products = st.session_state.get("df_products_raw")
    if isinstance(df_products, pd.DataFrame) and not df_products.empty:
        return False

    payload = load_sample_payload()
    if payload is None or payload.products.empty:
        return False

    products = payload.products.copy()
    st.session_state["df_products_raw"] = products

    sr_params = payload.sr_params.copy()
    sanitised_params, warnings = sanitize_params(sr_params or DEFAULT_PARAMS)
    for message in warnings:
        st.warning(message)
    st.session_state.setdefault("sr_params", sanitised_params.copy())
    st.session_state.setdefault("scenarios", {"ベース": sanitised_params.copy()})
    st.session_state.setdefault("current_scenario", "ベース")

    calc_params = payload.calc_params.copy()
    calc_params.update(
        {k: v for k, v in sanitised_params.items() if isinstance(v, (int, float))}
    )
    st.session_state.setdefault("calc_params", calc_params)
    st.session_state["using_sample_data"] = True
    st.session_state[notice_key] = True
    return True
