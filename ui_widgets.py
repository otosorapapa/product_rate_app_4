"""Streamlit UI helper widgets used across pages.

This module centralises small UI building blocks such as the main upload
form that appears on the dashboard as well as a sample CSV download
button.  The functions are intentionally kept lightweight so they can be
reused both inside the Streamlit app and from unit tests by injecting a
mocked Streamlit module.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import List, Optional, Protocol, Any

import pandas as pd

try:  # pragma: no cover - optional dependency guard for tests
    import streamlit as st  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - streamlit not available during tests
    st = None  # type: ignore


@dataclass(frozen=True)
class MainFormResult:
    """Container returned by :func:`main_form` with the collected values."""

    run: bool
    file: Any
    threshold: float
    errors: List[str]


class StreamlitLike(Protocol):
    """Protocol describing the subset of Streamlit used in this module."""

    def form(self, form_key: str, *, clear_on_submit: bool = ...) -> Any: ...

    def subheader(self, text: str) -> None: ...

    def file_uploader(self, label: str, *, type: List[str], help: str | None = None): ...

    def number_input(
        self,
        label: str,
        *,
        min_value: float,
        max_value: float,
        value: float,
        step: float,
    ) -> float: ...

    def form_submit_button(self, label: str, *, use_container_width: bool = ...) -> bool: ...

    def download_button(
        self,
        label: str,
        data: bytes | BytesIO,
        *,
        file_name: str,
        mime: str,
    ) -> None: ...


SAMPLE_TEMPLATE = pd.DataFrame(
    {"id": [1, 2], "date": ["2025-01-01", "2025-01-02"], "amount": [120000, 98000]}
)


def _get_streamlit(st_module: Optional[StreamlitLike]) -> StreamlitLike:
    """Return the active Streamlit-like module or raise an informative error."""

    module = st_module if st_module is not None else st
    if module is None:  # pragma: no cover - defensive branch
        raise RuntimeError("Streamlit is required to render the widgets.")
    return module


def main_form(st_module: Optional[StreamlitLike] = None) -> MainFormResult:
    """Render the main CSV upload form and return the submitted values."""

    ui = _get_streamlit(st_module)
    with ui.form("main_form", clear_on_submit=False):
        ui.subheader("入力データ")
        file = ui.file_uploader(
            "データCSVをアップロード", type=["csv"], help="UTF-8/Shift-JIS対応"
        )
        threshold = ui.number_input(
            "しきい値", min_value=0.0, max_value=1e9, value=1000.0, step=100.0
        )
        run = ui.form_submit_button("入力を確認して実行", use_container_width=True)

    errors: List[str] = []
    if run:
        if file is None:
            errors.append(
                "データCSVは必須です。サンプルをダウンロードして形式をご確認ください。"
            )
        if float(threshold) < 0:
            errors.append("しきい値は0以上で指定してください。")

    return MainFormResult(run=run, file=file, threshold=float(threshold), errors=errors)


def sample_download(
    st_module: Optional[StreamlitLike] = None,
    *,
    filename: str = "template.csv",
    sample: Optional[pd.DataFrame] = None,
) -> bytes:
    """Render a download button that provides a template CSV file.

    Parameters
    ----------
    st_module:
        Optional Streamlit-like interface.  Passing a mock makes the function easy
        to test outside of a real Streamlit session.
    filename:
        Name used for the downloaded file.
    sample:
        Optional DataFrame to serialise instead of the default template.

    Returns
    -------
    bytes
        The UTF-8 with BOM encoded CSV content that is passed to Streamlit.
    """

    ui = _get_streamlit(st_module)
    template = SAMPLE_TEMPLATE if sample is None else sample
    csv_bytes = template.to_csv(index=False).encode("utf-8-sig")
    ui.download_button(
        "テンプレCSVをダウンロード", csv_bytes, file_name=filename, mime="text/csv"
    )
    return csv_bytes
