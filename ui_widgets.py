"""Reusable Streamlit UI widgets for the product rate app."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st


__all__ = ["table_block"]


def table_block(df: pd.DataFrame, name: str = "result") -> None:
    """Render a dataframe with download actions for CSV and Excel formats.

    Parameters
    ----------
    df:
        The pandas ``DataFrame`` to display in Streamlit.
    name:
        Base filename used for the export buttons.
    """

    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "amount": st.column_config.NumberColumn("金額", format="¥,d"),
            "rate": st.column_config.NumberColumn("率", format="%.2f%%"),
            "date": st.column_config.DateColumn("日付"),
        },
    )

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "CSVダウンロード",
        csv,
        file_name=f"{name}.csv",
        mime="text/csv",
    )

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="data", index=False)
    st.download_button(
        "Excelダウンロード",
        buf.getvalue(),
        file_name=f"{name}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
