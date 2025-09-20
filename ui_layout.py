from typing import Optional

import streamlit as st


def sidebar_steps():
    st.sidebar.header("メニュー")
    use_new = st.sidebar.toggle("新UIを試す", value=True, help="不具合時はOFFで旧UIに戻せます")
    step = st.sidebar.radio("ステップ", ["① データ", "② 設定", "③ 結果", "④ 書き出し"], index=0)
    return use_new, step


def kpi_card(label: str, value, delta=None, help: Optional[str] = None) -> None:
    """Display a KPI block with optional delta and help text."""

    with st.container(border=True):
        st.caption(label)
        cols = st.columns([2, 1])
        cols[0].markdown(f"### {value}")
        if delta is not None:
            cols[1].metric(label="", value="", delta=delta)
        if help:
            st.caption(f"ℹ️ {help}")
