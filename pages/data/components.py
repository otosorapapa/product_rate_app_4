from components import render_stepper as _core_render_stepper
import streamlit as st


def render_stepper(current_step: int) -> None:
    """Proxy to the global stepper renderer used across the app."""

    _core_render_stepper(current_step)


def render_sidebar_nav() -> None:
    """Render sidebar navigation links across pages."""
    st.sidebar.header("ナビゲーション")
    st.sidebar.page_link("app.py", label="ホーム", icon="🏠")
    st.sidebar.page_link("pages/01_データ入力.py", label="① データ入力", icon="📥")
    st.sidebar.page_link("pages/02_ダッシュボード.py", label="② ダッシュボード", icon="📊")
    st.sidebar.page_link("pages/03_標準賃率計算.py", label="③ 標準賃率計算", icon="🧮")
