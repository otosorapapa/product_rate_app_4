import streamlit as st


def render_stepper(current_step: int) -> None:
    """Render a simple progress stepper for the import wizard.

    Parameters
    ----------
    current_step: int
        Zero-based index of the current step. The wizard steps are::

            0: ホーム
            1: 取り込み
            2: 自動検証
            3: 結果サマリ
            4: ダッシュボード
    """
    steps = ["ホーム", "取り込み", "自動検証", "結果サマリ", "ダッシュボード"]
    total = len(steps) - 1
    progress = min(max(current_step, 0), total) / total if total else 0.0
    st.progress(progress)
    cols = st.columns(len(steps))
    for idx, (col, label) in enumerate(zip(cols, steps)):
        prefix = "🔵" if idx <= current_step else "⚪️"
        col.markdown(f"{prefix} {label}")


def render_sidebar_nav() -> None:
    """Render sidebar navigation links across pages."""
    st.sidebar.header("ナビゲーション")
    st.sidebar.page_link("app.py", label="ホーム", icon="🏠")
    st.sidebar.page_link("pages/01_データ入力.py", label="① データ入力", icon="📥")
    st.sidebar.page_link("pages/02_ダッシュボード.py", label="② ダッシュボード", icon="📊")
    st.sidebar.page_link("pages/03_標準賃率計算.py", label="③ 標準賃率計算", icon="🧮")
