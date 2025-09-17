import streamlit as st

from components import (
    inject_mckinsey_style,
    render_page_header,
    render_sidebar_nav,
    render_stepper,
)

st.set_page_config(page_title="賃率ダッシュボード", layout="wide")

inject_mckinsey_style()
render_sidebar_nav()

render_page_header(
    "製品賃率ダッシュボード",
    "Excel（標賃 / R6.12）から賃率KPIを自動計算し、SKU別の達成状況を可視化します。",
    icon="📊",
)

# Progress stepper for wizard flow
render_stepper(0)

st.markdown(
    """
    <div class="mck-card">
        <h3>ダッシュボードの歩き方</h3>
        <p>製品別の採算状況を McKinsey らしい視点で段階的に把握できるよう、
        「インプット → 分析 → 感度検証」の順でナビゲートします。利用したい
        ページを下記のカードから選択してください。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

features = [
    {
        "icon": "📥",
        "title": "① データ入力 & 取り込み",
        "description": "標賃・R6.12 Excel を取り込み、製品マスターと賃率前提を整理します。",
        "path": "pages/01_データ入力.py",
    },
    {
        "icon": "📊",
        "title": "② ダッシュボード",
        "description": "SKU別の達成状況や未達要因をビジュアルで俯瞰し、改善余地を発見します。",
        "path": "pages/02_ダッシュボード.py",
    },
    {
        "icon": "🧮",
        "title": "③ 標準賃率 計算/感度分析",
        "description": "前提値を調整し、必要賃率への影響と感度を素早く検証します。",
        "path": "pages/03_標準賃率計算.py",
    },
]

cols = st.columns(len(features))
for col, feature in zip(cols, features):
    with col:
        st.markdown(
            f"""
            <div class=\"mck-feature-card\">
                <div class=\"mck-feature-icon\">{feature['icon']}</div>
                <div>
                    <h4>{feature['title']}</h4>
                    <p>{feature['description']}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.page_link(feature["path"], label="ページを開く", icon="↗️")

st.markdown(
    """
    <div class="mck-cta-banner">
        <span>💡</span>
        <strong>まずは『① データ入力 & 取り込み』で Excel を読み込むか、サンプルデータをご利用ください。</strong>
    </div>
    """,
    unsafe_allow_html=True,
)
