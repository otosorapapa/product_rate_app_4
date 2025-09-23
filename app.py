import streamlit as st
from components import (
    apply_user_theme,
    render_help_button,
    render_onboarding,
    render_page_tutorial,
    render_stepper,
    render_sidebar_nav,
)

st.set_page_config(page_title="賃率ダッシュボード", layout="wide")
apply_user_theme()

render_sidebar_nav(page_key="home")

header_col, help_col = st.columns([0.82, 0.18], gap="small")
with header_col:
    st.title("製品賃率ダッシュボード")
    st.caption("📊 Excel（標賃 / R6.12）から賃率KPIを自動計算し、SKU別の達成状況を可視化します。")

render_help_button("home", container=help_col)

render_onboarding()

# Page-specific tutorial with glossary support
render_page_tutorial("home")

# Progress stepper for wizard flow
render_stepper(0)

st.write("次のページから機能を選択してください。")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.page_link("pages/01_データ入力.py", label="① データ入力 & 取り込み", icon="📥")
with c2:
    st.page_link("pages/02_ダッシュボード.py", label="② ダッシュボード", icon="📊")
with c3:
    st.page_link("pages/03_標準賃率計算.py", label="③ 標準賃率 計算/感度分析", icon="🧮")
with c4:
    st.page_link("pages/04_チャットサポート.py", label="④ チャット/FAQ", icon="💬")

st.info(
    "まずは『データ入力 & 取り込み』で Excel を読み込むかサンプルを使用し、"
    "疑問があれば『チャット/FAQ』でAIに質問してください。"
)
