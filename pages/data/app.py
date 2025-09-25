import streamlit as st
from components import render_stepper, render_sidebar_nav, render_top_navbar

st.set_page_config(
    page_title="賃率ダッシュボード",
    layout="wide",
    initial_sidebar_state="collapsed",
)

render_sidebar_nav()

render_top_navbar(
    page_key="home",
    page_title="製品賃率ダッシュボード",
    subtitle="Product Rate Intelligence Suite",
    phase_label="Phase 3",
)

st.title("製品賃率ダッシュボード")
st.caption("📊 Excel（標賃 / R6.12）から賃率KPIを自動計算し、SKU別の達成状況を可視化します。")

# Progress stepper for wizard flow
render_stepper(0)

st.write("次のページから機能を選択してください。")

c1, c2, c3 = st.columns(3)
with c1:
    st.page_link("pages/01_データ入力.py", label="① データ入力 & 取り込み", icon="📥")
with c2:
    st.page_link("pages/02_ダッシュボード.py", label="② ダッシュボード", icon="📊")
with c3:
    st.page_link("pages/03_標準賃率計算.py", label="③ 標準賃率 計算/感度分析", icon="🧮")

st.info("まずは『データ入力 & 取り込み』で Excel を読み込むか、サンプルを使用してください。")
