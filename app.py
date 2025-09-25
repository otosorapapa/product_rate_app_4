import html

import streamlit as st
from components import (
    apply_user_theme,
    render_help_button,
    render_onboarding,
    render_page_tutorial,
    render_stepper,
    render_sidebar_nav,
    render_top_navbar,
)

st.set_page_config(
    page_title="賃率ダッシュボード",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_user_theme()

render_sidebar_nav(page_key="home")

render_top_navbar(
    page_key="home",
    page_title="製品賃率ダッシュボード",
    subtitle="Product Rate Intelligence Suite",
    phase_label="Phase 3",
)

hero_container = st.container()
hero_container.markdown("<section class='hero-card'>", unsafe_allow_html=True)
hero_cols = hero_container.columns([0.62, 0.38], gap="large")
with hero_cols[0]:
    st.markdown("<div class='hero-card__badge'>PHASE 3 / BRAND REFRESH</div>", unsafe_allow_html=True)
    st.markdown("<h1>製品賃率ダッシュボード</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='hero-card__lead'>📊 標準賃率シートをアップロードするだけで、KPIカード・達成状況ヒートマップ・感度分析がワンクリックで揃います。経営層向けの上質なレポート体験を届けましょう。</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="hero-card__meta">
            <span class="hero-chip">自動KPI集約</span>
            <span class="hero-chip">プレミアムUX</span>
            <span class="hero-chip">レスポンシブ対応</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hero_cols[1]:
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.markdown("<h4 class='content-card__title'>エグゼクティブスナップショット</h4>", unsafe_allow_html=True)
    st.markdown(
        "<p class='content-card__body'>最新の入力データから推計した主要KPIをサマリ表示します。詳細はダッシュボードタブでゲージやヒートマップとして深掘りできます。</p>",
        unsafe_allow_html=True,
    )
    metric_cols = st.columns(2, gap="medium")
    with metric_cols[0]:
        st.metric("SKU達成率", "82%", "先月比 +5pt")
    with metric_cols[1]:
        st.metric("稼働率", "94%", "計画比 +2pt")
    st.markdown("<div class='content-card__cta'>", unsafe_allow_html=True)
    help_container = st.container()
    render_help_button("home", container=help_container)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

hero_container.markdown("</section>", unsafe_allow_html=True)

render_onboarding()

# Page-specific tutorial with glossary support
render_page_tutorial("home")

# Progress stepper for wizard flow
render_stepper(0)

st.markdown("<h2>3ステップではじめましょう</h2>", unsafe_allow_html=True)
steps = [
    (
        "1. データを準備",
        "テンプレートを確認し、Excelをアップロードするかサンプルを読み込みます。",
    ),
    (
        "2. ダッシュボードで把握",
        "KPIカード・ゲージ・ヒートマップでSKUの達成状況を素早く確認します。",
    ),
    (
        "3. 標準賃率を検証",
        "ウィザードで固定費や稼働条件を見直し、必要賃率と感度をシミュレーションします。",
    ),
]
step_cards = "".join(
    f"""
    <div class='content-card'>
        <div class='content-card__title'>{html.escape(title)}</div>
        <p class='content-card__body'>{html.escape(body)}</p>
    </div>
    """
    for title, body in steps
)
st.markdown(f"<div class='card-grid card-grid--three'>{step_cards}</div>", unsafe_allow_html=True)
st.caption("※ 詳細ガイドや用語集は必要なときに展開して確認できます。")


def _go_to_data_page() -> None:
    """Navigate to the data intake screen with graceful fallback."""

    try:
        st.switch_page("pages/01_データ入力.py")
    except Exception:
        st.session_state["nav_intent"] = "data"
        st.experimental_set_query_params(next="data")
        st.experimental_rerun()


cta_container = st.container()
cta_cols = cta_container.columns([0.58, 0.42], gap="large")
with cta_cols[0]:
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.markdown("<h4 class='content-card__title'>主なページ</h4>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="content-card__body">
            <ul>
                <li>① データ入力: Excel取込 / サンプル / 直接編集</li>
                <li>② ダッシュボード: KPIカードと詳細グラフ</li>
                <li>③ 標準賃率計算: 5ステップウィザード</li>
                <li>④ チャット / FAQ: AIボットとマニュアル</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
with cta_cols[1]:
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.markdown("<h4 class='content-card__title'>まずはデータを準備</h4>", unsafe_allow_html=True)
    st.markdown(
        "<p class='content-card__body'>テンプレートを確認したら、アップロードかサンプル利用を選んでダッシュボードの土台を作成します。</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='content-card__cta'>", unsafe_allow_html=True)
    st.button("👉 データ入力へ進む", use_container_width=True, on_click=_go_to_data_page)
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("※ 初回はExcelを読み込むと自動的にダッシュボードへ遷移します。")
    st.markdown("</div>", unsafe_allow_html=True)

with st.expander("導入メリットの目安（フェルミ推定）", expanded=False):
    st.markdown(
        "- KPIカードを最上段に集約 → ダッシュボード確認時間が **約75分/月** 短縮（1人あたり）\n"
        "- データ取込のエラー即時通知 → 再入力時間を **約13.5分/月** 削減（5人利用時）\n"
        "- 標準賃率ウィザードの再利用 → 入力時間を **約60分/月** 削減（20人企業想定）\n"
        "> 合計で月 **約2.5時間** の意思決定時間短縮が見込めます。"
    )

st.markdown("<div class='content-card'>", unsafe_allow_html=True)
st.markdown("<h4 class='content-card__title'>クイックアクセス</h4>", unsafe_allow_html=True)
st.markdown(
    "<p class='content-card__body'>主要機能はこちらから移動できます。</p>",
    unsafe_allow_html=True,
)
link_cols = st.columns(4, gap="large")
with link_cols[0]:
    st.page_link("pages/01_データ入力.py", label="① データ入力 & 取り込み", icon="📥")
with link_cols[1]:
    st.page_link("pages/02_ダッシュボード.py", label="② ダッシュボード", icon="📊")
with link_cols[2]:
    st.page_link("pages/03_標準賃率計算.py", label="③ 標準賃率 計算/感度分析", icon="🧮")
with link_cols[3]:
    st.page_link("pages/04_チャットサポート.py", label="④ チャット/FAQ", icon="💬")
st.markdown("</div>", unsafe_allow_html=True)

st.info(
    "まずは『データ入力 & 取り込み』で Excel を読み込むかサンプルを使用し、"
    "疑問があれば『チャット/FAQ』でAIに質問してください。"
)
