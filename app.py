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
    st.caption(
        "📊 Excel（標賃 / R6.12）から賃率KPIを自動計算し、SKU別の達成状況を可視化します。"
    )

render_help_button("home", container=help_col)

render_onboarding()

# Page-specific tutorial with glossary support
render_page_tutorial("home")

# Progress stepper for wizard flow
render_stepper(0)

st.markdown(
    """
    ### ご利用の流れ
    1. **データ入力**でテンプレートを確認し、Excel取込またはサンプルで初期データを読み込みます。
    2. 取り込んだデータはダッシュボードと標準賃率ウィザードに共有され、KPIの監視と感度分析が可能になります。
    3. 疑問点はチャット/FAQからAIボットに相談し、必要な用語解説やマニュアルへ素早くアクセスしてください。
    """
)


def _go_to_data_page() -> None:
    """Navigate to the data intake screen with graceful fallback."""

    try:
        st.switch_page("pages/01_データ入力.py")
    except Exception:
        st.session_state["nav_intent"] = "data"
        st.experimental_rerun()


cta_col1, cta_col2 = st.columns([0.6, 0.4], gap="large")
with cta_col1:
    st.subheader("サイトマップ")
    st.markdown(
        """
        - ホーム / ガイド（このページ）
        - ① データ入力（Excel取込／サンプルデータ／直接編集）
        - ② ダッシュボード ─ KPIカード・トレンド・構成チャート、売上/粗利/在庫/資金/付加価値タブ
        - ③ 標準賃率計算 ─ 5ステップウィザードとシナリオ比較
        - ④ チャット / FAQ ─ AIボットとマニュアル・用語集
        """
    )
with cta_col2:
    st.subheader("まずはデータを準備しましょう")
    st.caption("テンプレートを確認したら、アップロードかサンプル利用を選んでダッシュボードの土台を作成します。")
    st.button("👉 データ入力へ進む", use_container_width=True, on_click=_go_to_data_page)
    st.caption("※ 初回はExcelを読み込むと自動的にダッシュボードへ遷移します。")

st.markdown(
    """
    #### 導線改善による想定効果（フェルミ推定）
    - KPIカードを最上段に集約 → ダッシュボード確認時間が **約75分/月** 短縮（1人あたり）
    - データ取込のエラー即時通知 → 再入力時間を **約13.5分/月** 削減（5人利用時）
    - 標準賃率ウィザードの再利用 → 入力時間を **約60分/月** 削減（20人企業想定）
    > 合計で月 **約2.5時間** の意思決定時間短縮が見込めます。
    """
)

st.write("主要機能はこちらから移動できます。")

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
