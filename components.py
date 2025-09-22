from typing import Any, Dict, List, Optional

import streamlit as st

_ONBOARDING_STATE_KEY = "onboarding_dismissed"
_PAGE_STATE_PREFIX = "tutorial_collapsed_"

_ONBOARDING_HEADLINE = (
    "最初にサンプルデータを読み込み、各KPIカードの意味を確認しましょう。"
    "必要賃率とは固定費＋必要利益を生産時間で割った値です。"
)
_ONBOARDING_STEPS = [
    "左側の『① データ入力 & 取り込み』から標賃Excelまたはサンプルを読み込みます。",
    "ステッパーの順にデータ検証 → KPIダッシュボード → 標準賃率の感度分析へ進みます。",
    "不明な用語は各ページのチュートリアルやサイドバーの用語集で確認できます。",
]
_ONBOARDING_EFFECT = "期待効果: 新規ユーザーの理解時間を約30分まで短縮（フェルミ推定 1時間→30分）"
_DEMO_URL = "https://appuctrateapp4-dqst3fvvfptjvavk2wbyfu.streamlit.app"

_GLOSSARY: Dict[str, str] = {
    "必要賃率": "固定費と確保したい利益の合計を年間の有効稼働時間で割った最低限必要な売上単価です。",
    "ブレークイーブン賃率": "材料費などの変動費を加味した損益分岐点の単価で、これを下回ると粗利がゼロ以下になります。",
    "付加価値/分": "製品1個あたりの付加価値（売価−材料費）を製造に要する分数で割った指標です。",
    "粗利/分": "付加価値/分と同義で、1分あたりに生み出す粗利益を示します。",
    "ギャップ": "必要賃率と現在の付加価値/分との差。プラス値ほど改善余地が大きいSKUを意味します。",
    "ROI": "想定投資回収期間（Return on Investment）。月数が小さいほど投資効果の立ち上がりが早いことを示します。",
    "固定費": "生産量にかかわらず発生する年間コスト（労務費や工場維持費など）。",
    "必要利益": "事業を健全に継続するために確保したい利益目標です。",
    "標準賃率": "前提コストと稼働時間から算出される基準の製造単価で、必要賃率と近い概念です。",
    "シナリオ": "前提条件のセットを保存したもの。複数登録するとダッシュボードで比較できます。",
    "感度分析": "前提条件を変えたときに賃率指標がどの程度変動するかを可視化する分析手法です。",
}

_PAGE_TUTORIALS: Dict[str, Dict[str, Any]] = {
    "home": {
        "goal": "アプリ全体の流れを素早く把握します。",
        "steps": [
            "左側の『① データ入力 & 取り込み』から標賃Excelまたはサンプルを読み込みます。",
            "読み込み後はステッパーの順番でデータ検証→KPIダッシュボード→標準賃率の感度分析へ進みます。",
            "迷ったらこのガイドとサイドバーの用語集を参照してください。",
        ],
        "tips": [
            "青いステッパーが現在地です。ホームに戻ると全体像を再確認できます。",
            f"[公開デモを見る]({_DEMO_URL}) から完成イメージを確認できます。",
        ],
        "terms": ["必要賃率", "ブレークイーブン賃率"],
    },
    "data": {
        "goal": "Excel原稿から製品マスタとコスト情報を取り込みます。",
        "steps": [
            "標準の『標賃』『R6.12』構成のExcelをアップロードします（未指定ならサンプルを使用）。",
            "読み込んだ内容は自動でバリデーションされ、固定費・必要利益などの前提値がセッションに保存されます。",
            "検索やフォームでSKUを確認・追加し、保存後にダッシュボードへ進みます。",
        ],
        "tips": [
            "読込エラーが出た場合はメッセージに沿って列名やシート構成を見直してください。",
            "追加した製品はセッション内に保持され、次のページの分析へ引き継がれます。",
        ],
        "terms": ["固定費", "必要利益", "必要賃率"],
    },
    "dashboard": {
        "goal": "SKU単位で必要賃率とのギャップや異常値を把握します。",
        "steps": [
            "左上のシナリオ選択やフィルタで比較したい条件を選びます。",
            "KPIカードで必要賃率・ブレークイーブン賃率・付加価値/分などの達成状況を確認します。",
            "要対策SKUリストや散布図でギャップの大きい製品を特定し、シナリオ反映で改善策を連携します。",
        ],
        "tips": [
            "ギャップは必要賃率−付加価値/分です。値が大きいほど改善余地が大きくなります。",
            "異常値検知タブでは欠損・外れ値などのデータ品質問題も確認できます。",
        ],
        "terms": ["必要賃率", "ブレークイーブン賃率", "付加価値/分", "ギャップ", "ROI"],
    },
    "standard_rate": {
        "goal": "前提コストを変更し、標準賃率やブレークイーブン賃率の感度を把握します。",
        "steps": [
            "必要固定費・必要利益・稼働時間などの前提値を入力します。",
            "シナリオを追加して仮定を保存し、ダッシュボードで比較できるようにします。",
            "下部のグラフで前提変更による賃率指標の変化を確認します。",
        ],
        "tips": [
            "複数シナリオを登録するとサイドバーで瞬時に切り替えられます。",
            "PDF出力ボタンから前提条件と感度分析結果を共有できます。",
        ],
        "terms": ["標準賃率", "必要賃率", "シナリオ", "感度分析"],
    },
}


def render_onboarding() -> None:
    """Display the first-time onboarding guidance until dismissed."""

    if st.session_state.get(_ONBOARDING_STATE_KEY, False):
        return

    container = st.container()
    with container:
        st.markdown("### 👋 はじめての方向けガイド")
        st.markdown(f"**{_ONBOARDING_HEADLINE}**")
        steps_md = "\n".join(f"- {step}" for step in _ONBOARDING_STEPS)
        st.markdown(steps_md)
        st.caption(_ONBOARDING_EFFECT)
        st.markdown(f"[公開デモを見る]({_DEMO_URL})")
        info_col, action_col = st.columns([5, 1])
        info_col.caption("ガイドはサイドバーの『チュートリアルを再表示』からいつでも開けます。")
        if action_col.button("閉じる", key="close_onboarding"):
            st.session_state[_ONBOARDING_STATE_KEY] = True
            toast = getattr(st, "toast", None)
            if callable(toast):
                toast("チュートリアルを折りたたみました。サイドバーから再表示できます。")


def render_page_tutorial(page_key: str) -> None:
    """Render a collapsible tutorial tailored to each page."""

    tutorial = _PAGE_TUTORIALS.get(page_key)
    if tutorial is None:
        return

    state_key = f"{_PAGE_STATE_PREFIX}{page_key}"
    collapsed = st.session_state.get(state_key, False)
    with st.expander("🎓 画面チュートリアル", expanded=not collapsed):
        st.markdown(f"**目的**: {tutorial['goal']}")

        steps: List[str] = tutorial.get("steps", [])
        if steps:
            steps_md = "\n".join(f"{idx}. {text}" for idx, text in enumerate(steps, start=1))
            st.markdown(f"**進め方**\n\n{steps_md}")

        tips: List[str] = tutorial.get("tips", [])
        if tips:
            tips_md = "\n".join(f"- {tip}" for tip in tips)
            st.markdown(f"**ヒント**\n\n{tips_md}")

        terms: List[str] = [term for term in tutorial.get("terms", []) if term in _GLOSSARY]
        if terms:
            st.markdown("**用語解説**")
            for term in terms:
                st.markdown(f"- **{term}**: {_GLOSSARY[term]}")

        if not collapsed:
            if st.button("次回は折りたたむ", key=f"collapse_tutorial_{page_key}"):
                st.session_state[state_key] = True
                toast = getattr(st, "toast", None)
                if callable(toast):
                    toast("ガイドはサイドバーから再表示できます。")
        else:
            st.caption("チュートリアルはサイドバーのボタンから再展開できます。")


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


def render_sidebar_nav(*, page_key: Optional[str] = None) -> None:
    """Render sidebar navigation links and tutorial shortcuts."""

    st.sidebar.header("ナビゲーション")
    st.sidebar.page_link("app.py", label="ホーム", icon="🏠")
    st.sidebar.page_link("pages/01_データ入力.py", label="① データ入力", icon="📥")
    st.sidebar.page_link("pages/02_ダッシュボード.py", label="② ダッシュボード", icon="📊")
    st.sidebar.page_link("pages/03_標準賃率計算.py", label="③ 標準賃率計算", icon="🧮")

    st.sidebar.divider()
    st.sidebar.subheader("チュートリアル")
    st.sidebar.caption("操作に迷ったらガイドを再表示してください。")
    button_key = f"show_tutorial_{page_key or 'global'}"
    if st.sidebar.button("👀 ガイドを再表示", use_container_width=True, key=button_key):
        st.session_state[_ONBOARDING_STATE_KEY] = False
        for key in _PAGE_TUTORIALS:
            st.session_state.pop(f"{_PAGE_STATE_PREFIX}{key}", None)
        toast = getattr(st, "toast", None)
        if callable(toast):
            toast("オンボーディングと各チュートリアルを再表示します。")

    tutorial = _PAGE_TUTORIALS.get(page_key or "")
    if tutorial:
        terms = [term for term in tutorial.get("terms", []) if term in _GLOSSARY]
        if terms:
            st.sidebar.markdown("**主要用語**")
            for term in terms:
                st.sidebar.caption(f"{term}: {_GLOSSARY[term]}")

    st.sidebar.caption(_ONBOARDING_EFFECT)
