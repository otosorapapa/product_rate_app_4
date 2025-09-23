import json
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from offline import render_offline_controls

try:  # pragma: no cover - optional bridge used only when available at runtime
    from streamlit_js_eval import streamlit_js_eval
except ModuleNotFoundError:  # pragma: no cover - allows tests to run without the dependency
    streamlit_js_eval = None  # type: ignore[misc]

_DEFAULT_THEME_KEY = "標準（ブルー）"
_DEFAULT_FONT_KEY = "ふつう"

_ACCESSIBILITY_STORAGE_KEY = "rate_app_accessibility_prefs_v1"
_ACCESSIBILITY_JS_PREFIX = "accessibility_prefs"
_ACCESSIBILITY_PREFS_FLAG = "_accessibility_prefs_loaded"

_THEME_PALETTES: Dict[str, Dict[str, str]] = {
    "標準（ブルー）": {
        "background": "#F4F7FA",
        "surface": "#FFFFFF",
        "text": "#1F2A44",
        "accent": "#2F6776",
        "border": "#CBD7E3",
        "muted": "#5F6B8A",
        "description": "やわらかなブルー基調の標準配色です。初めての方にも見やすく設計しています。",
    },
    "高コントラスト（濃紺×白）": {
        "background": "#0F172A",
        "surface": "#1F2937",
        "text": "#F9FAFB",
        "accent": "#F97316",
        "border": "#4B5563",
        "muted": "#E5E7EB",
        "description": "暗い背景と明るい文字でコントラストを最大化し、小さな文字も読みやすくします。",
    },
    "あたたかいセピア": {
        "background": "#F6F2EA",
        "surface": "#FFFBF5",
        "text": "#3F2F1E",
        "accent": "#B8631B",
        "border": "#E3D5C3",
        "muted": "#7B6651",
        "description": "目に優しい生成りカラー。長時間の閲覧でも疲れにくい落ち着いた配色です。",
    },
    "くっきり（白×黒）": {
        "background": "#FFFFFF",
        "surface": "#F8FAFC",
        "text": "#101828",
        "accent": "#B42318",
        "border": "#CBD5E1",
        "muted": "#475569",
        "description": "白地に濃色のテキストでコントラストを最大化したテーマです。印刷物と同じ感覚で閲覧できます。",
    },
}

_FONT_SCALE_OPTIONS: Dict[str, float] = {
    "ふつう": 1.0,
    "大きめ": 1.15,
    "特大": 1.3,
    "超特大": 1.45,
}


def _call_accessibility_js(expression: str, suffix: str) -> Optional[Any]:
    """Execute ``expression`` in the browser when :mod:`streamlit_js_eval` is available."""

    if streamlit_js_eval is None:  # pragma: no cover - bridge not installed during tests
        return None
    key = f"{_ACCESSIBILITY_JS_PREFIX}_{suffix}"
    try:
        return streamlit_js_eval(js_expressions=expression, key=key)
    except Exception:  # pragma: no cover - runtime JS errors should not break the app
        return None


def _load_accessibility_prefs() -> Optional[Dict[str, str]]:
    """Return stored accessibility preferences from the browser, if any."""

    raw = _call_accessibility_js(
        f"window.localStorage.getItem('{_ACCESSIBILITY_STORAGE_KEY}')",
        "get",
    )
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    result: Dict[str, str] = {}
    theme = data.get("theme")
    font = data.get("font")
    if isinstance(theme, str):
        result["theme"] = theme
    if isinstance(font, str):
        result["font"] = font
    return result or None


def _persist_accessibility_prefs(theme_key: str, font_key: str) -> None:
    """Persist the active theme and font selection to local storage."""

    if streamlit_js_eval is None:  # pragma: no cover - JS bridge unavailable in tests
        return
    payload = {"theme": theme_key, "font": font_key}
    payload_json = json.dumps(payload, ensure_ascii=False)
    _call_accessibility_js(
        f"window.localStorage.setItem('{_ACCESSIBILITY_STORAGE_KEY}', JSON.stringify({payload_json}))",
        "set",
    )

_HELP_CONTENT: Dict[str, Dict[str, Any]] = {
    "home": {
        "title": "ホーム画面のヘルプ",
        "intro": "アプリ全体の流れと主要な入口を確認できます。",
        "steps": [
            "左のナビゲーションから目的の画面を選択します。",
            "最初に『① データ入力 & 取り込み』でExcelを読み込みましょう。",
            "オンボーディングと画面チュートリアルで操作手順を確認できます。",
        ],
        "tips": [
            "サイドバー下部の『👁 アクセシビリティ設定』から文字サイズと配色を変更できます。",
            "ガイドを閉じてもサイドバーの『👀 ガイドを再表示』でいつでも呼び出せます。",
        ],
        "faqs": [
            {
                "question": "最初にどのページを操作すればよいですか？",
                "answer": "左サイドバーの『① データ入力 & 取り込み』からExcelを読み込むと、ダッシュボードや感度分析にもデータが共有されます。",
            },
            {
                "question": "ヘルプやガイドを後から見直す方法はありますか？",
                "answer": "サイドバー最下部付近の『👀 ガイドを再表示』ボタンを押すと、オンボーディングと各ページのチュートリアルをいつでも再表示できます。",
            },
        ],
    },
    "data": {
        "title": "データ入力画面のヘルプ",
        "intro": "Excel原稿を取り込み、製品マスタを整備する画面です。",
        "steps": [
            "『Excelテンプレート』でフォーマットを確認し、必要に応じてダウンロードします。",
            "ファイルをアップロードすると必須項目のチェックとクリーニングが自動で実行されます。",
            "検出されたエラーは修正してから再アップロードしてください。警告のみの場合は次に進めます。",
        ],
        "tips": [
            "検索ボックスで製品番号や名称を素早く絞り込みできます。",
            "『新規製品を追加』フォームから不足しているSKUを直接入力できます。",
        ],
        "faqs": [
            {
                "question": "Excelが読み込めない場合はどうすれば良いですか？",
                "answer": "テンプレートの列名やシート構成が変更されていないか確認し、必要に応じてサンプルテンプレートをダウンロードし直してコピーしてください。エラー一覧には想定される原因と対処のヒントが表示されます。",
            },
            {
                "question": "読み込んだ固定費や必要利益の設定は保存されますか？",
                "answer": "ブラウザのセッションに保存されるため、ページを移動しても同じ端末・ブラウザであれば設定が引き継がれます。通信が不安定な場合はサイドバーの『オフラインモード』から手動保存も可能です。",
            },
        ],
    },
    "dashboard": {
        "title": "ダッシュボード画面のヘルプ",
        "intro": "シナリオ別のKPIやギャップを俯瞰する分析ハブです。",
        "steps": [
            "上部のシナリオ選択で比較したい前提条件を指定します。",
            "KPIカードと要対策SKU表で必要賃率との差や優先度を確認します。",
            "グラフのフィルターと描画ツールを使うと改善ポイントを共有しやすくなります。",
        ],
        "tips": [
            "ダッシュボード画面右上の❓をクリックすると、各チャートの意味と使い方を確認できます。",
            "サイドバーの『グラフ操作オプション』でガイド線やレンジスライダーの表示を切り替えられます。",
        ],
        "faqs": [
            {
                "question": "『必要賃率差』はどのように解釈すれば良いですか？",
                "answer": "必要賃率から現在の付加価値/分を引いた値です。プラスの場合は必要賃率に届いていないため改善余地があり、マイナスの場合は達成済みを意味します。",
            },
            {
                "question": "グラフを資料に貼り付けたいときはどうすれば良いですか？",
                "answer": "Plotlyグラフは右上のカメラアイコンからPNGを保存できます。Altairグラフはメニューの『Download data』からデータをCSVとしてエクスポートし、PowerPoint等で再利用できます。",
            },
        ],
    },
    "standard_rate": {
        "title": "標準賃率計算画面のヘルプ",
        "intro": "固定費や必要利益の前提を変えながら感度分析を行います。",
        "steps": [
            "A〜Cの入力セクションで費用や稼働時間の前提値を調整します。",
            "右側のシナリオ管理で複数案を保存し、ダッシュボードと共有できます。",
            "グラフと表は入力値を変えると即座に再計算されます。",
        ],
        "tips": [
            "『PDF出力』で現在の前提条件と感度分析結果を資料として保存できます。",
            "感度グラフの凡例をクリックすると特定指標のみを強調表示できます。",
        ],
        "faqs": [
            {
                "question": "シナリオはどのように保存・切り替えできますか？",
                "answer": "右側のシナリオ管理で『シナリオを追加』を押すと現在の入力値を名前付きで保存できます。保存したシナリオはダッシュボード側でも選択でき、比較分析に活用できます。",
            },
            {
                "question": "固定費や必要利益の入力単位が分かりません。",
                "answer": "入力欄の単位は千円/年です。年間費用を千円単位で入力すると、必要賃率とブレークイーブン賃率が自動で再計算されます。",
            },
        ],
    },
    "chat": {
        "title": "チャットサポートのヘルプ",
        "intro": "AIに賃率や価格に関する疑問を質問できます。",
        "steps": [
            "画面下部の質問ボックスに知りたい内容を入力します。",
            "『損益分岐賃率の違い』などのFAQボタンを押すと定型質問を呼び出せます。",
            "回答にはインポート済みのデータと計算式を引用します。必要に応じてデータ取り込み画面で更新してください。",
        ],
        "tips": [
            "具体的な製品名や製品番号を含めると、必要販売単価を自動で算出します。",
            "回答履歴はセッション内で保持されます。『会話をリセット』で初期化できます。",
        ],
        "faqs": [
            {
                "question": "どのような質問に対応していますか？",
                "answer": "取り込んだ製品データと標準賃率計算の結果を参照して、必要販売単価の算出や賃率差の解釈などを回答します。製品番号やシナリオ名を含めると精度が高まります。",
            },
            {
                "question": "回答の根拠を確認するにはどうすれば良いですか？",
                "answer": "AIの回答には参照した指標や計算式を含めるよう設計されています。より詳しく知りたい場合は『根拠を詳しく教えて』と追い質問すると、追加の説明を得られます。",
            },
        ],
    },
}

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


def render_glossary_popover(
    terms: List[str],
    *,
    label: str = "用語の説明",
    container: Optional[DeltaGenerator] = None,
) -> None:
    """Display a popover listing glossary entries for the provided terms."""

    if not terms:
        return

    unique_terms: List[str] = []
    seen: set[str] = set()
    for term in terms:
        if term in _GLOSSARY and term not in seen:
            unique_terms.append(term)
            seen.add(term)

    if not unique_terms:
        return

    target = container or st
    title = f"📘 {label}"
    glossary_url_base = f"{_DEMO_URL}#glossary"

    with target.popover(title):
        st.caption("主要な用語の定義とリンクをまとめました。")
        for term in unique_terms:
            description = _GLOSSARY.get(term, "")
            st.markdown(f"**{term}**")
            if description:
                st.write(description)
            term_url = f"{glossary_url_base}?term={quote(term)}"
            st.markdown(f"[{term}の用語集を開く]({term_url})")


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
    "chat": {
        "goal": "AIとの対話で賃率の疑問を即時解消します。",
        "steps": [
            "下のFAQボタンを押すか質問を直接入力します。",
            "回答内の数式と引用値を確認し、意思決定に活用します。",
            "別シナリオを確認したい場合は『標準賃率計算』で前提を更新してから再度質問します。",
        ],
        "tips": [
            "製品名・品番を含めると必要販売単価を自動で計算します。",
            "右上のチャットヘルプから定型質問の例を参照できます。",
        ],
        "terms": ["必要賃率", "ブレークイーブン賃率", "付加価値/分"],
    },
}


def _ensure_theme_state() -> None:
    """Ensure theme-related options exist in :mod:`streamlit` session state."""

    if not st.session_state.get(_ACCESSIBILITY_PREFS_FLAG):
        stored = _load_accessibility_prefs()
        if stored:
            theme_pref = stored.get("theme")
            font_pref = stored.get("font")
            if isinstance(theme_pref, str) and theme_pref in _THEME_PALETTES:
                st.session_state["ui_theme"] = theme_pref
            if isinstance(font_pref, str) and font_pref in _FONT_SCALE_OPTIONS:
                st.session_state["ui_font_scale"] = font_pref
        st.session_state[_ACCESSIBILITY_PREFS_FLAG] = True

    theme_key = st.session_state.get("ui_theme", _DEFAULT_THEME_KEY)
    if theme_key not in _THEME_PALETTES:
        theme_key = _DEFAULT_THEME_KEY
    st.session_state["ui_theme"] = theme_key

    font_key = st.session_state.get("ui_font_scale", _DEFAULT_FONT_KEY)
    if font_key not in _FONT_SCALE_OPTIONS:
        font_key = _DEFAULT_FONT_KEY
    st.session_state["ui_font_scale"] = font_key


def _build_theme_css(theme: Dict[str, str], font_scale: float) -> str:
    """Return CSS for the selected theme and font scale."""

    base_font_px = round(16 * font_scale, 2)
    small_font_px = round(base_font_px * 0.85, 2)
    return f"""
    <style>
    :root {{
        --app-bg: {theme['background']};
        --app-surface: {theme['surface']};
        --app-text: {theme['text']};
        --app-accent: {theme['accent']};
        --app-border: {theme['border']};
        --app-muted: {theme['muted']};
        --app-font-base: {base_font_px}px;
        --app-font-small: {small_font_px}px;
    }}
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: var(--app-bg);
        color: var(--app-text);
        font-size: var(--app-font-base);
    }}
    body {{
        line-height: 1.6;
    }}
    h1 {{ font-size: calc(var(--app-font-base) * 1.7); }}
    h2 {{ font-size: calc(var(--app-font-base) * 1.45); }}
    h3 {{ font-size: calc(var(--app-font-base) * 1.25); }}
    h1, h2, h3, h4, h5, h6 {{
        color: var(--app-text);
        font-weight: 700;
    }}
    p, label, span, li {{
        color: var(--app-text);
    }}
    [data-testid="stHeader"] {{
        background-color: var(--app-surface);
        border-bottom: 1px solid var(--app-border);
    }}
    [data-testid="stSidebar"] {{
        background-color: var(--app-surface);
        border-right: 1px solid var(--app-border);
    }}
    [data-testid="stSidebar"] * {{
        color: var(--app-text);
    }}
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] .stMarkdown p {{
        color: var(--app-muted);
    }}
    .stCaption, caption {{
        color: var(--app-muted) !important;
        font-size: var(--app-font-small) !important;
    }}
    .stButton > button, .stDownloadButton > button {{
        background: var(--app-accent);
        color: #FFFFFF;
        border: none;
        border-radius: 999px;
        padding: 0.65rem 1.4rem;
        font-weight: 600;
        font-size: calc(var(--app-font-base) * 0.95);
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        filter: brightness(1.05);
    }}
    .stButton > button:focus-visible,
    .stDownloadButton > button:focus-visible {{
        outline: 3px solid var(--app-accent);
        outline-offset: 2px;
    }}
    input, textarea, select {{
        background-color: var(--app-surface);
        color: var(--app-text);
        border: 1px solid var(--app-border);
        border-radius: 8px;
    }}
    input:focus-visible, textarea:focus-visible, select:focus-visible {{
        outline: 2px solid var(--app-accent);
        outline-offset: 1px;
    }}
    [data-testid="stMetric"] {{
        background-color: var(--app-surface);
        border: 1px solid var(--app-border);
        border-radius: 18px;
        padding: 0.8rem 1rem;
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.08);
    }}
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] {{
        color: var(--app-text) !important;
    }}
    [data-testid="stMetricDelta"] span {{
        font-weight: 600;
    }}
    [data-testid="stAppViewContainer"] .stAlert {{
        border: 1px solid var(--app-border);
        background-color: var(--app-surface);
        color: var(--app-text);
    }}
    [data-testid="stExpander"] > div {{
        border: 1px solid var(--app-border);
        background-color: var(--app-surface);
    }}
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {{
        color: var(--app-text);
    }}
    [data-testid="dataframe-container"] * {{
        color: var(--app-text) !important;
    }}
    [data-testid="stTable"] th,
    [data-testid="stTable"] td {{
        color: var(--app-text);
        border-color: var(--app-border);
    }}
    [data-testid="stAppViewContainer"] a {{
        color: var(--app-accent);
        font-weight: 600;
    }}
    @media (max-width: 1080px) {{
        [data-testid="stSidebar"] {{
            width: 280px;
        }}
        .stButton > button,
        .stDownloadButton > button {{
            width: 100%;
        }}
    }}
    @media (max-width: 860px) {{
        [data-testid="stAppViewContainer"] {{
            padding-left: 0;
            padding-right: 0;
        }}
        [data-testid="block-container"] {{
            padding: 0.6rem 0.9rem 3rem;
        }}
        [data-testid="stHorizontalBlock"] {{
            flex-wrap: wrap;
            gap: 0.75rem;
        }}
        [data-testid="stHorizontalBlock"] > div {{
            flex: 1 1 100% !important;
            width: 100% !important;
        }}
        div[data-testid="column"] {{
            flex: 1 1 100% !important;
            width: 100% !important;
        }}
        [data-testid="stMetric"] {{
            width: 100%;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            flex-wrap: wrap;
        }}
    }}
    @media (max-width: 520px) {{
        h1 {{
            font-size: calc(var(--app-font-base) * 1.4);
        }}
        h2 {{
            font-size: calc(var(--app-font-base) * 1.25);
        }}
        .stButton > button,
        .stDownloadButton > button {{
            padding: 0.55rem 1.0rem;
        }}
    }}
    </style>
    """


def apply_user_theme() -> None:
    """Apply the active theme and font scale to the current Streamlit page."""

    _ensure_theme_state()
    theme_key = st.session_state["ui_theme"]
    font_key = st.session_state["ui_font_scale"]
    theme = _THEME_PALETTES.get(theme_key, _THEME_PALETTES[_DEFAULT_THEME_KEY])
    font_scale = _FONT_SCALE_OPTIONS.get(font_key, _FONT_SCALE_OPTIONS[_DEFAULT_FONT_KEY])
    css = _build_theme_css(theme, font_scale)
    st.markdown(css, unsafe_allow_html=True)
    st.session_state["_theme_css_injected"] = True


def get_active_theme_palette() -> Dict[str, str]:
    """Return the currently selected theme palette."""

    _ensure_theme_state()
    theme_key = st.session_state.get("ui_theme", _DEFAULT_THEME_KEY)
    return _THEME_PALETTES.get(theme_key, _THEME_PALETTES[_DEFAULT_THEME_KEY]).copy()


def render_help_button(
    page_key: str,
    *,
    align: str = "right",
    container: Optional[DeltaGenerator] = None,
) -> None:
    """Render a modal help button tailored to ``page_key``.

    Parameters
    ----------
    page_key:
        ページ固有のヘルプコンテンツを識別するキー。
    align:
        ``container`` を指定しない場合の配置。 ``"right"`` で右寄せ、 ``"left"`` で左寄せになります。
    container:
        ボタンを表示する ``streamlit`` のコンテナ。タイトル横に配置したい場合に列や空コンテナを渡します。
    """

    help_content = _HELP_CONTENT.get(page_key)
    if help_content is None:
        return

    state_key = f"help_modal_open_{page_key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = False

    if container is not None:
        button_container = container
    else:
        if align == "left":
            button_container, _ = st.columns([0.3, 0.7])
        else:
            _, button_container = st.columns([0.7, 0.3])

    if button_container.button(
        "❓ ヘルプ",
        key=f"help_button_{page_key}",
        use_container_width=True,
        help="画面の使い方とFAQを表示します。",
        type="primary",
    ):
        st.session_state[state_key] = True

    if not st.session_state.get(state_key):
        return

    def _render_help_sections() -> None:
        st.markdown(f"**{help_content['intro']}**")

        steps: List[str] = help_content.get("steps", [])
        if steps:
            st.markdown("**操作手順**")
            steps_md = "\n".join(
                f"{idx}. {text}" for idx, text in enumerate(steps, start=1)
            )
            st.markdown(steps_md)

        tips: List[str] = help_content.get("tips", [])
        if tips:
            st.markdown("**ヒント**")
            for tip in tips:
                st.markdown(f"- {tip}")

        faqs: List[Dict[str, str]] = help_content.get("faqs", [])
        if faqs:
            st.markdown("**よくある質問**")
            for faq in faqs:
                question = faq.get("question")
                answer = faq.get("answer")
                if not question or not answer:
                    continue
                st.markdown(f"**Q. {question}**")
                st.markdown(answer)

    modal = getattr(st, "modal", None)
    if callable(modal):
        with modal(help_content["title"]):
            _render_help_sections()

            if st.button(
                "閉じる",
                key=f"help_close_{page_key}",
                use_container_width=True,
            ):
                st.session_state[state_key] = False
    else:  # pragma: no cover - fallback for older Streamlit versions
        with st.expander(help_content["title"], expanded=True):
            _render_help_sections()
        st.session_state[state_key] = False


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

    _ensure_theme_state()
    if not st.session_state.get("_theme_css_injected"):
        apply_user_theme()

    st.sidebar.header("ナビゲーション")
    st.sidebar.page_link("app.py", label="ホーム", icon="🏠")
    st.sidebar.page_link("pages/01_データ入力.py", label="① データ入力", icon="📥")
    st.sidebar.page_link("pages/02_ダッシュボード.py", label="② ダッシュボード", icon="📊")
    st.sidebar.page_link("pages/03_標準賃率計算.py", label="③ 標準賃率計算", icon="🧮")
    st.sidebar.page_link("pages/04_チャットサポート.py", label="④ チャット/FAQ", icon="💬")

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

    st.sidebar.divider()
    st.sidebar.subheader("👁 アクセシビリティ設定")
    caption_lines = [
        "視認性が気になる場合は、ここから配色と文字サイズを調整してください。",
    ]
    if streamlit_js_eval is not None:
        caption_lines.append("設定は同じブラウザで保持されます。")
    else:
        caption_lines.append("ブラウザ保存が利用できない環境では、設定はセッション終了時にリセットされます。")
    st.sidebar.caption("\n".join(caption_lines))

    theme_options = list(_THEME_PALETTES.keys())
    selected_theme = st.sidebar.selectbox(
        "配色テーマ",
        theme_options,
        key="ui_theme",
        help="背景色とアクセントカラーの組み合わせを切り替えます。コントラストが強いテーマほど文字がくっきり表示されます。",
    )
    palette_preview = _THEME_PALETTES[selected_theme]
    st.sidebar.caption(palette_preview["description"])

    font_options = list(_FONT_SCALE_OPTIONS.keys())
    selected_font = st.sidebar.radio(
        "文字サイズ",
        font_options,
        key="ui_font_scale",
        help="本文・見出し・テーブルをまとめて拡大します。大きいほど読みやすくなります。",
    )
    if streamlit_js_eval is not None:
        persistence_note = "選択は同一ブラウザ内で保持されます。"
    else:
        persistence_note = "選択はページ再読み込みで初期化されます。"
    st.sidebar.caption(
        f"現在の文字サイズ: **{selected_font}** ／ {persistence_note}"
    )

    font_scale = _FONT_SCALE_OPTIONS[selected_font]
    preview_font_px = round(16 * font_scale, 1)
    preview_small_px = round(preview_font_px * 0.85, 1)
    st.sidebar.markdown(
        f"""
        <div style="margin-top:0.4rem; padding:0.7rem 0.85rem; border-radius:12px; border:1px solid {palette_preview['border']}; background:{palette_preview['surface']}; color:{palette_preview['text']}; font-size:{preview_font_px}px; line-height:1.6;">
            <div style="font-weight:700;">Aa あア 123</div>
            <div style="font-size:{preview_small_px}px; color:{palette_preview['muted']}; margin-top:0.25rem;">現在のテーマと文字サイズのプレビューです。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _persist_accessibility_prefs(selected_theme, selected_font)

    st.sidebar.caption(_ONBOARDING_EFFECT)

    st.sidebar.divider()
    render_offline_controls()
