import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    # Ensure our project root takes precedence so we import the local utils module
    # instead of any similarly named third-party package that might exist.
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from urllib.parse import urlencode
from datetime import date, datetime

from utils import compute_results, detect_quality_issues, detect_anomalies
from standard_rate_core import DEFAULT_PARAMS, sanitize_params, compute_rates
from components import (
    apply_user_theme,
    get_active_theme_palette,
    render_help_button,
    render_onboarding,
    render_page_tutorial,
    render_stepper,
    render_sidebar_nav,
)
import os
from typing import Dict, Any, List, Optional

from openai import OpenAI

PASTEL_PALETTE = [
    "#2F6776",
    "#79A3B1",
    "#F2C57C",
    "#9BC0A0",
    "#DDA0BC",
    "#AEC9EB",
]
PASTEL_ACCENT = "#2F6776"
PASTEL_BG = "#F4F7FA"
_PASTEL_THEME_NAME = "pastel_mck"
_PASTEL_THEME_CONFIG = {
    "config": {
        "background": PASTEL_BG,
        "view": {"stroke": "transparent"},
        "range": {"category": PASTEL_PALETTE},
        "title": {"color": "#1F2A44"},
        "axis": {
            "titleColor": "#1F2A44",
            "labelColor": "#30405B",
            "gridColor": "#D7E2EA",
        },
        "legend": {"labelColor": "#30405B", "titleColor": "#1F2A44"},
    }
}

_palette = get_active_theme_palette()
PASTEL_BG = _palette["surface"]
PASTEL_ACCENT = _palette["accent"]
_PASTEL_THEME_CONFIG["config"]["background"] = PASTEL_BG
_PASTEL_THEME_CONFIG["config"]["title"]["color"] = _palette["text"]
_PASTEL_THEME_CONFIG["config"]["axis"]["titleColor"] = _palette["text"]
_PASTEL_THEME_CONFIG["config"]["axis"]["labelColor"] = _palette["text"]
_PASTEL_THEME_CONFIG["config"]["axis"]["gridColor"] = _palette["border"]
_PASTEL_THEME_CONFIG["config"]["legend"]["labelColor"] = _palette["text"]
_PASTEL_THEME_CONFIG["config"]["legend"]["titleColor"] = _palette["text"]


apply_user_theme()


def _register_pastel_theme() -> None:
    """Register and enable the custom Altair theme across Altair versions."""

    try:
        theme_api = alt.theme
        if _PASTEL_THEME_NAME not in theme_api.names():

            @theme_api.register(_PASTEL_THEME_NAME, enable=False)
            def _pastel_theme():
                return _PASTEL_THEME_CONFIG

        theme_api.enable(_PASTEL_THEME_NAME)
    except (AttributeError, TypeError):
        if _PASTEL_THEME_NAME not in alt.themes.names():
            alt.themes.register(_PASTEL_THEME_NAME, _PASTEL_THEME_CONFIG)
        alt.themes.enable(_PASTEL_THEME_NAME)


_register_pastel_theme()

st.markdown(
    """
    <style>
    .main > div {
        background-color: var(--app-bg);
    }
    [data-testid="stMetric"] {
        background-color: var(--app-surface);
        border-radius: 18px;
        border: 1px solid var(--app-border);
        padding: 12px 16px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.06);
        color: var(--app-text);
    }
    [data-testid="stMetricDelta"] span {
        font-weight: 600;
    }
    .metric-badge {
        text-align: right;
        color: var(--app-accent);
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _apply_plotly_theme(fig: go.Figure, *, show_spikelines: bool = False, legend_bottom: bool = False) -> go.Figure:
    """Apply a consistent pastel style across plotly figures."""

    legend_conf = dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.0,
        bgcolor=_palette["surface"],
        bordercolor=_palette["border"],
        borderwidth=1,
    )
    if legend_bottom:
        legend_conf.update({"y": -0.2, "x": 0.5, "xanchor": "center"})

    fig.update_layout(
        plot_bgcolor=PASTEL_BG,
        paper_bgcolor=PASTEL_BG,
        font=dict(color=_palette["text"]),
        legend=legend_conf,
        margin=dict(l=40, r=30, t=60, b=60),
    )
    if show_spikelines:
        fig.update_layout(
            hovermode="x unified",
            xaxis=dict(showspikes=True, spikethickness=1, spikedash="dot"),
        )
        if "yaxis" in fig.layout:
            fig.update_layout(yaxis=dict(showspikes=True, spikethickness=1, spikedash="dot"))
    else:
        fig.update_layout(hovermode="closest")
    return fig


def _build_plotly_config() -> Dict[str, Any]:
    draw_tools = st.session_state.get(
        "plotly_draw_tools",
        ["drawline", "drawrect", "drawopenpath", "drawcircle", "eraseshape"],
    )
    return {
        "displaylogo": False,
        "responsive": True,
        "scrollZoom": True,
        "modeBarButtonsToAdd": draw_tools,
        "toImageButtonOptions": {"format": "png", "scale": 2},
    }


def _normalize_month(value: Any) -> Optional[pd.Timestamp]:
    """Convert arbitrary date-like input to the first day of its month."""

    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        ts = value
    elif isinstance(value, (datetime, date)):
        ts = pd.Timestamp(value)
    else:
        try:
            ts = pd.to_datetime(value)
        except Exception:
            return None
    if pd.isna(ts):
        return None
    return ts.to_period("M").to_timestamp()


def _pct_change(previous: float, current: float) -> float:
    """Compute percentage change while guarding against invalid denominators."""

    if previous in (None, 0) or pd.isna(previous):
        return np.nan
    if current in (None,) or pd.isna(current):
        return np.nan
    return (current / previous - 1.0) * 100.0


def _format_delta(value: float, suffix: str) -> str:
    """Format change metrics with sign and suffix, handling NaN gracefully."""

    if value is None or pd.isna(value) or not np.isfinite(value):
        return "N/A"
    return f"{value:+.1f}{suffix}"


def _upsert_trend_record(
    *,
    scenario: str,
    period: pd.Timestamp,
    ach_rate: float,
    va_per_min: float,
    required_rate: float,
    be_rate: float,
    note: str = "",
) -> pd.DataFrame:
    """Insert or update a monthly KPI snapshot for trend analysis."""

    record = {
        "scenario": scenario,
        "period": period,
        "ach_rate": float(ach_rate) if ach_rate is not None else np.nan,
        "va_per_min": float(va_per_min) if va_per_min is not None else np.nan,
        "required_rate": float(required_rate) if required_rate is not None else np.nan,
        "be_rate": float(be_rate) if be_rate is not None else np.nan,
        "note": note,
        "recorded_at": pd.Timestamp.utcnow(),
    }
    history = st.session_state.get("monthly_trend")
    if history is None or getattr(history, "empty", True):
        history = pd.DataFrame(columns=list(record.keys()))
    else:
        history = history.copy()
        history["period"] = pd.to_datetime(history["period"])
    mask = (history["scenario"] == scenario) & (history["period"] == period)
    history = history[~mask]
    history = pd.concat([history, pd.DataFrame([record])], ignore_index=True)
    history = history.sort_values(["period", "scenario"]).reset_index(drop=True)
    st.session_state["monthly_trend"] = history
    return history


def _prepare_trend_dataframe(trend_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Return data aggregated at the requested frequency for plotting."""

    if trend_df is None or trend_df.empty:
        return pd.DataFrame(columns=["scenario", "period", "ach_rate", "va_per_min", "required_rate", "be_rate"])
    df = trend_df.copy()
    df["period"] = pd.to_datetime(df["period"])
    df = df.dropna(subset=["period"])
    if freq == "四半期":
        df["period"] = df["period"].dt.to_period("Q").dt.to_timestamp()
        grouped = (
            df.groupby(["scenario", "period"], as_index=False)
            .agg(
                ach_rate=("ach_rate", "mean"),
                va_per_min=("va_per_min", "mean"),
                required_rate=("required_rate", "mean"),
                be_rate=("be_rate", "mean"),
            )
        )
        return grouped.sort_values(["period", "scenario"])
    df["period"] = df["period"].dt.to_period("M").dt.to_timestamp()
    df = df.sort_values(["period", "scenario"])
    return df[["scenario", "period", "ach_rate", "va_per_min", "required_rate", "be_rate", "note", "recorded_at"]]


def _build_yoy_summary(trend_df: pd.DataFrame, scenarios: List[str]) -> List[str]:
    """Create human-friendly YoY comparison sentences for the latest month."""

    if trend_df is None or trend_df.empty:
        return []
    df = trend_df.copy()
    df["period"] = pd.to_datetime(df["period"])
    df = df.dropna(subset=["period"])
    df["month"] = df["period"].dt.to_period("M")
    df = df.sort_values(["month", "scenario"])
    summaries: List[str] = []
    for scen in scenarios:
        scen_df = df[df["scenario"] == scen]
        if scen_df.empty:
            continue
        latest = scen_df.iloc[-1]
        prev_month = latest["month"] - 12
        prev_rows = scen_df[scen_df["month"] == prev_month]
        if prev_rows.empty:
            continue
        prev = prev_rows.iloc[-1]
        yoy_req = _pct_change(prev["required_rate"], latest["required_rate"])
        yoy_va = _pct_change(prev["va_per_min"], latest["va_per_min"])
        yoy_ach = latest["ach_rate"] - prev["ach_rate"]
        summaries.append(
            f"{scen}: 必要賃率 {_format_delta(yoy_req, '%')} / VA/分 {_format_delta(yoy_va, '%')} / 達成率 {_format_delta(yoy_ach, 'pt')}"
        )
    return summaries


def _generate_dashboard_comment(
    df: pd.DataFrame, metrics: Dict[str, float], insights: Dict[str, Any]
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OpenAI APIキーが設定されていません。"
    client = OpenAI(api_key=api_key)
    sample = df.head(5).to_markdown(index=False)
    top_gaps: List[Dict[str, Any]] = insights.get("top_underperformers", [])
    anomaly_summary: List[Dict[str, Any]] = insights.get("anomaly_summary", [])
    anomaly_details: List[Dict[str, Any]] = insights.get("anomaly_records", [])
    dq_summary = insights.get("dq_summary", {})

    top_gap_lines = []
    for row in top_gaps:
        roi = row.get("roi_months")
        roi_txt = "N/A" if roi is None or pd.isna(roi) else f"{float(roi):.1f}"
        gap_val = row.get("gap")
        gap_txt = "N/A" if gap_val is None or pd.isna(gap_val) else f"{float(gap_val):.2f}"
        top_gap_lines.append(
            f"- {row.get('product_name','不明')} (ギャップ {gap_txt}, ROI {roi_txt}ヶ月)"
        )
    top_gap_text = "\n".join(top_gap_lines) or "- 該当なし"

    anomaly_summary_text = "\n".join(
        [
            f"- {row['metric']}: {int(row['count'])}件 (平均逸脱 {row['severity_mean']:.1f})"
            for row in anomaly_summary
        ]
    ) or "- 大きな逸脱は検出されませんでした"

    anomaly_detail_lines = []
    for row in anomaly_details[:5]:
        value = row.get("value")
        median_val = row.get("median")
        val_txt = "N/A" if value is None or pd.isna(value) else f"{float(value):.2f}"
        median_txt = "N/A" if median_val is None or pd.isna(median_val) else f"{float(median_val):.2f}"
        anomaly_detail_lines.append(
            f"・{row.get('product_name','不明')} ({row.get('metric','-')}) = {val_txt} → 中央値 {median_txt}"
        )
    anomaly_detail_text = "\n".join(anomaly_detail_lines) or "・詳細サンプルなし"

    dq_text = (
        f"欠損{dq_summary.get('missing',0)}件 / 外れ値{dq_summary.get('negative',0)}件 / 重複{dq_summary.get('duplicate',0)}SKU"
        if dq_summary
        else "なし"
    )

    prompt = (
        "あなたは製造業向けの経営コンサルタントです。"
        "以下のKPIとデータサンプル、AIが抽出した追加インサイトを踏まえ、"
        "現状評価と優先アクション、リスクを3段落で構成し、最後に次の一歩を箇条書きで提案してください。\n"
        f"KPI: 達成率={metrics.get('ach_rate',0):.1f}%, "
        f"必要賃率={metrics.get('req_rate',0):.3f}, "
        f"損益分岐賃率={metrics.get('be_rate',0):.3f}\n"
        f"データ品質サマリ: {dq_text}\n"
        f"主要未達SKU:\n{top_gap_text}\n"
        f"異常検知サマリ:\n{anomaly_summary_text}\n"
        f"異常値サンプル:\n{anomaly_detail_text}\n"
        f"データサンプル:\n{sample}\n"
        "出力形式:\n"
        "1. 50文字以内の状況タイトル\n"
        "2. KPIの解釈 (箇条書き3点以内)\n"
        "3. 改善アクション提案 (箇条書き3点以内)\n"
        "4. リスク/ケアすべき点 (1-2点)\n"
        "5. 次の一歩 (1文)"
    )
    try:
        resp = client.responses.create(model="gpt-4o-mini", input=prompt)
        return resp.output_text.strip()
    except Exception as exc:
        return f"AIコメント生成に失敗しました: {exc}"

st.title("② ダッシュボード")
render_sidebar_nav(page_key="dashboard")
render_help_button("dashboard")

render_onboarding()
render_page_tutorial("dashboard")
render_stepper(4)
scenario_name = st.session_state.get("current_scenario", "ベース")
st.caption(f"適用中シナリオ: {scenario_name}")
scenario_options = ["ベース", "施策A"]
selected_scenarios = st.multiselect(
    "シナリオ選択", scenario_options, default=scenario_options
)
st.session_state.setdefault("quick_price", 0)
st.session_state.setdefault("quick_ct", 0)
st.session_state.setdefault("quick_material", 0)
st.session_state.setdefault(
    "plotly_draw_tools", ["drawline", "drawrect", "drawopenpath", "drawcircle", "eraseshape"]
)
st.session_state.setdefault("show_rangeslider", True)
st.session_state.setdefault("show_spikelines", True)

with st.sidebar.expander("グラフ操作オプション", expanded=False):
    st.session_state["show_spikelines"] = st.checkbox(
        "ホバー時にガイド線を表示", value=st.session_state["show_spikelines"], help="拡大モードでもX/Y方向のスパイクラインを表示します。"
    )
    st.session_state["show_rangeslider"] = st.checkbox(
        "時系列にレンジスライダーを表示", value=st.session_state["show_rangeslider"], help="月次トレンドなどを拡大表示した際にも範囲を素早く調整できます。"
    )
    st.session_state["plotly_draw_tools"] = st.multiselect(
        "描画ツール (拡大モードにも反映)",
        options=["drawline", "drawopenpath", "drawcircle", "drawrect", "eraseshape"],
        default=st.session_state["plotly_draw_tools"],
    )
    st.caption("設定は全Plotlyグラフのコントロールバーに適用されます。")


def reset_quick_params() -> None:
    """Reset quick simulation parameters to their default values."""
    st.session_state["quick_price"] = 0
    st.session_state["quick_ct"] = 0
    st.session_state["quick_material"] = 0

if "df_products_raw" not in st.session_state or st.session_state["df_products_raw"] is None or len(st.session_state["df_products_raw"]) == 0:
    st.info("先に『① データ入力 & 取り込み』でデータを準備してください。")
    st.stop()

df_raw_all = st.session_state["df_products_raw"]
excluded_skus = st.session_state.get("dq_exclude_skus", [])
df_products_raw = df_raw_all[~df_raw_all["product_no"].isin(excluded_skus)].copy()
dq_df = detect_quality_issues(df_products_raw)
miss_count = int((dq_df["type"] == "欠損").sum())
out_count = int((dq_df["type"] == "外れ値").sum())
dup_count = int((dq_df["type"] == "重複").sum())
affected_skus = dq_df["product_no"].nunique()
scenarios = st.session_state.get("scenarios", {scenario_name: st.session_state.get("sr_params", DEFAULT_PARAMS)})
st.session_state["scenarios"] = scenarios
base_params = scenarios.get(scenario_name, st.session_state.get("sr_params", DEFAULT_PARAMS))
base_params, warn_list = sanitize_params(base_params)
scenarios[scenario_name] = base_params
_, base_results = compute_rates(base_params)
be_rate = base_results["break_even_rate"]
req_rate = base_results["required_rate"]
for w in warn_list:
    st.warning(w)

# Baseline classification for reclassification counts
df_default = compute_results(df_products_raw, be_rate, req_rate)

# Threshold tuning slider within filter panel
dcol1, dcol2 = st.columns([2, 0.8])
delta_low, delta_high = dcol1.slider(
    "δ = VA/分 ÷ 必要賃率 の境界",
    min_value=0.5,
    max_value=1.5,
    value=(0.95, 1.05),
    step=0.01,
)
df = compute_results(df_products_raw, be_rate, req_rate, delta_low, delta_high)
reclassified = int((df["rate_class"] != df_default["rate_class"]).sum())
dcol2.metric("再分類SKU", reclassified)

with st.expander("表示設定", expanded=False):
    topn = int(
        st.slider("未達SKUの上位件数（テーブル/パレート）", min_value=5, max_value=50, value=20, step=1)
    )

# Global filters with view save/share
classes = df["rate_class"].dropna().unique().tolist()
global_mpu_min = float(np.nan_to_num(df["minutes_per_unit"].min(), nan=0.0))
global_mpu_max = float(np.nan_to_num(df["minutes_per_unit"].max(), nan=10.0))
global_v_min = float(np.nan_to_num(df["va_per_min"].replace([np.inf,-np.inf], np.nan).min(), nan=0.0))
global_v_max = float(np.nan_to_num(df["va_per_min"].replace([np.inf,-np.inf], np.nan).max(), nan=10.0))
qparams = dict(st.query_params)
default_classes = qparams.get("classes", ",".join(classes)).split(",")
default_classes = [c for c in default_classes if c in classes]
default_search = qparams.get("search", "")
mpu_param = qparams.get("mpu", f"{global_mpu_min},{global_mpu_max}")
try:
    m_min_q, m_max_q = [float(x) for x in mpu_param.split(",")]
except Exception:
    m_min_q, m_max_q = global_mpu_min, global_mpu_max
vapm_param = qparams.get("vapm", f"{global_v_min},{global_v_max}")
try:
    v_min_q, v_max_q = [float(x) for x in vapm_param.split(",")]
except Exception:
    v_min_q, v_max_q = global_v_min, global_v_max
fcol1, fcol2, fcol3, fcol4, fcol5, fcol6 = st.columns([1,1,2,2,0.5,0.5])
selected_classes = fcol1.multiselect("達成分類で絞り込み", classes, default=default_classes)
search = fcol2.text_input("製品名 検索（部分一致）", default_search)
mpu_min, mpu_max = fcol3.slider(
    "分/個（製造リードタイム）の範囲",
    global_mpu_min,
    global_mpu_max,
    value=(m_min_q, m_max_q)
)
vapm_min, vapm_max = fcol4.slider(
    "付加価値/分 の範囲",
    global_v_min,
    global_v_max,
    value=(v_min_q, v_max_q)
)
save_btn = fcol5.button("保存")
share_btn = fcol6.button("共有")
if save_btn or share_btn:
    state = {
        "classes": ",".join(selected_classes),
        "search": search,
        "mpu": f"{mpu_min},{mpu_max}",
        "vapm": f"{vapm_min},{vapm_max}"
    }
    st.query_params = state
    if share_btn:
        st.session_state["share_link"] = "?" + urlencode(state)
        st.session_state["show_share"] = True
    if save_btn:
        st.session_state["show_saved"] = True
    st.rerun()
if st.session_state.pop("show_saved", False):
    st.success("ビューを保存しました")
if st.session_state.pop("show_share", False):
    st.code(st.session_state.pop("share_link", ""), language=None)

mask = df["rate_class"].isin(selected_classes)
if search:
    mask &= df["product_name"].astype(str).str.contains(search, na=False)
mask &= df["minutes_per_unit"].fillna(0.0).between(mpu_min, mpu_max)
mask &= df["va_per_min"].replace([np.inf,-np.inf], np.nan).fillna(0.0).between(vapm_min, vapm_max)
df_view_filtered = df[mask].copy()

# Quick simulation toggles
qcol1, qcol2, qcol3, qcol4 = st.columns([1,1,1,0.6])
with qcol1:
    st.radio(
        "価格", options=[0,3,5,10], format_func=lambda x: f"+{x}%", key="quick_price", horizontal=True
    )
with qcol2:
    st.radio(
        "CT", options=[0,-5,-10], format_func=lambda x: f"{x}%", key="quick_ct", horizontal=True
    )
with qcol3:
    st.radio(
        "材料", options=[0,-3,-5], format_func=lambda x: f"{x}%", key="quick_material", horizontal=True
    )
with qcol4:
    st.button("Undo", on_click=reset_quick_params)

qp = st.session_state["quick_price"]
qc = st.session_state["quick_ct"]
qm = st.session_state["quick_material"]
df_base = df_view_filtered.copy()
base_ach_rate = (df_base["meets_required_rate"].mean()*100.0) if len(df_base)>0 else 0.0
base_avg_vapm = df_base["va_per_min"].replace([np.inf,-np.inf], np.nan).dropna().mean() if "va_per_min" in df_base else 0.0
df_sim = df_base.copy()
if qp:
    df_sim["actual_unit_price"] *= (1 + qp/100.0)
if qc:
    df_sim["minutes_per_unit"] *= (1 + qc/100.0)
if qm:
    df_sim["material_unit_cost"] *= (1 + qm/100.0)
df_sim["gp_per_unit"] = df_sim["actual_unit_price"] - df_sim["material_unit_cost"]
df_sim["daily_total_minutes"] = df_sim["minutes_per_unit"] * df_sim["daily_qty"]
df_sim["daily_va"] = df_sim["gp_per_unit"] * df_sim["daily_qty"]
with np.errstate(divide='ignore', invalid='ignore'):
    df_sim["va_per_min"] = df_sim["daily_va"] / df_sim["daily_total_minutes"]
df_view = compute_results(df_sim, be_rate, req_rate, delta_low, delta_high)
ach_rate = (df_view["meets_required_rate"].mean()*100.0) if len(df_view)>0 else 0.0
avg_vapm = df_view["va_per_min"].replace([np.inf,-np.inf], np.nan).dropna().mean() if "va_per_min" in df_view else 0.0
if qp or qc or qm:
    st.caption(f"Quick試算中: 価格{qp:+d}%, CT{qc:+d}%, 材料{qm:+d}%")

trend_history = st.session_state.get("monthly_trend")
if trend_history is None:
    trend_history = pd.DataFrame(
        columns=[
            "scenario",
            "period",
            "ach_rate",
            "va_per_min",
            "required_rate",
            "be_rate",
            "note",
            "recorded_at",
        ]
    )
    st.session_state["monthly_trend"] = trend_history

with st.expander("📈 月次スナップショットを記録", expanded=False):
    st.caption("現在表示中のKPIを対象月として保存します。再度同じ月を保存すると上書きされます。")
    default_month = st.session_state.get("trend_snapshot_month")
    if not isinstance(default_month, (datetime, date)):
        default_month = pd.Timestamp.today().to_pydatetime()
    col_t1, col_t2, col_t3, col_t4 = st.columns([1.3, 1.1, 1.1, 0.8])
    snapshot_month = col_t1.date_input("対象年月", value=default_month, key="trend_month_input")
    st.session_state["trend_snapshot_month"] = snapshot_month
    scen_default_idx = scenario_options.index(scenario_name) if scenario_name in scenario_options else 0
    scenario_for_snapshot = col_t2.selectbox(
        "対象シナリオ",
        options=scenario_options,
        index=scen_default_idx,
        key="trend_scenario_input",
    )
    note_value = col_t3.text_input("メモ (任意)", key="trend_note_input")
    save_snapshot = col_t4.button("保存/更新", key="trend_save_btn")

    if save_snapshot:
        period = _normalize_month(snapshot_month)
        if period is None:
            st.warning("対象年月を正しく指定してください。")
        else:
            metrics_map = {
                "ベース": (base_ach_rate, base_avg_vapm, df_base),
                "施策A": (ach_rate, avg_vapm, df_view),
            }
            ach_val, vapm_val, df_candidate = metrics_map.get(
                scenario_for_snapshot,
                (np.nan, np.nan, pd.DataFrame()),
            )
            if df_candidate.empty:
                st.warning("対象シナリオのデータがありません。フィルタ条件をご確認ください。")
            else:
                trend_history = _upsert_trend_record(
                    scenario=scenario_for_snapshot,
                    period=period,
                    ach_rate=ach_val,
                    va_per_min=vapm_val,
                    required_rate=req_rate,
                    be_rate=be_rate,
                    note=note_value,
                )
                st.success(f"{period.strftime('%Y-%m')} の {scenario_for_snapshot} を記録しました。")

    trend_history = st.session_state.get("monthly_trend", pd.DataFrame())
    if not trend_history.empty:
        history_display = trend_history.copy()
        history_display["period"] = pd.to_datetime(history_display["period"]).dt.strftime("%Y-%m")
        history_display["ach_rate"] = history_display["ach_rate"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
        history_display["va_per_min"] = history_display["va_per_min"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
        history_display["required_rate"] = history_display["required_rate"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
        history_display["be_rate"] = history_display["be_rate"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
        history_display = history_display[["period", "scenario", "ach_rate", "va_per_min", "required_rate", "be_rate", "note"]]
        st.dataframe(history_display, use_container_width=True)

        option_map = {
            f"{pd.to_datetime(row['period']).strftime('%Y-%m')}｜{row['scenario']}": (
                pd.to_datetime(row["period"]),
                row["scenario"],
            )
            for _, row in trend_history.sort_values(["period", "scenario"]).iterrows()
        }
        del_col1, del_col2 = st.columns([1.6, 0.4])
        delete_choice = del_col1.selectbox(
            "削除する記録",
            options=["選択なし"] + list(option_map.keys()),
            key="trend_delete_select",
        )
        if del_col2.button("削除", key="trend_delete_btn") and delete_choice != "選択なし":
            target_period, target_scenario = option_map[delete_choice]
            updated = trend_history[
                ~(
                    (trend_history["scenario"] == target_scenario)
                    & (pd.to_datetime(trend_history["period"]) == target_period)
                )
            ].reset_index(drop=True)
            st.session_state["monthly_trend"] = updated
            st.success(f"{target_period.strftime('%Y-%m')} の {target_scenario} を削除しました。")

# === KPI Targets & Cards ===
role = st.session_state.get("role", "一般")
st.session_state.setdefault("target_req_rate", req_rate)
st.session_state.setdefault("target_ach_rate", ach_rate)
with st.sidebar.expander("KPI目標設定", expanded=False):
    if role in ("経営者", "管理者"):
        st.session_state["target_req_rate"] = st.number_input(
            "目標必要賃率 (円/分)", value=st.session_state["target_req_rate"], format="%.3f"
        )
        st.session_state["target_ach_rate"] = st.number_input(
            "目標達成率 (%)", value=st.session_state["target_ach_rate"], format="%.1f"
        )
    else:
        st.number_input(
            "目標必要賃率 (円/分)", value=st.session_state["target_req_rate"], format="%.3f", disabled=True
        )
        st.number_input(
            "目標達成率 (%)", value=st.session_state["target_ach_rate"], format="%.1f", disabled=True
        )
target_req_rate = st.session_state["target_req_rate"]
target_ach_rate = st.session_state["target_ach_rate"]

anomaly_df = detect_anomalies(df_view)
if not anomaly_df.empty:
    anomaly_summary_stats = (
        anomaly_df.groupby("metric")
        .agg(count=("metric", "size"), severity_mean=("severity", "mean"))
        .reset_index()
        .sort_values(["count", "severity_mean"], ascending=[False, False])
    )
else:
    anomaly_summary_stats = pd.DataFrame(columns=["metric", "count", "severity_mean"])

gap_df = df_view.copy()
gap_df["gap"] = req_rate - gap_df["va_per_min"]
gap_df = gap_df[gap_df["gap"] > 0]
gap_df["price_improve"] = (gap_df["required_selling_price"] - gap_df["actual_unit_price"]).clip(lower=0)
gap_df["ct_improve"] = (gap_df["minutes_per_unit"] - (gap_df["gp_per_unit"] / req_rate)).clip(lower=0)
gap_df["material_improve"] = (
    gap_df["material_unit_cost"]
    - (gap_df["actual_unit_price"] - req_rate * gap_df["minutes_per_unit"])
).clip(lower=0)
gap_df["roi_months"] = gap_df["price_improve"].replace({0: np.nan}) / gap_df["gap"].replace({0: np.nan})
top_list = gap_df.sort_values("gap", ascending=False).head(20)
top_cards = top_list.head(5)


def _render_target_badge(col, text: str) -> None:
    col.markdown(
        f"<div class='metric-badge'><span style='background-color:#E0EEF4;padding:4px 10px;border-radius:999px;font-size:0.8em;'>🎯{text}</span></div>",
        unsafe_allow_html=True,
    )


col1, col2, col3, col5 = st.columns([1, 1, 1, 1])
_render_target_badge(col1, f"{target_req_rate:,.3f}")
col1.metric(
    "必要賃率 (円/分)", f"{req_rate:,.3f}", delta=f"{req_rate - target_req_rate:+.3f}"
)
_render_target_badge(col2, f"{target_ach_rate:.1f}%")
col2.metric(
    "必要賃率達成率 (%)", f"{ach_rate:.1f}", delta=f"{ach_rate - target_ach_rate:+.1f}"
)
col3.metric("損益分岐賃率 (円/分)", f"{be_rate:,.3f}")
with col5:
    dq_label = f"欠{miss_count} 外{out_count} 重{dup_count} / {affected_skus}SKU"
    st.markdown(
        f"<a href='#dq_errors' style='background-color:#F28B82;color:#1F2A44;padding:6px 10px;border-radius:999px;text-decoration:none;font-weight:600;display:inline-block;'>{dq_label}</a>",
        unsafe_allow_html=True,
    )

kpi_data = [
    {"scenario": "ベース", "KPI": "必要賃率達成SKU比率", "value": base_ach_rate},
    {"scenario": "ベース", "KPI": "平均 付加価値/分", "value": base_avg_vapm},
    {"scenario": "施策A", "KPI": "必要賃率達成SKU比率", "value": ach_rate},
    {"scenario": "施策A", "KPI": "平均 付加価値/分", "value": avg_vapm},
]
kpi_df = pd.DataFrame(kpi_data)
kpi_df = kpi_df[kpi_df["scenario"].isin(selected_scenarios)]
fig_kpi = px.bar(
    kpi_df,
    x="KPI",
    y="value",
    color="scenario",
    barmode="group",
    color_discrete_sequence=PASTEL_PALETTE,
)
fig_kpi.update_traces(opacity=0.85)
fig_kpi.update_yaxes(gridcolor="#D7E2EA")
fig_kpi.update_xaxes(gridcolor="#D7E2EA")
fig_kpi = _apply_plotly_theme(fig_kpi, legend_bottom=True)
st.plotly_chart(fig_kpi, use_container_width=True, config=_build_plotly_config())

ai_insights = {
    "top_underperformers": top_list[["product_name", "gap", "roi_months"]].head(3).to_dict("records")
    if not top_list.empty
    else [],
    "anomaly_summary": anomaly_summary_stats.to_dict("records"),
    "anomaly_records": anomaly_df.head(5).to_dict("records"),
    "dq_summary": {"missing": miss_count, "negative": out_count, "duplicate": dup_count},
}

st.subheader("AIコメント")
if st.button("AIコメント生成"):
    with st.spinner("生成中..."):
        st.session_state["dashboard_ai_comment"] = _generate_dashboard_comment(
            df_view,
            {"ach_rate": ach_rate, "req_rate": req_rate, "be_rate": be_rate},
            ai_insights,
        )
st.markdown(st.session_state.get("dashboard_ai_comment", ""))

st.markdown("<div id='dq_errors'></div>", unsafe_allow_html=True)
st.subheader("データ品質エラー一覧")
if dq_df.empty:
    st.success("エラーはありません。")
else:
    dq_display = dq_df.rename(
        columns={
            "product_no": "製品番号",
            "product_name": "製品名",
            "type": "種別",
            "column": "項目",
        }
    )
    dq_display.insert(0, "除外", dq_display["製品番号"].isin(excluded_skus))
    edited = st.data_editor(dq_display, use_container_width=True, key="dq_editor")
    new_excluded = edited[edited["除外"]]["製品番号"].unique().tolist()
    if set(new_excluded) != set(excluded_skus):
        st.session_state["dq_exclude_skus"] = new_excluded
        st.rerun()

st.subheader("異常値ハイライト")
if anomaly_df.empty:
    st.success("統計的な異常値は検出されませんでした。")
else:
    highlight = anomaly_df.head(3)
    cols = st.columns(len(highlight))
    for col, row in zip(cols, highlight.to_dict("records")):
        direction = "上振れ" if row.get("direction") == "high" else "下振れ"
        val_txt = "N/A" if pd.isna(row.get("value")) else f"{row['value']:.2f}"
        col.metric(
            f"{row.get('product_name', '不明')} ({row.get('metric', '-')})",
            val_txt,
            delta=f"{direction} z≈{row.get('severity', 0):.1f}",
        )

    if not anomaly_summary_stats.empty:
        summary_df = anomaly_summary_stats.rename(
            columns={"metric": "指標", "count": "件数", "severity_mean": "平均逸脱"}
        )
        st.dataframe(summary_df, use_container_width=True)

    detail_df = anomaly_df.head(20).rename(
        columns={
            "product_no": "製品番号",
            "product_name": "製品名",
            "metric": "指標",
            "value": "値",
            "direction": "方向",
            "severity": "逸脱度",
            "median": "中央値",
            "iqr_lower": "IQR下限",
            "iqr_upper": "IQR上限",
        }
    )
    with st.expander("異常値詳細 (上位20件)", expanded=False):
        st.dataframe(detail_df, use_container_width=True)

st.divider()

# Actionable SKU Top List
st.subheader("要対策SKUトップリスト")
st.caption("ギャップ = 必要賃率 - 付加価値/分")
top5 = top_cards
if len(top5) > 0:
    card_cols = st.columns(len(top5))
    for col, row in zip(card_cols, top5.to_dict("records")):
        roi_txt = "N/A" if pd.isna(row.get("roi_months")) else f"{row['roi_months']:.1f}"
        gap_txt = "N/A" if pd.isna(row.get("gap")) else f"{row['gap']:.2f}"
        col.metric(row["product_name"], gap_txt, delta=f"ROI {roi_txt}月")
        col.caption(
            " / ".join(
                [
                    f"価格+{row['price_improve']:.1f}" if not pd.isna(row.get("price_improve")) else "価格改善情報なし",
                    f"CT-{row['ct_improve']:.2f}" if not pd.isna(row.get("ct_improve")) else "CT改善情報なし",
                    f"材料-{row['material_improve']:.1f}" if not pd.isna(row.get("material_improve")) else "材料改善情報なし",
                ]
            )
        )

    table = top_list[[
        "product_no","product_name","gap","price_improve","ct_improve","material_improve","roi_months"
    ]].rename(columns={
        "product_no":"製品番号",
        "product_name":"製品名",
        "gap":"ギャップ",
        "price_improve":"価格改善",
        "ct_improve":"CT改善",
        "material_improve":"材料改善",
        "roi_months":"想定ROI(月)"
    })
    table.insert(0, "選択", False)
    edited = st.data_editor(table, use_container_width=True, key="action_sku_editor")
    csv_top = edited.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "CSV出力",
        data=csv_top,
        file_name="action_sku_top20.csv",
        mime="text/csv",
    )
    selected = edited[edited["選択"]]
    if st.button("シナリオに反映"):
        st.session_state["selected_action_skus"] = selected
        st.success(f"{len(selected)}件をシナリオに反映しました")
else:
    st.info("要対策SKUはありません。")

tabs = st.tabs(["全体分布（散布図）", "時系列", "達成状況（棒/円）", "未達SKU（パレート）", "SKUテーブル", "付加価値/分分布"])

with tabs[0]:
    st.caption(
        "横軸=分/個（製造リードタイム）, 縦軸=付加価値/分。必要賃率×δ帯と損益分岐賃率を表示。"
    )
    df_base["scenario"] = "ベース"
    df_view["scenario"] = "施策A"
    scatter_df = pd.concat([df_base, df_view], ignore_index=True)
    scatter_df = scatter_df[scatter_df["scenario"].isin(selected_scenarios)].copy()
    scatter_df["margin_to_req"] = req_rate - scatter_df["va_per_min"]
    fig = px.scatter(
        scatter_df,
        x="minutes_per_unit",
        y="va_per_min",
        color="scenario",
        hover_data={
            "product_name": True,
            "minutes_per_unit": ":.2f",
            "va_per_min": ":.2f",
            "margin_to_req": ":.2f",
        },
        opacity=0.8,
        color_discrete_sequence=PASTEL_PALETTE,
        height=420,
    )
    fig.update_traces(marker=dict(size=9, line=dict(color="#FFFFFF", width=0.6)))
    fig.add_hrect(
        y0=req_rate * delta_low,
        y1=req_rate * delta_high,
        line_width=0,
        fillcolor="#9BC0A0",
        opacity=0.15,
    )
    fig.add_hline(y=req_rate, line_color="#2F6776", line_width=2)
    fig.add_hline(y=be_rate, line_color="#E7A07A", line_dash="dash")
    fig.update_xaxes(title="分/個", gridcolor="#D7E2EA")
    fig.update_yaxes(title="付加価値/分", gridcolor="#D7E2EA")
    fig = _apply_plotly_theme(fig, show_spikelines=st.session_state["show_spikelines"])
    st.plotly_chart(fig, use_container_width=True, config=_build_plotly_config())

with tabs[1]:
    st.caption("月次・四半期のKPI推移を確認し、施策効果をトレースします。")
    trend_df = st.session_state.get("monthly_trend", pd.DataFrame())
    if trend_df.empty:
        st.info("『月次スナップショットを記録』からデータを保存すると時系列が表示されます。")
    else:
        available_scenarios = sorted(trend_df["scenario"].dropna().unique().tolist())
        filtered = trend_df[trend_df["scenario"].isin([s for s in selected_scenarios if s in available_scenarios])]
        if filtered.empty:
            st.warning("選択中のシナリオでは時系列データがまだ登録されていません。")
        else:
            st.session_state.setdefault("trend_freq", "月次")
            freq_choice = st.radio(
                "集計粒度",
                options=["月次", "四半期"],
                horizontal=True,
                key="trend_freq",
            )
            plot_df = _prepare_trend_dataframe(filtered, freq_choice)
            if plot_df.empty:
                st.warning("表示対象の時系列データが不足しています。")
            else:
                yoy_lines = _build_yoy_summary(
                    trend_df,
                    sorted(plot_df["scenario"].unique()),
                )
                if yoy_lines:
                    st.markdown("**前年同月比**")
                    st.markdown("\n".join(f"- {line}" for line in yoy_lines))
                fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
                scenario_colors = {
                    scen: PASTEL_PALETTE[idx % len(PASTEL_PALETTE)]
                    for idx, scen in enumerate(sorted(plot_df["scenario"].unique()))
                }
                for scen, group in plot_df.groupby("scenario"):
                    g = group.sort_values("period")
                    fig_ts.add_trace(
                        go.Scatter(
                            x=g["period"],
                            y=g["va_per_min"],
                            mode="lines+markers",
                            name=f"{scen} VA/分",
                            line=dict(color=scenario_colors.get(scen), width=2.5),
                            marker=dict(size=8),
                        ),
                        secondary_y=False,
                    )
                    fig_ts.add_trace(
                        go.Scatter(
                            x=g["period"],
                            y=g["required_rate"],
                            mode="lines+markers",
                            name=f"{scen} 必要賃率",
                            line=dict(color=scenario_colors.get(scen), width=2, dash="dash"),
                            marker=dict(size=7, symbol="diamond"),
                        ),
                        secondary_y=False,
                    )
                    fig_ts.add_trace(
                        go.Scatter(
                            x=g["period"],
                            y=g["ach_rate"],
                            mode="lines+markers",
                            name=f"{scen} 達成率",
                            line=dict(color=scenario_colors.get(scen), width=2, dash="dot"),
                            marker=dict(size=7),
                            opacity=0.8,
                        ),
                        secondary_y=True,
                    )
                fig_ts.update_yaxes(
                    title_text="VA/分・必要賃率 (円/分)",
                    secondary_y=False,
                    gridcolor="#D7E2EA",
                )
                fig_ts.update_yaxes(
                    title_text="必要賃率達成率 (%)",
                    range=[0, 100],
                    secondary_y=True,
                    gridcolor="#D7E2EA",
                )
                fig_ts.update_xaxes(
                    gridcolor="#D7E2EA",
                    rangeslider=dict(visible=st.session_state["show_rangeslider"]),
                    rangeselector=dict(
                        buttons=[
                            dict(count=3, label="3M", step="month", stepmode="backward"),
                            dict(count=6, label="6M", step="month", stepmode="backward"),
                            dict(step="all", label="ALL"),
                        ],
                        bgcolor="rgba(47,103,118,0.08)",
                        activecolor="#2F6776",
                        font=dict(color="#1F2A44"),
                    ),
                )
                fig_ts = _apply_plotly_theme(
                    fig_ts, show_spikelines=st.session_state["show_spikelines"]
                )
                st.plotly_chart(fig_ts, use_container_width=True, config=_build_plotly_config())

                display_df = plot_df.copy()
                if freq_choice == "四半期":
                    display_df["期間"] = display_df["period"].dt.to_period("Q").astype(str)
                else:
                    display_df["期間"] = display_df["period"].dt.strftime("%Y-%m")
                display_df = display_df.sort_values(["period", "scenario"])
                summary_table = pd.DataFrame(
                    {
                        "期間": display_df["期間"],
                        "シナリオ": display_df["scenario"],
                        "必要賃率 (円/分)": display_df["required_rate"].map(
                            lambda x: f"{x:.3f}" if pd.notna(x) else "-"
                        ),
                        "平均VA/分": display_df["va_per_min"].map(
                            lambda x: f"{x:.2f}" if pd.notna(x) else "-"
                        ),
                        "必要賃率達成率": display_df["ach_rate"].map(
                            lambda x: f"{x:.1f}%" if pd.notna(x) else "-"
                        ),
                        "損益分岐賃率": display_df["be_rate"].map(
                            lambda x: f"{x:.3f}" if pd.notna(x) else "-"
                        ),
                    }
                )
                st.dataframe(summary_table, use_container_width=True)

with tabs[2]:
    c1, c2 = st.columns([1.2,1])
    class_counts = df_view["rate_class"].value_counts().reset_index()
    class_counts.columns = ["rate_class", "count"]
    bar = alt.Chart(class_counts).mark_bar(color=PASTEL_ACCENT).encode(
        x=alt.X("rate_class:N", title="達成分類"),
        y=alt.Y("count:Q", title="件数"),
        tooltip=["rate_class","count"]
    ).properties(height=380)
    c1.altair_chart(bar, use_container_width=True)

    # Achievers vs Missed donut
    donut_df = pd.DataFrame({
        "group": ["達成", "未達"],
        "value": [ (df_view["meets_required_rate"].sum()), ( (~df_view["meets_required_rate"]).sum() ) ]
    })
    donut = (
        alt.Chart(donut_df)
        .mark_arc(innerRadius=80)
        .encode(
            theta="value:Q",
            color=alt.Color(
                "group:N",
                scale=alt.Scale(range=[PASTEL_ACCENT, "#DDA0BC"]),
                title="達成状況",
            ),
            tooltip=["group", "value"],
        )
    )
    c2.altair_chart(donut, use_container_width=True)

with tabs[3]:
    miss = df_view[df_view["meets_required_rate"] == False].copy()
    miss = miss.sort_values("rate_gap_vs_required").head(topn)
    st.caption("『必要賃率差』が小さい（またはマイナスが大）の順。右ほど改善余地が大。")
    if len(miss)==0:
        st.success("未達SKUはありません。")
    else:
        pareto = alt.Chart(miss).mark_bar(color=PASTEL_ACCENT).encode(
            x=alt.X("product_name:N", sort="-y", title="製品名"),
            y=alt.Y("rate_gap_vs_required:Q", title="必要賃率差（付加価値/分 - 必要賃率）"),
            tooltip=["product_name","rate_gap_vs_required"]
        ).properties(height=420)
        st.altair_chart(pareto, use_container_width=True)
        st.dataframe(miss[["product_no","product_name","minutes_per_unit","va_per_min","rate_gap_vs_required","price_gap_vs_required"]], use_container_width=True)

with tabs[4]:
    rename_map = {
        "product_no": "製品番号",
        "product_name": "製品名",
        "actual_unit_price": "実際売単価",
        "material_unit_cost": "材料原価",
        "minutes_per_unit": "分/個",
        "daily_qty": "日産数",
        "daily_total_minutes": "日産合計(分)",
        "gp_per_unit": "粗利/個",
        "daily_va": "付加価値(日産)",
        "va_per_min": "付加価値/分",
        "be_va_unit_price": "損益分岐付加価値単価",
        "req_va_unit_price": "必要付加価値単価",
        "required_selling_price": "必要販売単価",
        "price_gap_vs_required": "必要販売単価差額",
        "rate_gap_vs_required": "必要賃率差",
        "meets_required_rate": "必要賃率達成",
        "rate_class": "達成分類",
    }
    ordered_cols = [
        "製品番号","製品名","実際売単価","必要販売単価","必要販売単価差額","材料原価","粗利/個",
        "分/個","日産数","日産合計(分)","付加価値(日産)","付加価値/分",
        "損益分岐付加価値単価","必要付加価値単価","必要賃率差","必要賃率達成","達成分類",
    ]
    df_table = df_view.rename(columns=rename_map)
    df_table = df_table[[c for c in ordered_cols if c in df_table.columns]]

    st.dataframe(df_table, use_container_width=True, height=520)
    csv = df_table.to_csv(index=False).encode("utf-8-sig")
    st.download_button("結果をCSVでダウンロード", data=csv, file_name="calc_results.csv", mime="text/csv")

with tabs[5]:
    hist = alt.Chart(df_view).mark_bar(color=PASTEL_ACCENT).encode(
        x=alt.X("va_per_min:Q", bin=alt.Bin(maxbins=30), title="付加価値/分"),
        y=alt.Y("count()", title="件数"),
        tooltip=["count()"]
    ).properties(height=420)
    st.altair_chart(hist, use_container_width=True)
