import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    # Ensure our project root takes precedence so we import the local rate_utils module
    # instead of any similarly named third-party package that might exist.
    sys.path.insert(0, str(BASE_DIR))

from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from urllib.parse import urlencode
from datetime import date, datetime

from rate_utils import (
    compute_results,
    detect_quality_issues,
    detect_anomalies,
    summarize_segment_performance,
)
from standard_rate_core import DEFAULT_PARAMS, sanitize_params, compute_rates
from components import (
    apply_user_theme,
    get_active_theme_palette,
    render_help_button,
    render_onboarding,
    render_page_tutorial,
    render_stepper,
    render_sidebar_nav,
    render_indicator_cards,
)
from offline import restore_session_state_from_cache, sync_offline_cache
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

from openai import OpenAI

PASTEL_PALETTE = [
    "#0B1F3B",
    "#1E88E5",
    "#5A6B7A",
    "#69B36C",
    "#FCA333",
    "#EA615D",
]
PASTEL_ACCENT = "#1E88E5"
PASTEL_BG = "#F7F8FA"
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

restore_session_state_from_cache()


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

COLOR_ACCENT = _palette["accent"]
COLOR_ERROR = _palette.get("danger", "#B75C5C")
COLOR_SUCCESS = _palette.get("success", "#3B8363")
COLOR_SECONDARY = _palette.get("border", "#94A3B8")

STATUS_MESSAGES: Dict[str, Dict[str, Any]] = {
    "no_data": {
        "level": "warning",
        "text": "該当期間のデータはありません。期間や店舗を変更して再度お試しください。",
        "button": {"label": "別の期間を選ぶ", "action": "reset_period"},
        "persist": True,
    },
    "loading": {
        "level": "info",
        "text": "データを読み込んでいます…",
        "persist": True,
    },
    "error": {
        "level": "error",
        "text": "データ取得に失敗しました。数分後に再度お試しください。",
        "button": {"label": "再読み込み", "action": "rerun"},
        "persist": True,
    },
    "empty_filter": {
        "level": "warning",
        "text": "該当する項目がありません。フィルタ条件を緩めてください。",
        "button": {"label": "フィルタをリセット", "action": "reset_filters"},
        "persist": True,
    },
    "success_export": {
        "level": "success",
        "text": "ファイルをダウンロードしました。ご確認ください。",
        "persist": False,
    },
}

_STORE_NAME_MAP: Dict[str, str] = {
    "Square POS": "本店",
    "square": "本店",
    "スマレジPOS": "2号店",
    "smaregi": "2号店",
    "freee会計": "オンライン",
    "freee": "オンライン",
    "MFクラウド会計": "オンライン",
    "mf_cloud": "オンライン",
    "弥生会計": "本店",
    "yayoi": "本店",
}

_DEFAULT_STORE_OPTION = "全店舗"


def _set_status(key: Optional[str]) -> None:
    """Update the status flag in session state."""

    if key:
        st.session_state["status"] = key
    else:
        st.session_state.pop("status", None)


def _handle_status_action(action: Optional[str], default_period: pd.Timestamp, default_store: str) -> None:
    """Perform a follow-up action for status banner buttons."""

    if not action:
        _set_status(None)
        return
    if action == "reset_period":
        st.session_state["selected_period"] = default_period
        _set_status(None)
    elif action == "reset_filters":
        st.session_state["selected_store"] = default_store
        st.session_state["inventory_filter_mode"] = "不足のみ"
        _set_status(None)
    elif action == "rerun":
        _set_status(None)
        st.experimental_rerun()
    else:
        _set_status(None)


def _render_status_banner(default_period: pd.Timestamp, default_store: str) -> None:
    """Display contextual status messages with optional follow-up actions."""

    key = st.session_state.get("status")
    if not key:
        return
    info = STATUS_MESSAGES.get(key)
    if not info:
        _set_status(None)
        return

    level = info.get("level", "info")
    message = info.get("text", "")
    action = info.get("button")

    if level == "success":
        st.toast(message or "完了しました。", icon="📁")
        _set_status(None)
        return

    renderers = {
        "info": st.info,
        "warning": st.warning,
        "error": st.error,
    }
    renderer = renderers.get(level, st.info)

    renderer(message)
    if action:
        if st.button(action.get("label", "再試行"), key=f"status_action_{key}"):
            _handle_status_action(action.get("action"), default_period, default_store)
            return

    if not info.get("persist", False):
        _set_status(None)


def _safe_to_numeric(series: pd.Series) -> pd.Series:
    """Convert a Series to numeric values while filling NaNs with 0."""

    converted = pd.to_numeric(series, errors="coerce")
    return converted.fillna(0.0)


def _infer_category_label(name: Any) -> str:
    """Infer a coarse category label from the product name."""

    if not isinstance(name, str) or not name:
        return "その他"
    name = name.strip()
    if any(keyword in name for keyword in ["苺", "桃", "栗"]):
        return "季節限定"
    if "大福" in name or "饅頭" in name:
        return "定番商品"
    if "ギフト" in name or "詰め合わせ" in name:
        return "ギフト"
    return "その他"


def _infer_channel_from_product(product_no: Any) -> str:
    """Return a deterministic channel label from product numbers."""

    try:
        tail_digit = int(str(product_no)[-1])
    except (TypeError, ValueError):
        tail_digit = 0
    return "EC" if tail_digit % 2 == 0 else "店舗"


def _map_store_label(value: Any) -> str:
    """Normalise various vendor/source labels to store-friendly names."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "本店"
    text = str(value)
    return _STORE_NAME_MAP.get(text, _STORE_NAME_MAP.get(text.lower(), "本店"))


def _prepare_transactions_dataset(products: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Return a harmonised transaction dataset used across behaviour tabs."""

    raw = st.session_state.get("external_sync_last_transactions")
    if isinstance(raw, pd.DataFrame) and not raw.empty:
        tx = raw.copy()
    else:
        sample_path = Path(__file__).resolve().parents[1] / "data" / "external" / "pos_transactions_sample.csv"
        tx = pd.read_csv(sample_path)

    if "source" not in tx.columns:
        tx["source"] = "Square POS"

    tx["date"] = pd.to_datetime(tx.get("date"), errors="coerce")
    tx = tx.dropna(subset=["date", "product_no"])
    tx["product_no"] = tx["product_no"].astype(str)

    for column in ("sales_amount", "material_cost", "sold_qty"):
        if column in tx.columns:
            tx[column] = _safe_to_numeric(tx[column])
        else:
            tx[column] = 0.0

    tx["store"] = tx.get("store")
    tx["store"] = tx["store"].where(tx["store"].notna(), tx["source"])
    tx["store"] = tx["store"].map(_map_store_label)

    product_info = pd.DataFrame()
    if isinstance(products, pd.DataFrame) and not products.empty:
        product_info = products.copy()
        product_info["product_no"] = product_info["product_no"].astype(str)
        if "category" not in product_info.columns:
            product_info["category"] = product_info["product_name"].apply(_infer_category_label)
        else:
            product_info["category"] = product_info["category"].fillna(
                product_info["product_name"].apply(_infer_category_label)
            )
        product_info = product_info[["product_no", "product_name", "category"]]

    if not product_info.empty:
        tx = tx.merge(product_info, on="product_no", how="left")
    else:
        tx["product_name"] = tx["product_no"]
        tx["category"] = tx["product_name"].apply(_infer_category_label)

    tx["channel"] = tx["product_no"].apply(_infer_channel_from_product)
    tx["period"] = tx["date"].dt.to_period("M").dt.to_timestamp()
    tx["gross_profit"] = tx["sales_amount"] - tx["material_cost"]
    tx = tx.sort_values(["date", "product_no"]).reset_index(drop=True)
    return tx


def _prepare_inventory_snapshot(products: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Synthesise an inventory table from uploaded product data."""

    if not isinstance(products, pd.DataFrame) or products.empty:
        return pd.DataFrame(
            columns=[
                "product_no",
                "product_name",
                "store",
                "category",
                "on_hand",
                "safety_stock",
                "shortage",
                "coverage_days",
                "reorder_link",
            ]
        )

    inv = products.copy()
    inv["product_no"] = inv["product_no"].astype(str)
    if "category" not in inv.columns:
        inv["category"] = inv["product_name"].apply(_infer_category_label)
    else:
        inv["category"] = inv["category"].fillna(inv["product_name"].apply(_infer_category_label))

    inv["store"] = inv["product_no"].apply(lambda x: "本店" if int(str(x)[-1]) % 3 != 0 else "2号店")
    inv["daily_qty"] = _safe_to_numeric(inv.get("daily_qty", 0))
    inv["on_hand"] = (inv["daily_qty"] * 3).round().astype(int)
    inv["safety_stock"] = (inv["daily_qty"] * 4).round().astype(int)
    inv["shortage"] = inv["safety_stock"] - inv["on_hand"]
    inv["coverage_days"] = np.where(
        inv["daily_qty"] > 0,
        (inv["on_hand"] / inv["daily_qty"]).round(1),
        np.nan,
    )
    inv["reorder_link"] = inv.apply(
        lambda row: (
            "mailto:purchase@example.com"
            + f"?subject={row['product_name']}の追加発注&body=不足数: {max(int(row['shortage']), 0)}個"
        ),
        axis=1,
    )
    columns = [
        "product_no",
        "product_name",
        "store",
        "category",
        "on_hand",
        "safety_stock",
        "shortage",
        "coverage_days",
        "reorder_link",
    ]
    return inv[columns]


def _prepare_cashflow_dataset(
    transactions: pd.DataFrame, *, base_balance: float = 3_500_000.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Derive daily cash flow trends and transaction-level records."""

    if transactions.empty:
        empty = pd.DataFrame(
            columns=["date", "store", "cash_in", "cash_out", "net", "balance"]
        )
        return empty, empty

    daily = (
        transactions.groupby(["date", "store"], as_index=False)
        .agg(
            cash_in=("sales_amount", "sum"),
            material_out=("material_cost", "sum"),
            gross_profit=("gross_profit", "sum"),
        )
        .sort_values("date")
    )

    daily["operating_out"] = (daily["cash_in"] * 0.18).round()
    daily["cash_out"] = daily["material_out"] + daily["operating_out"]
    daily["net"] = daily["cash_in"] - daily["cash_out"]
    daily["balance"] = (
        daily.groupby("store")["net"].cumsum() + base_balance
    )

    records: List[Dict[str, Any]] = []
    for row in daily.itertuples():
        records.append(
            {
                "date": row.date,
                "store": row.store,
                "type": "売上入金",
                "direction": "入金",
                "amount": float(row.cash_in),
                "memo": f"{row.store}の売上入金",
            }
        )
        records.append(
            {
                "date": row.date,
                "store": row.store,
                "type": "材料支払",
                "direction": "出金",
                "amount": float(row.material_out),
                "memo": "仕入コストの支払",
            }
        )
        records.append(
            {
                "date": row.date,
                "store": row.store,
                "type": "人件費",
                "direction": "出金",
                "amount": float(row.operating_out),
                "memo": "スタッフ給与・諸経費",
            }
        )

    cash_records = pd.DataFrame(records).sort_values("date")
    return daily, cash_records


def _prepare_behavior_context(products: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Assemble shared data for the behaviour-first dashboard."""

    transactions = _prepare_transactions_dataset(products)
    monthly_summary = (
        transactions.groupby(["period", "store"], as_index=False)
        .agg(
            total_sales=("sales_amount", "sum"),
            total_gp=("gross_profit", "sum"),
            total_cost=("material_cost", "sum"),
            total_qty=("sold_qty", "sum"),
        )
        .sort_values("period")
    )
    trend_all = (
        monthly_summary.groupby("period", as_index=False)
        .agg(
            total_sales=("total_sales", "sum"),
            total_gp=("total_gp", "sum"),
            total_cost=("total_cost", "sum"),
        )
        .sort_values("period")
    )

    inventory = _prepare_inventory_snapshot(products)
    cash_daily, cash_records = _prepare_cashflow_dataset(transactions)

    period_options = sorted(transactions["period"].dropna().unique())
    if not period_options:
        period_options = [pd.Timestamp(pd.Timestamp.today().to_period("M").to_timestamp())]
    default_period = max(period_options)

    store_options = sorted(transactions["store"].dropna().unique())
    store_options = [_DEFAULT_STORE_OPTION] + store_options
    default_store = _DEFAULT_STORE_OPTION

    return {
        "transactions": transactions,
        "monthly_summary": monthly_summary,
        "trend_all": trend_all,
        "inventory": inventory,
        "cash_daily": cash_daily,
        "cash_records": cash_records,
        "period_options": period_options,
        "store_options": store_options,
        "default_period": default_period,
        "default_store": default_store,
    }


def _format_currency_short(value: Any) -> str:
    """Return a compact currency representation with yen symbol."""

    if value is None or pd.isna(value):
        return "-"
    return f"¥{float(value):,.0f}"


def _format_ratio(value: Any) -> str:
    """Format numeric ratio as percentage string."""

    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.1f}%"


def _build_pdf_from_dataframe(df: pd.DataFrame, title: str) -> bytes:
    """Generate a simple PDF document from a dataframe."""

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements: List[Any] = []
    styles = getSampleStyleSheet()
    elements.append(Paragraph(title, styles["Heading3"]))
    elements.append(Spacer(1, 12))
    table_data = [list(df.columns)] + df.astype(str).values.tolist()
    table = Table(table_data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1F2A44")),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FC")]),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D0D7E2")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer.read()


def _build_sales_trend_chart(chart_df: pd.DataFrame) -> alt.Chart:
    """Return a line chart describing the sales trend over time."""

    data = chart_df.copy()
    if data.empty:
        return alt.Chart(pd.DataFrame())

    data["period"] = pd.to_datetime(data["period"])
    latest = data["period"].max()
    data["is_latest"] = data["period"] == latest

    base = alt.Chart(data).encode(
        x=alt.X(
            "period:T",
            title="月",
            axis=alt.Axis(format="%Y-%m", labelAngle=-20),
        )
    )

    line = base.mark_line(color=COLOR_ACCENT, interpolate="monotone", strokeWidth=3).encode(
        y=alt.Y(
            "total_sales:Q",
            title="売上高 (円)",
            axis=alt.Axis(format=","),
            scale=alt.Scale(domainMin=0),
        ),
        tooltip=[
            alt.Tooltip("period:T", title="月"),
            alt.Tooltip("total_sales:Q", title="売上高", format=","),
            alt.Tooltip("total_gp:Q", title="粗利", format=","),
        ],
    )

    highlight = base.transform_filter(alt.datum.is_latest).mark_circle(size=80, color=COLOR_ACCENT).encode(
        y="total_sales:Q"
    )
    label = (
        base.transform_filter(alt.datum.is_latest)
        .mark_text(color=COLOR_ACCENT, dx=8, dy=-8, fontWeight="bold")
        .encode(text=alt.Text("total_sales:Q", format=","), y="total_sales:Q")
    )

    return line + highlight + label


def _build_sales_by_product_chart(product_summary: pd.DataFrame) -> alt.Chart:
    """Return a bar chart comparing sales by product."""

    data = product_summary.head(10).copy()
    if data.empty:
        return alt.Chart(pd.DataFrame())

    chart = (
        alt.Chart(data)
        .mark_bar(color=COLOR_ACCENT)
        .encode(
            x=alt.X(
                "sales:Q",
                title="売上高 (円)",
                axis=alt.Axis(format=","),
                scale=alt.Scale(domainMin=0),
            ),
            y=alt.Y("product_name:N", title="商品", sort="-x"),
            tooltip=[
                alt.Tooltip("product_name:N", title="商品"),
                alt.Tooltip("sales:Q", title="売上高", format=","),
                alt.Tooltip("gp:Q", title="粗利", format=","),
                alt.Tooltip("qty:Q", title="数量", format=","),
            ],
        )
    )
    return chart.properties(height=max(220, 32 * len(data)))


def _build_sales_by_channel_chart(channel_summary: pd.DataFrame) -> alt.Chart:
    """Return a horizontal bar chart comparing sales by channel."""

    data = channel_summary.copy()
    if data.empty:
        return alt.Chart(pd.DataFrame())

    chart = (
        alt.Chart(data)
        .mark_bar(color=COLOR_ACCENT)
        .encode(
            x=alt.X(
                "sales:Q",
                title="売上高 (円)",
                axis=alt.Axis(format=","),
                scale=alt.Scale(domainMin=0),
            ),
            y=alt.Y("channel:N", title="チャネル", sort="-x"),
            tooltip=[
                alt.Tooltip("channel:N", title="チャネル"),
                alt.Tooltip("sales:Q", title="売上高", format=","),
                alt.Tooltip("gp:Q", title="粗利", format=","),
            ],
        )
    )
    return chart.properties(height=max(180, 34 * len(data)))


def _build_sales_by_store_chart(store_summary: pd.DataFrame, avg_sales: float) -> alt.Chart:
    """Return a vertical bar chart with an average reference line for store comparison."""

    data = store_summary.copy()
    if data.empty:
        return alt.Chart(pd.DataFrame())

    bars = (
        alt.Chart(data)
        .mark_bar(color=COLOR_ACCENT)
        .encode(
            x=alt.X("store:N", title="店舗"),
            y=alt.Y(
                "sales:Q",
                title="売上高 (円)",
                axis=alt.Axis(format=","),
                scale=alt.Scale(domainMin=0),
            ),
            tooltip=[
                alt.Tooltip("store:N", title="店舗"),
                alt.Tooltip("sales:Q", title="売上高", format=","),
            ],
        )
    )

    reference = alt.Chart(pd.DataFrame({"avg": [avg_sales]})).mark_rule(
        color=COLOR_SECONDARY,
        strokeDash=[6, 4],
        size=2,
    ).encode(y="avg:Q")

    return bars + reference


def _build_gross_profit_trend_chart(trend_df: pd.DataFrame) -> alt.Chart:
    """Return an area chart illustrating sales vs cost and the resulting gross profit."""

    data = trend_df.copy()
    if data.empty:
        return alt.Chart(pd.DataFrame())

    data["period"] = pd.to_datetime(data["period"])
    rename_map = {"total_sales": "売上高", "total_cost": "原価", "total_gp": "粗利"}
    long_df = data.melt("period", value_vars=list(rename_map.keys()), var_name="metric", value_name="amount")
    long_df["metric_jp"] = long_df["metric"].map(rename_map)
    long_df["is_latest"] = long_df.groupby("metric_jp")["period"].transform(lambda x: x == x.max())

    color_scale = alt.Scale(
        domain=["売上高", "原価", "粗利"],
        range=[COLOR_ACCENT, COLOR_ERROR, COLOR_SUCCESS],
    )

    base = alt.Chart(long_df).encode(
        x=alt.X("period:T", title="月", axis=alt.Axis(format="%Y-%m", labelAngle=-20)),
        y=alt.Y(
            "amount:Q",
            title="金額 (円)",
            axis=alt.Axis(format=","),
            scale=alt.Scale(domainMin=0),
        ),
        color=alt.Color("metric_jp:N", title="指標", scale=color_scale),
        tooltip=[
            alt.Tooltip("period:T", title="月"),
            alt.Tooltip("metric_jp:N", title="指標"),
            alt.Tooltip("amount:Q", title="金額", format=","),
        ],
    )

    area = base.transform_filter(alt.datum.metric_jp != "粗利").mark_area(opacity=0.45).encode(
        order=alt.Order("metric_jp", sort="descending"),
        y=alt.Y(
            "amount:Q",
            title="金額 (円)",
            axis=alt.Axis(format=","),
            scale=alt.Scale(domainMin=0),
            stack=None,
        ),
    )

    line = base.transform_filter(alt.datum.metric_jp == "粗利").mark_line(
        strokeWidth=3,
        strokeDash=[6, 3],
    )

    point = (
        base.transform_filter((alt.datum.metric_jp == "粗利") & alt.datum.is_latest)
        .mark_circle(size=80, color=COLOR_SECONDARY)
    )
    label = (
        base.transform_filter((alt.datum.metric_jp == "粗利") & alt.datum.is_latest)
        .mark_text(color=COLOR_SECONDARY, dx=8, dy=-8, fontWeight="bold")
        .encode(text=alt.Text("amount:Q", format=","))
    )

    return area + line + point + label


def _build_product_profit_chart(product_gp: pd.DataFrame) -> alt.Chart:
    """Return a gradient bar chart for product-level gross profit."""

    data = product_gp.head(15).copy()
    if data.empty:
        return alt.Chart(pd.DataFrame())

    data["margin_rate"] = np.where(data["sales"] > 0, data["gp"] / data["sales"] * 100.0, np.nan)

    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(
                "gp:Q",
                title="粗利額 (円)",
                axis=alt.Axis(format=","),
                scale=alt.Scale(domainMin=0),
            ),
            y=alt.Y("product_name:N", title="商品", sort="-x"),
            color=alt.Color(
                "margin_rate:Q",
                title="粗利率 (%)",
                scale=alt.Scale(scheme="reds"),
            ),
            tooltip=[
                alt.Tooltip("product_name:N", title="商品"),
                alt.Tooltip("gp:Q", title="粗利額", format=","),
                alt.Tooltip("sales:Q", title="売上高", format=","),
                alt.Tooltip("margin_rate:Q", title="粗利率", format=".1f"),
            ],
        )
    )
    return chart.properties(height=max(220, 32 * len(data)))


def _build_channel_profit_chart(channel_gp: pd.DataFrame) -> alt.Chart:
    """Return a horizontal bar chart of gross profit by channel."""

    data = channel_gp.copy()
    if data.empty:
        return alt.Chart(pd.DataFrame())

    data["margin_rate"] = np.where(data["sales"] > 0, data["gp"] / data["sales"] * 100.0, np.nan)

    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(
                "gp:Q",
                title="粗利額 (円)",
                axis=alt.Axis(format=","),
                scale=alt.Scale(domainMin=0),
            ),
            y=alt.Y("channel:N", title="チャネル", sort="-x"),
            color=alt.Color(
                "margin_rate:Q",
                title="粗利率 (%)",
                scale=alt.Scale(scheme="blues"),
            ),
            tooltip=[
                alt.Tooltip("channel:N", title="チャネル"),
                alt.Tooltip("gp:Q", title="粗利額", format=","),
                alt.Tooltip("sales:Q", title="売上高", format=","),
                alt.Tooltip("margin_rate:Q", title="粗利率", format=".1f"),
            ],
        )
    )
    return chart.properties(height=max(200, 36 * len(data)))


def _create_inventory_projection_df(inventory: pd.DataFrame, horizon_days: int = 14) -> pd.DataFrame:
    """Return a projected inventory balance dataframe based on current stock and daily usage."""

    if inventory.empty:
        return pd.DataFrame(columns=["date", "projected_stock", "safety_stock"])

    on_hand = float(inventory["on_hand"].sum())
    safety_stock = float(inventory["safety_stock"].sum())
    if "daily_qty" in inventory:
        daily_usage = float(inventory["daily_qty"].sum())
    else:
        daily_usage = 0.0
    dates = pd.date_range(pd.Timestamp.today().normalize(), periods=horizon_days + 1, freq="D")

    records = []
    for idx, dt_value in enumerate(dates):
        projected = on_hand - daily_usage * idx
        records.append(
            {
                "date": dt_value,
                "projected_stock": max(projected, 0.0),
                "safety_stock": safety_stock,
            }
        )
    return pd.DataFrame(records)


def _build_inventory_projection_chart(projection_df: pd.DataFrame) -> alt.Chart:
    """Return a line chart for projected inventory versus safety stock."""

    data = projection_df.copy()
    if data.empty:
        return alt.Chart(pd.DataFrame())

    line = (
        alt.Chart(data)
        .mark_line(color=COLOR_ACCENT, strokeWidth=3)
        .encode(
            x=alt.X("date:T", title="日付"),
            y=alt.Y(
                "projected_stock:Q",
                title="在庫残数 (個)",
                axis=alt.Axis(format=","),
                scale=alt.Scale(domainMin=0),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="日付"),
                alt.Tooltip("projected_stock:Q", title="残数", format=","),
            ],
        )
    )

    area = (
        alt.Chart(data)
        .mark_area(color=COLOR_ACCENT, opacity=0.15)
        .encode(x="date:T", y="projected_stock:Q")
    )
    safety = (
        alt.Chart(data)
        .mark_rule(color=COLOR_SECONDARY, strokeDash=[6, 4])
        .encode(y="safety_stock:Q")
    )
    return area + line + safety


def _build_inventory_category_chart(category_summary: pd.DataFrame) -> alt.Chart:
    """Return a horizontal bar chart with color-coded stock status by category."""

    data = category_summary.copy()
    if data.empty:
        return alt.Chart(pd.DataFrame())

    status_order = ["不足", "適正", "過剰"]
    color_scale = alt.Scale(
        domain=status_order,
        range=[COLOR_ERROR, COLOR_ACCENT, COLOR_SUCCESS],
    )

    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(
                "on_hand:Q",
                title="在庫数 (個)",
                axis=alt.Axis(format=","),
                scale=alt.Scale(domainMin=0),
            ),
            y=alt.Y("category:N", title="カテゴリ", sort="-x"),
            color=alt.Color("status:N", title="在庫状況", scale=color_scale),
            tooltip=[
                alt.Tooltip("category:N", title="カテゴリ"),
                alt.Tooltip("on_hand:Q", title="在庫数", format=","),
                alt.Tooltip("status:N", title="状況"),
            ],
        )
    )
    return chart.properties(height=max(200, 34 * len(data)))


def _compute_inventory_turnover(
    transactions: pd.DataFrame, inventory: pd.DataFrame, store: str
) -> pd.DataFrame:
    """Calculate monthly inventory turnover using material cost and current stock value."""

    tx = _filter_by_store(transactions, store)
    if tx.empty or inventory.empty:
        return pd.DataFrame(columns=["period", "turnover"])

    cost_per_unit = (
        tx.groupby("product_no", as_index=False)
        .agg(cost=("material_cost", "sum"), qty=("sold_qty", "sum"))
        .assign(unit_cost=lambda df: np.where(df["qty"] > 0, df["cost"] / df["qty"], np.nan))
    )
    overall_unit_cost = cost_per_unit["unit_cost"].replace(0, np.nan).mean()
    if pd.isna(overall_unit_cost):
        overall_unit_cost = float(tx["material_cost"].sum() / max(tx["sold_qty"].sum(), 1))

    inv = inventory.merge(cost_per_unit[["product_no", "unit_cost"]], on="product_no", how="left")
    inv["unit_cost"] = inv["unit_cost"].fillna(overall_unit_cost)
    inventory_value = float((inv["on_hand"] * inv["unit_cost"]).sum())
    if inventory_value <= 0:
        return pd.DataFrame(columns=["period", "turnover"])

    monthly = (
        tx.groupby(tx["date"].dt.to_period("M"))
        .agg(material_cost=("material_cost", "sum"))
        .reset_index()
    )
    monthly["period"] = monthly["date"].dt.to_timestamp()
    monthly["turnover"] = monthly["material_cost"] / inventory_value
    return monthly[["period", "turnover"]].sort_values("period")


def _build_inventory_turnover_chart(turnover_df: pd.DataFrame, target: float = 4.0) -> alt.Chart:
    """Return a line chart of inventory turnover with a target line."""

    data = turnover_df.copy()
    if data.empty:
        return alt.Chart(pd.DataFrame())

    data["is_latest"] = data["period"] == data["period"].max()

    line = (
        alt.Chart(data)
        .mark_line(color=COLOR_ACCENT, strokeWidth=3)
        .encode(
            x=alt.X("period:T", title="月", axis=alt.Axis(format="%Y-%m", labelAngle=-20)),
            y=alt.Y(
                "turnover:Q",
                title="在庫回転率 (回/月)",
                axis=alt.Axis(format=".1f"),
                scale=alt.Scale(domainMin=0),
            ),
            tooltip=[
                alt.Tooltip("period:T", title="月"),
                alt.Tooltip("turnover:Q", title="回転率", format=".2f"),
            ],
        )
    )

    points = line.transform_filter(alt.datum.is_latest).mark_circle(size=80)
    target_line = alt.Chart(pd.DataFrame({"target": [target]})).mark_rule(
        color=COLOR_SECONDARY,
        strokeDash=[4, 4],
    ).encode(y="target:Q")

    return line + points + target_line


def _build_cash_balance_chart(cash_chart: pd.DataFrame) -> alt.Chart:
    """Return a line chart for cash balance with daily net bars."""

    data = cash_chart.copy()
    if data.empty:
        return alt.Chart(pd.DataFrame())

    balance_line = (
        alt.Chart(data)
        .mark_line(color=COLOR_ACCENT, strokeWidth=3)
        .encode(
            x=alt.X("date:T", title="日付"),
            y=alt.Y(
                "balance:Q",
                title="残高 (円)",
                axis=alt.Axis(format=","),
                scale=alt.Scale(domainMin=0),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="日付"),
                alt.Tooltip("balance:Q", title="残高", format=","),
            ],
        )
    )

    net_bar = (
        alt.Chart(data)
        .mark_bar(color="#9BC0A0", opacity=0.4)
        .encode(
            x="date:T",
            y=alt.Y("net:Q", title="日次ネット (円)", axis=alt.Axis(format=",")),
            tooltip=[
                alt.Tooltip("date:T", title="日付"),
                alt.Tooltip("net:Q", title="日次ネット", format=","),
            ],
        )
    )

    return balance_line + net_bar


def _build_cash_flow_bars(monthly_flow: pd.DataFrame) -> alt.Chart:
    """Return a grouped bar chart for monthly cash-in versus cash-out."""

    data = monthly_flow.copy()
    if data.empty:
        return alt.Chart(pd.DataFrame())

    long_df = data.melt("month", value_vars=["cash_in", "cash_out"], var_name="type", value_name="amount")
    color_scale = alt.Scale(domain=["cash_in", "cash_out"], range=[COLOR_ACCENT, COLOR_ERROR])

    chart = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X("month:T", title="月", axis=alt.Axis(format="%Y-%m")),
            y=alt.Y(
                "amount:Q",
                title="金額 (円)",
                axis=alt.Axis(format=","),
                scale=alt.Scale(domainMin=0),
            ),
            color=alt.Color("type:N", title="区分", scale=color_scale, legend=alt.Legend(orient="top")),
            tooltip=[
                alt.Tooltip("month:T", title="月"),
                alt.Tooltip("type:N", title="区分"),
                alt.Tooltip("amount:Q", title="金額", format=","),
            ],
        )
    )
    return chart


def _build_cash_composition_chart(composition: pd.DataFrame) -> alt.Chart:
    """Return a bar chart showing inflow/outflow composition by type."""

    data = composition.copy()
    if data.empty:
        return alt.Chart(pd.DataFrame())

    color_scale = alt.Scale(domain=["入金", "出金"], range=[COLOR_ACCENT, COLOR_ERROR])

    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("type:N", title="区分", sort="-y"),
            y=alt.Y(
                "amount:Q",
                title="金額 (円)",
                axis=alt.Axis(format=","),
                scale=alt.Scale(domainMin=0),
            ),
            color=alt.Color("direction:N", title="入出金", scale=color_scale),
            tooltip=[
                alt.Tooltip("type:N", title="区分"),
                alt.Tooltip("direction:N", title="入出金"),
                alt.Tooltip("amount:Q", title="金額", format=","),
            ],
        )
    )
    return chart

def _filter_by_store(df: pd.DataFrame, store: str) -> pd.DataFrame:
    """Return a filtered dataframe for the selected store option."""

    if not isinstance(df, pd.DataFrame) or df.empty:
        if isinstance(df, pd.DataFrame):
            return df.copy()
        return pd.DataFrame()
    if store == _DEFAULT_STORE_OPTION or "store" not in df.columns:
        return df.copy()
    return df[df["store"] == store].copy()


def _compute_cash_balance(
    cash_daily: pd.DataFrame, store: str, *, base_balance: float = 3_500_000.0
) -> float:
    """Compute the latest cash balance for the selected store."""

    if cash_daily.empty:
        return float("nan")
    if store == _DEFAULT_STORE_OPTION:
        aggregated = (
            cash_daily.groupby("date", as_index=False)
            .agg(net=("net", "sum"))
            .sort_values("date")
        )
        aggregated["balance"] = base_balance + aggregated["net"].cumsum()
        return float(aggregated["balance"].iloc[-1]) if not aggregated.empty else float("nan")

    filtered = cash_daily[cash_daily["store"] == store].sort_values("date")
    if filtered.empty:
        return float("nan")
    return float(filtered["balance"].iloc[-1])


def _compute_cash_balance_for_period(
    cash_daily: pd.DataFrame,
    store: str,
    period: Optional[pd.Timestamp],
    *,
    base_balance: float = 3_500_000.0,
) -> float:
    """Return the month-end cash balance for a given period and store."""

    if cash_daily.empty or period is None:
        return float("nan")

    if store == _DEFAULT_STORE_OPTION:
        data = (
            cash_daily.groupby("date", as_index=False)
            .agg(net=("net", "sum"))
            .sort_values("date")
        )
        data["balance"] = base_balance + data["net"].cumsum()
    else:
        data = cash_daily[cash_daily["store"] == store].sort_values("date")

    if data.empty:
        return float("nan")

    period_end = pd.Timestamp(period).to_period("M").to_timestamp() + pd.offsets.MonthEnd(0)
    subset = data[data["date"] <= period_end]
    if subset.empty:
        return float("nan")
    return float(subset["balance"].iloc[-1])


def _prepare_cash_chart(
    cash_daily: pd.DataFrame, store: str, *, base_balance: float = 3_500_000.0
) -> pd.DataFrame:
    """Return a cashflow dataframe suitable for plotting."""

    if cash_daily.empty:
        return cash_daily
    if store == _DEFAULT_STORE_OPTION:
        aggregated = (
            cash_daily.groupby("date", as_index=False)
            .agg(
                cash_in=("cash_in", "sum"),
                cash_out=("cash_out", "sum"),
                net=("net", "sum"),
            )
            .sort_values("date")
        )
        aggregated["balance"] = base_balance + aggregated["net"].cumsum()
        aggregated["store"] = store
        return aggregated
    return cash_daily[cash_daily["store"] == store].sort_values("date")


def _render_sales_tab(
    context: Dict[str, Any],
    *,
    selected_period: pd.Timestamp,
    previous_period: Optional[pd.Timestamp],
    selected_store: str,
    current_period_df: pd.DataFrame,
    previous_period_df: pd.DataFrame,
) -> None:
    """Render the sales tab with metrics, trend and breakdown views."""

    transactions = _filter_by_store(context["transactions"], selected_store)
    if transactions.empty:
        _set_status("no_data")
        st.info("売上データがありません。先にデータを取り込み、再度ご確認ください。")
        return

    if current_period_df.empty:
        _set_status("empty_filter")
        st.warning("選択した期間・店舗に一致する売上データがありません。フィルタ条件を見直してください。")
        return

    if st.session_state.get("status") in {"no_data", "empty_filter"}:
        _set_status(None)

    period_label = _format_period_label(selected_period, "月次")
    sales_now = float(current_period_df["sales_amount"].sum())
    qty_now = float(current_period_df["sold_qty"].sum())
    gp_now = float(current_period_df["gross_profit"].sum())

    sales_prev = float(previous_period_df["sales_amount"].sum()) if not previous_period_df.empty else np.nan
    sales_delta = _pct_change(sales_prev, sales_now)

    avg_unit_price = sales_now / qty_now if qty_now else float("nan")
    gp_ratio = (gp_now / sales_now * 100.0) if sales_now else float("nan")

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "売上高",
        _format_currency_short(sales_now),
        delta=f"{sales_delta:+.1f}%" if not pd.isna(sales_delta) else "-",
        help=f"{period_label}の売上合計。前月比を右側に表示します。",
    )
    col2.metric(
        "平均客単価",
        _format_currency_short(avg_unit_price),
        help="売上高÷数量で算出。販売単価の傾向を把握できます。",
    )
    col3.metric(
        "粗利率",
        _format_ratio(gp_ratio),
        help="売上高に対する粗利の割合。",
    )

    monthly_summary = context["monthly_summary"]
    if selected_store == _DEFAULT_STORE_OPTION:
        chart_df = (
            monthly_summary.groupby("period", as_index=False)
            .agg(
                total_sales=("total_sales", "sum"),
                total_gp=("total_gp", "sum"),
                total_cost=("total_cost", "sum"),
            )
            .sort_values("period")
        )
    else:
        chart_df = monthly_summary[monthly_summary["store"] == selected_store]

    if not chart_df.empty:
        st.altair_chart(_build_sales_trend_chart(chart_df), use_container_width=True)

        weekly_summary = (
            current_period_df.set_index("date")["sales_amount"].resample("W-MON").sum().reset_index()
            if not current_period_df.empty
            else pd.DataFrame(columns=["date", "sales_amount"])
        )
        if not weekly_summary.empty:
            peak_idx = weekly_summary["sales_amount"].idxmax()
            peak_week = weekly_summary.loc[peak_idx, "date"]
            peak_value = weekly_summary.loc[peak_idx, "sales_amount"]
            week_no = int(((peak_week.day - 1) // 7) + 1)
            st.caption(
                f"今月の売上は{_format_currency_short(sales_now)}で、前月比 {sales_delta:+.1f}% の伸び。"
                f"週次では第{week_no}週が{_format_currency_short(peak_value)}でピークとなり、施策効果が確認できます。"
                if not pd.isna(sales_delta)
                else f"今月の売上は{_format_currency_short(sales_now)}。週次では第{week_no}週が{_format_currency_short(peak_value)}で最大です。"
            )

    tab_product, tab_channel = st.tabs(["商品別", "チャネル別"])

    with tab_product:
        product_summary = (
            current_period_df.groupby("product_name", as_index=False)
            .agg(
                sales=("sales_amount", "sum"),
                gp=("gross_profit", "sum"),
                qty=("sold_qty", "sum"),
            )
            .sort_values("sales", ascending=False)
        )
        if product_summary.empty:
            st.info("商品別の売上データがありません。")
        else:
            st.altair_chart(
                _build_sales_by_product_chart(product_summary),
                use_container_width=True,
            )
            top3_share = (
                product_summary.head(3)["sales"].sum() / sales_now
                if sales_now
                else np.nan
            )
            if not pd.isna(top3_share):
                st.caption(
                    f"上位3商品で総売上の{top3_share * 100:.1f}%を占めています。重点SKUの在庫と販促を優先的に確認しましょう。"
                )
            display = product_summary.rename(
                columns={"product_name": "商品", "sales": "売上", "gp": "粗利", "qty": "数量"}
            )
            product_column_config = {
                "売上": st.column_config.NumberColumn("売上", format="¥%,d"),
                "粗利": st.column_config.NumberColumn("粗利", format="¥%,d"),
                "数量": st.column_config.NumberColumn("数量", format="%,d"),
            }
            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
                column_config=product_column_config,
            )

    with tab_channel:
        channel_summary = (
            current_period_df.groupby("channel", as_index=False)
            .agg(
                sales=("sales_amount", "sum"),
                gp=("gross_profit", "sum"),
            )
            .sort_values("sales", ascending=False)
        )
        if channel_summary.empty:
            st.info("チャネル別の売上データがありません。")
        else:
            st.altair_chart(
                _build_sales_by_channel_chart(channel_summary),
                use_container_width=True,
            )
            lead_channel = channel_summary.iloc[0]
            share = lead_channel["sales"] / sales_now if sales_now else np.nan
            if not pd.isna(share):
                st.caption(
                    f"{lead_channel['channel']}チャネルが構成比{share * 100:.1f}%で最大です。構成比維持と他チャネルの底上げを検討しましょう。"
                )
            display_ch = channel_summary.rename(columns={"channel": "チャネル", "sales": "売上", "gp": "粗利"})
            channel_column_config = {
                "売上": st.column_config.NumberColumn("売上", format="¥%,d"),
                "粗利": st.column_config.NumberColumn("粗利", format="¥%,d"),
            }
            st.dataframe(
                display_ch,
                use_container_width=True,
                hide_index=True,
                column_config=channel_column_config,
            )

    store_summary = (
        current_period_df.groupby("store", as_index=False)["sales_amount"].sum().rename(columns={"sales_amount": "sales"})
    )
    if len(store_summary) > 1:
        avg_sales = store_summary["sales"].mean()
        st.markdown("#### 店舗別売上")
        st.altair_chart(
            _build_sales_by_store_chart(store_summary, avg_sales),
            use_container_width=True,
        )
        st.caption(
            f"平均売上 {_format_currency_short(avg_sales)} を灰色線で表示。未達の店舗は販促や在庫補充を重点化しましょう。"
        )

    detail = current_period_df[
        [
            "date",
            "product_no",
            "product_name",
            "category",
            "channel",
            "store",
            "sold_qty",
            "sales_amount",
            "gross_profit",
        ]
    ].copy()
    detail["date"] = detail["date"].dt.strftime("%Y-%m-%d")
    detail = detail.rename(
        columns={
            "date": "日付",
            "product_no": "製品番号",
            "product_name": "製品名",
            "category": "カテゴリ",
            "channel": "チャネル",
            "store": "店舗",
            "sold_qty": "数量",
            "sales_amount": "売上",
            "gross_profit": "粗利",
        }
    )
    if detail.empty:
        st.caption("この期間の明細はありません。")
    else:
        detail_column_config = {
            "数量": st.column_config.NumberColumn("数量", format="%,d"),
            "売上": st.column_config.NumberColumn("売上", format="¥%,d"),
            "粗利": st.column_config.NumberColumn("粗利", format="¥%,d"),
        }
        st.dataframe(
            detail,
            use_container_width=True,
            hide_index=True,
            column_config=detail_column_config,
        )
        period_str = selected_period.strftime("%Y-%m")
        store_label = selected_store if selected_store != _DEFAULT_STORE_OPTION else "全店舗"
        csv_name = f"売上_{period_str}_{store_label}.csv"
        pdf_name = f"売上_{period_str}_{store_label}.pdf"
        csv_bytes = detail.to_csv(index=False).encode("utf-8-sig")
        col_csv, col_pdf = st.columns(2)
        if col_csv.download_button(
            "CSVダウンロード",
            data=csv_bytes,
            file_name=csv_name,
            mime="text/csv",
        ):
            _set_status("success_export")
        if col_pdf.download_button(
            "PDFダウンロード",
            data=_build_pdf_from_dataframe(detail, title=f"{period_label} 売上明細 ({store_label})"),
            file_name=pdf_name,
            mime="application/pdf",
        ):
            _set_status("success_export")


def _render_profit_tab(
    context: Dict[str, Any],
    *,
    selected_period: pd.Timestamp,
    previous_period: Optional[pd.Timestamp],
    selected_store: str,
    current_period_df: pd.DataFrame,
    previous_period_df: pd.DataFrame,
) -> None:
    """Render the gross profit analysis tab."""

    if current_period_df.empty:
        st.info("粗利データがありません。売上タブのフィルタ条件をご確認ください。")
        return

    period_label = _format_period_label(selected_period, "月次")
    gp_now = float(current_period_df["gross_profit"].sum())
    gp_prev = float(previous_period_df["gross_profit"].sum()) if not previous_period_df.empty else np.nan
    gp_delta = _pct_change(gp_prev, gp_now)

    sales_now = float(current_period_df["sales_amount"].sum())
    sales_prev = float(previous_period_df["sales_amount"].sum()) if not previous_period_df.empty else np.nan
    margin_now = (gp_now / sales_now * 100.0) if sales_now else float("nan")
    margin_prev = (gp_prev / sales_prev * 100.0) if sales_prev else np.nan
    margin_delta = margin_now - margin_prev if not pd.isna(margin_now) and not pd.isna(margin_prev) else np.nan

    cost_now = float(current_period_df["material_cost"].sum())

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "粗利額",
        _format_currency_short(gp_now),
        delta=f"{gp_delta:+.1f}%" if not pd.isna(gp_delta) else "-",
    )
    col2.metric(
        "粗利率",
        _format_ratio(margin_now),
        delta=f"{margin_delta:+.1f}pt" if not pd.isna(margin_delta) else "-",
    )
    col3.metric("材料費", _format_currency_short(cost_now))

    monthly_summary = context["monthly_summary"]
    if selected_store == _DEFAULT_STORE_OPTION:
        gp_trend_df = (
            monthly_summary.groupby("period", as_index=False)
            .agg(
                total_sales=("total_sales", "sum"),
                total_gp=("total_gp", "sum"),
                total_cost=("total_cost", "sum"),
            )
            .sort_values("period")
        )
    else:
        gp_trend_df = monthly_summary[monthly_summary["store"] == selected_store]

    if not gp_trend_df.empty:
        st.altair_chart(_build_gross_profit_trend_chart(gp_trend_df), use_container_width=True)
        growth_text = f"前月比 {gp_delta:+.1f}%" if not pd.isna(gp_delta) else "前月比データなし"
        margin_text = (
            f"粗利率は {margin_now:.1f}%" if not pd.isna(margin_now) else "粗利率データなし"
        )
        st.caption(
            f"粗利額は{_format_currency_short(gp_now)}で{growth_text}。{margin_text}の水準で推移しており、原価線との乖離で季節要因を把握できます。"
        )

    tab_product, tab_channel = st.tabs(["商品別粗利", "チャネル別粗利"])

    with tab_product:
        product_gp = (
            current_period_df.groupby("product_name", as_index=False)
            .agg(
                gp=("gross_profit", "sum"),
                sales=("sales_amount", "sum"),
            )
            .sort_values("gp", ascending=False)
        )
        if product_gp.empty:
            st.info("商品別粗利のデータがありません。")
        else:
            st.altair_chart(_build_product_profit_chart(product_gp), use_container_width=True)
            top_item = product_gp.iloc[0]
            top_margin = (top_item["gp"] / top_item["sales"] * 100.0) if top_item["sales"] else float("nan")
            top_margin_text = _format_ratio(top_margin)
            st.caption(
                f"{top_item['product_name']}が粗利額{_format_currency_short(top_item['gp'])}、粗利率{top_margin_text}で最も貢献しています。高粗利商品の販売機会を逃さないようフォローしましょう。"
            )
            display = product_gp.rename(columns={"product_name": "商品", "gp": "粗利", "sales": "売上"})
            gp_column_config = {
                "粗利": st.column_config.NumberColumn("粗利", format="¥%,d"),
                "売上": st.column_config.NumberColumn("売上", format="¥%,d"),
            }
            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
                column_config=gp_column_config,
            )

    with tab_channel:
        channel_gp = (
            current_period_df.groupby("channel", as_index=False)
            .agg(
                gp=("gross_profit", "sum"),
                sales=("sales_amount", "sum"),
            )
        )
        if channel_gp.empty:
            st.info("チャネル別の粗利データがありません。")
        else:
            st.altair_chart(_build_channel_profit_chart(channel_gp), use_container_width=True)
            lead_channel = channel_gp.sort_values("gp", ascending=False).iloc[0]
            lead_margin = (
                lead_channel["gp"] / lead_channel["sales"] * 100.0
                if lead_channel["sales"]
                else float("nan")
            )
            lead_margin_text = _format_ratio(lead_margin)
            st.caption(
                f"{lead_channel['channel']}チャネルが粗利{_format_currency_short(lead_channel['gp'])}、粗利率{lead_margin_text}でトップ。構成比が低いチャネルの改善余地も併せて検討しましょう。"
            )
            display = channel_gp.rename(columns={"channel": "チャネル", "gp": "粗利", "sales": "売上"})
            channel_gp_config = {
                "粗利": st.column_config.NumberColumn("粗利", format="¥%,d"),
                "売上": st.column_config.NumberColumn("売上", format="¥%,d"),
            }
            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
                column_config=channel_gp_config,
            )

    detail = current_period_df[
        ["product_name", "category", "channel", "sold_qty", "sales_amount", "gross_profit"]
    ].copy()
    detail = detail.rename(
        columns={
            "product_name": "商品",
            "category": "カテゴリ",
            "channel": "チャネル",
            "sold_qty": "数量",
            "sales_amount": "売上",
            "gross_profit": "粗利",
        }
    )
    profit_detail_config = {
        "数量": st.column_config.NumberColumn("数量", format="%,d"),
        "売上": st.column_config.NumberColumn("売上", format="¥%,d"),
        "粗利": st.column_config.NumberColumn("粗利", format="¥%,d"),
    }
    st.dataframe(
        detail,
        use_container_width=True,
        hide_index=True,
        column_config=profit_detail_config,
    )


def _render_inventory_tab(
    context: Dict[str, Any], *, selected_store: str, threshold_days: float, mode: str
) -> None:
    """Render the inventory tab with shortage highlights and table."""

    inventory = _filter_by_store(context["inventory"], selected_store)
    if inventory.empty:
        st.info("在庫データがありません。製品マスタに安全在庫情報を追加してください。")
        return

    inventory = inventory.copy()
    inventory["shortage_flag"] = (inventory["coverage_days"].fillna(0) < threshold_days) | (inventory["shortage"] > 0)
    if mode == "不足のみ":
        filtered = inventory[inventory["shortage_flag"]]
    else:
        filtered = inventory

    shortage_count = int((inventory["shortage_flag"]).sum())
    total_items = len(inventory)
    col1, col2, col3 = st.columns(3)
    col1.metric("不足SKU数", f"{shortage_count}/{total_items}")
    avg_days = inventory["coverage_days"].replace([np.inf, -np.inf], np.nan).mean()
    col2.metric("平均在庫日数", f"{avg_days:.1f}日" if not pd.isna(avg_days) else "-")
    col3.metric(
        "対象店舗",
        selected_store if selected_store != _DEFAULT_STORE_OPTION else "全店舗",
    )

    projection_df = _create_inventory_projection_df(inventory)
    if not projection_df.empty:
        st.altair_chart(_build_inventory_projection_chart(projection_df), use_container_width=True)
        below = projection_df[projection_df["projected_stock"] < projection_df["safety_stock"]]
        if not below.empty:
            alert_day = below.iloc[0]["date"].strftime("%m/%d")
            st.caption(
                f"{alert_day}に安全在庫を下回る見込みです。リードタイムを考慮した前倒し補充を検討しましょう。"
            )
        else:
            st.caption("予測期間内は安全在庫を上回っています。販売増加時は補充リードタイムに注意してください。")

    category_summary = (
        inventory.groupby("category", as_index=False)
        .agg(on_hand=("on_hand", "sum"), safety=("safety_stock", "sum"))
        .assign(
            status=lambda df: np.select(
                [df["on_hand"] < df["safety"], df["on_hand"] > df["safety"] * 1.3],
                ["不足", "過剰"],
                default="適正",
            )
        )
        .sort_values("on_hand", ascending=False)
    )
    if not category_summary.empty:
        st.altair_chart(_build_inventory_category_chart(category_summary), use_container_width=True)
        lead_category = category_summary.iloc[0]
        st.caption(
            f"{lead_category['category']}カテゴリの在庫が{int(lead_category['on_hand']):,}個で最大です。状況は{lead_category['status']}のため補充・削減計画を確認してください。"
        )

    turnover_df = _compute_inventory_turnover(context["transactions"], inventory, selected_store)
    if not turnover_df.empty:
        st.altair_chart(_build_inventory_turnover_chart(turnover_df), use_container_width=True)
        latest_turnover = float(turnover_df.iloc[-1]["turnover"])
        st.caption(
            f"直近の在庫回転率は{latest_turnover:.2f}回/月。目標4.0回との差を確認し、滞留在庫の圧縮や販売強化の必要性を議論しましょう。"
        )

    if filtered.empty:
        st.success("閾値未満の不足在庫はありません。")
        return

    top_shortages = filtered.sort_values("shortage", ascending=False).head(3)
    for _, row in top_shortages.iterrows():
        st.info(
            f"{row['product_name']}｜残数 {int(row['on_hand'])}個｜安全在庫 {int(row['safety_stock'])}個"
            f"｜不足 {int(max(row['shortage'], 0))}個"
        )

    table = filtered[[
        "product_no",
        "product_name",
        "category",
        "store",
        "on_hand",
        "safety_stock",
        "shortage",
        "coverage_days",
        "reorder_link",
    ]].rename(
        columns={
            "product_no": "製品番号",
            "product_name": "製品名",
            "category": "カテゴリ",
            "store": "店舗",
            "on_hand": "在庫数",
            "safety_stock": "安全在庫",
            "shortage": "不足数",
            "coverage_days": "残日数",
            "reorder_link": "発注",
        }
    )
    table["不足数"] = table["不足数"].map(lambda v: int(max(v, 0)))
    table["在庫数"] = table["在庫数"].astype(int)
    table["安全在庫"] = table["安全在庫"].astype(int)
    table["残日数"] = table["残日数"].map(lambda v: f"{float(v):.1f}" if not pd.isna(v) else "-")

    column_config = {
        "発注": st.column_config.LinkColumn("発注", help="在庫担当へのメールリンクを開きます。"),
    }
    st.data_editor(
        table,
        hide_index=True,
        use_container_width=True,
        column_config=column_config,
    )

    store_label = selected_store if selected_store != _DEFAULT_STORE_OPTION else "全店舗"
    csv_name = f"在庫_{store_label}.csv"
    pdf_name = f"在庫_{store_label}.pdf"
    csv_bytes = table.to_csv(index=False).encode("utf-8-sig")
    col_csv, col_pdf = st.columns(2)
    if col_csv.download_button("CSVダウンロード", data=csv_bytes, file_name=csv_name, mime="text/csv"):
        _set_status("success_export")
    if col_pdf.download_button(
        "PDFダウンロード",
        data=_build_pdf_from_dataframe(table, title=f"在庫一覧 ({store_label})"),
        file_name=pdf_name,
        mime="application/pdf",
    ):
        _set_status("success_export")


def _render_cash_tab(
    context: Dict[str, Any],
    *,
    selected_period: pd.Timestamp,
    selected_store: str,
) -> None:
    """Render the cash management view with balance and transaction table."""

    cash_daily = context["cash_daily"]
    cash_chart = _prepare_cash_chart(cash_daily, selected_store)
    if cash_chart.empty:
        st.info("キャッシュフローデータがありません。外部連携で入出金を同期してください。")
        return

    cash_balance = _compute_cash_balance(cash_daily, selected_store)
    mask_period = cash_chart["date"].dt.to_period("M").dt.to_timestamp() == selected_period
    net_period = cash_chart.loc[mask_period, "net"].sum()

    col1, col2 = st.columns(2)
    col1.metric(
        "現預金残高",
        _format_currency_short(cash_balance),
        help="期間末時点の残高。複数店舗を集計する場合は全体のネットキャッシュを計算します。",
    )
    col2.metric(
        "月次キャッシュフロー",
        _format_currency_short(net_period),
        help="選択期間の入金−出金。",
    )

    st.altair_chart(_build_cash_balance_chart(cash_chart), use_container_width=True)
    trend_text = (
        f"月次ネットは{_format_currency_short(net_period)}で、"
        if not pd.isna(net_period)
        else ""
    )
    st.caption(
        f"現預金残高は{_format_currency_short(cash_balance)}まで積み上がっています。{trend_text}資金ショートリスクの有無を日次で確認できます。"
    )

    monthly_flow = (
        cash_chart.assign(month=cash_chart["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)
        .agg(cash_in=("cash_in", "sum"), cash_out=("cash_out", "sum"))
        .sort_values("month")
    )
    if not monthly_flow.empty:
        st.altair_chart(_build_cash_flow_bars(monthly_flow), use_container_width=True)
        latest_row = monthly_flow[monthly_flow["month"] == monthly_flow["month"].max()].iloc[0]
        balance_msg = latest_row["cash_in"] - latest_row["cash_out"]
        st.caption(
            f"直近月は入金{_format_currency_short(latest_row['cash_in'])}、出金{_format_currency_short(latest_row['cash_out'])}でネット{_format_currency_short(balance_msg)}でした。入出金のバランスを維持できているか確認しましょう。"
        )

    cash_records = _filter_by_store(context["cash_records"], selected_store)
    mask_records = cash_records["date"].dt.to_period("M").dt.to_timestamp() == selected_period
    cash_records = cash_records[mask_records]
    if cash_records.empty:
        st.caption("選択期間の入出金明細はありません。")
        return

    composition = (
        cash_records.groupby(["type", "direction"], as_index=False)["amount"].sum().sort_values("amount", ascending=False)
    )
    if not composition.empty:
        st.altair_chart(_build_cash_composition_chart(composition), use_container_width=True)
        major_row = composition.iloc[0]
        st.caption(
            f"{major_row['type']}が{major_row['direction']}の中心で{_format_currency_short(major_row['amount'])}。費目別の構成比から削減余地や追加投資の判断材料を得られます。"
        )

    display = cash_records.copy()
    display["date"] = display["date"].dt.strftime("%Y-%m-%d")
    display = display.rename(
        columns={
            "date": "日付",
            "store": "店舗",
            "type": "区分",
            "direction": "入出金",
            "amount": "金額",
            "memo": "メモ",
        }
    )
    cash_column_config = {
        "金額": st.column_config.NumberColumn("金額", format="¥%,d"),
    }
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config=cash_column_config,
    )

    period_str = selected_period.strftime("%Y-%m")
    store_label = selected_store if selected_store != _DEFAULT_STORE_OPTION else "全店舗"
    csv_name = f"資金_{period_str}_{store_label}.csv"
    pdf_name = f"資金_{period_str}_{store_label}.pdf"
    csv_bytes = display.to_csv(index=False).encode("utf-8-sig")
    col_csv, col_pdf = st.columns(2)
    if col_csv.download_button("CSVダウンロード", data=csv_bytes, file_name=csv_name, mime="text/csv"):
        _set_status("success_export")
    if col_pdf.download_button(
        "PDFダウンロード",
        data=_build_pdf_from_dataframe(display, title=f"入出金明細 {period_str} ({store_label})"),
        file_name=pdf_name,
        mime="application/pdf",
    ):
        _set_status("success_export")


def _render_value_added_tab(
    context: Dict[str, Any],
    *,
    selected_period: pd.Timestamp,
    previous_period: Optional[pd.Timestamp],
    selected_store: str,
    current_period_df: pd.DataFrame,
    previous_period_df: pd.DataFrame,
) -> None:
    """Show value-added contribution and composition analysis."""

    if current_period_df.empty:
        st.info("付加価値データがありません。Excelを更新するか、期間・店舗フィルタを調整してください。")
        return

    gross_now = float(current_period_df["gross_profit"].sum())
    delta_text: Optional[str] = None
    if previous_period is not None and not previous_period_df.empty:
        gross_prev = float(previous_period_df["gross_profit"].sum())
        diff = gross_now - gross_prev
        delta_text = f"{diff:+,.0f}円"

    qty_now = float(current_period_df.get("sold_qty", pd.Series(dtype=float)).sum())
    va_per_unit = gross_now / qty_now if qty_now else float("nan")

    metric_cols = st.columns(2)
    metric_cols[0].metric(
        "付加価値額 (月次)",
        _format_currency_short(gross_now),
        delta_text,
    )
    va_display = "-" if not np.isfinite(va_per_unit) else f"¥{va_per_unit:,.0f}"
    metric_cols[1].metric(
        "付加価値/個",
        va_display,
        help="当月粗利を数量で割った1個当たりの付加価値です。",
    )
    st.caption("KGI(誤判断減)と連動。付加価値が落ちているカテゴリは粗利タブで詳細確認してください。")

    category_comp = (
        current_period_df.groupby("category", dropna=False)["gross_profit"].sum().reset_index()
    )
    category_comp = category_comp.sort_values("gross_profit", ascending=False)
    if category_comp.empty:
        st.caption("カテゴリ情報が不足しています。データ入力画面でカテゴリー列を設定してください。")
        return

    chart = (
        alt.Chart(category_comp)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("gross_profit:Q", title="付加価値 (円)", axis=alt.Axis(format="s")),
            y=alt.Y("category:N", title="カテゴリ", sort="-x"),
            tooltip=[
                alt.Tooltip("category:N", title="カテゴリ"),
                alt.Tooltip("gross_profit:Q", title="付加価値", format=",")
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)

    top_table = category_comp.copy()
    top_table = top_table.rename(columns={"category": "カテゴリ", "gross_profit": "付加価値 (円)"})
    value_column_config = {
        "付加価値 (円)": st.column_config.NumberColumn("付加価値 (円)", format="¥%,d"),
    }
    st.dataframe(
        top_table,
        use_container_width=True,
        hide_index=True,
        column_config=value_column_config,
    )

    st.caption("CSV/PDFは粗利タブからエクスポートできます。フィルタ条件が共通で反映されます。")


def _render_behavior_dashboard(products: Optional[pd.DataFrame]) -> None:
    """Render the behaviour-first dashboard view introduced in Step4."""

    context = _prepare_behavior_context(products)
    default_period: pd.Timestamp = context["default_period"]
    default_store: str = context["default_store"]

    st.session_state.setdefault("status", None)
    st.session_state.setdefault("inventory_filter_mode", "不足のみ")
    st.session_state.setdefault("inventory_threshold", 3.0)

    scenario_store = st.session_state.get("scenarios")
    if not scenario_store:
        base_params_default = st.session_state.get("sr_params", DEFAULT_PARAMS)
        scenario_store = {"ベース": base_params_default}
        st.session_state["scenarios"] = scenario_store
    scenario_keys = list(scenario_store.keys()) or ["ベース"]
    current_scenario = st.session_state.get("current_scenario", scenario_keys[0])
    if current_scenario not in scenario_keys:
        current_scenario = scenario_keys[0]
        st.session_state["current_scenario"] = current_scenario

    if (
        "selected_period" not in st.session_state
        or st.session_state["selected_period"] not in context["period_options"]
    ):
        st.session_state["selected_period"] = default_period
    if (
        "selected_store" not in st.session_state
        or st.session_state["selected_store"] not in context["store_options"]
    ):
        st.session_state["selected_store"] = default_store

    _render_status_banner(default_period, default_store)

    selector_left, selector_right = st.columns([0.68, 0.32], gap="large")
    with selector_right:
        st.markdown("#### セグメント選択")
        selected_period = st.selectbox(
            "期間",
            options=context["period_options"],
            format_func=lambda v: _format_period_label(v, "月次"),
            key="selected_period",
            help="前回選択した期間を記憶し、再訪時に自動で復元します。",
        )
        selected_store = st.selectbox(
            "店舗",
            options=context["store_options"],
            key="selected_store",
            help="全店舗または特定店舗を切り替えて分析します。",
        )
        if scenario_keys:
            selected_scenario = st.selectbox(
                "シナリオ",
                options=scenario_keys,
                index=scenario_keys.index(current_scenario),
                key="dashboard_behavior_scenario",
                help="標準賃率ウィザードで保存したシナリオを切り替えます。",
            )
            if selected_scenario != current_scenario:
                st.session_state["current_scenario"] = selected_scenario
                current_scenario = selected_scenario

    period_options = context["period_options"]
    try:
        idx = period_options.index(selected_period)
    except ValueError:
        idx = len(period_options) - 1
    previous_period = period_options[idx - 1] if idx > 0 else None

    base_params = scenario_store.get(
        current_scenario,
        st.session_state.get("sr_params", DEFAULT_PARAMS),
    )
    base_params, base_warnings = sanitize_params(base_params)
    for msg in base_warnings:
        st.warning(msg)
    scenario_store[current_scenario] = base_params
    st.session_state["scenarios"] = scenario_store
    _, base_results = compute_rates(base_params)
    be_rate = base_results["break_even_rate"]
    req_rate = base_results["required_rate"]

    dq_overview = (
        detect_quality_issues(products)
        if isinstance(products, pd.DataFrame) and not products.empty
        else pd.DataFrame()
    )

    ach_rate = np.nan
    avg_va = np.nan
    gap_to_req = np.nan
    kpi_df = pd.DataFrame()
    if isinstance(products, pd.DataFrame) and not products.empty:
        kpi_df = compute_results(products, be_rate, req_rate)
        meets_series = kpi_df.get("meets_required_rate")
        if meets_series is not None:
            if meets_series.dtype == bool:
                ach_rate = float(meets_series.mean() * 100.0)
            else:
                ach_rate = float(
                    pd.to_numeric(meets_series, errors="coerce").fillna(0).mean() * 100.0
                )
        avg_va = _safe_series_mean(kpi_df.get("va_per_min"))
        if np.isfinite(avg_va):
            gap_to_req = req_rate - avg_va
        avg_gap_series = kpi_df.get("rate_gap_vs_required")
        avg_gap = _safe_series_mean(avg_gap_series) if avg_gap_series is not None else np.nan
    else:
        avg_gap = np.nan

    kpi_state = st.session_state.setdefault("behavior_kpi_snapshot", {})
    kgi_state = st.session_state.setdefault("behavior_kgi_snapshot", {})

    def _store_delta(state: Dict[str, float], key: str, value: float) -> Optional[float]:
        if value is None or not np.isfinite(value):
            return None
        prev = state.get(key)
        delta_val: Optional[float] = None
        if prev is not None and np.isfinite(prev):
            delta_val = value - prev
        state[key] = value
        return delta_val

    decision_time_saved_min = 148.5
    decision_hours = decision_time_saved_min / 60.0
    time_delta = _store_delta(kgi_state, "decision_hours", decision_hours)

    total_products = len(products) if isinstance(products, pd.DataFrame) else 0
    baseline_issue_ratio = st.session_state.setdefault(
        "kgi_baseline_issue_ratio",
        0.0 if total_products == 0 else (len(dq_overview) / max(total_products, 1)) * 100.0,
    )
    issue_ratio = (
        0.0
        if total_products == 0
        else (len(dq_overview) / max(total_products, 1)) * 100.0
    )
    issue_delta = None
    prev_issue = kgi_state.get("issue_ratio")
    if prev_issue is not None and np.isfinite(prev_issue) and np.isfinite(issue_ratio):
        issue_delta = issue_ratio - prev_issue
    kgi_state["issue_ratio"] = issue_ratio

    ach_delta = _store_delta(kpi_state, "ach_rate", ach_rate)
    req_delta = _store_delta(kpi_state, "req_rate", req_rate)
    be_delta = _store_delta(kpi_state, "be_rate", be_rate)
    va_delta = _store_delta(kpi_state, "avg_va", avg_va)
    gap_delta = _store_delta(kpi_state, "gap_to_req", gap_to_req)

    diagnostic_cards: List[Dict[str, Any]] = []
    kpi_cards: List[Dict[str, Any]] = []

    def _render_metric(
        column: Any,
        label: str,
        value: float,
        value_formatter: Callable[[float], str],
        *,
        delta_value: Optional[float] = None,
        delta_formatter: Optional[Callable[[float], str]] = None,
        delta_color: str = "normal",
    ) -> None:
        if value is None or not np.isfinite(value):
            column.metric(label, "-", "", delta_color=delta_color)
            return
        display_value = value_formatter(value)
        delta_text = ""
        if delta_formatter and delta_value is not None and np.isfinite(delta_value):
            delta_text = delta_formatter(delta_value)
        column.metric(label, display_value, delta_text, delta_color=delta_color)

    summary_cols = st.columns([3, 1, 1, 1], gap="medium")
    with summary_cols[0]:
        st.markdown("#### KPIハイライト")
        st.write(
            "主要指標を一列で確認し、詳細カードとタブでドリルダウンできます。"
        )
        st.caption(
            "期間・店舗・シナリオの選択は st.session_state に保持され、直近比較のΔに反映されます。"
        )

    _render_metric(
        summary_cols[1],
        "必要賃率達成率",
        ach_rate,
        lambda v: f"{v:.1f}%",
        delta_value=ach_delta,
        delta_formatter=lambda v: f"{v:+.1f}pt",
        delta_color="normal",
    )
    _render_metric(
        summary_cols[2],
        "必要賃率との差",
        gap_to_req,
        lambda v: f"{v:+.2f} 円/分",
        delta_value=gap_delta,
        delta_formatter=lambda v: f"{v:+.2f}",
        delta_color="inverse",
    )
    _render_metric(
        summary_cols[3],
        "平均付加価値/分",
        avg_va,
        lambda v: f"{v:.2f} 円/分",
        delta_value=va_delta,
        delta_formatter=lambda v: f"{v:+.2f}",
        delta_color="normal",
    )

    diagnostic_cards.append(
        {
            "title": "KGI｜意思決定時間を短縮",
            "value": f"{decision_hours:.1f}時間/月",
            "delta": time_delta,
            "delta_format": "{:+.1f}h",
            "note": "KPIカード集約とガイド導線でダッシュボード確認を15秒以内に短縮（約75分/月の削減想定）。",
            "positive_is_good": True,
        }
    )
    diagnostic_cards.append(
        {
            "title": "KGI｜誤判断を減らす",
            "value": f"品質リスク {issue_ratio:.1f}%",
            "delta": issue_delta,
            "delta_format": "{:+.1f}pt",
            "note": f"初期基準 {baseline_issue_ratio:.1f}% → 現在 {issue_ratio:.1f}%。品質アラート {len(dq_overview)} 件。",
            "positive_is_good": False,
        }
    )

    if np.isfinite(ach_rate):
        target_ach_rate = st.session_state.get("target_ach_rate", np.nan)
        note = "KGI(時間短縮)と直結。目標達成率の差は戦略会議で共有してください。"
        if np.isfinite(target_ach_rate):
            note = f"目標 {target_ach_rate:.1f}% と比較。{note}"
        kpi_cards.append(
            {
                "title": "KPI｜必要賃率達成率",
                "value": f"{ach_rate:.1f}%",
                "delta": ach_delta,
                "delta_format": "{:+.1f}pt",
                "note": note,
                "positive_is_good": True,
            }
        )
    if np.isfinite(gap_to_req):
        kpi_cards.append(
            {
                "title": "KPI｜必要賃率との差",
                "value": f"{gap_to_req:+.2f}円/分",
                "delta": gap_delta,
                "delta_format": "{:+.2f}",
                "note": "正の値は未達。改善で誤判断リスクを抑え、在庫/粗利タブへドリルダウン。",
                "positive_is_good": False,
            }
        )
    if np.isfinite(be_rate):
        kpi_cards.append(
            {
                "title": "KPI｜損益分岐賃率",
                "value": f"{be_rate:.2f}円/分",
                "delta": be_delta,
                "delta_format": "{:+.2f}",
                "note": "固定費・必要利益から算出。下がるほど経営余力が増えます。",
                "positive_is_good": False,
            }
        )
    if np.isfinite(avg_va):
        kpi_cards.append(
            {
                "title": "KPI｜平均付加価値/分",
                "value": f"{avg_va:.2f}円/分",
                "delta": va_delta,
                "delta_format": "{:+.2f}",
                "note": "付加価値貢献度。付加価値タブでカテゴリ別の構成を確認できます。",
                "positive_is_good": True,
            }
        )

    if diagnostic_cards:
        st.markdown("#### KPIスナップショット")
        render_indicator_cards(diagnostic_cards)
    if kpi_cards:
        st.markdown("#### KPI詳細")
        render_indicator_cards(kpi_cards)

    st.markdown("#### トレンド & 異常検知ハイライト")
    trend_cols = st.columns([0.7, 0.3], gap="large")
    with trend_cols[0]:
        trend_df = context.get("trend_all", pd.DataFrame()).copy()
        if trend_df.empty:
            st.info("トレンドデータがまだありません。期間フィルタを見直すか、データ入力を更新してください。")
        else:
            trend_df = trend_df.sort_values("period")
            melted = trend_df.melt(
                id_vars="period",
                value_vars=["total_sales", "total_gp"],
                var_name="metric",
                value_name="value",
            )
            metric_label = {"total_sales": "売上高", "total_gp": "粗利"}
            melted["指標"] = melted["metric"].map(metric_label)
            chart = (
                alt.Chart(melted)
                .mark_line(point=True, strokeWidth=3)
                .encode(
                    x=alt.X("period:T", title="期間", axis=alt.Axis(format="%Y-%m")),
                    y=alt.Y("value:Q", title="金額 (円)", axis=alt.Axis(format="s")),
                    color=alt.Color("指標:N", legend=alt.Legend(title="")),
                    tooltip=[
                        alt.Tooltip("period:T", title="期間", format="%Y-%m"),
                        alt.Tooltip("指標:N"),
                        alt.Tooltip("value:Q", title="金額", format=",")
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)
            st.caption("期間・店舗フィルタの変更は全チャートに即時反映されます。")

    with trend_cols[1]:
        st.metric(
            "検出された品質アラート",
            f"{len(dq_overview)}件",
            help="Excel取込時に検出された欠損・外れ値・重複の件数です。",
        )
        if total_products:
            st.caption(
                f"対象SKU {total_products} 件中 {issue_ratio:.1f}% が再確認推奨です。詳細は下段タブ（在庫/粗利）で確認できます。"
            )
        else:
            st.caption("まだ製品データが登録されていません。サンプルを読み込むかExcelをアップロードしてください。")

    store_transactions = _filter_by_store(context["transactions"], selected_store)
    current_period_df = (
        store_transactions[store_transactions["period"] == selected_period]
        if not store_transactions.empty
        else pd.DataFrame(columns=store_transactions.columns)
    )
    if previous_period is not None and not store_transactions.empty:
        previous_period_df = store_transactions[store_transactions["period"] == previous_period]
    else:
        previous_period_df = pd.DataFrame(columns=store_transactions.columns if not store_transactions.empty else [])

    sales_now = float(current_period_df["sales_amount"].sum()) if not current_period_df.empty else 0.0
    gp_now = float(current_period_df["gross_profit"].sum()) if not current_period_df.empty else 0.0
    cash_balance = _compute_cash_balance(context["cash_daily"], selected_store)
    sales_prev = float(previous_period_df["sales_amount"].sum()) if not previous_period_df.empty else np.nan
    gp_prev = float(previous_period_df["gross_profit"].sum()) if not previous_period_df.empty else np.nan
    cash_prev = _compute_cash_balance_for_period(
        context["cash_daily"], selected_store, previous_period
    )

    primary_cards = [
        {
            "title": "KGI｜売上高",
            "value": _format_currency_short(sales_now),
            "delta": _pct_change(sales_prev, sales_now),
            "delta_format": "{:+.1f}%",
            "note": "期間合計の売上。伸び率は前期比で表示します。",
            "positive_is_good": True,
        },
        {
            "title": "KGI｜粗利",
            "value": _format_currency_short(gp_now),
            "delta": _pct_change(gp_prev, gp_now),
            "delta_format": "{:+.1f}%",
            "note": "粗利額の推移は原価や値引きの影響を確認する指標です。",
            "positive_is_good": True,
        },
        {
            "title": "KGI｜資金残高",
            "value": _format_currency_short(cash_balance),
            "delta": cash_balance - cash_prev if np.isfinite(cash_prev) else None,
            "delta_format": "{:+,.0f}円",
            "note": "日次キャッシュフローから算出した月末残高です。",
            "positive_is_good": True,
        },
    ]

    with selector_left:
        st.markdown("#### KGIダッシュボード")
        render_indicator_cards(primary_cards)
        st.caption("上位指標→トレンド→明細の順で重要情報を確認できます。")

    tabs = st.tabs(["売上", "粗利", "在庫", "資金"])
    with tabs[0]:
        _render_sales_tab(
            context,
            selected_period=selected_period,
            previous_period=previous_period,
            selected_store=selected_store,
            current_period_df=current_period_df,
            previous_period_df=previous_period_df,
        )
    with tabs[1]:
        _render_profit_tab(
            context,
            selected_period=selected_period,
            previous_period=previous_period,
            selected_store=selected_store,
            current_period_df=current_period_df,
            previous_period_df=previous_period_df,
        )
    with tabs[2]:
        control_fn = getattr(st, "segmented_control", st.radio)
        mode = control_fn(
            "表示対象",
            options=["不足のみ", "すべて"],
            key="inventory_filter_mode",
            help="不足しているSKUのみ、または全件表示を切り替えます。",
        )
        threshold = st.slider(
            "安全在庫までの残日数",
            min_value=1.0,
            max_value=14.0,
            value=float(st.session_state.get("inventory_threshold", 3.0)),
            step=0.5,
            key="inventory_threshold",
        )
        _render_inventory_tab(
            context,
            selected_store=selected_store,
            threshold_days=threshold,
            mode=mode,
        )
    with tabs[3]:
        _render_cash_tab(
            context,
            selected_period=selected_period,
            selected_store=selected_store,
        )


METRIC_LABELS = {
    "actual_unit_price": "実際売単価",
    "material_unit_cost": "材料原価",
    "minutes_per_unit": "分/個",
    "daily_qty": "日産数",
    "va_per_min": "付加価値/分",
    "rate_gap_vs_required": "必要賃率差",
    "required_selling_price": "必要販売単価",
}

ANOMALY_REVIEW_CHOICES: List[Dict[str, Any]] = [
    {
        "key": "exception",
        "label": "例外的な値として許容",
        "description": "商流の急変や季節要因など正当な理由がある異常値として扱います。",
        "requires_value": False,
    },
    {
        "key": "input_error",
        "label": "誤入力として修正",
        "description": "入力ミスと判断し訂正値を登録します。保存すると即時にダッシュボードへ反映されます。",
        "requires_value": True,
    },
    {
        "key": "monitor",
        "label": "要調査として記録",
        "description": "原因調査中の異常値として扱い、メモだけ残しておきます。",
        "requires_value": False,
    },
]

ANOMALY_REVIEW_LABELS: Dict[str, str] = {
    choice["key"]: choice["label"] for choice in ANOMALY_REVIEW_CHOICES
}
ANOMALY_REVIEW_DESCRIPTIONS: Dict[str, str] = {
    choice["key"]: choice.get("description", "") for choice in ANOMALY_REVIEW_CHOICES
}
ANOMALY_REVIEW_REQUIRES_VALUE: Dict[str, bool] = {
    choice["key"]: bool(choice.get("requires_value")) for choice in ANOMALY_REVIEW_CHOICES
}
ANOMALY_REVIEW_UNSET_LABEL = "未分類"


def _normalize_review_classification(record: Optional[Dict[str, Any]]) -> Optional[str]:
    """Resolve the stored review classification from legacy or new schema."""

    if not record:
        return None
    classification = record.get("classification")
    if classification:
        return classification
    decision = record.get("decision")
    if decision == "corrected":
        return "input_error"
    return decision

st.markdown(
    """
    <style>
    .main > div {
        background-color: var(--app-bg);
    }
    [data-testid="stMetric"] {
        background-color: var(--app-surface);
        border-radius: 12px;
        border: 1px solid rgba(11, 31, 59, 0.15);
        padding: calc(var(--spacing-unit) * 1.5);
        box-shadow: 0 2px 6px rgba(11, 31, 59, 0.12);
        color: var(--app-text);
    }
    [data-testid="stMetricDelta"] span {
        font-weight: 600;
        color: var(--app-accent);
    }
    .metric-badge {
        text-align: right;
        color: var(--app-accent);
        font-weight: 600;
        letter-spacing: 0.02em;
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


def _determine_pdca_comment(
    *, required_rate: float, va_per_min: float, delta_va: float, delta_ach: float
) -> str:
    """Generate a PDCA oriented commentary for KPI trend tracking."""

    tol = 0.05  # 5銭単位の微細な変化はノイズとして扱う
    if pd.isna(va_per_min) or pd.isna(required_rate):
        return "Plan: KPIが不足しているため基準値の再確認が必要です。"

    gap = va_per_min - required_rate
    improving = np.isfinite(delta_va) and delta_va > tol
    worsening = np.isfinite(delta_va) and delta_va < -tol

    if gap >= tol:
        if improving:
            return "Act: 必要賃率超過幅が広がっており改善を定着させています。"
        return "Act: 必要賃率を上回っており現状維持フェーズです。"

    if gap <= -tol:
        if improving:
            return "Check: まだ未達ですが改善傾向を確認できました。"
        return "Do: 必要賃率を下回っているため追加施策の実行が必要です。"

    if improving:
        return "Check: 基準付近で改善が進んでいます。"
    if worsening:
        return "Do: 基準付近ですが悪化傾向のため注意が必要です。"

    if np.isfinite(delta_ach) and abs(delta_ach) > 0.2:
        if delta_ach > 0:
            return "Check: 達成率が上昇しており施策効果を確認できています。"
        return "Do: 達成率が低下しているため原因分析が求められます。"

    return "Plan: 大きな変化はなく基準水準を維持しています。"


def _build_pdca_summary(trend_df: pd.DataFrame) -> pd.DataFrame:
    """Build PDCA commentary rows for each scenario-period combination."""

    columns = [
        "scenario",
        "period",
        "required_rate",
        "va_per_min",
        "ach_rate",
        "delta_va",
        "delta_ach",
        "pdca_comment",
    ]
    if trend_df is None or trend_df.empty:
        return pd.DataFrame(columns=columns)

    df = trend_df.copy()
    df["period"] = pd.to_datetime(df["period"])
    df = df.dropna(subset=["period"])
    if df.empty:
        return pd.DataFrame(columns=columns)

    records: List[Dict[str, Any]] = []
    for scenario, group in df.groupby("scenario"):
        g = group.sort_values("period")
        prev_va = np.nan
        prev_ach = np.nan
        for _, row in g.iterrows():
            va = float(row.get("va_per_min", np.nan))
            req = float(row.get("required_rate", np.nan))
            ach = float(row.get("ach_rate", np.nan))
            delta_va = (
                va - prev_va
                if np.isfinite(va) and np.isfinite(prev_va)
                else np.nan
            )
            delta_ach = (
                ach - prev_ach
                if np.isfinite(ach) and np.isfinite(prev_ach)
                else np.nan
            )
            comment = _determine_pdca_comment(
                required_rate=req,
                va_per_min=va,
                delta_va=delta_va,
                delta_ach=delta_ach,
            )
            records.append(
                {
                    "scenario": scenario,
                    "period": row["period"],
                    "required_rate": req,
                    "va_per_min": va,
                    "ach_rate": ach,
                    "delta_va": delta_va,
                    "delta_ach": delta_ach,
                    "pdca_comment": comment,
                }
            )
            prev_va = va if np.isfinite(va) else prev_va
            prev_ach = ach if np.isfinite(ach) else prev_ach

    return pd.DataFrame(records, columns=columns)


def _sku_to_str(value: Any) -> str:
    """Normalize product numbers for consistent dictionary keys."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _format_number(value: Any) -> str:
    """Format numeric values for concise display in anomaly prompts."""

    if value is None or pd.isna(value):
        return "N/A"
    val = float(value)
    if abs(val) >= 1000:
        return f"{val:,.0f}"
    return f"{val:,.2f}"


def _format_currency(value: Any, unit: str = "円", decimals: int = 0) -> str:
    """Format numeric values as currency text with thousands separators."""

    if value is None or pd.isna(value):
        return "N/A"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "N/A"
    fmt = f"{{:,.{decimals}f}}" if decimals > 0 else "{:,}"  # type: ignore[str-format]
    return f"{fmt.format(val)}{unit}"


def _format_roi(value: Any) -> str:
    """Format ROI months with guard for non-finite numbers."""

    if value is None or pd.isna(value):
        return "N/A"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(val):
        return "∞"
    return f"{val:.1f}"


def _format_delta(value: float, suffix: str) -> str:
    """Format change metrics with sign and suffix, handling NaN gracefully."""

    if value is None or pd.isna(value) or not np.isfinite(value):
        return "N/A"
    return f"{value:+.1f}{suffix}"


SIMULATION_PRESETS: Dict[str, Dict[str, Any]] = {
    "販売価格+5%": {
        "adjustments": {
            "quick_price": 5,
            "quick_ct": 0,
            "quick_material": 0,
            "quick_volume": 0,
        },
        "description": "すべての製品で販売単価を一律5%引き上げるケース",
    },
    "リードタイム-10%": {
        "adjustments": {
            "quick_price": 0,
            "quick_ct": -10,
            "quick_material": 0,
            "quick_volume": 0,
        },
        "description": "1製品当たりの製造時間（分/個）を10%圧縮するケース",
    },
    "材料費-3%": {
        "adjustments": {
            "quick_price": 0,
            "quick_ct": 0,
            "quick_material": -3,
            "quick_volume": 0,
        },
        "description": "原材料コストを平均で3%削減するケース",
    },
    "増産+15%": {
        "adjustments": {
            "quick_price": 0,
            "quick_ct": 0,
            "quick_material": 0,
            "quick_volume": 15,
        },
        "description": "日産数を15%拡大するケース",
    },
}


def apply_simulation_preset(label: str) -> None:
    """Apply predefined what-if adjustments to quick simulation controls."""

    preset = SIMULATION_PRESETS.get(label)
    if not preset:
        return
    adjustments = preset.get("adjustments", {})
    for key, value in adjustments.items():
        st.session_state[key] = value
    st.session_state["active_simulation"] = label


def _detect_simulation_label(qp: int, qc: int, qm: int, qv: int) -> str:
    """Return a human friendly label for the current quick adjustments."""

    for label, preset in SIMULATION_PRESETS.items():
        adjustments = preset.get("adjustments", {})
        if (
            adjustments.get("quick_price", 0) == qp
            and adjustments.get("quick_ct", 0) == qc
            and adjustments.get("quick_material", 0) == qm
            and adjustments.get("quick_volume", 0) == qv
        ):
            return label
    if qp == 0 and qc == 0 and qm == 0 and qv == 0:
        return "ベース"
    return "カスタム設定"


def _resolve_scenario_label(
    qp: int,
    qc: int,
    qm: int,
    qv: int,
    saved: Optional[Dict[str, Dict[str, Any]]],
) -> str:
    """Determine the most relevant scenario label for quick simulation values."""

    if saved:
        for name, config in saved.items():
            if (
                int(config.get("quick_price", 0)) == int(qp)
                and int(config.get("quick_ct", 0)) == int(qc)
                and int(config.get("quick_material", 0)) == int(qm)
                and int(config.get("quick_volume", 0)) == int(qv)
            ):
                return str(name)
    return _detect_simulation_label(qp, qc, qm, qv)


def _simulate_scenario(
    df_template: pd.DataFrame,
    *,
    price_pct: float,
    ct_pct: float,
    volume_pct: float,
    material_pct: float,
    be_rate: float,
    req_rate: float,
    delta_low: float,
    delta_high: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Apply percentage adjustments and compute KPI metrics for a scenario."""

    df_sim = df_template.copy()
    if price_pct:
        df_sim["actual_unit_price"] *= 1 + float(price_pct) / 100.0
    if ct_pct:
        df_sim["minutes_per_unit"] *= 1 + float(ct_pct) / 100.0
    if volume_pct:
        df_sim["daily_qty"] *= 1 + float(volume_pct) / 100.0
    if material_pct:
        df_sim["material_unit_cost"] *= 1 + float(material_pct) / 100.0

    df_sim["gp_per_unit"] = df_sim["actual_unit_price"] - df_sim["material_unit_cost"]
    df_sim["daily_total_minutes"] = df_sim["minutes_per_unit"] * df_sim["daily_qty"]
    df_sim["daily_va"] = df_sim["gp_per_unit"] * df_sim["daily_qty"]
    with np.errstate(divide="ignore", invalid="ignore"):
        df_sim["va_per_min"] = df_sim["daily_va"] / df_sim["daily_total_minutes"]

    df_result = compute_results(df_sim, be_rate, req_rate, delta_low, delta_high)
    if len(df_result) > 0:
        ach_rate = float(df_result["meets_required_rate"].mean() * 100.0)
    else:
        ach_rate = float("nan")

    avg_vapm = _safe_series_mean(df_result.get("va_per_min"))
    avg_gap = _safe_series_mean(df_result.get("rate_gap_vs_required"))
    avg_required_price = _safe_series_mean(df_result.get("required_selling_price"))

    if "daily_va" in df_result:
        daily_total = float(
            np.nansum(pd.to_numeric(df_result["daily_va"], errors="coerce"))
        )
    else:
        daily_total = float("nan")

    return df_result, {
        "ach_rate": ach_rate,
        "avg_vapm": float(avg_vapm) if np.isfinite(avg_vapm) else float("nan"),
        "daily_va_total": daily_total,
        "avg_gap": float(avg_gap) if np.isfinite(avg_gap) else float("nan"),
        "avg_required_price": float(avg_required_price)
        if np.isfinite(avg_required_price)
        else float("nan"),
    }


def _sanitize_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """Normalize metric dictionary to safe numeric values for display."""

    ach = metrics.get("ach_rate", float("nan"))
    if not np.isfinite(ach):
        ach = 0.0
    avg = metrics.get("avg_vapm", float("nan"))
    if not np.isfinite(avg):
        avg = np.nan
    daily = metrics.get("daily_va_total", float("nan"))
    if not np.isfinite(daily):
        daily = np.nan
    avg_gap = metrics.get("avg_gap", float("nan"))
    if not np.isfinite(avg_gap):
        avg_gap = np.nan
    avg_required_price = metrics.get("avg_required_price", float("nan"))
    if not np.isfinite(avg_required_price):
        avg_required_price = np.nan
    return {
        "ach_rate": ach,
        "avg_vapm": avg,
        "daily_va_total": daily,
        "avg_gap": avg_gap,
        "avg_required_price": avg_required_price,
    }


def _format_adjustment_summary(adjustments: Dict[str, Any]) -> str:
    """Return a compact text description of percentage adjustments."""

    qp = int(adjustments.get("quick_price", 0))
    qc = int(adjustments.get("quick_ct", 0))
    qv = int(adjustments.get("quick_volume", 0))
    qm = int(adjustments.get("quick_material", 0))
    return f"価格{qp:+d}%・リードタイム{qc:+d}%・生産量{qv:+d}%・材料{qm:+d}%"


def _summarize_scenario_effect(
    *,
    ach_delta: float,
    vapm_delta: float,
    daily_delta: float,
    annual_delta: float,
    req_price_delta: float,
    gap_delta: float,
) -> str:
    """Build a one-line summary describing KPI deltas for the active scenario."""

    pieces: List[str] = []
    if np.isfinite(ach_delta) and abs(ach_delta) >= 0.05:
        pieces.append(f"達成率 {ach_delta:+.1f}pt")
    if np.isfinite(vapm_delta) and abs(vapm_delta) >= 0.01:
        pieces.append(f"平均VA/分 {vapm_delta:+.2f}円")
    if np.isfinite(daily_delta) and abs(daily_delta) >= 100.0:
        pieces.append(f"日次付加価値 {daily_delta:+,.0f}円")
    if np.isfinite(annual_delta) and abs(annual_delta) >= 1000.0:
        pieces.append(f"年間利益 {annual_delta / 10000:+,.1f}万円")
    if np.isfinite(req_price_delta) and abs(req_price_delta) >= 1.0:
        direction = "低下" if req_price_delta < 0 else "上昇"
        pieces.append(
            f"平均必要販売単価 {abs(req_price_delta):,.0f}円{direction}"
        )
    if np.isfinite(gap_delta) and abs(gap_delta) >= 0.01:
        direction = "改善" if gap_delta >= 0 else "悪化"
        pieces.append(
            f"必要賃率との差 {abs(gap_delta):.2f}円/分{direction}"
        )
    return " / ".join(pieces)


def _safe_series_mean(series: Optional[pd.Series]) -> float:
    """Return a finite mean value for the provided numeric series if possible."""

    if series is None:
        return float("nan")
    cleaned = pd.to_numeric(series, errors="coerce")
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        return float("nan")
    return float(cleaned.mean())


def _analyze_driver_impacts(
    df_template: pd.DataFrame,
    df_base: pd.DataFrame,
    base_metrics: Dict[str, float],
    *,
    price_pct: float,
    ct_pct: float,
    volume_pct: float,
    material_pct: float,
    be_rate: float,
    req_rate: float,
    delta_low: float,
    delta_high: float,
    working_days: float,
) -> Tuple[pd.DataFrame, List[str]]:
    """Simulate each adjustment factor independently and summarise KPI deltas."""

    base_daily = float(base_metrics.get("daily_va_total", float("nan")))
    if not np.isfinite(base_daily):
        base_daily = float("nan")
    base_avg_va = float(base_metrics.get("avg_vapm", float("nan")))
    if not np.isfinite(base_avg_va):
        base_avg_va = float("nan")
    base_avg_gap = float(base_metrics.get("avg_gap", float("nan")))
    if not np.isfinite(base_avg_gap):
        base_avg_gap = float("nan")
    base_avg_req_price = float(
        base_metrics.get("avg_required_price", float("nan"))
    )
    if not np.isfinite(base_avg_req_price):
        base_avg_req_price = float("nan")

    driver_configs = [
        ("販売価格", price_pct, "price"),
        ("リードタイム", ct_pct, "lead_time"),
        ("生産量", volume_pct, "volume"),
        ("材料費", material_pct, "material"),
    ]

    records: List[Dict[str, Any]] = []
    insights: List[str] = []
    valid_working_days = (
        working_days is not None and np.isfinite(working_days) and working_days > 0
    )

    for label, pct_value, key in driver_configs:
        if pct_value in (None, 0) or not np.isfinite(float(pct_value)):
            continue

        kwargs = {
            "price_pct": 0.0,
            "ct_pct": 0.0,
            "volume_pct": 0.0,
            "material_pct": 0.0,
        }
        kwargs["price_pct"] = kwargs["price_pct"] if key != "price" else pct_value
        kwargs["ct_pct"] = kwargs["ct_pct"] if key != "lead_time" else pct_value
        kwargs["volume_pct"] = kwargs["volume_pct"] if key != "volume" else pct_value
        kwargs["material_pct"] = (
            kwargs["material_pct"] if key != "material" else pct_value
        )

        df_driver, driver_metrics = _simulate_scenario(
            df_template,
            price_pct=kwargs["price_pct"],
            ct_pct=kwargs["ct_pct"],
            volume_pct=kwargs["volume_pct"],
            material_pct=kwargs["material_pct"],
            be_rate=be_rate,
            req_rate=req_rate,
            delta_low=delta_low,
            delta_high=delta_high,
        )
        driver_metrics_clean = _sanitize_metrics(driver_metrics)

        daily_candidate = driver_metrics_clean.get("daily_va_total", float("nan"))
        daily_delta = daily_candidate - base_daily
        if not np.isfinite(daily_delta):
            daily_delta = float("nan")
        annual_delta = float("nan")
        if valid_working_days and np.isfinite(daily_delta):
            annual_delta = daily_delta * float(working_days)

        avg_va = float(driver_metrics_clean.get("avg_vapm", float("nan")))
        avg_gap = float(driver_metrics_clean.get("avg_gap", float("nan")))
        avg_req_price = float(
            driver_metrics_clean.get("avg_required_price", float("nan"))
        )

        avg_va_delta = (
            avg_va - base_avg_va
            if np.isfinite(avg_va) and np.isfinite(base_avg_va)
            else float("nan")
        )
        avg_gap_delta = (
            avg_gap - base_avg_gap
            if np.isfinite(avg_gap) and np.isfinite(base_avg_gap)
            else float("nan")
        )
        req_price_delta = (
            avg_req_price - base_avg_req_price
            if np.isfinite(avg_req_price) and np.isfinite(base_avg_req_price)
            else float("nan")
        )

        records.append(
            {
                "施策": label,
                "変化率(%)": float(pct_value),
                "日次付加価値差(円)": daily_delta,
                "年間利益差(万円)": annual_delta / 10000.0
                if np.isfinite(annual_delta)
                else float("nan"),
                "平均VA/分差(円)": avg_va_delta,
                "平均必要賃率差(円/分)": avg_gap_delta,
                "平均必要販売単価差(円)": req_price_delta,
            }
        )

        if key in {"price", "volume", "material"} and np.isfinite(daily_delta):
            direction_daily = "増加" if daily_delta >= 0 else "減少"
            daily_abs = abs(daily_delta)
            annual_phrase = ""
            if np.isfinite(annual_delta):
                annual_abs = abs(annual_delta) / 10000.0
                direction_annual = "増加" if annual_delta >= 0 else "減少"
                annual_phrase = (
                    f"、年間利益が{annual_abs:,.1f}万円{direction_annual}"
                )
            insights.append(
                f"{label}を{int(pct_value):+d}%調整すると日次付加価値が{daily_abs:,.0f}円{direction_daily}{annual_phrase}します。"
            )
        elif key == "lead_time":
            parts: List[str] = []
            if np.isfinite(avg_va_delta):
                direction = "増加" if avg_va_delta >= 0 else "減少"
                parts.append(
                    f"平均VA/分が{abs(avg_va_delta):.2f}円{direction}"
                )
            if np.isfinite(avg_gap_delta):
                direction = "改善" if avg_gap_delta >= 0 else "悪化"
                parts.append(
                    f"必要賃率との差が{abs(avg_gap_delta):.2f}円/分{direction}"
                )
            if np.isfinite(req_price_delta):
                direction = "低下" if req_price_delta < 0 else "上昇"
                parts.append(
                    f"必要販売単価が{abs(req_price_delta):,.0f}円{direction}"
                )
            if parts:
                joined = "、".join(parts)
                insights.append(
                    f"{label}を{int(pct_value):+d}%調整すると{joined}します。"
                )

    if records:
        df_summary = pd.DataFrame(records)
    else:
        df_summary = pd.DataFrame(
            columns=[
                "施策",
                "変化率(%)",
                "日次付加価値差(円)",
                "年間利益差(万円)",
                "平均VA/分差(円)",
                "平均必要賃率差(円/分)",
                "平均必要販売単価差(円)",
            ]
        )

    return df_summary, insights


def _format_fermi_estimate(delta_daily_va: float, working_days: float, scenario_label: str) -> str:
    """Build a short Fermi style estimate text for annual profit impact."""

    if working_days is None or working_days <= 0:
        return "稼働日数の情報が不足しているため年間影響を概算できません。"
    if delta_daily_va is None or not np.isfinite(delta_daily_va):
        return "シミュレーション結果から日次付加価値の変化を取得できませんでした。"
    if abs(delta_daily_va) < 1:
        return "日次付加価値の変化がごく小さいため年間影響は限定的と推定されます。"

    annual_change = float(delta_daily_va) * float(working_days)
    lower = abs(annual_change) * 0.8
    upper = abs(annual_change) * 1.2
    sign = "増加" if annual_change >= 0 else "減少"
    scenario = scenario_label or "カスタム設定"
    return (
        f"{scenario} を適用すると日次の付加価値(粗利相当)が {delta_daily_va:+,.0f} 円変化 → "
        f"年間利益インパクトは{sign}方向に概ね {lower:,.0f} ～ {upper:,.0f} 円と推定されます。"
    )


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


def _format_period_label(value: Any, freq: str) -> str:
    """Format a timestamp for display based on the selected frequency."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return str(value)
    if freq == "四半期":
        return str(ts.to_period("Q"))
    return ts.strftime("%Y-%m")


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
        roi_txt = _format_roi(row.get("roi_months"))
        gap_val = row.get("gap")
        gap_txt = "N/A" if gap_val is None or pd.isna(gap_val) else f"{float(gap_val):.2f}円/分"
        action_label = row.get("best_action_label")
        if action_label and action_label != "推奨なし":
            action_txt = f", 推奨 {action_label}"
        else:
            action_txt = ""
        benefit_txt = _format_currency(row.get("best_monthly_benefit"))
        top_gap_lines.append(
            f"- {row.get('product_name','不明')} (ギャップ {gap_txt}, ROI {roi_txt}ヶ月{action_txt}, 月次効果 {benefit_txt})"
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

    def _format_segment_line(row: Dict[str, Any]) -> str:
        segment = row.get("segment", "不明")
        pieces = []
        avg_va = row.get("avg_va_per_min")
        gap_val = row.get("avg_gap")
        ach_val = row.get("ach_rate_pct")
        roi_val = row.get("avg_roi_months")
        if avg_va is not None and not pd.isna(avg_va):
            pieces.append(f"VA/分 {float(avg_va):.1f}円")
        if gap_val is not None and not pd.isna(gap_val):
            pieces.append(f"差 {float(gap_val):+.1f}円")
        if ach_val is not None and not pd.isna(ach_val):
            pieces.append(f"達成率 {float(ach_val):.1f}%")
        if roi_val is not None and not pd.isna(roi_val):
            pieces.append(f"ROI {float(roi_val):.1f}月")
        detail = " / ".join(pieces) if pieces else "データ不足"
        return f"- {segment}: {detail}"

    category_text = "\n".join(
        [_format_segment_line(row) for row in insights.get("segment_category", [])[:3]]
    ) or "- 情報不足"
    customer_text = "\n".join(
        [_format_segment_line(row) for row in insights.get("segment_customer", [])[:3]]
    ) or "- 情報不足"

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
        f"カテゴリー別サマリ:\n{category_text}\n"
        f"主要顧客別サマリ:\n{customer_text}\n"
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

render_sidebar_nav(page_key="dashboard")

header_col, help_col = st.columns([0.76, 0.24], gap="small")
with header_col:
    st.title("② ダッシュボード")

render_help_button("dashboard", container=help_col)

render_onboarding()
render_page_tutorial("dashboard")
render_stepper(4)
scenario_name = st.session_state.get("current_scenario", "ベース")
st.caption(f"適用中シナリオ: {scenario_name}")
st.session_state.setdefault("quick_price", 0)
st.session_state.setdefault("quick_ct", 0)
st.session_state.setdefault("quick_volume", 0)
st.session_state.setdefault("quick_material", 0)
st.session_state.setdefault("active_simulation", "ベース")
st.session_state.setdefault(
    "plotly_draw_tools", ["drawline", "drawrect", "drawopenpath", "drawcircle", "eraseshape"]
)
st.session_state.setdefault("show_rangeslider", True)
st.session_state.setdefault("show_spikelines", True)
scenario_store = st.session_state.setdefault("whatif_scenarios", {})

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
    st.session_state["quick_volume"] = 0
    st.session_state["quick_material"] = 0
    st.session_state["active_simulation"] = "ベース"

if "df_products_raw" not in st.session_state or st.session_state["df_products_raw"] is None or len(st.session_state["df_products_raw"]) == 0:
    st.info("先に『① データ入力 & 取り込み』でデータを準備してください。")
    st.stop()

df_raw_all = st.session_state["df_products_raw"]

view_options = ["行動設計ビュー", "詳細分析ビュー"]
st.session_state.setdefault("dashboard_view_mode", view_options[0])
segmented = getattr(st, "segmented_control", None)
if callable(segmented):
    selected_view = segmented(
        "表示モード",
        options=view_options,
        key="dashboard_view_mode",
        help="日次業務に最適化したビューと詳細分析ビューを切り替えます。",
    )
else:
    selected_view = st.radio(
        "表示モード",
        options=view_options,
        key="dashboard_view_mode",
        horizontal=True,
        help="日次業務に最適化したビューと詳細分析ビューを切り替えます。",
    )

if selected_view == "行動設計ビュー":
    _render_behavior_dashboard(df_raw_all)
    sync_offline_cache()
    st.stop()
st.session_state.setdefault("anomaly_review", {})
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

with st.expander("ダッシュボードの表示調整", expanded=False):
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

# Quick simulation presets & toggles
st.markdown("#### 🎯 What-ifシミュレーション")
preset_cols = st.columns(len(SIMULATION_PRESETS))
for col, (label, preset) in zip(preset_cols, SIMULATION_PRESETS.items()):
    desc = preset.get("description")
    if col.button(label, help=desc):
        apply_simulation_preset(label)
        st.rerun()

qcol1, qcol2, qcol3, qcol4, qcol5 = st.columns([1.1, 1.1, 1.1, 1.1, 0.8])
with qcol1:
    st.slider(
        "販売価格",
        min_value=-10,
        max_value=15,
        value=int(st.session_state.get("quick_price", 0)),
        step=1,
        format="%d%%",
        key="quick_price",
        help="製品価格を一律で増減させる簡易試算です。",
    )
with qcol2:
    st.slider(
        "リードタイム (分/個)",
        min_value=-30,
        max_value=30,
        value=int(st.session_state.get("quick_ct", 0)),
        step=1,
        format="%d%%",
        key="quick_ct",
        help="製品1個当たりの所要時間（分/個）を短縮/延長した場合を想定します。",
    )
with qcol3:
    st.slider(
        "生産量 (日産数)",
        min_value=-30,
        max_value=30,
        value=int(st.session_state.get("quick_volume", 0)),
        step=1,
        format="%d%%",
        key="quick_volume",
        help="日産数を一律で増減させたときの影響を試算します。",
    )
with qcol4:
    st.slider(
        "材料費",
        min_value=-10,
        max_value=10,
        value=int(st.session_state.get("quick_material", 0)),
        step=1,
        format="%d%%",
        key="quick_material",
        help="原材料コストを全SKUで同じ割合だけ増減させます。",
    )
with qcol5:
    st.button("リセット", on_click=reset_quick_params)

qp = st.session_state["quick_price"]
qc = st.session_state["quick_ct"]
qv = st.session_state["quick_volume"]
qm = st.session_state["quick_material"]
active_label = _resolve_scenario_label(qp, qc, qm, qv, scenario_store)
st.session_state["active_simulation"] = active_label
preset_desc = SIMULATION_PRESETS.get(active_label, {}).get("description", "")
summary_text = (
    f"販売価格{qp:+d}%｜リードタイム{qc:+d}%｜生産量{qv:+d}%｜材料費{qm:+d}%"
)
if active_label == "ベース":
    st.caption(f"シミュレーション: ベースライン（{summary_text}）")
else:
    detail = f"｜{preset_desc}" if preset_desc else ""
    st.caption(f"シミュレーション: {active_label}（{summary_text}）{detail}")

feedback = st.session_state.pop("scenario_manager_feedback", None)
if feedback:
    level = feedback.get("type", "info") if isinstance(feedback, dict) else "info"
    message = feedback.get("message", "") if isinstance(feedback, dict) else str(feedback)
    notify = {"success": st.success, "warning": st.warning, "info": st.info}.get(level, st.info)
    if message:
        notify(message)

with st.expander("💾 シナリオ管理", expanded=False):
    st.caption("現在のクイック調整を名前を付けて保存し、後から呼び出して比較できます。")
    saved_names = list(scenario_store.keys())
    manage_cols = None
    selected_saved: Optional[str] = None
    if saved_names:
        selected_saved = st.selectbox(
            "保存済みシナリオ",
            ["選択なし"] + saved_names,
            key="scenario_manager_select",
        )
        manage_cols = st.columns(2)
        if manage_cols[0].button("適用", key="scenario_manager_load"):
            if selected_saved and selected_saved != "選択なし":
                config = scenario_store.get(selected_saved, {})
                st.session_state["quick_price"] = int(config.get("quick_price", 0))
                st.session_state["quick_ct"] = int(config.get("quick_ct", 0))
                st.session_state["quick_volume"] = int(config.get("quick_volume", 0))
                st.session_state["quick_material"] = int(config.get("quick_material", 0))
                st.session_state["scenario_manager_feedback"] = {
                    "type": "success",
                    "message": f"{selected_saved} を適用しました。",
                }
                st.rerun()
        if manage_cols[1].button("削除", key="scenario_manager_delete"):
            if selected_saved and selected_saved != "選択なし":
                scenario_store.pop(selected_saved, None)
                st.session_state["whatif_scenarios"] = scenario_store
                st.session_state["scenario_manager_feedback"] = {
                    "type": "info",
                    "message": f"{selected_saved} を削除しました。",
                }
                st.rerun()
    else:
        st.caption("保存済みシナリオはまだありません。下で名前を入力して保存してください。")

    if st.session_state.pop("scenario_manager_clear_input", False):
        st.session_state["scenario_save_name"] = ""

    new_name = st.text_input(
        "シナリオ名",
        key="scenario_save_name",
        help="例: 施策A (価格+5%)、施策B (CT-10%) など",
    )
    if st.button("保存/上書き", key="scenario_manager_save"):
        trimmed = new_name.strip()
        if not trimmed:
            st.warning("シナリオ名を入力してください。")
        else:
            scenario_store[trimmed] = {
                "quick_price": int(qp),
                "quick_ct": int(qc),
                "quick_volume": int(qv),
                "quick_material": int(qm),
            }
            st.session_state["whatif_scenarios"] = scenario_store
            st.session_state["scenario_manager_feedback"] = {
                "type": "success",
                "message": f"{trimmed} を保存しました。",
            }
            st.session_state["scenario_manager_clear_input"] = True
            st.rerun()

scenario_template = df_view_filtered.copy()
df_base, base_metrics = _simulate_scenario(
    scenario_template,
    price_pct=0,
    ct_pct=0,
    volume_pct=0,
    material_pct=0,
    be_rate=be_rate,
    req_rate=req_rate,
    delta_low=delta_low,
    delta_high=delta_high,
)
base_metrics_clean = _sanitize_metrics(base_metrics)
base_ach_rate = base_metrics_clean["ach_rate"]
base_avg_vapm = base_metrics_clean["avg_vapm"]
base_daily_va_total = base_metrics_clean["daily_va_total"]
base_avg_gap = base_metrics_clean.get("avg_gap", float("nan"))
base_avg_req_price = base_metrics_clean.get("avg_required_price", float("nan"))

df_view, active_metrics = _simulate_scenario(
    scenario_template,
    price_pct=qp,
    ct_pct=qc,
    volume_pct=qv,
    material_pct=qm,
    be_rate=be_rate,
    req_rate=req_rate,
    delta_low=delta_low,
    delta_high=delta_high,
)
active_metrics_clean = _sanitize_metrics(active_metrics)
ach_rate = active_metrics_clean["ach_rate"]
avg_vapm = active_metrics_clean["avg_vapm"]
sim_daily_va_total = active_metrics_clean["daily_va_total"]
avg_gap = active_metrics_clean.get("avg_gap", float("nan"))
avg_req_price = active_metrics_clean.get("avg_required_price", float("nan"))

daily_delta = (
    sim_daily_va_total - base_daily_va_total
    if np.isfinite(sim_daily_va_total) and np.isfinite(base_daily_va_total)
    else float("nan")
)
ach_delta = (
    ach_rate - base_ach_rate
    if np.isfinite(ach_rate) and np.isfinite(base_ach_rate)
    else float("nan")
)
vapm_delta = (
    avg_vapm - base_avg_vapm
    if np.isfinite(avg_vapm) and np.isfinite(base_avg_vapm)
    else float("nan")
)
gap_delta_metric = (
    avg_gap - base_avg_gap
    if np.isfinite(avg_gap) and np.isfinite(base_avg_gap)
    else float("nan")
)
req_price_delta_metric = (
    avg_req_price - base_avg_req_price
    if np.isfinite(avg_req_price) and np.isfinite(base_avg_req_price)
    else float("nan")
)

raw_working_days = base_params.get("working_days", DEFAULT_PARAMS["working_days"])
try:
    working_days = float(raw_working_days)
except (TypeError, ValueError):
    working_days = float("nan")
if not np.isfinite(working_days) or working_days <= 0:
    working_days = float("nan")

annual_base_va = (
    base_daily_va_total * working_days
    if np.isfinite(base_daily_va_total) and np.isfinite(working_days)
    else float("nan")
)
annual_sim_va = (
    sim_daily_va_total * working_days
    if np.isfinite(sim_daily_va_total) and np.isfinite(working_days)
    else float("nan")
)
annual_delta = (
    annual_sim_va - annual_base_va
    if np.isfinite(annual_sim_va) and np.isfinite(annual_base_va)
    else float("nan")
)

scenario_results: Dict[str, Dict[str, Any]] = {
    "ベース": {
        "df": df_base,
        "metrics": base_metrics_clean,
        "adjustments": {
            "quick_price": 0,
            "quick_ct": 0,
            "quick_volume": 0,
            "quick_material": 0,
        },
    }
}

for name, config in scenario_store.items():
    saved_df, saved_metrics = _simulate_scenario(
        scenario_template,
        price_pct=config.get("quick_price", 0),
        ct_pct=config.get("quick_ct", 0),
        volume_pct=config.get("quick_volume", 0),
        material_pct=config.get("quick_material", 0),
        be_rate=be_rate,
        req_rate=req_rate,
        delta_low=delta_low,
        delta_high=delta_high,
    )
    scenario_results[name] = {
        "df": saved_df,
        "metrics": _sanitize_metrics(saved_metrics),
        "adjustments": {
            "quick_price": int(config.get("quick_price", 0)),
            "quick_ct": int(config.get("quick_ct", 0)),
            "quick_volume": int(config.get("quick_volume", 0)),
            "quick_material": int(config.get("quick_material", 0)),
        },
    }

if active_label != "ベース":
    scenario_results[active_label] = {
        "df": df_view,
        "metrics": active_metrics_clean,
        "adjustments": {
            "quick_price": int(qp),
            "quick_ct": int(qc),
            "quick_volume": int(qv),
            "quick_material": int(qm),
        },
    }

option_candidates = ["ベース"] + list(scenario_store.keys())
if active_label and active_label not in option_candidates:
    option_candidates.append(active_label)
scenario_options = list(dict.fromkeys(option_candidates))

compare_key = "scenario_compare_selection"
if compare_key in st.session_state:
    current_selection = [
        scen for scen in st.session_state.get(compare_key, []) if scen in scenario_options
    ]
    if not current_selection:
        current_selection = scenario_options
    if set(current_selection) != set(st.session_state.get(compare_key, [])):
        st.session_state[compare_key] = current_selection
else:
    st.session_state[compare_key] = scenario_options

selected_scenarios = st.multiselect(
    "シナリオ選択",
    scenario_options,
    default=scenario_options,
    key=compare_key,
)

st.markdown("#### 📁 シナリオ比較レポート")

comparison_records: List[Dict[str, Any]] = []
base_ach = base_metrics_clean["ach_rate"]
base_avg = base_metrics_clean["avg_vapm"]
base_daily = base_metrics_clean["daily_va_total"]

def _delta_or_nan(current: float, base_value: float) -> float:
    if np.isfinite(current) and np.isfinite(base_value):
        return float(current) - float(base_value)
    return float("nan")

for scen_name in selected_scenarios:
    scen_data = scenario_results.get(scen_name)
    if not scen_data:
        continue
    metrics = scen_data.get("metrics", {})
    adjustments = scen_data.get("adjustments", {})
    ach_val = float(metrics.get("ach_rate", np.nan))
    avg_val = float(metrics.get("avg_vapm", np.nan))
    daily_val = float(metrics.get("daily_va_total", np.nan))
    avg_req_price_val = float(metrics.get("avg_required_price", np.nan))
    avg_gap_val = float(metrics.get("avg_gap", np.nan))
    comparison_records.append(
        {
            "シナリオ": scen_name,
            "調整サマリ": _format_adjustment_summary(adjustments),
            "販売価格調整(%)": int(adjustments.get("quick_price", 0)),
            "リードタイム調整(%)": int(adjustments.get("quick_ct", 0)),
            "生産量調整(%)": int(adjustments.get("quick_volume", 0)),
            "材料費調整(%)": int(adjustments.get("quick_material", 0)),
            "必要賃率達成率(%)": ach_val,
            "達成率差分(pts)": 0.0
            if scen_name == "ベース" and np.isfinite(base_ach)
            else _delta_or_nan(ach_val, base_ach),
            "平均VA/分(円)": avg_val,
            "平均VA/分差分(円)": 0.0
            if scen_name == "ベース" and np.isfinite(base_avg)
            else _delta_or_nan(avg_val, base_avg),
            "日次付加価値(円)": daily_val,
            "日次付加価値差分(円)": 0.0
            if scen_name == "ベース" and np.isfinite(base_daily)
            else _delta_or_nan(daily_val, base_daily),
            "平均必要販売単価(円)": avg_req_price_val,
            "平均必要販売単価差分(円)": 0.0
            if scen_name == "ベース" and np.isfinite(base_avg_req_price)
            else _delta_or_nan(avg_req_price_val, base_avg_req_price),
            "平均必要賃率差(円/分)": avg_gap_val,
            "平均必要賃率差分(円/分)": 0.0
            if scen_name == "ベース" and np.isfinite(base_avg_gap)
            else _delta_or_nan(avg_gap_val, base_avg_gap),
        }
    )

if comparison_records:
    comparison_df = pd.DataFrame(comparison_records)
    styled = comparison_df.style.format(
        {
            "販売価格調整(%)": "{:+d}",
            "リードタイム調整(%)": "{:+d}",
            "生産量調整(%)": "{:+d}",
            "材料費調整(%)": "{:+d}",
            "必要賃率達成率(%)": "{:.1f}",
            "達成率差分(pts)": "{:+.1f}",
            "平均VA/分(円)": "{:.2f}",
            "平均VA/分差分(円)": "{:+.2f}",
            "日次付加価値(円)": "{:,.0f}",
            "日次付加価値差分(円)": "{:+,.0f}",
            "平均必要販売単価(円)": "{:,.0f}",
            "平均必要販売単価差分(円)": "{:+,.0f}",
            "平均必要賃率差(円/分)": "{:+.2f}",
            "平均必要賃率差分(円/分)": "{:+.2f}",
        },
        na_rep="-",
    )
    st.dataframe(styled, use_container_width=True)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        comparison_df.to_excel(writer, sheet_name="比較サマリ", index=False)
        meta_df = pd.DataFrame(
            {
                "生成日時": [now_str],
                "選択シナリオ": [", ".join(selected_scenarios)],
                "基準シナリオ": ["ベース"],
            }
        )
        meta_df.to_excel(writer, sheet_name="メタ情報", index=False)
    excel_buffer.seek(0)

    try:
        pdfmetrics.getFont("HeiseiMin-W3")
    except KeyError:
        pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))

    def _fmt(value: float, fmt: str) -> str:
        if value is None or not np.isfinite(value):
            return "-"
        return fmt.format(value)

    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=A4, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    styles["Heading1"].fontName = "HeiseiMin-W3"
    styles["Heading2"].fontName = "HeiseiMin-W3"
    styles["Normal"].fontName = "HeiseiMin-W3"

    story = [
        Paragraph("シナリオ比較レポート", styles["Heading1"]),
        Spacer(1, 12),
        Paragraph(f"生成日時: {now_str}", styles["Normal"]),
        Paragraph(f"基準シナリオ: ベース", styles["Normal"]),
        Paragraph(f"比較対象: {', '.join(selected_scenarios)}", styles["Normal"]),
        Spacer(1, 12),
    ]

    table_header = [
        "シナリオ",
        "調整サマリ",
        "必要賃率達成率(%)",
        "平均VA/分(円)",
        "日次付加価値(円)",
        "達成率差分(pts)",
        "VA/分差分(円)",
        "日次付加価値差分(円)",
        "平均必要販売単価(円)",
        "平均必要販売単価差分(円)",
        "平均必要賃率差(円/分)",
        "平均必要賃率差分(円/分)",
    ]
    table_rows = [table_header]
    for record in comparison_records:
        table_rows.append(
            [
                record["シナリオ"],
                record["調整サマリ"],
                _fmt(record["必要賃率達成率(%)"], "{:.1f}"),
                _fmt(record["平均VA/分(円)"], "{:.2f}"),
                _fmt(record["日次付加価値(円)"], "{:,.0f}"),
                _fmt(record["達成率差分(pts)"], "{:+.1f}"),
                _fmt(record["平均VA/分差分(円)"], "{:+.2f}"),
                _fmt(record["日次付加価値差分(円)"], "{:+,.0f}"),
                _fmt(record["平均必要販売単価(円)"], "{:,.0f}"),
                _fmt(record["平均必要販売単価差分(円)"], "{:+,.0f}"),
                _fmt(record["平均必要賃率差(円/分)"], "{:+.2f}"),
                _fmt(record["平均必要賃率差分(円/分)"], "{:+.2f}"),
            ]
        )

    table = Table(table_rows, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2F6776")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "HeiseiMin-W3"),
                ("FONTNAME", (0, 1), (-1, -1), "HeiseiMin-W3"),
                ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#F4F7FA")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D7E2EA")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 12))

    for record in comparison_records:
        if record["シナリオ"] == "ベース":
            continue
        parts = [
            f"達成率 {_fmt(record['達成率差分(pts)'], '{:+.1f}')}pt",
            f"平均VA {_fmt(record['平均VA/分差分(円)'], '{:+.2f}')}円",
            f"日次VA {_fmt(record['日次付加価値差分(円)'], '{:+,.0f}')}円",
        ]
        req_price_part = _fmt(record["平均必要販売単価差分(円)"], "{:+,.0f}")
        if req_price_part != "-":
            parts.append(f"必要販売単価 {req_price_part}円")
        gap_part = _fmt(record["平均必要賃率差分(円/分)"], "{:+.2f}")
        if gap_part != "-":
            parts.append(f"必要賃率差 {gap_part}円/分")
        summary_line = f"{record['シナリオ']}: " + " / ".join(parts)
        story.append(Paragraph(summary_line, styles["Normal"]))

    doc.build(story)
    pdf_buffer.seek(0)

    download_cols = st.columns(2)
    with download_cols[0]:
        st.download_button(
            "Excelエクスポート",
            data=excel_buffer.getvalue(),
            file_name="scenario_comparison.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with download_cols[1]:
        st.download_button(
            "PDFエクスポート",
            data=pdf_buffer.getvalue(),
            file_name="scenario_comparison.pdf",
            mime="application/pdf",
        )
else:
    st.info("比較対象のシナリオを選択してください。")

st.markdown("##### 📊 感度分析ハイライト")
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
mcol1.metric(
    "必要賃率達成率",
    f"{ach_rate:.1f}%" if np.isfinite(ach_rate) else "N/A",
    delta=f"{ach_delta:+.1f}pt" if np.isfinite(ach_delta) else "N/A",
)
mcol2.metric(
    "平均VA/分",
    f"{avg_vapm:.2f}円" if np.isfinite(avg_vapm) else "N/A",
    delta=f"{vapm_delta:+.2f}円" if np.isfinite(vapm_delta) else "N/A",
)
mcol3.metric(
    "日次付加価値",
    f"{sim_daily_va_total:,.0f}円" if np.isfinite(sim_daily_va_total) else "N/A",
    delta=f"{daily_delta:+,.0f}円" if np.isfinite(daily_delta) else "N/A",
)
annual_value = (
    f"{annual_sim_va / 10000:,.1f}万円" if np.isfinite(annual_sim_va) else "N/A"
)
annual_delta_text = (
    f"{annual_delta / 10000:+,.1f}万円" if np.isfinite(annual_delta) else "N/A"
)
mcol4.metric("年間利益見込", annual_value, delta=annual_delta_text)

gcol1, gcol2 = st.columns(2)
gcol1.metric(
    "平均必要販売単価",
    f"{avg_req_price:,.0f}円" if np.isfinite(avg_req_price) else "N/A",
    delta=(
        f"{req_price_delta_metric:+,.0f}円"
        if np.isfinite(req_price_delta_metric)
        else "N/A"
    ),
)
gcol2.metric(
    "平均必要賃率との差",
    f"{avg_gap:+.2f}円/分" if np.isfinite(avg_gap) else "N/A",
    delta=(
        f"{gap_delta_metric:+.2f}円/分"
        if np.isfinite(gap_delta_metric)
        else "N/A"
    ),
)

scenario_summary_text = _summarize_scenario_effect(
    ach_delta=ach_delta,
    vapm_delta=vapm_delta,
    daily_delta=daily_delta,
    annual_delta=annual_delta,
    req_price_delta=req_price_delta_metric,
    gap_delta=gap_delta_metric,
)
if scenario_summary_text:
    st.markdown(f"**KPI変化サマリ:** {scenario_summary_text}")

if active_label == "ベース" and not any([qp, qc, qv, qm]):
    st.caption("シミュレーション条件を変更すると年間インパクトの概算を表示します。")
else:
    st.info(
        f"フェルミ推定: {_format_fermi_estimate(daily_delta, working_days, active_label)}"
    )

driver_df, driver_messages = _analyze_driver_impacts(
    scenario_template,
    df_base,
    base_metrics_clean,
    price_pct=qp,
    ct_pct=qc,
    volume_pct=qv,
    material_pct=qm,
    be_rate=be_rate,
    req_rate=req_rate,
    delta_low=delta_low,
    delta_high=delta_high,
    working_days=working_days,
)

if driver_messages or not driver_df.empty:
    st.markdown("##### 🧮 感度分析サマリ")
    for msg in driver_messages:
        st.markdown(f"- {msg}")
    if not driver_df.empty:
        st.caption("各施策を単独で適用した場合の主要KPI差分です（他の変数はベース値を使用）。")
        driver_styled = driver_df.style.format(
            {
                "変化率(%)": "{:+.0f}",
                "日次付加価値差(円)": "{:+,.0f}",
                "年間利益差(万円)": "{:+,.1f}",
                "平均VA/分差(円)": "{:+.2f}",
                "平均必要賃率差(円/分)": "{:+.2f}",
                "平均必要販売単価差(円)": "{:+,.0f}",
            },
            na_rep="-",
        )
        st.dataframe(driver_styled, use_container_width=True)

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
                name: (
                    data.get("metrics", {}).get("ach_rate", np.nan),
                    data.get("metrics", {}).get("avg_vapm", np.nan),
                    data.get("df", pd.DataFrame()),
                )
                for name, data in scenario_results.items()
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

anomaly_all_df = detect_anomalies(df_view)
review_state = st.session_state.get("anomaly_review", {})
legacy_refreshed = False
for _key, record in list(review_state.items()):
    if not isinstance(record, dict):
        continue
    resolved = _normalize_review_classification(record)
    if resolved and record.get("classification") != resolved:
        record["classification"] = resolved
        legacy_refreshed = True
    if resolved and not record.get("classification_label"):
        record["classification_label"] = ANOMALY_REVIEW_LABELS.get(resolved)
        legacy_refreshed = True
if legacy_refreshed:
    st.session_state["anomaly_review"] = review_state
if not anomaly_all_df.empty:
    key_series = anomaly_all_df["product_no"].apply(_sku_to_str) + "::" + anomaly_all_df["metric"].astype(str)
    anomaly_all_df = anomaly_all_df.assign(key=key_series)
    anomaly_all_df["decision"] = anomaly_all_df["key"].map(
        lambda k: review_state.get(k, {}).get("decision")
    )
    anomaly_all_df["classification"] = anomaly_all_df["key"].map(
        lambda k: _normalize_review_classification(review_state.get(k, {}))
    )
    anomaly_all_df["classification_label"] = anomaly_all_df["classification"].map(
        lambda code: ANOMALY_REVIEW_LABELS.get(code)
    )
    anomaly_all_df["note"] = anomaly_all_df["key"].map(lambda k: review_state.get(k, {}).get("note"))
    anomaly_all_df["corrected_value"] = anomaly_all_df["key"].map(
        lambda k: review_state.get(k, {}).get("corrected_value")
    )
    anomaly_all_df["last_decided_at"] = anomaly_all_df["key"].map(
        lambda k: review_state.get(k, {}).get("timestamp")
    )
    anomaly_df = anomaly_all_df[anomaly_all_df["classification"] != "exception"].copy()
else:
    anomaly_df = anomaly_all_df

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
gap_df["va_per_min"] = pd.to_numeric(gap_df.get("va_per_min"), errors="coerce")
gap_df["gap"] = req_rate - gap_df["va_per_min"]
gap_df = gap_df[gap_df["gap"] > 0].copy()

empty_series = pd.Series(np.nan, index=gap_df.index, dtype="float64")
actual_price = pd.to_numeric(gap_df.get("actual_unit_price", empty_series), errors="coerce")
material_cost = pd.to_numeric(gap_df.get("material_unit_cost", empty_series), errors="coerce")
minutes_per_unit = pd.to_numeric(gap_df.get("minutes_per_unit", empty_series), errors="coerce")
daily_qty = pd.to_numeric(gap_df.get("daily_qty", empty_series), errors="coerce")
required_price = pd.to_numeric(gap_df.get("required_selling_price", empty_series), errors="coerce")
gp_per_unit = pd.to_numeric(gap_df.get("gp_per_unit", empty_series), errors="coerce")

gap_df["price_improve"] = (required_price - actual_price).clip(lower=0)
if req_rate:
    target_minutes = gp_per_unit / req_rate
else:
    target_minutes = pd.Series(np.nan, index=gap_df.index, dtype="float64")
gap_df["ct_improve"] = (minutes_per_unit - target_minutes).clip(lower=0)
gap_df["material_improve"] = (
    material_cost - (actual_price - req_rate * minutes_per_unit)
).clip(lower=0)

gap_df["price_improve"] = gap_df["price_improve"].fillna(0.0)
gap_df["ct_improve"] = gap_df["ct_improve"].fillna(0.0)
gap_df["material_improve"] = gap_df["material_improve"].fillna(0.0)

annual_days = float(base_params.get("working_days", DEFAULT_PARAMS["working_days"]))
default_monthly_days = max(1.0, round(annual_days / 12.0, 1))
priority_state = st.session_state.setdefault("priority_controls", {})
priority_state.setdefault("working_days_per_month", default_monthly_days)
priority_state.setdefault("price_cost", 200000.0)
priority_state.setdefault("ct_cost", 600000.0)
priority_state.setdefault("material_cost", 400000.0)
priority_state.setdefault("roi_limit", 3.0)
priority_state.setdefault("apply_roi_filter", False)
priority_state.setdefault("roi_priority_high", 2.0)
priority_state.setdefault("roi_priority_medium", 4.0)
priority_state.setdefault("investment_executable", 500000.0)
priority_state.setdefault(
    "action_filter",
    ["価格改善", "リードタイム改善", "材料改善"],
)

action_labels = {
    "price": "価格改善",
    "ct": "リードタイム改善",
    "material": "材料改善",
    "none": "推奨なし",
}
action_primary = [action_labels["price"], action_labels["ct"], action_labels["material"]]
action_all_options = action_primary + [action_labels["none"]]

with st.expander("優先順位付けロジック & フィルター設定", expanded=False):
    st.markdown(
        """
        **算出方法**
        - ギャップ（月次不足額）= (必要賃率 − 現状VA/分) × 分/個 × 日産数 × 稼働日数
        - 価格/材料改善の月次効果 = 単価差額 × 日産数 × 稼働日数
        - リードタイム改善の月次効果 = 改善分(分/個) × 日産数 × 稼働日数 × 必要賃率
        - 優先度スコア = 月次効果 ÷ 想定投資額（= 1か月あたりのROI）
        - 想定ROI(月) = 想定投資額 ÷ 月次効果
        """
    )
    conf_left, conf_right = st.columns(2)
    with conf_left:
        priority_state["working_days_per_month"] = st.number_input(
            "月あたり稼働日数",
            min_value=1.0,
            max_value=31.0,
            value=float(priority_state["working_days_per_month"]),
            step=1.0,
        )
        priority_state["price_cost"] = st.number_input(
            "価格改善の想定投資額 (円)",
            min_value=1.0,
            value=float(priority_state["price_cost"]),
            step=50000.0,
        )
        priority_state["ct_cost"] = st.number_input(
            "リードタイム改善の想定投資額 (円)",
            min_value=1.0,
            value=float(priority_state["ct_cost"]),
            step=50000.0,
        )
        priority_state["material_cost"] = st.number_input(
            "材料改善の想定投資額 (円)",
            min_value=1.0,
            value=float(priority_state["material_cost"]),
            step=50000.0,
        )
    with conf_right:
        priority_state["roi_limit"] = st.number_input(
            "ROI上限 (月)",
            min_value=0.5,
            value=float(priority_state["roi_limit"]),
            step=0.5,
            format="%.1f",
        )
        priority_state["apply_roi_filter"] = st.checkbox(
            "ROI上限で絞り込む",
            value=bool(priority_state["apply_roi_filter"]),
        )
        roi_high_input = st.number_input(
            "優先度『高』と判定するROI閾値 (月)",
            min_value=0.5,
            value=float(priority_state["roi_priority_high"]),
            step=0.5,
            format="%.1f",
        )
        priority_state["roi_priority_high"] = roi_high_input
        roi_medium_default = max(
            float(priority_state.get("roi_priority_medium", roi_high_input)),
            roi_high_input,
        )
        roi_medium_input = st.number_input(
            "優先度『中』の上限ROI閾値 (月)",
            min_value=roi_high_input,
            value=roi_medium_default,
            step=0.5,
            format="%.1f",
        )
        priority_state["roi_priority_medium"] = max(roi_medium_input, roi_high_input)
        priority_state["investment_executable"] = st.number_input(
            "即実行できる投資額の上限 (円)",
            min_value=0.0,
            value=float(priority_state["investment_executable"]),
            step=50000.0,
            format="%.0f",
        )
        default_actions = [
            opt
            for opt in priority_state.get("action_filter", action_primary)
            if opt in action_all_options
        ]
        if not default_actions:
            default_actions = action_primary
        priority_state["action_filter"] = st.multiselect(
            "表示する施策タイプ",
            options=action_all_options,
            default=default_actions,
        )
    st.markdown(
        f"- ROIが{priority_state['roi_priority_high']:.1f}ヶ月未満なら優先度『高』、"
        f"{priority_state['roi_priority_medium']:.1f}ヶ月までは『中』、それ以上は『低』と定義します。"
    )
    st.markdown(
        f"- 想定投資額が{priority_state['investment_executable']:,.0f}円以下なら『即実行可』、"
        "超える場合は『要投資検討』と表示します。"
    )
    st.caption(
        "投資額はSKUごとの月次効果に一律で適用します。ROI = 想定投資額 ÷ 月次効果。"
    )

working_days_per_month = float(priority_state.get("working_days_per_month", default_monthly_days))
price_cost = float(priority_state.get("price_cost", 1.0))
ct_cost = float(priority_state.get("ct_cost", 1.0))
material_cost = float(priority_state.get("material_cost", 1.0))
roi_priority_high = float(priority_state.get("roi_priority_high", 2.0))
roi_priority_medium = max(
    float(priority_state.get("roi_priority_medium", roi_priority_high * 2.0)),
    roi_priority_high,
)
investment_threshold = float(priority_state.get("investment_executable", 500000.0))
roi_limit = float(priority_state.get("roi_limit", 3.0))
raw_selected_actions = priority_state.get("action_filter", action_primary)
if not raw_selected_actions:
    selected_actions = action_all_options
else:
    selected_actions = [
        opt for opt in raw_selected_actions if opt in action_all_options
    ]
    if not selected_actions:
        selected_actions = action_all_options

gap_vals = pd.to_numeric(gap_df["gap"], errors="coerce").fillna(0.0)
minutes_vals = minutes_per_unit.fillna(0.0)
qty_vals = daily_qty.fillna(0.0)
price_vals = pd.to_numeric(gap_df["price_improve"], errors="coerce").fillna(0.0)
ct_vals = pd.to_numeric(gap_df["ct_improve"], errors="coerce").fillna(0.0)
material_vals = pd.to_numeric(gap_df["material_improve"], errors="coerce").fillna(0.0)

gap_df["monthly_shortfall_value"] = gap_vals * minutes_vals * qty_vals * working_days_per_month
gap_df["price_monthly_benefit"] = price_vals * qty_vals * working_days_per_month
gap_df["ct_monthly_benefit"] = ct_vals * req_rate * qty_vals * working_days_per_month
gap_df["material_monthly_benefit"] = material_vals * qty_vals * working_days_per_month

price_benefit = pd.to_numeric(gap_df["price_monthly_benefit"], errors="coerce")
ct_benefit = pd.to_numeric(gap_df["ct_monthly_benefit"], errors="coerce")
material_benefit = pd.to_numeric(gap_df["material_monthly_benefit"], errors="coerce")

gap_df["price_roi_months"] = np.where(price_benefit > 0, price_cost / price_benefit, np.nan)
gap_df["ct_roi_months"] = np.where(ct_benefit > 0, ct_cost / ct_benefit, np.nan)
gap_df["material_roi_months"] = np.where(
    material_benefit > 0, material_cost / material_benefit, np.nan
)

gap_df["price_score"] = np.where(price_benefit > 0, price_benefit / price_cost, 0.0)
gap_df["ct_score"] = np.where(ct_benefit > 0, ct_benefit / ct_cost, 0.0)
gap_df["material_score"] = np.where(
    material_benefit > 0, material_benefit / material_cost, 0.0
)

action_map = {
    "price": {
        "label": action_labels["price"],
        "score_col": "price_score",
        "roi_col": "price_roi_months",
        "benefit_col": "price_monthly_benefit",
        "cost": price_cost,
    },
    "ct": {
        "label": action_labels["ct"],
        "score_col": "ct_score",
        "roi_col": "ct_roi_months",
        "benefit_col": "ct_monthly_benefit",
        "cost": ct_cost,
    },
    "material": {
        "label": action_labels["material"],
        "score_col": "material_score",
        "roi_col": "material_roi_months",
        "benefit_col": "material_monthly_benefit",
        "cost": material_cost,
    },
}

score_cols = [meta["score_col"] for meta in action_map.values()]
score_df = gap_df[score_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
best_idx = score_df.idxmax(axis=1)
best_scores = score_df.max(axis=1)
gap_df["best_score"] = best_scores
gap_df["best_action_key"] = np.where(
    best_scores > 0, best_idx.str.replace("_score", "", regex=False), "none"
)
gap_df["best_action_label"] = gap_df["best_action_key"].map(
    {key: meta["label"] for key, meta in action_map.items()}
).fillna(action_labels["none"])
gap_df.loc[gap_df["best_action_key"] == "none", "best_action_label"] = action_labels["none"]

gap_df["best_roi_months"] = np.nan
gap_df["best_monthly_benefit"] = 0.0
gap_df["best_investment"] = np.nan
for key, meta in action_map.items():
    mask = gap_df["best_action_key"] == key
    gap_df.loc[mask, "best_roi_months"] = gap_df.loc[mask, meta["roi_col"]]
    gap_df.loc[mask, "best_monthly_benefit"] = gap_df.loc[mask, meta["benefit_col"]]
    gap_df.loc[mask, "best_investment"] = meta["cost"]
gap_df["best_roi_months"] = gap_df["best_roi_months"].replace([np.inf, -np.inf], np.nan)
gap_df["roi_months"] = gap_df["best_roi_months"]


def _classify_priority_rank(roi_value: Any) -> str:
    if roi_value is None or pd.isna(roi_value):
        return "情報不足"
    if roi_value <= roi_priority_high:
        return "高"
    if roi_value <= roi_priority_medium:
        return "中"
    return "低"


def _classify_execution(cost_value: Any) -> str:
    if cost_value is None or pd.isna(cost_value):
        return "投資額未設定"
    if cost_value <= investment_threshold:
        return "即実行可"
    return "要投資検討"


gap_df["priority_rank"] = gap_df["best_roi_months"].apply(_classify_priority_rank)
gap_df["execution_feasibility"] = gap_df["best_investment"].apply(_classify_execution)

filtered_gap_df = gap_df.copy()
if selected_actions:
    filtered_gap_df = filtered_gap_df[filtered_gap_df["best_action_label"].isin(selected_actions)]
if priority_state.get("apply_roi_filter", False):
    filtered_gap_df = filtered_gap_df[
        filtered_gap_df["best_roi_months"].notna() & (filtered_gap_df["best_roi_months"] <= roi_limit)
    ]

top_list = filtered_gap_df.sort_values(
    ["best_score", "best_monthly_benefit", "gap"], ascending=[False, False, False]
).head(20)
top_cards = top_list.head(5)

filter_summaries: List[str] = []
if priority_state.get("apply_roi_filter", False):
    filter_summaries.append(f"ROI≦{roi_limit:.1f}ヶ月")
if set(selected_actions) != set(action_primary):
    filter_summaries.append("施策タイプ: " + ", ".join(selected_actions))

category_summary = summarize_segment_performance(df_view, req_rate, "category")
customer_summary = summarize_segment_performance(df_view, req_rate, "major_customer")


def _render_target_badge(col, text: str) -> None:
    col.markdown(
        f"<div class='metric-badge'><span style='background-color:#E0EEF4;padding:4px 10px;border-radius:999px;font-size:0.8em;'>🎯{text}</span></div>",
        unsafe_allow_html=True,
    )


def _format_segment_prefix(segment: Any, label: str) -> str:
    """Return a natural language prefix for segment commentary."""

    seg = "未設定" if segment in [None, "", "nan"] else str(segment)
    if label == "カテゴリー":
        return f"カテゴリー『{seg}』"
    if label == "主要顧客":
        return f"主要顧客『{seg}』向け商品"
    return f"{seg}{label}"


def _compose_segment_insight(summary_df: pd.DataFrame, label: str) -> str:
    if summary_df is None or summary_df.empty:
        return f"{label}別のデータがありません。Excelに{label}列を追加してください。"

    df = summary_df.dropna(subset=["avg_va_per_min"]).copy()
    if df.empty:
        return f"{label}別の平均VA/分を計算できません。"

    tol = 0.05
    df = df.sort_values("avg_gap", ascending=False).reset_index(drop=True)
    best = df.iloc[0]
    diff_best = float(best.get("avg_gap", 0.0))
    abs_best = abs(diff_best)

    if abs_best <= tol:
        first = (
            f"{_format_segment_prefix(best['segment'], label)}は平均VA/分が{best['avg_va_per_min']:.1f}円で"
            f"必要賃率とほぼ同水準です（達成率{best['ach_rate_pct']:.1f}%）。"
        )
    elif diff_best > 0:
        first = (
            f"{_format_segment_prefix(best['segment'], label)}は平均VA/分が{best['avg_va_per_min']:.1f}円で"
            f"必要賃率を{abs_best:.1f}円上回っているため利益率が高い"
            f"（達成率{best['ach_rate_pct']:.1f}%）。"
        )
    else:
        first = (
            f"{_format_segment_prefix(best['segment'], label)}は平均VA/分が{best['avg_va_per_min']:.1f}円で"
            f"必要賃率を{abs_best:.1f}円下回っているため収益性に課題があります"
            f"（達成率{best['ach_rate_pct']:.1f}%）。"
        )

    if len(df) == 1:
        return first

    negatives = df[df["avg_gap"] < -tol]
    if not negatives.empty:
        worst = negatives.sort_values("avg_gap").iloc[0]
        diff_worst = float(abs(worst.get("avg_gap", 0.0)))
        second = (
            f"一方、{_format_segment_prefix(worst['segment'], label)}は平均VA/分が{worst['avg_va_per_min']:.1f}円で"
            f"必要賃率を{diff_worst:.1f}円下回っています"
        )
        roi = worst.get("avg_roi_months")
        if roi is not None and not pd.isna(roi):
            second += f"（未達SKUの平均ROI回復期間は{float(roi):.1f}ヶ月）"
        second += "。"
        return f"{first} {second}"

    if (df["avg_gap"] > tol).all():
        return f"{first} 全てのセグメントが必要賃率をクリアしています。"

    worst = df.sort_values("avg_gap").iloc[0]
    diff_worst = abs(float(worst.get("avg_gap", 0.0)))
    second = (
        f"他のセグメントも必要賃率との差は最大でも{diff_worst:.1f}円に収まっています。"
    )
    return f"{first} {second}"


def _build_segment_highlights(summary_df: pd.DataFrame, label: str) -> List[str]:
    """Create bullet style highlights for segment performance."""

    if summary_df is None or summary_df.empty:
        return []

    df = summary_df.dropna(subset=["avg_gap"]).copy()
    if df.empty:
        return []

    tol = 0.05
    highlights: List[str] = []

    positive = df[df["avg_gap"] > tol].sort_values("avg_gap", ascending=False)
    if not positive.empty:
        row = positive.iloc[0]
        gap_val = abs(float(row["avg_gap"]))
        highlights.append(
            f"{_format_segment_prefix(row['segment'], label)}は必要賃率を{gap_val:.1f}円上回り、達成率は{row['ach_rate_pct']:.1f}%です。"
        )

    negative = df[df["avg_gap"] < -tol].sort_values("avg_gap")
    if not negative.empty:
        row = negative.iloc[0]
        gap_val = abs(float(row["avg_gap"]))
        roi = row.get("avg_roi_months")
        roi_txt = ""
        if roi is not None and not pd.isna(roi):
            roi_txt = f"（未達SKUの平均ROI {float(roi):.1f}ヶ月）"
        highlights.append(
            f"{_format_segment_prefix(row['segment'], label)}は必要賃率を{gap_val:.1f}円下回っており改善余地があります{roi_txt}。"
        )

    if not highlights:
        highlights.append(f"{label}別では必要賃率との差が小さく概ね基準水準です。")

    return highlights


def _render_segment_tab(
    summary_df: pd.DataFrame, label: str, req_rate: float
) -> None:
    if summary_df is None or summary_df.empty:
        st.info(f"{label}情報が不足しています。Excelに{label}列を追加してください。")
        return

    chart_df = summary_df.copy()
    chart = (
        alt.Chart(chart_df)
        .mark_bar(color=PASTEL_ACCENT)
        .encode(
            x=alt.X("segment:N", sort="-y", title=label),
            y=alt.Y("avg_va_per_min:Q", title="平均VA/分 (円)"),
            tooltip=[
                alt.Tooltip("segment:N", title=label),
                alt.Tooltip("avg_va_per_min:Q", title="平均VA/分", format=".1f"),
                alt.Tooltip("ach_rate_pct:Q", title="達成率", format=".1f"),
                alt.Tooltip("avg_gap:Q", title="必要賃率差", format="+.1f"),
            ],
        )
        .properties(height=360)
    )
    rule_df = pd.DataFrame({"req_rate": [req_rate]})
    rule = alt.Chart(rule_df).mark_rule(color="#E07A5F", strokeDash=[6, 4]).encode(
        y="req_rate:Q"
    )
    st.altair_chart(chart + rule, use_container_width=True)

    display = summary_df.copy()
    display = display.rename(columns={"segment": label, "sku_count": "SKU数"})
    display["達成率"] = display["ach_rate_pct"].map(
        lambda x: f"{x:.1f}%" if pd.notna(x) else "-"
    )
    display["平均VA/分"] = display["avg_va_per_min"].map(
        lambda x: f"{x:.1f}" if pd.notna(x) else "-"
    )
    display["必要賃率差"] = display["avg_gap"].map(
        lambda x: f"{x:+.1f}" if pd.notna(x) else "-"
    )
    display["平均ROI(月)"] = display["avg_roi_months"].map(
        lambda x: "-" if pd.isna(x) else f"{x:.1f}"
    )
    display = display[
        [label, "SKU数", "達成率", "平均VA/分", "必要賃率差", "平均ROI(月)"]
    ]
    st.dataframe(display, use_container_width=True)
    st.caption("※ 平均ROI(月)は未達SKUのみを対象としたギャップ解消の目安です。")
    st.info(_compose_segment_insight(summary_df, label))


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

kpi_records: List[Dict[str, Any]] = []
for scen_name in selected_scenarios:
    scen_data = scenario_results.get(scen_name)
    if not scen_data:
        continue
    metrics = scen_data.get("metrics", {})
    adjustments = scen_data.get("adjustments", {})
    display_name = scen_name
    if scen_name != "ベース":
        display_name = f"{scen_name} ({_format_adjustment_summary(adjustments)})"
    kpi_records.append(
        {
            "scenario": scen_name,
            "display": display_name,
            "KPI": "必要賃率達成SKU比率",
            "value": metrics.get("ach_rate", np.nan),
        }
    )
    kpi_records.append(
        {
            "scenario": scen_name,
            "display": display_name,
            "KPI": "平均 付加価値/分",
            "value": metrics.get("avg_vapm", np.nan),
        }
    )
kpi_df = pd.DataFrame(kpi_records)
if kpi_df.empty:
    st.info("比較対象のシナリオを選択してください。")
else:
    fig_kpi = px.bar(
        kpi_df,
        x="KPI",
        y="value",
        color="display",
        barmode="group",
        color_discrete_sequence=PASTEL_PALETTE,
    )
    fig_kpi.update_traces(opacity=0.85)
    fig_kpi.update_yaxes(gridcolor="#D7E2EA")
    fig_kpi.update_xaxes(gridcolor="#D7E2EA")
    fig_kpi.update_layout(legend_title_text="シナリオ")
    fig_kpi = _apply_plotly_theme(fig_kpi, legend_bottom=True)
    st.plotly_chart(fig_kpi, use_container_width=True, config=_build_plotly_config())

ai_insights = {
    "top_underperformers": top_list[
        ["product_name", "gap", "roi_months", "best_action_label", "best_monthly_benefit"]
    ].head(3).to_dict("records")
    if not top_list.empty
    else [],
    "anomaly_summary": anomaly_summary_stats.to_dict("records"),
    "anomaly_records": anomaly_df.sort_values("severity", ascending=False).head(5).to_dict("records")
    if not anomaly_df.empty
    else [],
    "dq_summary": {"missing": miss_count, "negative": out_count, "duplicate": dup_count},
    "segment_category": category_summary.head(5).to_dict("records"),
    "segment_customer": customer_summary.head(5).to_dict("records"),
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
    if anomaly_all_df.empty:
        st.success("統計的な異常値は検出されませんでした。")
    else:
        st.info("検出された異常値はすべて『例外的な値』として除外済みです。必要に応じて下部のレビューから再評価できます。")
else:
    highlight = anomaly_df.sort_values("severity", ascending=False).head(3)
    if not highlight.empty:
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

    detail_source = (
        anomaly_df.sort_values("severity", ascending=False)
        .head(20)
        .drop(
            columns=[
                "key",
                "decision",
                "classification",
                "classification_label",
                "note",
                "corrected_value",
                "last_decided_at",
            ],
            errors="ignore",
        )
    )
    detail_df = detail_source.rename(
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

if not anomaly_all_df.empty:
    with st.expander("異常値レビュー / 処置", expanded=not anomaly_df.empty):
        review_candidates = anomaly_all_df.sort_values("severity", ascending=False).head(20)
        if review_candidates.empty:
            st.caption("現在レビュー対象の異常値はありません。")
        else:
            records = review_candidates.to_dict("records")
            classification_labels = [choice["label"] for choice in ANOMALY_REVIEW_CHOICES]
            label_to_key = {choice["label"]: choice["key"] for choice in ANOMALY_REVIEW_CHOICES}
            options = [ANOMALY_REVIEW_UNSET_LABEL] + classification_labels
            decisions: List[Dict[str, Any]] = []
            with st.form("anomaly_review_form"):
                for idx, row in enumerate(records):
                    key = row["key"]
                    metric_label = METRIC_LABELS.get(row.get("metric"), row.get("metric"))
                    product_no = row.get("product_no")
                    product_no_display = _sku_to_str(product_no)
                    product_name = row.get("product_name") or "不明"
                    value = row.get("value")
                    median_val = row.get("median")
                    ratio = None
                    if (
                        value is not None
                        and median_val is not None
                        and not pd.isna(value)
                        and not pd.isna(median_val)
                        and median_val != 0
                    ):
                        ratio = float(value) / float(median_val)
                    row["ratio"] = ratio
                    ratio_txt = f"{ratio:.1f}倍" if ratio is not None else "中央値情報なし"
                    severity = row.get("severity")
                    severity_txt = "N/A" if severity is None or pd.isna(severity) else f"{float(severity):.1f}"
                    direction = row.get("direction")
                    direction_txt = "上振れ" if direction == "high" else "下振れ"
                    median_txt = _format_number(median_val)
                    value_txt = _format_number(value)
                    question = (
                        f"製品番号{product_no_display}（{product_name}）の{metric_label}が"
                        f"{median_txt}に対して{value_txt}（{ratio_txt}）です。"
                    )
                    row["question"] = question
                    st.markdown(f"**{product_no_display}｜{product_name}**")
                    st.caption(
                        f"{metric_label}: 現在値 {value_txt} / 中央値 {median_txt}｜{direction_txt}｜Z≈{severity_txt}"
                    )
                    st.caption(question)
                    existing = review_state.get(key, {})
                    existing_classification = _normalize_review_classification(existing)
                    default_index = 0
                    if existing_classification:
                        existing_label = ANOMALY_REVIEW_LABELS.get(existing_classification)
                        if existing_label and existing_label in classification_labels:
                            default_index = classification_labels.index(existing_label) + 1
                    choice_label = st.radio(
                        "分類を選択してください。",
                        options=options,
                        index=default_index,
                        horizontal=True,
                        key=f"decision_{key}",
                    )
                    if choice_label == ANOMALY_REVIEW_UNSET_LABEL:
                        classification_key = None
                    else:
                        classification_key = label_to_key.get(choice_label)
                    if classification_key:
                        desc = ANOMALY_REVIEW_DESCRIPTIONS.get(classification_key)
                        if desc:
                            st.caption(f"分類メモ: {desc}")
                    corrected_value = None
                    if classification_key == "input_error":
                        default_value = existing.get("corrected_value", value)
                        if default_value is None or pd.isna(default_value):
                            default_value = (
                                median_val if median_val is not None and not pd.isna(median_val) else 0.0
                            )
                        step = max(abs(float(default_value)) * 0.01, 0.01)
                        corrected_value = st.number_input(
                            f"訂正後の値（{metric_label}） - {product_no_display}",
                            value=float(default_value),
                            step=float(step),
                            format="%.3f",
                            key=f"corrected_{key}",
                        )
                    note = st.text_input(
                        f"メモ（任意） - {product_no_display}",
                        value=existing.get("note", ""),
                        key=f"note_{key}",
                    )
                    decisions.append(
                        {
                            "key": key,
                            "classification": classification_key,
                            "corrected_value": corrected_value,
                            "note": note,
                            "row": row,
                        }
                    )
                    if idx < len(records) - 1:
                        st.markdown("---")
                submitted = st.form_submit_button("レビュー結果を保存")

            if submitted:
                review_map = dict(review_state)
                df_full = st.session_state["df_products_raw"].copy()
                product_no_keys = (
                    df_full["product_no"].apply(_sku_to_str)
                    if "product_no" in df_full.columns
                    else None
                )
                dataset_changed = False
                decision_changed = False

                def _revert_previous(record: Dict[str, Any]) -> bool:
                    if (
                        not record
                        or _normalize_review_classification(record) != "input_error"
                        or record.get("original_value") is None
                        or product_no_keys is None
                    ):
                        return False
                    metric_prev = record.get("metric")
                    if metric_prev not in df_full.columns:
                        return False
                    sku_prev = _sku_to_str(record.get("product_no"))
                    mask_prev = product_no_keys == sku_prev
                    if mask_prev.any():
                        df_full.loc[mask_prev, metric_prev] = record.get("original_value")
                        return True
                    return False

                for entry in decisions:
                    key = entry["key"]
                    classification = entry["classification"]
                    corrected_value = entry.get("corrected_value")
                    note = entry.get("note")
                    row = entry.get("row", {})
                    existing = review_state.get(key)
                    if classification is None:
                        if key in review_map:
                            if _revert_previous(existing):
                                dataset_changed = True
                            del review_map[key]
                            decision_changed = True
                        continue

                    record = {
                        "product_no": row.get("product_no"),
                        "product_no_display": _sku_to_str(row.get("product_no")),
                        "product_name": row.get("product_name"),
                        "metric": row.get("metric"),
                        "metric_label": METRIC_LABELS.get(row.get("metric"), row.get("metric")),
                        "median": None if pd.isna(row.get("median")) else float(row.get("median")),
                        "severity": None if pd.isna(row.get("severity")) else float(row.get("severity")),
                        "direction": row.get("direction"),
                        "note": note,
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "question": row.get("question"),
                    }
                    ratio_val = row.get("ratio")
                    if ratio_val is not None and not pd.isna(ratio_val):
                        record["ratio_vs_median"] = float(ratio_val)
                    if (
                        existing
                        and existing.get("decision") == "corrected"
                        and existing.get("original_value") is not None
                    ):
                        record["original_value"] = existing.get("original_value")
                    else:
                        value_now = row.get("value")
                        record["original_value"] = (
                            None if pd.isna(value_now) else float(value_now)
                        )

                    record["classification"] = classification
                    record["classification_label"] = ANOMALY_REVIEW_LABELS.get(classification)
                    if classification == "exception":
                        record["decision"] = "exception"
                        if _revert_previous(existing):
                            dataset_changed = True
                        review_map[key] = record
                        decision_changed = True
                        continue
                    if classification == "monitor":
                        record["decision"] = "monitor"
                        if _revert_previous(existing):
                            dataset_changed = True
                        review_map[key] = record
                        decision_changed = True
                        continue
                    if classification != "input_error":
                        record["decision"] = classification or ""
                        review_map[key] = record
                        decision_changed = True
                        continue

                    if corrected_value is None:
                        continue
                    corrected_numeric = float(corrected_value)
                    record["decision"] = "corrected"
                    record["corrected_value"] = corrected_numeric
                    metric = row.get("metric")
                    if product_no_keys is not None and metric in df_full.columns:
                        sku_key = _sku_to_str(row.get("product_no"))
                        mask = product_no_keys == sku_key
                        if mask.any():
                            df_full.loc[mask, metric] = corrected_numeric
                            dataset_changed = True
                    review_map[key] = record
                    decision_changed = True

                if decision_changed:
                    st.session_state["anomaly_review"] = review_map
                    if dataset_changed:
                        st.session_state["df_products_raw"] = df_full
                    st.success("レビュー結果を保存しました。")
                    st.rerun()
                else:
                    st.info("変更はありませんでした。")

history_state = st.session_state.get("anomaly_review", {})
if history_state:
    history_records = []
    for key, info in history_state.items():
        classification_code = _normalize_review_classification(info)
        classification_label = info.get("classification_label") or ANOMALY_REVIEW_LABELS.get(
            classification_code, "-"
        )
        history_records.append(
            {
                "decision": info.get("decision"),
                "classification": classification_code,
                "分類": classification_label,
                "製品番号": info.get("product_no_display")
                or _sku_to_str(info.get("product_no")),
                "製品名": info.get("product_name"),
                "指標": info.get("metric_label")
                or METRIC_LABELS.get(info.get("metric"), info.get("metric")),
                "元の値": info.get("original_value"),
                "訂正値": info.get("corrected_value"),
                "中央値": info.get("median"),
                "中央値比": info.get("ratio_vs_median"),
                "逸脱度": info.get("severity"),
                "判定日時": info.get("timestamp"),
                "メモ": info.get("note"),
            }
        )
    history_df = pd.DataFrame(history_records)
    if not history_df.empty:
        history_df = history_df.sort_values("判定日時", ascending=False)

        def _prepare_history(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            df = df.copy()
            if "classification" in df.columns:
                df = df.drop(columns=["classification"])
            df["中央値比"] = df["中央値比"].apply(
                lambda v: f"{float(v):.2f}倍" if v is not None and not pd.isna(v) else "-"
            )
            df["元の値"] = df["元の値"].apply(_format_number)
            df["訂正値"] = df["訂正値"].apply(
                lambda v: "-" if v is None or pd.isna(v) else _format_number(v)
            )
            df["中央値"] = df["中央値"].apply(_format_number)
            df["逸脱度"] = df["逸脱度"].apply(
                lambda v: "-" if v is None or pd.isna(v) else f"{float(v):.1f}"
            )
            return df

        exceptions_history = history_df[history_df["classification"] == "exception"].copy()
        corrections_history = history_df[history_df["classification"] == "input_error"].copy()
        monitor_history = history_df[history_df["classification"] == "monitor"].copy()

        if not exceptions_history.empty:
            cols = [
                "分類",
                "製品番号",
                "製品名",
                "指標",
                "元の値",
                "中央値",
                "中央値比",
                "逸脱度",
                "判定日時",
                "メモ",
            ]
            with st.expander("例外として扱う異常値", expanded=False):
                st.dataframe(_prepare_history(exceptions_history)[cols], use_container_width=True)

        if not corrections_history.empty:
            cols = [
                "分類",
                "製品番号",
                "製品名",
                "指標",
                "元の値",
                "訂正値",
                "中央値",
                "中央値比",
                "逸脱度",
                "判定日時",
                "メモ",
            ]
            with st.expander("訂正済みの異常値", expanded=False):
                st.dataframe(_prepare_history(corrections_history)[cols], use_container_width=True)

        if not monitor_history.empty:
            cols = [
                "分類",
                "製品番号",
                "製品名",
                "指標",
                "元の値",
                "中央値",
                "中央値比",
                "逸脱度",
                "判定日時",
                "メモ",
            ]
            with st.expander("要調査として記録した異常値", expanded=False):
                st.dataframe(_prepare_history(monitor_history)[cols], use_container_width=True)

st.divider()

# Actionable SKU Top List
st.subheader("要対策SKUトップリスト")
st.caption(
    "ギャップ = 必要賃率 - 付加価値/分。優先度は推定月次効果 ÷ 想定投資額で算出しています。\n"
    f"優先度ランク: ROI≦{roi_priority_high:.1f}ヶ月は『高』、ROI≦{roi_priority_medium:.1f}ヶ月までは『中』、それ以上は『低』。\n"
    f"実行可否: 想定投資額≦{investment_threshold:,.0f}円なら『即実行可』と判定します。"
)
if filter_summaries:
    st.caption("適用中フィルター: " + " / ".join(filter_summaries))
top5 = top_cards
if len(top5) > 0:
    card_cols = st.columns(len(top5))
    for col, row in zip(card_cols, top5.to_dict("records")):
        roi_txt = _format_roi(row.get("best_roi_months"))
        gap_val = row.get("gap")
        gap_txt = "N/A" if pd.isna(gap_val) else f"{float(gap_val):.2f}円/分"
        action_label = row.get("best_action_label") or "推奨なし"
        if action_label == "推奨なし":
            delta_label = "推奨施策なし"
        else:
            delta_label = f"{action_label}｜ROI {roi_txt}月"
        col.metric(row.get("product_name", "不明"), gap_txt, delta=delta_label)
        badge_parts: List[str] = []
        priority_label = row.get("priority_rank")
        execution_label = row.get("execution_feasibility")
        if priority_label and isinstance(priority_label, str):
            badge_parts.append(f"優先度:{priority_label}")
        if execution_label and isinstance(execution_label, str):
            badge_parts.append(execution_label)
        if badge_parts:
            _render_target_badge(col, " / ".join(badge_parts))

        price_val = row.get("price_improve")
        ct_val = row.get("ct_improve")
        material_val = row.get("material_improve")
        price_txt = (
            f"価格+{float(price_val):,.0f}円"
            if price_val is not None and not pd.isna(price_val) and float(price_val) > 0
            else "価格改善情報なし"
        )
        ct_txt = (
            f"CT-{float(ct_val):.2f}分"
            if ct_val is not None and not pd.isna(ct_val) and float(ct_val) > 0
            else "CT改善情報なし"
        )
        material_txt = (
            f"材料-{float(material_val):,.0f}円"
            if material_val is not None and not pd.isna(material_val) and float(material_val) > 0
            else "材料改善情報なし"
        )
        benefit_txt = _format_currency(row.get("best_monthly_benefit"))
        col.caption(f"{' / '.join([price_txt, ct_txt, material_txt])}｜月次効果 ≈ {benefit_txt}")

    rename_map = {
        "product_no": "製品番号",
        "product_name": "製品名",
        "best_action_label": "推奨施策",
        "gap": "ギャップ(円/分)",
        "monthly_shortfall_value": "不足額/月(円)",
        "price_improve": "価格改善(円/個)",
        "ct_improve": "CT改善(分/個)",
        "material_improve": "材料改善(円/個)",
        "best_monthly_benefit": "推定月次効果(円)",
        "best_investment": "想定投資額(円)",
        "best_roi_months": "想定ROI(月)",
        "best_score": "優先度スコア(1/月)",
        "priority_rank": "優先度ランク",
        "execution_feasibility": "実行可否",
    }
    columns = [
        "product_no",
        "product_name",
        "best_action_label",
        "gap",
        "monthly_shortfall_value",
        "price_improve",
        "ct_improve",
        "material_improve",
        "best_monthly_benefit",
        "best_investment",
        "best_roi_months",
        "priority_rank",
        "execution_feasibility",
        "best_score",
    ]
    table = top_list[columns].copy().rename(columns=rename_map)
    numeric_columns = [
        "ギャップ(円/分)",
        "不足額/月(円)",
        "価格改善(円/個)",
        "CT改善(分/個)",
        "材料改善(円/個)",
        "推定月次効果(円)",
        "想定投資額(円)",
        "想定ROI(月)",
        "優先度スコア(1/月)",
    ]
    table[numeric_columns] = table[numeric_columns].apply(pd.to_numeric, errors="coerce")
    table.insert(0, "選択", False)
    column_config = {
        "選択": st.column_config.CheckboxColumn("選択", help="シナリオに転送するSKUを選択"),
        "製品番号": st.column_config.TextColumn("製品番号"),
        "製品名": st.column_config.TextColumn("製品名"),
        "推奨施策": st.column_config.TextColumn("推奨施策"),
        "ギャップ(円/分)": st.column_config.NumberColumn("ギャップ(円/分)", format="%.2f"),
        "不足額/月(円)": st.column_config.NumberColumn(
            "不足額/月(円)",
            format="%.0f",
            help="(必要賃率−現状VA/分)×分/個×日産数×稼働日数",
        ),
        "価格改善(円/個)": st.column_config.NumberColumn(
            "価格改善(円/個)",
            format="%.0f",
            help="必要販売単価 − 現在の販売単価",
        ),
        "CT改善(分/個)": st.column_config.NumberColumn(
            "CT改善(分/個)",
            format="%.2f",
            help="現状分/個 − 達成に必要な分/個",
        ),
        "材料改善(円/個)": st.column_config.NumberColumn(
            "材料改善(円/個)",
            format="%.0f",
            help="現状材料費 − 目標材料費",
        ),
        "推定月次効果(円)": st.column_config.NumberColumn(
            "推定月次効果(円)",
            format="%.0f",
            help="推奨施策を実行した場合の月次インパクト",
        ),
        "想定投資額(円)": st.column_config.NumberColumn(
            "想定投資額(円)",
            format="%.0f",
            help="設定した施策別の想定投資額",
        ),
        "想定ROI(月)": st.column_config.NumberColumn(
            "想定ROI(月)",
            format="%.1f",
            help="想定投資額 ÷ 推定月次効果",
        ),
        "優先度ランク": st.column_config.TextColumn(
            "優先度ランク",
            help=(
                f"ROI≦{roi_priority_high:.1f}ヶ月は『高』、ROI≦{roi_priority_medium:.1f}ヶ月までは『中』、"
                "それ以上は『低』として判定"
            ),
        ),
        "実行可否": st.column_config.TextColumn(
            "実行可否",
            help=(
                f"想定投資額≦{investment_threshold:,.0f}円で『即実行可』、超える場合は『要投資検討』。"
                "投資額が未設定の施策は『投資額未設定』。"
            ),
        ),
        "優先度スコア(1/月)": st.column_config.NumberColumn(
            "優先度スコア(1/月)",
            format="%.2f",
            help="推定月次効果 ÷ 想定投資額。1.0で1か月回収",
        ),
    }
    edited = st.data_editor(
        table,
        use_container_width=True,
        key="action_sku_editor",
        column_config=column_config,
        hide_index=True,
    )
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
elif gap_df.empty:
    st.info("要対策SKUはありません。")
else:
    st.info("設定した条件に合致する要対策SKUはありません。")

st.subheader("セグメント分析（カテゴリー/顧客）")
st.caption("平均VA/分と必要賃率との差、達成率、ROIをセグメント単位で比較します。")

insight_sections = [
    ("カテゴリー", _build_segment_highlights(category_summary, "カテゴリー")),
    ("主要顧客", _build_segment_highlights(customer_summary, "主要顧客")),
]
for section_label, bullets in insight_sections:
    if not bullets:
        continue
    st.markdown(f"**{section_label}の注目ポイント**")
    st.markdown("\n".join(f"- {line}" for line in bullets))

segment_tabs = st.tabs(["カテゴリー別", "主要顧客別"])
with segment_tabs[0]:
    _render_segment_tab(category_summary, "カテゴリー", req_rate)
with segment_tabs[1]:
    _render_segment_tab(customer_summary, "主要顧客", req_rate)

tabs = st.tabs(["全体分布（散布図）", "時系列", "達成状況（棒/円）", "未達SKU（パレート）", "SKUテーブル", "付加価値/分分布"])

with tabs[0]:
    st.caption(
        "横軸=分/個（製造リードタイム）, 縦軸=付加価値/分。必要賃率×δ帯と損益分岐賃率を表示。"
    )
    scatter_frames: List[pd.DataFrame] = []
    for scen_name in selected_scenarios:
        scen_data = scenario_results.get(scen_name)
        if not scen_data:
            continue
        scen_df = scen_data.get("df")
        if scen_df is None or scen_df.empty:
            continue
        scen_copy = scen_df.copy()
        scen_copy["scenario"] = scen_name
        scatter_frames.append(scen_copy)
    if not scatter_frames:
        st.info("表示可能なシナリオがありません。")
    else:
        scatter_df = pd.concat(scatter_frames, ignore_index=True)
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
                pdca_df = _build_pdca_summary(plot_df)
                yoy_lines = _build_yoy_summary(
                    trend_df,
                    sorted(plot_df["scenario"].unique()),
                )
                if yoy_lines:
                    st.markdown("**前年同月比**")
                    st.markdown("\n".join(f"- {line}" for line in yoy_lines))
                if not pdca_df.empty:
                    latest_records = (
                        pdca_df.sort_values("period")
                        .groupby("scenario", as_index=False)
                        .tail(1)
                        .sort_values("scenario")
                    )
                    if not latest_records.empty:
                        st.markdown("**最新PDCAステータス**")
                        st.markdown(
                            "\n".join(
                                f"- {row['scenario']}（{_format_period_label(row['period'], freq_choice)}）: {row['pdca_comment']}"
                                for _, row in latest_records.iterrows()
                            )
                        )
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
                display_df["期間"] = display_df["period"].map(
                    lambda v: _format_period_label(v, freq_choice)
                )
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

                if not pdca_df.empty:
                    display_pdca = pdca_df.copy()
                    display_pdca["期間"] = display_pdca["period"].map(
                        lambda v: _format_period_label(v, freq_choice)
                    )
                    display_pdca["P(必要賃率)"] = display_pdca["required_rate"].map(
                        lambda x: f"{x:.3f}" if pd.notna(x) else "-"
                    )
                    display_pdca["D(VA/分)"] = display_pdca["va_per_min"].map(
                        lambda x: f"{x:.2f}" if pd.notna(x) else "-"
                    )
                    display_pdca["C(達成率)"] = display_pdca["ach_rate"].map(
                        lambda x: f"{x:.1f}%" if pd.notna(x) else "-"
                    )
                    display_pdca["ΔVA/分"] = display_pdca["delta_va"].map(
                        lambda x: f"{x:+.2f}" if pd.notna(x) else "-"
                    )
                    display_pdca["Δ達成率"] = display_pdca["delta_ach"].map(
                        lambda x: f"{x:+.1f}pt" if pd.notna(x) else "-"
                    )
                    pdca_display_cols = [
                        "シナリオ",
                        "期間",
                        "P(必要賃率)",
                        "D(VA/分)",
                        "C(達成率)",
                        "ΔVA/分",
                        "Δ達成率",
                        "PDCAコメント",
                    ]
                    display_pdca = display_pdca.rename(
                        columns={"scenario": "シナリオ", "pdca_comment": "PDCAコメント"}
                    )[pdca_display_cols]
                    st.markdown("**PDCAチェックリスト**")
                    st.dataframe(display_pdca, use_container_width=True)

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
        "category": "カテゴリー",
        "major_customer": "主要顧客",
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
        "製品番号","製品名","カテゴリー","主要顧客","実際売単価","必要販売単価","必要販売単価差額","材料原価","粗利/個",
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

sync_offline_cache()
