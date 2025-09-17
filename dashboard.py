"""ダッシュボード描画と各種ビジュアル生成を担うモジュール."""
from __future__ import annotations

import io
import os
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from openai import OpenAI

from calculations import OperationalMetrics, estimate_roi_table


PALETTE = ["#2F4F6E", "#4F6D7A", "#839AA8", "#D9E2EC", "#102A43"]


def _format_metric(value: float, unit: str = "") -> str:
    """数値を読みやすい文字列へ整形するヘルパー."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if unit == "円/分":
        return f"{value:,.2f}"
    if unit == "円":
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def scenario_summary_table(scenarios: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """シナリオごとの主要KPIを表形式で整形する."""

    rows: List[Dict[str, Any]] = []
    for name, payload in scenarios.items():
        df: pd.DataFrame = payload.get("products", pd.DataFrame())
        metrics: OperationalMetrics = payload.get("metrics")
        if df.empty or metrics is None:
            continue
        meets_ratio = float(df.get("meets_required_rate", pd.Series(dtype=float)).mean()) if "meets_required_rate" in df else np.nan
        avg_va = float(df.get("va_per_min", pd.Series(dtype=float)).mean()) if "va_per_min" in df else np.nan
        avg_gap = float(df.get("rate_gap_vs_required", pd.Series(dtype=float)).mean()) if "rate_gap_vs_required" in df else np.nan
        rows.append(
            {
                "シナリオ": name,
                "必要負担率(円/分)": metrics.required_rate,
                "損益分岐負担率(円/分)": metrics.break_even_rate,
                "付加価値/分 平均": avg_va,
                "必要負担率との差": avg_gap,
                "達成SKU比率": meets_ratio,
            }
        )
    return pd.DataFrame(rows)


def prepare_scatter_source(scenarios: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """散布図用にシナリオを縦結合し `scenario` 列を付与する."""

    frames = []
    for name, payload in scenarios.items():
        df = payload.get("products")
        if df is None or df.empty:
            continue
        tmp = df.copy()
        tmp["scenario"] = name
        frames.append(tmp)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def scatter_plot(df: pd.DataFrame, required_rate: float, translator) -> go.Figure:
    """加工時間×付加価値の散布図を生成する."""

    if df.empty:
        return go.Figure()
    fig = px.scatter(
        df,
        x="minutes_per_unit",
        y="va_per_min",
        color="scenario",
        hover_data=["product_no", "product_name", "rate_class", "rate_gap_vs_required"],
        color_discrete_sequence=PALETTE,
        labels={
            "minutes_per_unit": translator("scatter_x"),
            "va_per_min": translator("scatter_y"),
        },
    )
    fig.add_hline(
        y=required_rate,
        line_dash="dash",
        line_color="#F25C54",
        annotation_text=translator("kpi_required_rate"),
    )
    fig.update_layout(
        title=translator("scatter_title"),
        paper_bgcolor="#F7F9FB",
        plot_bgcolor="#FFFFFF",
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
    )
    return fig


def scenario_bar_chart(summary: pd.DataFrame, translator) -> go.Figure:
    """必要負担率差異や平均付加価値を棒グラフで比較する."""

    if summary.empty:
        return go.Figure()
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(translator("bar_gap_title"), translator("bar_va_title")),
    )
    fig.add_trace(
        go.Bar(
            x=summary["シナリオ"],
            y=summary["必要負担率との差"],
            marker_color=PALETTE[0],
            name=translator("bar_gap_title"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=summary["シナリオ"],
            y=summary["付加価値/分 平均"],
            marker_color=PALETTE[2],
            name=translator("bar_va_title"),
        ),
        row=1,
        col=2,
    )
    fig.update_layout(height=400, paper_bgcolor="#F7F9FB", plot_bgcolor="#FFFFFF")
    return fig


def distribution_histogram(df: pd.DataFrame, translator) -> go.Figure:
    """付加価値/分 と 加工時間 の分布ヒストグラムを生成する."""

    if df.empty:
        return go.Figure()
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(translator("hist_va_title"), translator("hist_mpu_title")),
    )
    vap = pd.to_numeric(df.get("va_per_min"), errors="coerce")
    mpu = pd.to_numeric(df.get("minutes_per_unit"), errors="coerce")
    fig.add_trace(
        go.Histogram(
            x=vap.dropna(),
            nbinsx=25,
            name=translator("hist_va_title"),
            marker_color=PALETTE[1],
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=mpu.dropna(),
            nbinsx=25,
            name=translator("hist_mpu_title"),
            marker_color=PALETTE[3],
        ),
        row=1,
        col=2,
    )
    fig.update_layout(barmode="overlay", paper_bgcolor="#F7F9FB", plot_bgcolor="#FFFFFF")
    fig.update_traces(opacity=0.75)
    return fig


def generate_ai_commentary(df: pd.DataFrame, metrics: OperationalMetrics) -> str:
    """OpenAI API を呼び出しコンサルタント風のコメントを生成する."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OpenAI API キーが設定されていません。環境変数 OPENAI_API_KEY を登録してください。"

    client = OpenAI(api_key=api_key)
    top_products = df.sort_values(by="rate_gap_vs_required").head(5)
    prompt = (
        "あなたは中小製造業の経営コンサルタントです。"
        "以下の指標から経営者向けの提言を日本語でまとめてください。"
        "- 必要負担率: {required_rate:.2f} 円/分\n"
        "- 損益分岐負担率: {break_even_rate:.2f} 円/分\n"
        "- 正味直接工員数: {net_workers:.2f} 人\n"
        "- 上位課題SKU: {top_list}\n"
        "3段落で現状評価→原因→打ち手を述べ、最後に優先アクションを3つ箇条書きしてください。"
    ).format(
        required_rate=metrics.required_rate,
        break_even_rate=metrics.break_even_rate,
        net_workers=metrics.net_workers,
        top_list=top_products[["product_name", "rate_gap_vs_required"]].to_dict("records"),
    )
    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            max_output_tokens=400,
            temperature=0.3,
        )
        return response.output_text
    except Exception as exc:
        return f"AI コメント生成中にエラーが発生しました: {exc}"


def build_pdf_report(name: str, df: pd.DataFrame, metrics: OperationalMetrics) -> bytes:
    """ReportLab を用いて PDF レポートを生成する."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 20 * mm
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, height - margin, "標準賦率ダッシュボード レポート")
    c.setFont("Helvetica", 10)
    c.drawString(margin, height - margin - 20, f"シナリオ: {name}")
    c.drawString(margin, height - margin - 35, f"作成日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    metrics_y = height - margin - 60
    c.drawString(margin, metrics_y, f"必要負担率: {metrics.required_rate:,.2f} 円/分")
    c.drawString(margin, metrics_y - 15, f"損益分岐負担率: {metrics.break_even_rate:,.2f} 円/分")
    c.drawString(margin, metrics_y - 30, f"年間標準稼働時間: {metrics.annual_minutes:,.0f} 分")
    c.drawString(margin, metrics_y - 45, f"正味直接工員数: {metrics.net_workers:,.2f} 人")

    header_y = metrics_y - 80
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, header_y, "ROI 上位 SKU")
    c.setFont("Helvetica", 9)
    top = df.sort_values(by="rate_gap_vs_required").head(10)
    y = header_y - 15
    for _, row in top.iterrows():
        line = f"{row.get('product_no')} {row.get('product_name')} ギャップ {row.get('rate_gap_vs_required', 0):.2f}"
        c.drawString(margin, y, line)
        y -= 12
        if y < margin + 40:
            c.showPage()
            y = height - margin
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


def render_dashboard(
    current_scenario: str,
    scenarios: Dict[str, Dict[str, Any]],
    translator,
) -> None:
    """メインダッシュボードを構築して表示する."""

    payload = scenarios.get(current_scenario)
    if not payload:
        st.info(translator("no_data"))
        return
    df_products: pd.DataFrame = payload.get("products", pd.DataFrame())
    metrics: OperationalMetrics = payload.get("metrics")
    if metrics is None:
        st.warning("計算指標が見つかりません。データ入力画面で設定を保存してください。")
        return

    st.subheader(translator("dashboard_title"))
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(translator("kpi_required_rate"), _format_metric(metrics.required_rate, "円/分"))
    col2.metric(translator("kpi_break_even"), _format_metric(metrics.break_even_rate, "円/分"))
    col3.metric(translator("kpi_avg_va"), _format_metric(df_products["va_per_min"].mean()))
    if "meets_required_rate" in df_products:
        meets_ratio = pd.to_numeric(df_products["meets_required_rate"], errors="coerce").mean()
        meets_text = f"{meets_ratio:.0%}" if not np.isnan(meets_ratio) else "-"
    else:
        meets_text = "-"
    col4.metric(translator("kpi_meet_ratio"), meets_text)

    all_data = prepare_scatter_source(scenarios)
    st.plotly_chart(scatter_plot(all_data, metrics.required_rate, translator), use_container_width=True)

    summary = scenario_summary_table(scenarios)
    st.plotly_chart(scenario_bar_chart(summary, translator), use_container_width=True)

    st.plotly_chart(distribution_histogram(df_products, translator), use_container_width=True)

    roi_df = estimate_roi_table(df_products, metrics.required_rate)
    st.subheader(translator("roi_header"))
    st.dataframe(roi_df.head(20), use_container_width=True)
    st.download_button(
        label="ROI テーブルをCSVでダウンロード",
        data=roi_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="roi_simulation.csv",
        mime="text/csv",
    )

    with st.expander(translator("ai_commentary"), expanded=False):
        commentary = generate_ai_commentary(df_products, metrics)
        st.write(commentary)
    pdf_bytes = build_pdf_report(current_scenario, df_products, metrics)
    st.download_button(
        label=translator("pdf_button"),
        data=pdf_bytes,
        file_name=f"standard_rate_{current_scenario}.pdf",
        mime="application/pdf",
    )
