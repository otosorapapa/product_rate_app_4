"""感度分析と可視化コンポーネントをまとめたモジュール."""
from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from calculations import (
    OperationalMetrics,
    compute_operational_metrics,
    list_parameter_impacts,
    sanitize_operational_inputs,
)


def compute_waterfall(base: OperationalMetrics, updated: OperationalMetrics) -> pd.DataFrame:
    """固定費・利益・稼働条件の変化をウォーターフォール形式で整形する."""

    base_rate = base.required_rate
    step_fixed = ((updated.fixed_total + base.required_profit_total) / base.annual_minutes) - base_rate
    step_profit = ((updated.fixed_total + updated.required_profit_total) / base.annual_minutes) - (
        (updated.fixed_total + base.required_profit_total) / base.annual_minutes
    )
    step_minutes = updated.required_rate - (
        (updated.fixed_total + updated.required_profit_total) / base.annual_minutes
    )
    data = pd.DataFrame(
        [
            {"name": "基準必要負担率", "type": "base", "value": base_rate},
            {"name": "固定費の変化", "type": "relative", "value": step_fixed},
            {"name": "必要利益の変化", "type": "relative", "value": step_profit},
            {"name": "稼働効率の変化", "type": "relative", "value": step_minutes},
            {"name": "結果必要負担率", "type": "total", "value": updated.required_rate},
        ]
    )
    return data


def build_sensitivity_table(base: OperationalMetrics, updated: OperationalMetrics) -> pd.DataFrame:
    """主要指標のビフォー・アフターと差分を表形式にする."""

    items = {
        "損益分岐負担率(円/分)": (base.break_even_rate, updated.break_even_rate),
        "必要負担率(円/分)": (base.required_rate, updated.required_rate),
        "年間標準稼働時間(分)": (base.annual_minutes, updated.annual_minutes),
        "正味直接工員数(人)": (base.net_workers, updated.net_workers),
        "固定費合計(円)": (base.fixed_total, updated.fixed_total),
        "必要利益合計(円)": (base.required_profit_total, updated.required_profit_total),
    }
    rows = []
    for label, (before, after) in items.items():
        rows.append(
            {
                "指標": label,
                "Before": before,
                "After": after,
                "差分": after - before,
            }
        )
    return pd.DataFrame(rows)


def render_waterfall_chart(data: pd.DataFrame, translator) -> go.Figure:
    """Plotly を用いてウォーターフォールチャートを描画する."""

    fig = go.Figure(
        go.Waterfall(
            name="Required Rate",
            orientation="v",
            measure=data["type"].tolist(),
            x=data["name"].tolist(),
            y=data["value"].tolist(),
            connector={"line": {"color": "#7F8FA6"}},
        )
    )
    fig.update_layout(
        title=translator("waterfall_title"),
        yaxis_title="円/分",
        paper_bgcolor="#F7F9FB",
        plot_bgcolor="#FFFFFF",
    )
    return fig


def render_sensitivity_form(
    base_params: Dict[str, float],
    translator,
) -> Tuple[Dict[str, float], OperationalMetrics, OperationalMetrics, pd.DataFrame]:
    """Streamlit フォームを用いて感度分析の入力と結果を提示する."""

    sanitized, warn = sanitize_operational_inputs(base_params)
    base_metrics = compute_operational_metrics(sanitized)

    if warn:
        for msg in warn:
            st.warning(msg)

    st.subheader(translator("sensitivity_inputs"))
    with st.form("sensitivity_form"):
        col_a, col_b = st.columns(2)
        fixed_slider = col_a.slider(
            translator("sensitivity_fixed_slider"),
            min_value=50,
            max_value=150,
            value=100,
            step=5,
            help="固定費を ±50% の範囲で仮想的に変更します。",
        )
        profit_slider = col_b.slider(
            translator("sensitivity_profit_slider"),
            min_value=50,
            max_value=150,
            value=100,
            step=5,
            help="必要利益を ±50% の範囲で仮想的に変更します。",
        )
        col_c1, col_c2 = st.columns(2)
        net_workers = col_c1.slider(
            translator("sensitivity_net_workers"),
            min_value=max(0.5, base_metrics.net_workers * 0.5),
            max_value=max(base_metrics.net_workers * 1.5, 1.0),
            value=float(base_metrics.net_workers),
            step=0.1,
            help="人員構成を増減させると必要負担率がどう変動するかを検証できます。",
        )
        operation_rate = col_c1.slider(
            translator("sensitivity_operation_rate"),
            min_value=50,
            max_value=110,
            value=int(base_params.get("operation_rate", 0.75) * 100),
            step=5,
            help="実際の稼働効率を想定し、操業度を調整してください。",
        )
        working_days = col_c2.slider(
            translator("sensitivity_working_days"),
            min_value=180,
            max_value=280,
            value=int(base_params.get("working_days", 236)),
            step=1,
        )
        daily_hours = col_c2.slider(
            translator("sensitivity_daily_hours"),
            min_value=6.0,
            max_value=10.0,
            value=float(base_params.get("daily_hours", 8.68)),
            step=0.25,
        )
        submitted = st.form_submit_button(translator("sensitivity_submit"))

    if not submitted:
        updated_params = sanitized
    else:
        updated_params = sanitized.copy()
        fixed_ratio = fixed_slider / 100
        profit_ratio = profit_slider / 100
        if base_metrics.fixed_total > 0:
            updated_params["labor_cost"] = sanitized["labor_cost"] * fixed_ratio
            updated_params["sga_cost"] = sanitized["sga_cost"] * fixed_ratio
        if base_metrics.required_profit_total > 0:
            updated_params["loan_repayment"] = sanitized["loan_repayment"] * profit_ratio
            updated_params["tax_payment"] = sanitized["tax_payment"] * profit_ratio
            updated_params["future_business"] = sanitized["future_business"] * profit_ratio
        if base_metrics.net_workers > 0:
            scale = net_workers / base_metrics.net_workers
            updated_params["fulltime_workers"] = sanitized["fulltime_workers"] * scale
            updated_params["part1_workers"] = sanitized["part1_workers"] * scale
            updated_params["part2_workers"] = sanitized["part2_workers"] * scale
        updated_params["operation_rate"] = operation_rate / 100
        updated_params["working_days"] = float(working_days)
        updated_params["daily_hours"] = float(daily_hours)

    updated_metrics = compute_operational_metrics(updated_params)
    wf_data = compute_waterfall(base_metrics, updated_metrics)

    st.metric(
        label=translator("kpi_required_rate"),
        value=f"{updated_metrics.required_rate:,.2f}",
        delta=f"{updated_metrics.required_rate - base_metrics.required_rate:+.2f}",
    )
    st.metric(
        label=translator("kpi_break_even"),
        value=f"{updated_metrics.break_even_rate:,.2f}",
        delta=f"{updated_metrics.break_even_rate - base_metrics.break_even_rate:+.2f}",
    )
    st.metric(
        label=translator("kpi_annual_minutes"),
        value=f"{updated_metrics.annual_minutes:,.0f}",
        delta=f"{updated_metrics.annual_minutes - base_metrics.annual_minutes:+,.0f}",
    )
    st.metric(
        label=translator("kpi_net_workers"),
        value=f"{updated_metrics.net_workers:,.2f}",
        delta=f"{updated_metrics.net_workers - base_metrics.net_workers:+.2f}",
    )

    st.plotly_chart(render_waterfall_chart(wf_data, translator), use_container_width=True)

    summary_table = build_sensitivity_table(base_metrics, updated_metrics)
    st.dataframe(summary_table, use_container_width=True)

    mapping = list_parameter_impacts()
    st.write(translator("sensitivity_mapping_title"), mapping)

    return updated_params, base_metrics, updated_metrics, summary_table
