import sys
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parents[1])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from components import (
    apply_user_theme,
    render_help_button,
    render_onboarding,
    render_page_tutorial,
    render_stepper,
    render_sidebar_nav,
)
import os
from textwrap import dedent
from typing import Any
from openai import OpenAI
from offline import restore_session_state_from_cache, sync_offline_cache


def _explain_standard_rate(
    params: dict[str, float], results: dict[str, float], detail: str
) -> tuple[str, list[dict[str, Any]]]:
    sanitized_params, _ = sanitize_params(params)

    def _safe_pct(part: float, whole: float) -> float:
        if whole <= 0:
            return 0.0
        return part / whole * 100

    def _format_currency(value: float) -> str:
        return f"{value:,.0f}"

    def _format_currency_delta(value: float) -> str:
        return f"{value:+,.0f}"

    def _format_rate(value: float) -> str:
        return f"{value:.3f}"

    def _format_rate_delta(value: float) -> str:
        return f"{value:+.3f}"

    def _format_percent(value: float) -> str:
        return f"{value:.1f}%"

    def _format_percent_delta(value: float) -> str:
        return f"{value:+.1f}pt"

    def _format_minutes(value: float) -> str:
        return f"{value:,.0f}"

    def _format_minutes_delta(value: float) -> str:
        return f"{value:+,.0f}"

    fixed_total = results.get("fixed_total", 0.0)
    required_profit_total = results.get("required_profit_total", 0.0)
    break_even_rate = results.get("break_even_rate", 0.0)
    required_rate = results.get("required_rate", 0.0)
    annual_minutes = results.get("annual_minutes", 0.0)
    minutes_per_day = results.get("minutes_per_day", 0.0)
    standard_daily_minutes = results.get("standard_daily_minutes", 0.0)
    net_workers = results.get("net_workers", 0.0)

    labor_cost = sanitized_params.get("labor_cost", 0.0)
    sga_cost = sanitized_params.get("sga_cost", 0.0)
    loan_repayment = sanitized_params.get("loan_repayment", 0.0)
    tax_payment = sanitized_params.get("tax_payment", 0.0)
    future_business = sanitized_params.get("future_business", 0.0)
    fulltime_workers = sanitized_params.get("fulltime_workers", 0.0)
    part1_workers = sanitized_params.get("part1_workers", 0.0)
    part2_workers = sanitized_params.get("part2_workers", 0.0)
    part2_coefficient = sanitized_params.get("part2_coefficient", 0.0)
    working_days = sanitized_params.get("working_days", 0.0)
    daily_hours = sanitized_params.get("daily_hours", 0.0)
    operation_rate = sanitized_params.get("operation_rate", 0.0)

    labor_share_pct = _safe_pct(labor_cost, fixed_total)
    sga_share_pct = _safe_pct(sga_cost, fixed_total)
    loan_share_pct = _safe_pct(loan_repayment, required_profit_total)
    tax_share_pct = _safe_pct(tax_payment, required_profit_total)
    future_share_pct = _safe_pct(future_business, required_profit_total)

    headcount_total = fulltime_workers + part1_workers + part2_workers
    part_ratio_pct = _safe_pct(part1_workers + part2_workers, headcount_total)
    part_ratio_ratio = (
        (part1_workers + part2_workers) / headcount_total if headcount_total > 0 else 0.0
    )

    scenario_data: list[dict[str, Any]] = []

    def register_scenario(title: str, narrative: str, mutate) -> None:
        scenario_params = sanitized_params.copy()
        extra = mutate(scenario_params) or {}
        scenario_params, _ = sanitize_params(scenario_params)
        _, scenario_results = compute_rates(scenario_params)
        headcount_after = (
            scenario_params["fulltime_workers"]
            + scenario_params["part1_workers"]
            + scenario_params["part2_workers"]
        )
        part_ratio_after = _safe_pct(
            scenario_params["part1_workers"] + scenario_params["part2_workers"],
            headcount_after,
        )
        entry: dict[str, Any] = {
            "title": title,
            "narrative": narrative,
            "focus": extra.get("focus", ""),
            "assumption": extra.get("assumption", ""),
            "notes": extra.get("notes", ""),
            "required_rate": scenario_results["required_rate"],
            "delta_required_rate": scenario_results["required_rate"] - required_rate,
            "break_even_rate": scenario_results["break_even_rate"],
            "delta_break_even_rate": scenario_results["break_even_rate"]
            - break_even_rate,
            "fixed_total": scenario_results["fixed_total"],
            "delta_fixed_total": scenario_results["fixed_total"] - fixed_total,
            "annual_minutes": scenario_results["annual_minutes"],
            "delta_annual_minutes": scenario_results["annual_minutes"] - annual_minutes,
            "labor_cost": scenario_params["labor_cost"],
            "delta_labor_cost": scenario_params["labor_cost"] - labor_cost,
            "part_ratio_before": part_ratio_pct,
            "part_ratio_after": part_ratio_after,
            "part_ratio_delta": part_ratio_after - part_ratio_pct,
            "fulltime_before": fulltime_workers,
            "fulltime_after": scenario_params["fulltime_workers"],
            "delta_fulltime": scenario_params["fulltime_workers"] - fulltime_workers,
            "part1_before": part1_workers,
            "part1_after": scenario_params["part1_workers"],
            "delta_part1": scenario_params["part1_workers"] - part1_workers,
            "part2_before": part2_workers,
            "part2_after": scenario_params["part2_workers"],
            "delta_part2": scenario_params["part2_workers"] - part2_workers,
            "operation_rate_before": operation_rate,
            "operation_rate_after": scenario_params["operation_rate"],
            "operation_rate_delta": scenario_params["operation_rate"] - operation_rate,
        }
        if "converted_workers" in extra:
            entry["converted_workers"] = extra["converted_workers"]
        if "param_changes" in extra:
            entry["param_changes"] = extra["param_changes"]
        scenario_data.append(entry)

    if labor_cost > 0:
        def _reduce_labor_cost(p: dict[str, float]) -> dict[str, Any]:
            before = p["labor_cost"]
            p["labor_cost"] = max(before * 0.95, 0.0)
            delta = p["labor_cost"] - before
            return {
                "focus": "労務費削減",
                "assumption": "人員配置の最適化で平均人件費を5%圧縮する想定。",
                "param_changes": {"labor_cost": f"{delta:+,.0f}円 (-5%)"},
            }

        register_scenario(
            "平均賃率を5%圧縮",
            "正社員シフトの一部をパート化して労務費を抑える想定",
            _reduce_labor_cost,
        )

    if sga_cost > 0:
        def _trim_sga(p: dict[str, float]) -> dict[str, Any]:
            before = p["sga_cost"]
            p["sga_cost"] = max(before * 0.95, 0.0)
            delta = p["sga_cost"] - before
            return {
                "focus": "販管費最適化",
                "assumption": "共通管理費を5%削減できると仮定。",
                "param_changes": {"sga_cost": f"{delta:+,.0f}円 (-5%)"},
            }

        register_scenario(
            "共通管理費を5%削減",
            "間接部門コストを見直して固定費を圧縮する想定",
            _trim_sga,
        )

    op_increment = min(0.05, max(0.0, 1.0 - operation_rate))
    if op_increment > 0:
        def _raise_operation(p: dict[str, float], inc: float = op_increment) -> dict[str, Any]:
            before = p["operation_rate"]
            p["operation_rate"] = min(before + inc, 1.0)
            return {
                "focus": "操業度改善",
                "assumption": f"段取り短縮などで操業度を{inc * 100:.1f}ポイント改善する想定。",
                "param_changes": {
                    "operation_rate": f"{before:.2f}→{p['operation_rate']:.2f}"
                },
            }

        register_scenario(
            f"段取り改善で操業度を{op_increment * 100:.1f}pt改善",
            "ライン停止を減らして有効稼働率を高める想定",
            _raise_operation,
        )

    coeff_increment = min(0.1, max(0.0, 1.0 - part2_coefficient))
    if part2_workers > 0 and coeff_increment > 0:
        def _raise_part2_coeff(p: dict[str, float], inc: float = coeff_increment) -> dict[str, Any]:
            before = p["part2_coefficient"]
            p["part2_coefficient"] = min(before + inc, 1.0)
            return {
                "focus": "稼働効率改善",
                "assumption": f"準社員Bの稼働係数を{inc * 100:.1f}ポイント改善できると仮定。",
                "param_changes": {
                    "part2_coefficient": f"{before:.2f}→{p['part2_coefficient']:.2f}"
                },
            }

        register_scenario(
            f"準社員Bの稼働係数を{coeff_increment * 100:.1f}pt改善",
            "柔軟シフトの調整で同じ人員の稼働効率を引き上げる想定",
            _raise_part2_coeff,
        )

    part_ratio_increment = min(0.1, max(0.0, 0.95 - part_ratio_ratio))
    potential_convert = min(fulltime_workers, headcount_total * part_ratio_increment)
    if labor_cost > 0 and potential_convert > 0:
        def _shift_to_part(p: dict[str, float], convert: float = potential_convert) -> dict[str, Any]:
            headcount = (
                sanitized_params["fulltime_workers"]
                + sanitized_params["part1_workers"]
                + sanitized_params["part2_workers"]
            )
            wage_discount = 0.3
            avg_full_cost = labor_cost / headcount if headcount > 0 else 0.0
            cost_reduction = avg_full_cost * wage_discount * convert
            p["fulltime_workers"] = max(p["fulltime_workers"] - convert, 0.0)
            p["part1_workers"] = p.get("part1_workers", 0.0) + convert
            p["labor_cost"] = max(p["labor_cost"] - cost_reduction, 0.0)
            return {
                "focus": "人員シフト",
                "assumption": f"正社員 {convert:.2f}人をパートにシフトし、パート賃率を正社員比{(1 - wage_discount) * 100:.0f}%と仮定。",
                "param_changes": {
                    "fulltime_workers": f"-{convert:.2f}",
                    "part1_workers": f"+{convert:.2f}",
                    "labor_cost": f"{-cost_reduction:,.0f}円",
                },
                "converted_workers": convert,
            }

        register_scenario(
            f"パート比率を{part_ratio_increment * 100:.0f}pt引き上げて人件費最適化",
            "正社員シフトの一部をパート化して労務費を抑える想定",
            _shift_to_part,
        )

    base_info_lines = [
        f"- 必要賃率: {_format_rate(required_rate)}円/分（時給換算 {required_rate * 60:,.1f}円）",
        f"- 損益分岐賃率: {_format_rate(break_even_rate)}円/分（時給換算 {break_even_rate * 60:,.1f}円）",
        f"- 年間標準稼働分: {_format_minutes(annual_minutes)}分",
        f"- 固定費計: {_format_currency(fixed_total)}円",
        f"- 必要利益計: {_format_currency(required_profit_total)}円",
    ]

    cost_lines = [
        f"- 労務費: {_format_currency(labor_cost)}円（固定費の{_format_percent(labor_share_pct)}）",
        f"- 販管費: {_format_currency(sga_cost)}円（固定費の{_format_percent(sga_share_pct)}）",
    ]

    profit_lines = [
        f"- 借入返済: {_format_currency(loan_repayment)}円（必要利益の{_format_percent(loan_share_pct)}）",
        f"- 納税・納付: {_format_currency(tax_payment)}円（必要利益の{_format_percent(tax_share_pct)}）",
        f"- 未来事業費: {_format_currency(future_business)}円（必要利益の{_format_percent(future_share_pct)}）",
    ]

    workforce_lines = [
        f"- 正味直接工員数: {net_workers:.2f}人（正社員 {fulltime_workers:.2f}人、準社員A {part1_workers:.2f}人、準社員B {part2_workers:.2f}人、準社員B稼働係数 {part2_coefficient:.2f}）",
        f"- パート比率: {_format_percent(part_ratio_pct)}（パート人員 {part1_workers + part2_workers:.2f}人 / 総人員 {headcount_total:.2f}人）",
        f"- 年間稼働日数: {working_days:,.0f}日、1日稼働時間 {daily_hours:.2f}時間（1日稼働分 {_format_minutes(minutes_per_day)}分、標準稼働分 {_format_minutes(standard_daily_minutes)}分、操業度 {_format_percent(operation_rate * 100)}）",
    ]

    if scenario_data:
        scenario_entries: list[str] = []
        for idx, sc in enumerate(scenario_data, 1):
            focus_txt = f"（重点領域: {sc['focus']}）" if sc.get("focus") else ""
            entry_lines = [
                f"{idx}. {sc['title']}{focus_txt}｜{sc['narrative']}",
                "   効果: "
                f"必要賃率 {_format_rate(sc['required_rate'])}円/分 (Δ{_format_rate_delta(sc['delta_required_rate'])}円/分) / "
                f"損益分岐賃率 {_format_rate(sc['break_even_rate'])}円/分 (Δ{_format_rate_delta(sc['delta_break_even_rate'])}円/分)",
                "   コスト: "
                f"固定費 {_format_currency(sc['fixed_total'])}円 (Δ{_format_currency_delta(sc['delta_fixed_total'])}円) / "
                f"労務費 {_format_currency(sc['labor_cost'])}円 (Δ{_format_currency_delta(sc['delta_labor_cost'])}円)",
                "   稼働: "
                f"年間稼働分 {_format_minutes(sc['annual_minutes'])}分 (Δ{_format_minutes_delta(sc['delta_annual_minutes'])}分) / "
                f"パート比率 {_format_percent(sc['part_ratio_after'])} (Δ{_format_percent_delta(sc['part_ratio_delta'])})",
                "   人員: "
                f"正社員 {sc['fulltime_after']:.2f}人 (Δ{sc['delta_fulltime']:+.2f}) / "
                f"準社員A {sc['part1_after']:.2f}人 (Δ{sc['delta_part1']:+.2f}) / 準社員B {sc['part2_after']:.2f}人 (Δ{sc['delta_part2']:+.2f})",
            ]
            if sc.get("assumption"):
                entry_lines.append(f"   想定: {sc['assumption']}")
            if sc.get("param_changes"):
                change_desc = ", ".join(
                    f"{k}: {v}" for k, v in sc["param_changes"].items()
                )
                entry_lines.append(f"   主な入力変更: {change_desc}")
            if sc.get("notes"):
                entry_lines.append(f"   補足: {sc['notes']}")
            scenario_entries.append("\n".join(entry_lines))
        scenario_block = "\n".join(scenario_entries)
    else:
        scenario_block = "- 改善シミュレーションは生成できませんでした。"

    detail_rules = {
        "simple": "- 専門用語を控え、経営層が意思決定で押さえたいポイントを端的に述べてください。\n- 数値は丸めつつ単位を明記し、施策ごとの効果（金額・ポイント）を具体的に示してください。",
        "detailed": "- 管理会計やCVP分析の用語を活用し、施策のロジックと前提条件を丁寧に記述してください。\n- 施策ごとに必要賃率・損益分岐賃率・コスト差分などの根拠を引用し、経営会議の議事メモとして読める密度にしてください。",
    }
    detail_key = detail if detail in detail_rules else "simple"
    detail_label_map = {"simple": "経営者向け", "detailed": "管理会計担当向け"}
    detail_label = detail_label_map[detail_key]
    style_rules = detail_rules[detail_key]

    prompt = dedent(
        f"""
        あなたは製造業の管理会計に精通したアドバイザーです。以下の標準賃率の計算結果を読み取り、経営層が意思決定に使える解説と改善策を提示してください。

        【前提（計算結果）】
        {'\n'.join(base_info_lines)}
        【コスト構成】
        {'\n'.join(cost_lines)}
        【利益確保の内訳】
        {'\n'.join(profit_lines)}
        【人員と稼働前提】
        {'\n'.join(workforce_lines)}
        【アクションシミュレーション】
        {scenario_block}

        指示:
        1. 計算結果から読み取れる現状の要点を2行程度でまとめてください。
        2. 上記アクションシミュレーションや比率を根拠に、具体的な施策を最低2つ提示してください。各施策では必要賃率または損益分岐賃率が何円変化するか、主要なコスト・人員指標の変化も明示してください。
        3. 施策ごとに優先度や実行留意点にも触れ、意思決定に役立つ洞察を添えてください。

        粒度指定: {detail_label}
        表現ルール:
        {style_rules}

        出力フォーマット:
        【サマリー】
        - ...
        【アクションプラン】
        1. ...（施策名と効果）
        2. ...
        """
    )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OpenAI APIキーが設定されていません。", scenario_data

    client = OpenAI(api_key=api_key)

    try:
        resp = client.responses.create(model="gpt-4o-mini", input=prompt)
        return resp.output_text.strip(), scenario_data
    except Exception as exc:
        return f"AI説明の生成に失敗しました: {exc}", scenario_data

from standard_rate_core import (
    DEFAULT_PARAMS,
    sanitize_params,
    compute_rates,
    build_reverse_index,
    plot_sensitivity,
    generate_pdf,
)

apply_user_theme()

restore_session_state_from_cache()

render_sidebar_nav(page_key="standard_rate")

header_col, help_col = st.columns([0.76, 0.24], gap="small")
with header_col:
    st.title("③ 標準賃率 計算/感度分析")

render_help_button("standard_rate", container=help_col)

render_onboarding()
render_page_tutorial("standard_rate")
render_stepper(4)

st.markdown(
    """
    <style>
    .sr-section {
        background: var(--app-surface);
        border-radius: 20px;
        border: 1px solid var(--app-border);
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 20px 36px rgba(0, 0, 0, 0.08);
        color: var(--app-text);
    }
    .sr-section h4 {
        color: var(--app-text);
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .sr-section p,
    .sr-section .sr-helper {
        color: var(--app-muted);
        margin-bottom: 0.6rem;
    }
    .sr-section div[data-baseweb="input"] > input,
    .sr-section textarea,
    .sr-section select,
    .sr-section input[type="number"],
    .sr-section input[type="text"] {
        background-color: var(--app-surface) !important;
        color: var(--app-text) !important;
        border-radius: 12px;
        border: 1px solid var(--app-border);
        font-weight: 600;
    }
    .sr-section label {
        color: var(--app-text) !important;
        font-weight: 600 !important;
    }
    .sr-section .stSlider > div > div > div[data-testid="stTickBar"] {
        background-color: var(--app-border);
    }
    .sr-section .stSlider > div > div > div > div {
        background: linear-gradient(90deg, var(--app-accent), rgba(255, 255, 255, 0));
    }
    .sr-section .stSlider [data-testid="stThumbValue"] > div {
        color: var(--app-text) !important;
        font-weight: 700;
    }
    div[data-testid="metric-container"] {
        background: var(--app-surface);
        border-radius: 20px;
        border: 1px solid var(--app-border);
        padding: 1.2rem;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05), 0 18px 32px rgba(0, 0, 0, 0.08);
    }
    div[data-testid="metric-container"] label {
        color: var(--app-muted) !important;
        font-weight: 600;
    }
    div[data-testid="metric-container"] > div:nth-child(2) {
        color: var(--app-text) !important;
        font-weight: 700;
    }
    .sr-metric-caption {
        margin-top: -0.6rem;
        font-size: var(--app-font-small);
        color: var(--app-muted);
    }
    .sr-highlight {
        background: var(--app-surface);
        border-radius: 16px;
        border: 1px solid var(--app-border);
        padding: 0.8rem 1rem;
        color: var(--app-text);
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.08);
    }
    .sr-highlight strong {
        color: var(--app-accent);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

scenarios = st.session_state.setdefault("scenarios", {"ベース": st.session_state.get("sr_params", DEFAULT_PARAMS)})
current = st.session_state.setdefault("current_scenario", "ベース")
st.caption(f"適用中シナリオ: {current}")

params = scenarios.get(current, st.session_state.get("sr_params", DEFAULT_PARAMS)).copy()

st.sidebar.header("シナリオ")
names = list(scenarios.keys())
selected = st.sidebar.selectbox("シナリオ選択", names, index=names.index(current))
if selected != current:
    st.session_state["current_scenario"] = selected
    st.session_state["sr_params"] = scenarios[selected].copy()
    st.rerun()

new_name = st.sidebar.text_input("新規シナリオ名", "")
if st.sidebar.button("追加") and new_name:
    scenarios[new_name] = params.copy()
    st.session_state["current_scenario"] = new_name
    st.session_state["sr_params"] = params.copy()
    st.rerun()

if current != "ベース" and st.sidebar.button("削除"):
    del scenarios[current]
    st.session_state["current_scenario"] = "ベース"
    st.session_state["sr_params"] = scenarios["ベース"].copy()
    st.rerun()

st.markdown(
    """
    <div class="sr-highlight">
        標準賃率は「工場を動かすのに最低限必要な売上単価」です。下記の前提を変えると、主要な賃率指標が即座にアップデートされます。
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### 入力条件")
st.caption("数字を直接入力すると即時に再計算され、影響を受ける指標が項目ごとに表示されます。")

placeholders = {}

section_container = st.container()
section_container.markdown('<div class="sr-section">', unsafe_allow_html=True)
with section_container.container():
    st.markdown("#### A) 必要固定費（円/年）")
    st.caption("製造を維持するために必ず発生する年間コストです。")
    fixed_col1, fixed_col2 = st.columns(2, gap="large")
    with fixed_col1:
        params["labor_cost"] = st.number_input(
            "労務費（製造担当の年間人件費）",
            value=float(params["labor_cost"]),
            step=1000.0,
            format="%.0f",
            min_value=0.0,
            help="工場で働くメンバーに支払う給与・賞与・社会保険料などの合計です。",
        )
        placeholders["labor_cost"] = st.empty()
    with fixed_col2:
        params["sga_cost"] = st.number_input(
            "販管費（共通管理コスト）",
            value=float(params["sga_cost"]),
            step=1000.0,
            format="%.0f",
            min_value=0.0,
            help="製造以外に必要な管理部門・営業部門の共通費用を含みます。",
        )
        placeholders["sga_cost"] = st.empty()
section_container.markdown("</div>", unsafe_allow_html=True)

section_container = st.container()
section_container.markdown('<div class="sr-section">', unsafe_allow_html=True)
with section_container.container():
    st.markdown("#### B) 必要利益（円/年）")
    st.caption("事業を健全に続けるために確保したい利益水準です。")
    profit_col1, profit_col2, profit_col3 = st.columns(3, gap="large")
    with profit_col1:
        params["loan_repayment"] = st.number_input(
            "借入返済（年間返済額）",
            value=float(params["loan_repayment"]),
            step=1000.0,
            format="%.0f",
            min_value=0.0,
            help="銀行等への年間返済額。資金繰り計画に沿って設定してください。",
        )
        placeholders["loan_repayment"] = st.empty()
    with profit_col2:
        params["tax_payment"] = st.number_input(
            "納税・納付（年間見込額）",
            value=float(params["tax_payment"]),
            step=1000.0,
            format="%.0f",
            min_value=0.0,
            help="法人税や社会保険料など、利益確保後に支払う義務がある金額です。",
        )
        placeholders["tax_payment"] = st.empty()
    with profit_col3:
        params["future_business"] = st.number_input(
            "未来事業費（投資原資）",
            value=float(params["future_business"]),
            step=1000.0,
            format="%.0f",
            min_value=0.0,
            help="新設備や人材育成など、将来に向けて確保したい資金を入力します。",
        )
        placeholders["future_business"] = st.empty()
section_container.markdown("</div>", unsafe_allow_html=True)

section_container = st.container()
section_container.markdown('<div class="sr-section">', unsafe_allow_html=True)
with section_container.container():
    st.markdown("#### C) 人員・稼働前提")
    st.caption("工場の人員体制と稼働日数/時間の想定です。『稼働係数』はフルタイム勤務を1.00とした場合の働き方の比率を表します。")
    staff_col1, staff_col2, staff_col3 = st.columns(3, gap="large")
    with staff_col1:
        params["fulltime_workers"] = st.number_input(
            "正社員の人数（常勤）",
            value=float(params["fulltime_workers"]),
            step=0.5,
            format="%.2f",
            min_value=0.0,
            help="常勤で働く正社員人数。1人あたり稼働係数は1.00です。",
        )
        placeholders["fulltime_workers"] = st.empty()
    with staff_col2:
        params["part1_workers"] = st.number_input(
            "準社員Aの人数（短時間勤務）",
            value=float(params["part1_workers"]),
            step=0.5,
            format="%.2f",
            min_value=0.0,
            help="短時間勤務の準社員人数。想定稼働係数は0.75です。",
        )
        placeholders["part1_workers"] = st.empty()
    with staff_col3:
        params["part2_workers"] = st.number_input(
            "準社員Bの人数（柔軟シフト）",
            value=float(params["part2_workers"]),
            step=0.5,
            format="%.2f",
            min_value=0.0,
            help="シフトが柔軟な準社員人数。稼働係数は下のスライダーで調整します。",
        )
        placeholders["part2_workers"] = st.empty()

    coeff_col, days_col, hours_col, rate_col = st.columns(4, gap="large")
    with coeff_col:
        params["part2_coefficient"] = st.slider(
            "準社員Bの稼働係数",
            value=float(params["part2_coefficient"]),
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            help="フルタイムを1.00とした場合の働きぶりの割合です。0.60なら6割稼働を意味します。",
        )
        placeholders["part2_coefficient"] = st.empty()
    with days_col:
        params["working_days"] = st.number_input(
            "年間稼働日数",
            value=float(params["working_days"]),
            step=1.0,
            format="%.0f",
            min_value=1.0,
            help="1年間で工場を動かす日数です（休日や定期点検日を除きます）。",
        )
        placeholders["working_days"] = st.empty()
    with hours_col:
        params["daily_hours"] = st.number_input(
            "1日あたりの稼働時間",
            value=float(params["daily_hours"]),
            step=0.1,
            format="%.2f",
            min_value=0.1,
            help="工場を動かす時間帯の合計です。たとえば8時間稼働なら『8.0』を入力します。",
        )
        placeholders["daily_hours"] = st.empty()
    with rate_col:
        params["operation_rate"] = st.slider(
            "1日の稼働率",
            value=float(params["operation_rate"]),
            min_value=0.5,
            max_value=1.0,
            step=0.01,
            help="実働時間のうち、生産に充てられる割合です。0.85なら85%の時間が有効稼働になります。",
        )
        placeholders["operation_rate"] = st.empty()
section_container.markdown("</div>", unsafe_allow_html=True)

params, warn_list = sanitize_params(params)
for w in warn_list:
    st.sidebar.warning(w)
st.session_state["sr_params"] = params
scenarios[current] = params
st.session_state["scenarios"] = scenarios

nodes, results = compute_rates(params)
reverse_index = build_reverse_index(nodes)
for k, ph in placeholders.items():
    affected = ", ".join(reverse_index.get(k, []))
    if affected:
        ph.caption(f"この入力が影響する指標: {affected}")

c1, c2, c3, c4 = st.columns(4, gap="large")
with c1:
    st.metric("損益分岐賃率（円/分）", f"{results['break_even_rate']:.3f}")
    st.caption("売上単価がこの水準を上回ると、固定費を回収して黒字化します。")
with c2:
    st.metric("必要賃率（円/分）", f"{results['required_rate']:.3f}")
    st.caption("借入返済や将来投資を含め、目標利益を確保するための最低単価です。")
with c3:
    st.metric("年間標準稼働時間（分）", f"{results['annual_minutes']:.0f}")
    st.caption("人員構成と稼働率から算出した、年間で確保できる生産可能時間です。")
with c4:
    st.metric("正味直接工員数合計", f"{results['net_workers']:.2f}")
    st.caption("稼働係数を考慮した実働ベースの生産要員数です。")

st.subheader("AI解説・アクションプラン")
st.caption("AIが標準賃率の背景と数値根拠つきの改善策を提示します。表現モードを選んでから生成してください。")

explain_options = {
    "経営者向け（簡易表現）": "simple",
    "管理会計担当向け（詳細表現）": "detailed",
}
selected_label = st.radio(
    "表現モード",
    list(explain_options.keys()),
    index=0,
    horizontal=True,
    help="経営者向けは意思決定ポイントを平易に整理し、管理会計担当向けは専門用語を交えて深掘りします。",
)
detail_key = explain_options[selected_label]

if "sr_ai_comment" not in st.session_state:
    st.session_state["sr_ai_comment"] = {}
if "sr_ai_action_plan" not in st.session_state:
    st.session_state["sr_ai_action_plan"] = {}

if st.button("AI解説・アクションプラン生成"):
    with st.spinner("生成中..."):
        ai_text, plan_payload = _explain_standard_rate(params, results, detail_key)
        st.session_state["sr_ai_comment"][detail_key] = ai_text
        st.session_state["sr_ai_action_plan"][detail_key] = plan_payload

ai_comment = st.session_state["sr_ai_comment"].get(detail_key, "")
ai_plan_data = st.session_state["sr_ai_action_plan"].get(detail_key, [])
if ai_comment:
    if ai_comment.startswith("OpenAI APIキー"):
        st.warning(ai_comment)
    elif ai_comment.startswith("AI説明の生成に失敗しました"):
        st.error(ai_comment)
    else:
        st.markdown(ai_comment)
else:
    st.caption("※ボタンを押すと、選択した表現モードでAIの解説と施策案が表示されます。")

if ai_plan_data:
    st.markdown("#### AI提案のアクションプラン試算")
    base_headcount = (
        params["fulltime_workers"] + params["part1_workers"] + params["part2_workers"]
    )
    base_part_ratio = (
        (params["part1_workers"] + params["part2_workers"]) / base_headcount * 100
        if base_headcount > 0
        else 0.0
    )
    plan_rows = []
    for sc in ai_plan_data:
        param_changes = sc.get("param_changes") or {}
        change_desc = (
            ", ".join(f"{k}: {v}" for k, v in param_changes.items()) if param_changes else "－"
        )
        plan_rows.append(
            {
                "施策": sc.get("title", ""),
                "重点領域": sc.get("focus") or "－",
                "狙い": sc.get("narrative", ""),
                "必要賃率差 (円/分)": f"{sc.get('delta_required_rate', 0.0):+.3f}",
                "損益分岐差 (円/分)": f"{sc.get('delta_break_even_rate', 0.0):+.3f}",
                "固定費差 (円/年)": f"{sc.get('delta_fixed_total', 0.0):+,.0f}",
                "労務費差 (円/年)": f"{sc.get('delta_labor_cost', 0.0):+,.0f}",
                "年間稼働分差 (分/年)": f"{sc.get('delta_annual_minutes', 0.0):+,.0f}",
                "パート比率": f"{base_part_ratio:.1f}%→{sc.get('part_ratio_after', base_part_ratio):.1f}% (Δ{sc.get('part_ratio_delta', 0.0):+.1f}pt)",
                "主な操作値": change_desc,
                "想定/前提": sc.get("assumption") or sc.get("notes") or "－",
            }
        )
    if plan_rows:
        plan_df = pd.DataFrame(plan_rows)
        st.dataframe(plan_df, use_container_width=True, hide_index=True)
        st.caption("※シミュレーション値はAI説明に合わせた簡易試算です。現場条件での検証を前提にご利用ください。")

_, wf_col = st.columns([3, 1])
with wf_col:
    with st.expander("必要賃率ウォーターフォール", expanded=False):
        prev_params = st.session_state.get("prev_month_params")
        if prev_params is not None:
            _, prev_res = compute_rates(prev_params)
            f_prev = prev_res["fixed_total"]
            p_prev = prev_res["required_profit_total"]
            m_prev = prev_res["annual_minutes"]
            r_prev = prev_res["required_rate"]
            f_cur = results["fixed_total"]
            p_cur = results["required_profit_total"]
            m_cur = results["annual_minutes"]
            r_cur = results["required_rate"]
            diff_fixed = (f_cur - f_prev) / m_prev
            diff_profit = (p_cur - p_prev) / m_prev
            diff_minutes = r_cur - r_prev - diff_fixed - diff_profit
            wf_fig = go.Figure(
                go.Waterfall(
                    x=["前月必要賃率", "固定費差分", "必要利益差分", "年間稼働分差分", "当月必要賃率"],
                    measure=["absolute", "relative", "relative", "relative", "total"],
                    y=[r_prev, diff_fixed, diff_profit, diff_minutes, r_cur],
                    increasing={"marker": {"color": "#D55E00"}},
                    decreasing={"marker": {"color": "#009E73"}},
                    totals={"marker": {"color": "#0072B2"}},
                )
            )
            wf_fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(wf_fig, use_container_width=True)
            comp_table = pd.DataFrame(
                {
                    "項目": ["固定費計", "必要利益計", "年間標準稼働分", "必要賃率"],
                    "前月": [f_prev, p_prev, m_prev, r_prev],
                    "当月": [f_cur, p_cur, m_cur, r_cur],
                }
            )
            comp_table["差額"] = comp_table["当月"] - comp_table["前月"]
            styled = comp_table.style.applymap(
                lambda v: "color:red" if v > 0 else "color:blue", subset=["差額"]
            )
            st.dataframe(styled, use_container_width=True)
        else:
            st.info("前月データがありません。")

st.subheader("ブレークダウン")
st.caption("各指標の計算式と、どの入力が影響しているかを一覧で確認できます。")
cat_map = {
    "fixed_total": "固定費",
    "required_profit_total": "必要利益",
    "net_workers": "工数前提",
    "minutes_per_day": "工数前提",
    "standard_daily_minutes": "工数前提",
    "annual_minutes": "工数前提",
    "break_even_rate": "賃率",
    "required_rate": "賃率",
    "daily_be_va": "付加価値",
    "daily_req_va": "付加価値",
}
df_break = pd.DataFrame(
    [
        (
            cat_map.get(n["key"], ""),
            n["label"],
            n["value"],
            n.get("unit", ""),
            n["formula"],
            ", ".join(n["depends_on"]),
        )
        for n in nodes.values()
    ],
    columns=["区分", "項目", "値", "単位", "式", "依存要素"],
)
st.dataframe(df_break, use_container_width=True)

st.subheader("感度分析")
st.caption("主要な入力を増減させたときに賃率がどのように変わるかを可視化します。")
fig = plot_sensitivity(params)
st.pyplot(fig)

df_csv = pd.DataFrame(list(nodes.values()))
df_csv["depends_on"] = df_csv["depends_on"].apply(lambda x: ",".join(x))
csv = df_csv.to_csv(index=False, encoding="utf-8-sig")
st.download_button("CSVエクスポート", data=csv, file_name=f"standard_rate__{current}.csv", mime="text/csv")

pdf_bytes = generate_pdf(nodes, fig)
st.download_button("PDFエクスポート", data=pdf_bytes, file_name=f"standard_rate_summary__{current}.pdf", mime="application/pdf")

sync_offline_cache()
