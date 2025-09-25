import sys
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parents[1])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from components import (
    apply_user_theme,
    render_help_button,
    render_onboarding,
    render_page_tutorial,
    render_stepper,
    render_sidebar_nav,
)
import os
from io import BytesIO
from textwrap import dedent
from typing import Any, Optional
from openai import OpenAI
from offline import restore_session_state_from_cache, sync_offline_cache
from legal_updates import (
    build_compliance_alerts,
    fetch_labor_standards_updates,
)

WIZARD_STEPS: list[tuple[str, str]] = [
    ("従業員情報", "従業員区分ごとの人数と稼働係数を入力します。"),
    ("標準作業時間", "年間稼働日数や1日の操業条件を設定します。"),
    ("法定費用・間接費", "労務費や販管費など固定的に発生する費用を入力します。"),
    ("利益率設定", "借入返済や納税・納付など確保したい利益額を登録します。"),
    ("結果表示", "標準賃率と差異分析を確認し、改善ポイントを検討します。"),
]

PARAM_METADATA: dict[str, dict[str, str]] = {
    "labor_cost": {"label": "労務費", "category": "固定費", "unit": "円/年"},
    "sga_cost": {"label": "販管費", "category": "固定費", "unit": "円/年"},
    "loan_repayment": {"label": "借入返済", "category": "利益確保", "unit": "円/年"},
    "tax_payment": {"label": "納税・納付", "category": "利益確保", "unit": "円/年"},
    "future_business": {"label": "未来事業費", "category": "利益確保", "unit": "円/年"},
    "fulltime_workers": {"label": "正社員数", "category": "従業員", "unit": "人"},
    "part1_workers": {"label": "準社員A数", "category": "従業員", "unit": "人"},
    "part2_workers": {"label": "準社員B数", "category": "従業員", "unit": "人"},
    "part2_coefficient": {"label": "準社員B稼働係数", "category": "従業員", "unit": "係数"},
    "working_days": {"label": "年間稼働日数", "category": "稼働条件", "unit": "日"},
    "daily_hours": {"label": "1日稼働時間", "category": "稼働条件", "unit": "時間"},
    "operation_rate": {"label": "1日の稼働率", "category": "稼働条件", "unit": "比率"},
}

FORMULA_TIPS: tuple[str, ...] = (
    "標準労務費 ＝ 標準時間 × 標準賃率",
    "賃率差異 ＝ 実際時間 × (標準賃率 − 実際賃率)",
    "効率差異 ＝ 標準賃率 × (標準時間 − 実際時間)",
)

LANGUAGE_CHOICES: dict[str, str] = {"日本語": "ja", "English": "en", "简体中文": "zh"}
LANGUAGE_DEFAULT = "ja"

TEXTS: dict[str, dict[str, str]] = {
    "ja": {
        "language_label": "表示言語 / Language",
        "language_help": "外国人スタッフ向けに英語・中国語のガイドも表示できます。",
        "legal_alert_header": "法改正アラート",
        "legal_alert_caption": "最新の最低賃金や社会保険料率の改定を反映して標準賃率を確認しましょう。",
        "legal_alert_no_data": "最新の法改正情報を取得できませんでした（サンプルデータを表示しています）。",
        "legal_alert_min_wage_warning": "平均人件費時給 {current_hourly:,.1f} 円が{region}の最低賃金 {value:,.0f} 円を下回っています。{effective} 施行の改定に備えて賃率や人員計画の見直しが必要です。",
        "legal_alert_min_wage_ok": "平均人件費時給 {current_hourly:,.1f} 円は{region}の最低賃金 {value:,.0f} 円を上回っています（施行日 {effective}）。",
        "legal_alert_social_info": "{region} は {effective} から {value:.2f}{unit} に改定されます。",
        "legal_alert_source_prefix": "情報源: {source}",
        "legal_alert_api_note": "※ 将来的に厚生労働省等のAPIと連携し、改定情報を自動反映する計画です。",
        "scenario_header": "シナリオ分析",
        "scenario_caption": "賃率や労働時間を変化させたときの必要賃率・労務費率・利益配分のシミュレーションです。",
        "scenario_tab_simulation": "シミュレーション",
        "scenario_tab_chart": "感度チャート",
        "wage_change_label": "平均賃率の変化率",
        "wage_change_help": "基準値: +10%（労務費が10%上昇するシナリオ）",
        "hours_change_label": "1日稼働時間の変化率",
        "hours_change_help": "段取り改善や残業抑制による稼働時間の増減を想定します。",
        "scenario_required_rate_metric": "必要賃率 (円/分)",
        "scenario_labor_share_metric": "労務費率 (%)",
        "scenario_profit_margin_metric": "利益確保比率 (%)",
        "scenario_metric_caption": "基準との差分はΔ欄で表示されます。",
        "scenario_table_col_label": "シナリオ",
        "scenario_table_col_required_rate": "必要賃率 (円/分)",
        "scenario_table_col_labor_share": "労務費率 (%)",
        "scenario_table_col_profit_share": "利益確保比率 (%)",
        "scenario_table_col_minutes": "年間標準稼働分 (分)",
        "scenario_table_label_base": "基準シナリオ",
        "scenario_table_label_sim": "調整後シナリオ",
        "sensitivity_chart_caption": "主要パラメータを±20%変化させた場合の必要賃率と比率の推移です。赤点は+10%の状況を示します。",
        "sensitivity_title_labor": "賃率変動と労務費率",
        "sensitivity_title_hours": "稼働時間変動と利益配分",
        "sensitivity_axis_change_pct": "変化率 (%)",
        "sensitivity_axis_required_rate": "必要賃率 (円/分)",
        "sensitivity_axis_labor_share": "労務費率 (%)",
        "sensitivity_axis_profit_share": "利益確保比率 (%)",
        "sensitivity_annotation_labor": "+10%で労務費率 {value:.1f}%",
        "sensitivity_annotation_hours": "+10%で利益確保比率 {value:.1f}%",
        "sensitivity_table_label_labor": "賃率+10%時",
        "sensitivity_table_label_hours": "稼働時間+10%時",
        "sensitivity_summary_caption": "10%増加時の主要指標。必要賃率への影響と比率の変化を確認できます。",
        "education_header": "教育コンテンツ",
        "education_caption": "標準原価計算や差異分析を学ぶための外部リソースです（外部サイトへ移動します）。",
        "pdca_header": "PDCA改善ログ",
        "pdca_caption": "施策アイデアや検証結果を記録し、PDCAサイクルを回しましょう。",
        "pdca_stage_label": "ステージ",
        "pdca_note_label": "メモ",
        "pdca_save_button": "ログを追加",
        "pdca_saved_message": "PDCAログを保存しました。",
        "pdca_note_required": "メモを入力してください。",
        "pdca_empty": "まだログがありません。新しいアクションを記録しましょう。",
        "pdca_log_header": "最近の記録",
        "pdca_column_stage": "ステージ",
        "pdca_column_note": "内容",
        "pdca_column_timestamp": "記録日時",
    },
    "en": {
        "language_label": "Interface language",
        "language_help": "Display supporting copy in English or Chinese for non-Japanese staff.",
        "legal_alert_header": "Regulation alerts",
        "legal_alert_caption": "Review minimum wage and social insurance updates before finalising the standard rate.",
        "legal_alert_no_data": "No live regulatory data was retrieved; showing bundled sample updates instead.",
        "legal_alert_min_wage_warning": "Estimated wage JPY {current_hourly:,.1f}/hour is below the {region} minimum of JPY {value:,.0f}. Adjust rates or staffing before {effective}.",
        "legal_alert_min_wage_ok": "Estimated wage JPY {current_hourly:,.1f}/hour stays above the {region} minimum of JPY {value:,.0f} (effective {effective}).",
        "legal_alert_social_info": "{region} rate changes to {value:.2f}{unit} from {effective}.",
        "legal_alert_source_prefix": "Source: {source}",
        "legal_alert_api_note": "Future releases will connect to open government APIs (e.g. MHLW) for automatic updates.",
        "scenario_header": "Scenario analysis",
        "scenario_caption": "Simulate how wage or working-hour changes impact the required rate, labour share and profit share.",
        "scenario_tab_simulation": "Simulation",
        "scenario_tab_chart": "Sensitivity chart",
        "wage_change_label": "Average wage change",
        "wage_change_help": "Default +10% represents a labour cost increase scenario.",
        "hours_change_label": "Daily working time change",
        "hours_change_help": "Model overtime controls or productivity programmes.",
        "scenario_required_rate_metric": "Required rate (JPY/min)",
        "scenario_labor_share_metric": "Labour share (%)",
        "scenario_profit_margin_metric": "Profit share (%)",
        "scenario_metric_caption": "Δ shows the gap versus the baseline assumptions.",
        "scenario_table_col_label": "Scenario",
        "scenario_table_col_required_rate": "Required rate (JPY/min)",
        "scenario_table_col_labor_share": "Labour share (%)",
        "scenario_table_col_profit_share": "Profit share (%)",
        "scenario_table_col_minutes": "Annual standard minutes",
        "scenario_table_label_base": "Baseline",
        "scenario_table_label_sim": "Adjusted scenario",
        "sensitivity_chart_caption": "Parameter sweep (±20%) for required rate and ratios. Red markers emphasise the +10% point.",
        "sensitivity_title_labor": "Wage change vs labour share",
        "sensitivity_title_hours": "Time change vs profit share",
        "sensitivity_axis_change_pct": "Change (%)",
        "sensitivity_axis_required_rate": "Required rate (JPY/min)",
        "sensitivity_axis_labor_share": "Labour share (%)",
        "sensitivity_axis_profit_share": "Profit share (%)",
        "sensitivity_annotation_labor": "+10% ⇒ labour share {value:.1f}%",
        "sensitivity_annotation_hours": "+10% ⇒ profit share {value:.1f}%",
        "sensitivity_table_label_labor": "Wage +10%",
        "sensitivity_table_label_hours": "Hours +10%",
        "sensitivity_summary_caption": "Key metrics when inputs rise by 10%. Use it to brief management quickly.",
        "education_header": "Learning resources",
        "education_caption": "Curated videos and tutorials on standard costing and variance analysis (external links).",
        "pdca_header": "PDCA improvement log",
        "pdca_caption": "Record actions and findings to continuously improve the model.",
        "pdca_stage_label": "Stage",
        "pdca_note_label": "Notes",
        "pdca_save_button": "Save entry",
        "pdca_saved_message": "Entry added to the PDCA log.",
        "pdca_note_required": "Please enter a note before saving.",
        "pdca_empty": "No entries yet — capture your first improvement idea.",
        "pdca_log_header": "Recent entries",
        "pdca_column_stage": "Stage",
        "pdca_column_note": "Details",
        "pdca_column_timestamp": "Logged at",
    },
    "zh": {
        "language_label": "界面语言 / Language",
        "language_help": "可切换为英文或中文，方便外籍员工理解关键指标。",
        "legal_alert_header": "法规更新提醒",
        "legal_alert_caption": "请结合最新的最低工资与社会保险费率调整，检讨标准工资率。",
        "legal_alert_no_data": "未能取得实时法规数据，现展示随附的样本资讯。",
        "legal_alert_min_wage_warning": "估算的平均工资 {current_hourly:,.1f} 日元/小时低于 {region} 的最低工资 {value:,.0f} 日元。请在 {effective} 生效前调整工资或人员计划。",
        "legal_alert_min_wage_ok": "估算的平均工资 {current_hourly:,.1f} 日元/小时高于 {region} 的最低工资 {value:,.0f} 日元（生效日 {effective}）。",
        "legal_alert_social_info": "{region} 将自 {effective} 起调整为 {value:.2f}{unit}。",
        "legal_alert_source_prefix": "信息来源：{source}",
        "legal_alert_api_note": "未来版本将与厚生劳动省等公开API对接，实现自动更新。",
        "scenario_header": "情景分析",
        "scenario_caption": "模拟工资或工时变化对必要工资率、劳务成本比例及利润分配的影响。",
        "scenario_tab_simulation": "模拟",
        "scenario_tab_chart": "敏感度图",
        "wage_change_label": "平均工资变动",
        "wage_change_help": "默认 +10% 代表人工成本上升的情景。",
        "hours_change_label": "每日工时变动",
        "hours_change_help": "用于评估加班控制或效率提升的影响。",
        "scenario_required_rate_metric": "必要工资率 (日元/分)",
        "scenario_labor_share_metric": "劳务成本率 (%)",
        "scenario_profit_margin_metric": "利润保留率 (%)",
        "scenario_metric_caption": "Δ 显示与基准方案的差异。",
        "scenario_table_col_label": "方案",
        "scenario_table_col_required_rate": "必要工资率 (日元/分)",
        "scenario_table_col_labor_share": "劳务成本率 (%)",
        "scenario_table_col_profit_share": "利润保留率 (%)",
        "scenario_table_col_minutes": "年度标准工时 (分)",
        "scenario_table_label_base": "基准方案",
        "scenario_table_label_sim": "调整后方案",
        "sensitivity_chart_caption": "主要参数在 ±20% 变动时的走势，红点表示 +10% 情况。",
        "sensitivity_title_labor": "工资变动与劳务成本率",
        "sensitivity_title_hours": "工时变动与利润分配",
        "sensitivity_axis_change_pct": "变动率 (%)",
        "sensitivity_axis_required_rate": "必要工资率 (日元/分)",
        "sensitivity_axis_labor_share": "劳务成本率 (%)",
        "sensitivity_axis_profit_share": "利润保留率 (%)",
        "sensitivity_annotation_labor": "+10% → 劳务成本率 {value:.1f}%",
        "sensitivity_annotation_hours": "+10% → 利润保留率 {value:.1f}%",
        "sensitivity_table_label_labor": "工资 +10%",
        "sensitivity_table_label_hours": "工时 +10%",
        "sensitivity_summary_caption": "关注输入增加 10% 时的关键指标，快速把握对经营的影响。",
        "education_header": "学习资源",
        "education_caption": "标准成本计算与差异分析的教学视频/文章（外部链接）。",
        "pdca_header": "PDCA 改善记录",
        "pdca_caption": "记录行动与复盘，持续推动改善循环。",
        "pdca_stage_label": "阶段",
        "pdca_note_label": "备注",
        "pdca_save_button": "新增记录",
        "pdca_saved_message": "已保存到 PDCA 记录。",
        "pdca_note_required": "请先输入备注内容。",
        "pdca_empty": "尚无记录，欢迎先登记第一条改善计划。",
        "pdca_log_header": "最新记录",
        "pdca_column_stage": "阶段",
        "pdca_column_note": "内容",
        "pdca_column_timestamp": "记录时间",
    },
}

PDCA_STAGE_ORDER = ["plan", "do", "check", "act"]
PDCA_STAGE_TRANSLATIONS = {
    "plan": {"ja": "Plan（計画）", "en": "Plan", "zh": "计划"},
    "do": {"ja": "Do（実行）", "en": "Do", "zh": "执行"},
    "check": {"ja": "Check（評価）", "en": "Check", "zh": "检查"},
    "act": {"ja": "Action（改善）", "en": "Act", "zh": "改善"},
}

EDUCATIONAL_RESOURCES = [
    {
        "url": "https://j-net21.smrj.go.jp/qa/expand/entry/qa137.html",
        "translations": {
            "ja": {
                "title": "J-Net21: 標準原価計算の基礎",
                "description": "中小機構がまとめた標準原価計算の概要と導入手順の解説記事。",
            },
            "en": {
                "title": "J-Net21: Standard Costing Overview (Japanese)",
                "description": "Outline produced by Japan's SME agency explaining the steps for standard costing (Japanese content).",
            },
            "zh": {
                "title": "J-Net21：标准成本计算基础（日语）",
                "description": "日本中小企业支援机构提供的标准成本计算入门文章。",
            },
        },
    },
    {
        "url": "https://www.udemy.com/course/standard-costing-and-variance-analysis/",
        "translations": {
            "ja": {
                "title": "Udemy: Standard Costing & Variance Analysis",
                "description": "英語のオンデマンド講座で、差異分析の計算手順と活用方法を実務目線で学べます。",
            },
            "en": {
                "title": "Udemy: Standard Costing & Variance Analysis",
                "description": "Hands-on video lessons covering variance calculations and interpretation.",
            },
            "zh": {
                "title": "Udemy：标准成本与差异分析",
                "description": "英文线上课程，透过案例学习差异计算与解读。",
            },
        },
    },
    {
        "url": "https://www.coursera.org/learn/managerial-accounting-fundamentals",
        "translations": {
            "ja": {
                "title": "Coursera: Managerial Accounting Fundamentals",
                "description": "英語の基礎講座。コスト管理やCVP分析の全体像を体系的に学習できます。",
            },
            "en": {
                "title": "Coursera: Managerial Accounting Fundamentals",
                "description": "Introductory course (English) covering managerial accounting, CVP and variance topics.",
            },
            "zh": {
                "title": "Coursera：管理会计基础",
                "description": "英文课程，系统学习管理会计、成本-数量-利润分析等主题。",
            },
        },
    },
]


def _get_language_code() -> str:
    return st.session_state.get("sr_language", LANGUAGE_DEFAULT)


def _t(key: str, **kwargs: Any) -> str:
    lang = _get_language_code()
    translations = TEXTS.get(lang, TEXTS[LANGUAGE_DEFAULT])
    template = translations.get(key) or TEXTS[LANGUAGE_DEFAULT].get(key, key)
    return template.format(**kwargs) if kwargs else template


def _stage_label(stage_key: str) -> str:
    lang = _get_language_code()
    options = PDCA_STAGE_TRANSLATIONS.get(stage_key, {})
    return options.get(lang) or options.get(LANGUAGE_DEFAULT, stage_key)


def _pdca_options() -> list[tuple[str, str]]:
    return [(key, _stage_label(key)) for key in PDCA_STAGE_ORDER]


def _resource_text(resource: dict[str, Any]) -> tuple[str, str]:
    lang = _get_language_code()
    payload = resource.get("translations", {}).get(lang)
    if not payload:
        payload = resource.get("translations", {}).get(LANGUAGE_DEFAULT, {})
    return payload.get("title", ""), payload.get("description", "")


def render_wizard_stepper(current_step: int) -> None:
    """Render a responsive step indicator for the guided input flow."""

    total_steps = len(WIZARD_STEPS)
    progress = 0.0 if total_steps <= 1 else current_step / (total_steps - 1)
    st.progress(progress)
    st.caption(f"ステップ {current_step + 1} / {total_steps}")

    blocks: list[str] = ["<div class=\"sr-stepper\">"]
    for idx, (title, desc) in enumerate(WIZARD_STEPS):
        status = "is-active" if idx == current_step else "is-complete" if idx < current_step else "is-pending"
        indicator = idx + 1
        detail = desc if idx == current_step else ""
        block = (
            "<div class=\"sr-step {status}\">"
            "<span class=\"sr-step-index\">{indicator}</span>"
            "<div class=\"sr-step-body\"><strong>{title}</strong>"
            "<p class=\"sr-step-desc\">{detail}</p></div>"
            "</div>"
        ).format(status=status, indicator=indicator, title=title, detail=detail)
        blocks.append(block)
    blocks.append("</div>")
    st.markdown("\n".join(blocks), unsafe_allow_html=True)


def classify_variance(value: float) -> str:
    """Return a textual judgement (F/A) for variance analysis."""

    if abs(value) < 1e-6:
        return "±0"
    return "有利 (F)" if value < 0 else "不利 (A)"


def build_excel_report(
    params: dict[str, float],
    nodes: dict[str, dict[str, Any]],
    variance_inputs: dict[str, float],
    variance_table: pd.DataFrame,
) -> bytes:
    """Generate an Excel workbook summarising assumptions and results."""

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        param_rows = []
        for key, meta in PARAM_METADATA.items():
            if key not in params:
                continue
            param_rows.append(
                {
                    "カテゴリ": meta["category"],
                    "項目": meta["label"],
                    "値": params[key],
                    "単位": meta["unit"],
                }
            )
        if param_rows:
            pd.DataFrame(param_rows).to_excel(writer, sheet_name="入力前提", index=False)

        result_rows = []
        for node in nodes.values():
            result_rows.append(
                {
                    "指標": node.get("label"),
                    "値": node.get("value"),
                    "単位": node.get("unit"),
                    "計算式": node.get("formula"),
                    "依存要素": ", ".join(node.get("depends_on", [])),
                }
            )
        if result_rows:
            pd.DataFrame(result_rows).to_excel(writer, sheet_name="標準賃率指標", index=False)

        variance_inputs_rows = [
            {"区分": "標準時間 (分)", "値": variance_inputs.get("standard_minutes", 0.0)},
            {"区分": "標準賃率 (円/分)", "値": variance_inputs.get("standard_rate", 0.0)},
            {"区分": "標準労務費 (円)", "値": variance_inputs.get("standard_labor_cost", 0.0)},
            {"区分": "実際時間 (分)", "値": variance_inputs.get("actual_minutes", 0.0)},
            {"区分": "実際賃率 (円/分)", "値": variance_inputs.get("actual_rate", 0.0)},
            {"区分": "実際労務費 (円)", "値": variance_inputs.get("actual_labor_cost", 0.0)},
        ]
        variance_sheet = pd.DataFrame(variance_inputs_rows)
        variance_sheet.to_excel(writer, sheet_name="差異分析", index=False)
        start_row = len(variance_inputs_rows) + 2
        variance_table.to_excel(writer, sheet_name="差異分析", index=False, startrow=start_row)

    buffer.seek(0)
    return buffer.read()


def render_info_popover(label: str, content: str, container: Optional[Any] = None) -> None:
    """Render an information popover with graceful fallback."""

    target = container if container is not None else st
    popover = getattr(target, "popover", None)
    if callable(popover):
        with popover(label):
            target.markdown(content)
    else:
        info_fn = getattr(target, "info", None)
        if callable(info_fn):
            info_fn(content)
        else:
            st.info(content)


def render_wizard_nav(
    current_step: int,
    location: str = "top",
    *,
    next_disabled: bool = False,
    next_disabled_help: Optional[str] = None,
) -> None:
    """Render navigation buttons for the guided wizard."""

    total_steps = len(WIZARD_STEPS)
    nav_container = st.container()
    nav_container.markdown('<div class="sr-nav-buttons">', unsafe_allow_html=True)
    prev_col, next_col = nav_container.columns(2, gap="small")

    prev_disabled = current_step <= 0
    if prev_col.button(
        "← 戻る",
        disabled=prev_disabled,
        use_container_width=True,
        key=f"sr_nav_prev_{location}_{current_step}",
    ):
        st.session_state["sr_wizard_step"] = max(current_step - 1, 0)
        st.rerun()

    is_last_step = current_step >= total_steps - 1
    disable_next = next_disabled or is_last_step
    next_label = "次へ →" if not is_last_step else "ウィザード完了"
    if next_col.button(
        next_label,
        disabled=disable_next,
        use_container_width=True,
        key=f"sr_nav_next_{location}_{current_step}",
        help=next_disabled_help if (next_disabled and not is_last_step) else None,
    ):
        st.session_state["sr_wizard_step"] = min(current_step + 1, total_steps - 1)
        st.rerun()

    nav_container.markdown("</div>", unsafe_allow_html=True)


def validate_wizard_step(
    current_step: int, params: dict[str, float]
) -> tuple[list[str], dict[str, str]]:
    """Return validation messages and field-level hints for the wizard."""

    messages: list[str] = []
    field_messages: dict[str, str] = {}

    if current_step == 0:
        headcount = (
            params.get("fulltime_workers", 0.0)
            + params.get("part1_workers", 0.0)
            + params.get("part2_workers", 0.0) * params.get("part2_coefficient", 0.0)
        )
        if headcount <= 0:
            msg = "正社員または準社員の人数を1名以上入力してください。"
            messages.append(msg)
            field_messages["fulltime_workers"] = msg
            field_messages["part1_workers"] = "準社員Aの人数を入力すると正味工数を計算できます。"
            field_messages["part2_workers"] = "準社員Bの人数を入力すると稼働係数を反映できます。"
    elif current_step == 2:
        labor_cost = params.get("labor_cost", 0.0)
        sga_cost = params.get("sga_cost", 0.0)
        if labor_cost <= 0 and sga_cost <= 0:
            msg = "労務費または販管費を入力してください。"
            messages.append(msg)
            field_messages["labor_cost"] = "労務費が0のままです。標準人件費を入力してください。"
            field_messages["sga_cost"] = "販管費が0のままです。固定的に発生する費用を入力してください。"
    elif current_step == 3:
        loan = params.get("loan_repayment", 0.0)
        tax = params.get("tax_payment", 0.0)
        future = params.get("future_business", 0.0)
        if loan <= 0 and tax <= 0 and future <= 0:
            msg = "確保したい利益額を少なくとも1項目入力してください。"
            messages.append(msg)
            shared_hint = "借入返済・納税・未来事業費のいずれかを入力すると必要賃率に反映されます。"
            field_messages["loan_repayment"] = shared_hint
            field_messages["tax_payment"] = shared_hint
            field_messages["future_business"] = shared_hint

    return messages, field_messages


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
    build_sensitivity_table,
    compute_profit_margin_share,
)

apply_user_theme()

restore_session_state_from_cache()

render_sidebar_nav(page_key="standard_rate")

if "df_products_raw" not in st.session_state or st.session_state.get("df_products_raw") is None:
    st.info("データがまだ取り込まれていません。『① データ入力』でExcelを読み込むかサンプルを使用してください。")
    st.page_link("pages/01_データ入力.py", label="データ入力ページを開く", icon="📥")
    st.stop()

if "sr_language" not in st.session_state:
    st.session_state["sr_language"] = LANGUAGE_DEFAULT

language_keys = list(LANGUAGE_CHOICES.keys())
current_lang_code = _get_language_code()
current_name = next(
    (name for name, code in LANGUAGE_CHOICES.items() if code == current_lang_code),
    language_keys[0],
)
selected_name = st.sidebar.selectbox(
    _t("language_label"),
    language_keys,
    index=language_keys.index(current_name),
    help=_t("language_help"),
)
selected_code = LANGUAGE_CHOICES[selected_name]
if selected_code != current_lang_code:
    st.session_state["sr_language"] = selected_code
    st.rerun()

header_col, help_col = st.columns([0.76, 0.24], gap="small")
with header_col:
    st.title("③ 標準賃率 計算/感度分析")

render_help_button("standard_rate", container=help_col)

render_onboarding()
render_page_tutorial("standard_rate")
render_stepper(5)

st.markdown(
    """
    <style>
    .sr-section {
        background: var(--app-surface);
        border-radius: 12px;
        border: 1px solid rgba(11, 31, 59, 0.12);
        padding: calc(var(--spacing-unit) * 2);
        margin-bottom: calc(var(--spacing-unit) * 2.5);
        box-shadow: 0 2px 10px rgba(11, 31, 59, 0.12);
        color: var(--app-text);
    }
    .sr-section h4 {
        color: var(--app-text);
        font-weight: 700;
        margin-bottom: calc(var(--spacing-unit));
    }
    .sr-section p,
    .sr-section .sr-helper {
        color: rgba(11, 31, 59, 0.7);
        margin-bottom: calc(var(--spacing-unit) * 1.2);
        line-height: 1.6;
    }
    .sr-section div[data-baseweb="input"] > input,
    .sr-section textarea,
    .sr-section select,
    .sr-section input[type="number"],
    .sr-section input[type="text"] {
        background-color: var(--app-surface) !important;
        color: var(--app-text) !important;
        border-radius: 10px;
        border: 1px solid rgba(11, 31, 59, 0.16);
        font-weight: 600;
    }
    .sr-section label {
        color: var(--app-text) !important;
        font-weight: 600 !important;
    }
    .sr-section .stSlider > div > div > div[data-testid="stTickBar"] {
        background-color: rgba(11, 31, 59, 0.15);
    }
    .sr-section .stSlider > div > div > div > div {
        background: linear-gradient(90deg, rgba(30, 136, 229, 0.95), rgba(30, 136, 229, 0.1));
    }
    .sr-section .stSlider [data-testid="stThumbValue"] > div {
        color: var(--app-text) !important;
        font-weight: 700;
    }
    .sr-stepper {
        display: flex;
        gap: calc(var(--spacing-unit) * 1.5);
        flex-wrap: wrap;
        margin: calc(var(--spacing-unit) * 1.5) 0;
    }
    .sr-step {
        flex: 1 1 200px;
        background: var(--app-surface);
        border-radius: 12px;
        border: 1px solid rgba(11, 31, 59, 0.12);
        padding: calc(var(--spacing-unit) * 1.5);
        display: flex;
        align-items: center;
        gap: calc(var(--spacing-unit) * 1);
        box-shadow: 0 2px 8px rgba(11, 31, 59, 0.12);
        transition: border 0.2s ease, box-shadow 0.2s ease;
    }
    .sr-step-index {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        color: #FFFFFF;
        background: rgba(30, 136, 229, 0.45);
    }
    .sr-step.is-active {
        border-color: var(--app-accent);
        box-shadow: 0 6px 16px rgba(30, 136, 229, 0.2);
    }
    .sr-step.is-active .sr-step-index {
        background: var(--app-accent);
    }
    .sr-step.is-complete .sr-step-index {
        background: rgba(30, 136, 229, 0.65);
    }
    .sr-step-body strong {
        color: var(--app-text);
        display: block;
        margin-bottom: calc(var(--spacing-unit) * 0.5);
    }
    .sr-step-desc {
        margin: 0;
        color: rgba(11, 31, 59, 0.65);
        font-size: var(--app-font-small);
    }
    .sr-nav-buttons {
        display: flex;
        gap: calc(var(--spacing-unit) * 1);
        margin: calc(var(--spacing-unit) * 1.5) 0;
    }
    .sr-nav-buttons > div {
        flex: 1;
    }
    .sr-nav-buttons > div button {
        width: 100%;
        border-radius: 12px;
        font-weight: 600;
    }
    div[data-testid="metric-container"] {
        background: var(--app-surface);
        border-radius: 12px;
        border: 1px solid rgba(11, 31, 59, 0.12);
        padding: calc(var(--spacing-unit) * 1.5);
        box-shadow: 0 2px 8px rgba(11, 31, 59, 0.12);
    }
    div[data-testid="metric-container"] label {
        color: rgba(11, 31, 59, 0.65) !important;
        font-weight: 600;
    }
    div[data-testid="metric-container"] > div:nth-child(2) {
        color: var(--app-text) !important;
        font-weight: 700;
    }
    .sr-metric-caption {
        margin-top: calc(var(--spacing-unit) * -0.75);
        font-size: var(--app-font-small);
        color: rgba(11, 31, 59, 0.6);
    }
    .sr-highlight {
        background: var(--app-surface);
        border-radius: 12px;
        border: 1px solid rgba(11, 31, 59, 0.12);
        padding: calc(var(--spacing-unit) * 1.5);
        color: var(--app-text);
        box-shadow: 0 2px 8px rgba(11, 31, 59, 0.12);
    }
    .sr-highlight strong {
        color: var(--app-accent);
    }
    @media (max-width: 860px) {
        .sr-section {
            padding: calc(var(--spacing-unit) * 1.5);
        }
        .sr-stepper {
            flex-direction: column;
        }
        .sr-step {
            flex: 1 1 100%;
        }
        .sr-step-index {
            width: 28px;
            height: 28px;
            font-size: 0.85rem;
        }
        .sr-nav-buttons {
            flex-direction: column;
        }
        .sr-nav-buttons > div button {
            width: 100%;
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }
        div[data-testid="metric-container"] {
            margin-bottom: calc(var(--spacing-unit) * 1);
        }
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

st.sidebar.divider()
st.sidebar.subheader("計算式チートシート")
for tip in FORMULA_TIPS:
    st.sidebar.caption(f"・{tip}")

st.markdown(
    """
    <div class="sr-highlight">
        標準賃率は「工場を動かすのに最低限必要な売上単価」です。入力条件を段階的に見直すと主要な指標が即座にアップデートされます。
    </div>
    """,
    unsafe_allow_html=True,
)

guide_info = (
    "- 標準賃率は技能・所要時間・業界賃率・労使協定・法律などの基準をもとに決定します。\n"
    "- 本ウィザードでは固定費と稼働条件を整理し、必要賃率や損益分岐賃率を算出します。\n"
    "**よく使う公式**\n"
    + "\n".join(f"- {tip}" for tip in FORMULA_TIPS)
)
render_info_popover("ℹ️ 標準賃率の考え方", guide_info)

total_steps = len(WIZARD_STEPS)
current_step = st.session_state.setdefault("sr_wizard_step", 0)
if current_step < 0:
    current_step = 0
if current_step >= total_steps:
    current_step = total_steps - 1
st.session_state["sr_wizard_step"] = current_step

st.markdown("### ガイド付き入力")
st.caption("※ * は必須入力です。")
render_wizard_stepper(current_step)

placeholders: dict[str, Any] = {}

step_container = st.container()
step_container.markdown('<div class="sr-section">', unsafe_allow_html=True)
with step_container.container():
    if current_step == 0:
        st.markdown("#### ステップ1: 従業員情報の入力")
        st.caption("技能や勤務形態ごとの人数と稼働係数を登録します。")
        render_info_popover(
            "ℹ️ 人員区分のヒント",
            "- 正社員: フルタイムで技能水準が高いメンバー。\n"
            "- 準社員A: パート・アルバイトなど短時間勤務者（標準係数0.75で換算）。\n"
            "- 準社員B: シフトが柔軟な人員。稼働係数を調整して実働換算します。",
        )
        staff_cols = st.columns(3, gap="large")
        with staff_cols[0]:
            params["fulltime_workers"] = st.number_input(
                "正社員の人数 *",
                value=float(params["fulltime_workers"]),
                step=0.5,
                format="%.2f",
                min_value=0.0,
                help="技能・資格を備えた常勤従業員数。標準賃率の技能基準を反映します。",
            )
            placeholders["fulltime_workers"] = st.empty()
        with staff_cols[1]:
            params["part1_workers"] = st.number_input(
                "準社員Aの人数（短時間） *",
                value=float(params["part1_workers"]),
                step=0.5,
                format="%.2f",
                min_value=0.0,
                help="短時間勤務の準社員人数。標準では稼働係数0.75で正社員換算します。",
            )
            placeholders["part1_workers"] = st.empty()
        with staff_cols[2]:
            params["part2_workers"] = st.number_input(
                "準社員Bの人数（柔軟シフト） *",
                value=float(params["part2_workers"]),
                step=0.5,
                format="%.2f",
                min_value=0.0,
                help="曜日・時間帯でシフトを最適化する人員。稼働係数は下のスライダーで調整します。",
            )
            placeholders["part2_workers"] = st.empty()

        params["part2_coefficient"] = st.slider(
            "準社員Bの稼働係数",
            value=float(params["part2_coefficient"]),
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            help="フルタイムを1.00とした場合の働きぶり。0.60なら6割稼働を意味します。",
        )
        placeholders["part2_coefficient"] = st.empty()

    elif current_step == 1:
        st.markdown("#### ステップ2: 標準作業時間の設定")
        st.caption("年間稼働日数と1日の操業条件を設定します。")
        render_info_popover(
            "ℹ️ 標準時間の決め方",
            "- 標準時間は熟練者が安全に達成できる時間を基準に設定します。\n"
            "- 段取り替えや休憩、保全時間も操業度に織り込んでください。",
        )
        time_cols = st.columns(3, gap="large")
        with time_cols[0]:
            params["working_days"] = st.number_input(
                "年間稼働日数",
                value=float(params["working_days"]),
                step=1.0,
                format="%.0f",
                min_value=1.0,
                help="休日・点検日を除いた年間の操業日数です。",
            )
            placeholders["working_days"] = st.empty()
        with time_cols[1]:
            params["daily_hours"] = st.number_input(
                "1日あたりの稼働時間",
                value=float(params["daily_hours"]),
                step=0.1,
                format="%.2f",
                min_value=0.1,
                help="シフトを通じて確保する標準稼働時間。休憩を除いた純稼働時間を入力します。",
            )
            placeholders["daily_hours"] = st.empty()
        with time_cols[2]:
            params["operation_rate"] = st.slider(
                "1日の稼働率",
                value=float(params["operation_rate"]),
                min_value=0.5,
                max_value=1.0,
                step=0.01,
                help="有効稼働時間の割合。段取り替え・打ち合わせ時間などを差し引いた実効稼働率です。",
            )
            placeholders["operation_rate"] = st.empty()

    elif current_step == 2:
        st.markdown("#### ステップ3: 法定費用・間接費の入力")
        st.caption("標準賃率の基礎となる固定費を登録します。")
        render_info_popover(
            "ℹ️ 固定費の内訳",
            "- 労務費: 技能・所要時間・業界賃率・労使協定・法律を基準に算出した標準人件費。\n"
            "- 販管費: 法定福利費や共通間接費など、操業に不可欠な間接費を含みます。",
        )
        cost_cols = st.columns(2, gap="large")
        with cost_cols[0]:
            params["labor_cost"] = st.number_input(
                "労務費（年間） *",
                value=float(params["labor_cost"]),
                step=1000.0,
                format="%.0f",
                min_value=0.0,
                help="技能・所要時間・業界賃率を根拠に設定した標準人件費の合計です。",
            )
            placeholders["labor_cost"] = st.empty()
        with cost_cols[1]:
            params["sga_cost"] = st.number_input(
                "販管費（年間） *",
                value=float(params["sga_cost"]),
                step=1000.0,
                format="%.0f",
                min_value=0.0,
                help="法定福利費や共通管理費など、製造以外に必須となる固定的コストです。",
            )
            placeholders["sga_cost"] = st.empty()

    elif current_step == 3:
        st.markdown("#### ステップ4: 利益率設定")
        st.caption("借入返済や納税・将来投資に必要な利益額を設定します。")
        render_info_popover(
            "ℹ️ 目標利益の考え方",
            "- 必要賃率＝(固定費 + 必要利益) ÷ 年間標準稼働分 で算出します。\n"
            "- 返済・納税・投資の計画を金額ベースで入力し、賃率に落とし込みます。",
        )
        profit_cols = st.columns(3, gap="large")
        with profit_cols[0]:
            params["loan_repayment"] = st.number_input(
                "借入返済（年間） *",
                value=float(params["loan_repayment"]),
                step=1000.0,
                format="%.0f",
                min_value=0.0,
                help="金融機関などへの年間返済額。キャッシュフロー計画を反映します。",
            )
            placeholders["loan_repayment"] = st.empty()
        with profit_cols[1]:
            params["tax_payment"] = st.number_input(
                "納税・納付（年間） *",
                value=float(params["tax_payment"]),
                step=1000.0,
                format="%.0f",
                min_value=0.0,
                help="法人税や社会保険料など、法律で義務付けられた支出です。",
            )
            placeholders["tax_payment"] = st.empty()
        with profit_cols[2]:
            params["future_business"] = st.number_input(
                "未来事業費（投資原資） *",
                value=float(params["future_business"]),
                step=1000.0,
                format="%.0f",
                min_value=0.0,
                help="新規事業や設備更新など、将来に向けて確保したい利益額です。",
            )
            placeholders["future_business"] = st.empty()

    else:
        st.markdown("#### ステップ5: 結果表示")
        st.caption("設定した前提をもとに標準賃率と差異分析を確認できます。下部で実績データを入力してください。")

step_container.markdown("</div>", unsafe_allow_html=True)

validation_messages, field_messages = validate_wizard_step(current_step, params)

params, warn_list = sanitize_params(params)
for w in warn_list:
    st.sidebar.warning(w)
st.session_state["sr_params"] = params
scenarios[current] = params
st.session_state["scenarios"] = scenarios

nodes, results = compute_rates(params)
reverse_index = build_reverse_index(nodes)
for k, ph in placeholders.items():
    if field_messages.get(k):
        ph.error(field_messages[k])
        continue
    affected = ", ".join(reverse_index.get(k, []))
    if affected:
        ph.caption(f"この入力が影響する指標: {affected}")

headcount_total = params["fulltime_workers"] + params["part1_workers"] + params["part2_workers"]
part_workers = params["part1_workers"] + params["part2_workers"]
part_ratio_pct = part_workers / headcount_total * 100 if headcount_total > 0 else 0.0

if current_step == 0:
    summary_cols = st.columns(3, gap="large")
    with summary_cols[0]:
        st.metric("正味直接工員数", f"{results['net_workers']:.2f} 人")
    with summary_cols[1]:
        st.metric("総人員", f"{headcount_total:.2f} 人")
    with summary_cols[2]:
        st.metric("パート比率", f"{part_ratio_pct:.1f} %")
    st.caption("人員構成を見直すと標準時間と固定費の割付が変わります。")
elif current_step == 1:
    time_cols = st.columns(3, gap="large")
    with time_cols[0]:
        st.metric("1日稼働分", f"{results['minutes_per_day']:.0f} 分")
    with time_cols[1]:
        st.metric("1日標準稼働分", f"{results['standard_daily_minutes']:.0f} 分")
    with time_cols[2]:
        st.metric("年間標準稼働分", f"{results['annual_minutes']:.0f} 分")
    st.caption("標準労務費＝標準時間×標準賃率 の基礎となる指標です。")
elif current_step == 2:
    cost_cols = st.columns(2, gap="large")
    with cost_cols[0]:
        st.metric("固定費計", f"{results['fixed_total']:,.0f} 円")
    with cost_cols[1]:
        st.metric("1日当り損益分岐付加価値", f"{results['daily_be_va']:,.0f} 円")
    st.caption("固定費の圧縮は損益分岐賃率の改善に直結します。")
elif current_step == 3:
    profit_cols = st.columns(3, gap="large")
    with profit_cols[0]:
        st.metric("必要利益計", f"{results['required_profit_total']:,.0f} 円")
    with profit_cols[1]:
        st.metric("損益分岐賃率", f"{results['break_even_rate']:.3f} 円/分")
    with profit_cols[2]:
        st.metric("必要賃率", f"{results['required_rate']:.3f} 円/分")
    st.caption("目標単価が必要賃率を上回るかをチェックしましょう。")
else:
    st.caption("下部に標準賃率の結果と差異分析を表示します。")

should_block_next = bool(validation_messages) and current_step < total_steps - 1
if should_block_next:
    warning_lines = "\n".join(f"- {msg}" for msg in validation_messages)
    st.warning(f"次へ進む前に以下をご確認ください:\n{warning_lines}")

render_wizard_nav(
    current_step,
    location="main",
    next_disabled=should_block_next,
    next_disabled_help="必須項目を入力すると次のステップへ進めます。",
)

if current_step >= 4:
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

    base_fixed_total = results.get("fixed_total", 0.0)
    base_labor_share = (
        params["labor_cost"] / base_fixed_total * 100.0 if base_fixed_total else 0.0
    )
    base_profit_share = compute_profit_margin_share(results)

    updates_df = fetch_labor_standards_updates()

    st.subheader(_t("legal_alert_header"))
    st.caption(_t("legal_alert_caption"))
    if updates_df.empty:
        st.info(_t("legal_alert_no_data"))
    else:
        alerts = build_compliance_alerts(
            params,
            results,
            updates_df,
            preferred_regions=["東京都", "全国加重平均"],
        )
        for alert in alerts:
            effective_val = alert.get("effective_from")
            effective_label = (
                pd.to_datetime(effective_val, errors="coerce").strftime("%Y-%m-%d")
                if effective_val
                else "-"
            )
            if alert.get("category") == "最低賃金":
                message_key = (
                    "legal_alert_min_wage_warning"
                    if alert.get("severity") == "warning"
                    else "legal_alert_min_wage_ok"
                )
                message_fn = st.warning if alert.get("severity") == "warning" else st.info
                message_fn(
                    _t(
                        message_key,
                        current_hourly=alert.get("current_hourly_wage", 0.0),
                        region=alert.get("region", ""),
                        value=alert.get("value", 0.0),
                        effective=effective_label,
                    )
                )
            else:
                st.info(
                    _t(
                        "legal_alert_social_info",
                        region=alert.get("region", ""),
                        value=alert.get("value", 0.0),
                        unit=alert.get("unit", ""),
                        effective=effective_label,
                    )
                )
            source = alert.get("source") or ""
            if source:
                st.caption(_t("legal_alert_source_prefix", source=source))
        display_df = updates_df.copy()
        if "effective_from" in display_df.columns:
            display_df["effective_from"] = pd.to_datetime(
                display_df["effective_from"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
        if "last_updated" in display_df.columns:
            display_df["last_updated"] = pd.to_datetime(
                display_df["last_updated"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
        lang = _get_language_code()
        if lang == "en":
            column_map = {
                "category": "Category",
                "region": "Region",
                "effective_from": "Effective from",
                "value": "Value",
                "unit": "Unit",
                "source": "Source",
                "last_updated": "Last updated",
                "notes": "Notes",
                "url": "URL",
            }
        elif lang == "zh":
            column_map = {
                "category": "类别",
                "region": "地区",
                "effective_from": "生效日",
                "value": "数值",
                "unit": "单位",
                "source": "来源",
                "last_updated": "更新日",
                "notes": "备注",
                "url": "链接",
            }
        else:
            column_map = {
                "category": "カテゴリ",
                "region": "地域",
                "effective_from": "施行日",
                "value": "数値",
                "unit": "単位",
                "source": "情報源",
                "last_updated": "更新日",
                "notes": "備考",
                "url": "URL",
            }
        display_df = display_df.rename(columns=column_map).fillna("")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.caption(_t("legal_alert_api_note"))

    st.subheader(_t("scenario_header"))
    st.caption(_t("scenario_caption"))
    sim_tab, chart_tab = st.tabs([
        _t("scenario_tab_simulation"),
        _t("scenario_tab_chart"),
    ])

    with sim_tab:
        sim_cols = st.columns(2, gap="large")
        wage_change = sim_cols[0].slider(
            _t("wage_change_label"),
            min_value=-0.2,
            max_value=0.2,
            value=0.1,
            step=0.05,
            format="%+0.0f%%",
            help=_t("wage_change_help"),
        )
        hours_change = sim_cols[1].slider(
            _t("hours_change_label"),
            min_value=-0.2,
            max_value=0.2,
            value=0.0,
            step=0.05,
            format="%+0.0f%%",
            help=_t("hours_change_help"),
        )

        adjusted_params = params.copy()
        adjusted_params["labor_cost"] = adjusted_params["labor_cost"] * (1 + wage_change)
        adjusted_params["daily_hours"] = adjusted_params["daily_hours"] * (1 + hours_change)
        adjusted_params, _ = sanitize_params(adjusted_params)
        _, adjusted_results = compute_rates(adjusted_params)

        adjusted_required = adjusted_results["required_rate"]
        adjusted_labor_share = (
            adjusted_params["labor_cost"] / adjusted_results["fixed_total"] * 100.0
            if adjusted_results["fixed_total"]
            else 0.0
        )
        adjusted_profit_share = compute_profit_margin_share(adjusted_results)

        sim_metrics = st.columns(3, gap="large")
        sim_metrics[0].metric(
            _t("scenario_required_rate_metric"),
            f"{adjusted_required:.3f}",
            delta=f"{adjusted_required - results['required_rate']:+.3f}",
        )
        sim_metrics[1].metric(
            _t("scenario_labor_share_metric"),
            f"{adjusted_labor_share:.1f}%",
            delta=f"{adjusted_labor_share - base_labor_share:+.1f}pt",
        )
        sim_metrics[2].metric(
            _t("scenario_profit_margin_metric"),
            f"{adjusted_profit_share:.1f}%",
            delta=f"{adjusted_profit_share - base_profit_share:+.1f}pt",
        )
        st.caption(_t("scenario_metric_caption"))

        summary_df = pd.DataFrame(
            [
                {
                    "scenario": _t("scenario_table_label_base"),
                    "required_rate": results["required_rate"],
                    "labor_share": base_labor_share,
                    "profit_share": base_profit_share,
                    "annual_minutes": results["annual_minutes"],
                },
                {
                    "scenario": _t("scenario_table_label_sim"),
                    "required_rate": adjusted_required,
                    "labor_share": adjusted_labor_share,
                    "profit_share": adjusted_profit_share,
                    "annual_minutes": adjusted_results["annual_minutes"],
                },
            ]
        )
        summary_df = summary_df.rename(
            columns={
                "scenario": _t("scenario_table_col_label"),
                "required_rate": _t("scenario_table_col_required_rate"),
                "labor_share": _t("scenario_table_col_labor_share"),
                "profit_share": _t("scenario_table_col_profit_share"),
                "annual_minutes": _t("scenario_table_col_minutes"),
            }
        )
        summary_style = summary_df.style.format(
            {
                _t("scenario_table_col_required_rate"): "{:.3f}",
                _t("scenario_table_col_labor_share"): "{:.1f}%",
                _t("scenario_table_col_profit_share"): "{:.1f}%",
                _t("scenario_table_col_minutes"): "{:,.0f}",
            }
        )
        st.dataframe(summary_style, use_container_width=True, hide_index=True)

    with chart_tab:
        st.caption(_t("sensitivity_chart_caption"))
        sensitivity_df = build_sensitivity_table(params)
        labor_df = sensitivity_df[sensitivity_df["factor"] == "labor_cost"]
        hours_df = sensitivity_df[sensitivity_df["factor"] == "daily_hours"]

        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"secondary_y": True}, {"secondary_y": True}]],
            subplot_titles=(
                _t("sensitivity_title_labor"),
                _t("sensitivity_title_hours"),
            ),
            horizontal_spacing=0.12,
        )

        fig.add_trace(
            go.Scatter(
                x=labor_df["change_pct"],
                y=labor_df["required_rate"],
                name=_t("scenario_required_rate_metric"),
                line=dict(color="#1f77b4"),
            ),
            row=1,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=labor_df["change_pct"],
                y=labor_df["labor_share_pct"],
                name=_t("scenario_labor_share_metric"),
                line=dict(color="#ff7f0e", dash="dash"),
            ),
            row=1,
            col=1,
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=hours_df["change_pct"],
                y=hours_df["required_rate"],
                name=_t("scenario_required_rate_metric"),
                line=dict(color="#1f77b4"),
                showlegend=False,
            ),
            row=1,
            col=2,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=hours_df["change_pct"],
                y=hours_df["profit_margin_pct"],
                name=_t("scenario_profit_margin_metric"),
                line=dict(color="#2ca02c", dash="dash"),
            ),
            row=1,
            col=2,
            secondary_y=True,
        )

        highlight_labor = labor_df[labor_df["change_pct"] == 10.0]
        if not highlight_labor.empty:
            fig.add_trace(
                go.Scatter(
                    x=highlight_labor["change_pct"],
                    y=highlight_labor["labor_share_pct"],
                    mode="markers+text",
                    text=[
                        _t(
                            "sensitivity_annotation_labor",
                            value=float(highlight_labor["labor_share_pct"].iloc[0]),
                        )
                    ],
                    textposition="top center",
                    marker=dict(size=10, color="#d62728"),
                    showlegend=False,
                ),
                row=1,
                col=1,
                secondary_y=True,
            )
        highlight_hours = hours_df[hours_df["change_pct"] == 10.0]
        if not highlight_hours.empty:
            fig.add_trace(
                go.Scatter(
                    x=highlight_hours["change_pct"],
                    y=highlight_hours["profit_margin_pct"],
                    mode="markers+text",
                    text=[
                        _t(
                            "sensitivity_annotation_hours",
                            value=float(highlight_hours["profit_margin_pct"].iloc[0]),
                        )
                    ],
                    textposition="bottom center",
                    marker=dict(size=10, color="#d62728"),
                    showlegend=False,
                ),
                row=1,
                col=2,
                secondary_y=True,
            )

        fig.update_xaxes(title_text=_t("sensitivity_axis_change_pct"), row=1, col=1)
        fig.update_xaxes(title_text=_t("sensitivity_axis_change_pct"), row=1, col=2)
        fig.update_yaxes(
            title_text=_t("sensitivity_axis_required_rate"),
            row=1,
            col=1,
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text=_t("sensitivity_axis_labor_share"),
            row=1,
            col=1,
            secondary_y=True,
        )
        fig.update_yaxes(
            title_text=_t("sensitivity_axis_required_rate"),
            row=1,
            col=2,
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text=_t("sensitivity_axis_profit_share"),
            row=1,
            col=2,
            secondary_y=True,
        )
        fig.add_vline(x=10, line_dash="dot", line_color="#d62728", row=1, col=1)
        fig.add_vline(x=10, line_dash="dot", line_color="#d62728", row=1, col=2)
        fig.update_layout(
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        highlight_rows: list[dict[str, float]] = []
        if not highlight_labor.empty:
            highlight_rows.append(
                {
                    "scenario": _t("sensitivity_table_label_labor"),
                    "required_rate": float(highlight_labor["required_rate"].iloc[0]),
                    "labor_share": float(highlight_labor["labor_share_pct"].iloc[0]),
                    "profit_share": float(highlight_labor["profit_margin_pct"].iloc[0]),
                    "annual_minutes": float(highlight_labor["annual_minutes"].iloc[0]),
                }
            )
        if not highlight_hours.empty:
            highlight_rows.append(
                {
                    "scenario": _t("sensitivity_table_label_hours"),
                    "required_rate": float(highlight_hours["required_rate"].iloc[0]),
                    "labor_share": float(highlight_hours["labor_share_pct"].iloc[0]),
                    "profit_share": float(highlight_hours["profit_margin_pct"].iloc[0]),
                    "annual_minutes": float(highlight_hours["annual_minutes"].iloc[0]),
                }
            )
        if highlight_rows:
            highlight_df = pd.DataFrame(highlight_rows).rename(
                columns={
                    "scenario": _t("scenario_table_col_label"),
                    "required_rate": _t("scenario_table_col_required_rate"),
                    "labor_share": _t("scenario_table_col_labor_share"),
                    "profit_share": _t("scenario_table_col_profit_share"),
                    "annual_minutes": _t("scenario_table_col_minutes"),
                }
            )
            highlight_style = highlight_df.style.format(
                {
                    _t("scenario_table_col_required_rate"): "{:.3f}",
                    _t("scenario_table_col_labor_share"): "{:.1f}%",
                    _t("scenario_table_col_profit_share"): "{:.1f}%",
                    _t("scenario_table_col_minutes"): "{:,.0f}",
                }
            )
            st.dataframe(highlight_style, use_container_width=True, hide_index=True)
            st.caption(_t("sensitivity_summary_caption"))

    st.subheader(_t("education_header"))
    st.caption(_t("education_caption"))
    for resource in EDUCATIONAL_RESOURCES:
        title, description = _resource_text(resource)
        if not title:
            continue
        if description:
            st.markdown(f"- [{title}]({resource['url']}) — {description}")
        else:
            st.markdown(f"- [{title}]({resource['url']})")

    st.subheader(_t("pdca_header"))
    st.caption(_t("pdca_caption"))
    pdca_log: list[dict[str, Any]] = st.session_state.setdefault("pdca_log", [])
    with st.form("pdca_log_form", clear_on_submit=True):
        options = _pdca_options()
        option_labels = [label for _, label in options]
        selected_stage_label = st.selectbox(
            _t("pdca_stage_label"),
            option_labels,
            index=0,
        )
        selected_stage_key = next(
            key for key, label in options if label == selected_stage_label
        )
        note_text = st.text_area(_t("pdca_note_label"), height=100)
        submitted = st.form_submit_button(_t("pdca_save_button"))
    if submitted:
        if note_text.strip():
            pdca_log.append(
                {
                    "stage": selected_stage_key,
                    "note": note_text.strip(),
                    "timestamp": datetime.now().isoformat(),
                }
            )
            st.session_state["pdca_log"] = pdca_log
            st.success(_t("pdca_saved_message"))
        else:
            st.warning(_t("pdca_note_required"))

    if pdca_log:
        st.markdown(f"**{_t('pdca_log_header')}**")
        log_df = pd.DataFrame(pdca_log)
        log_df["stage"] = log_df["stage"].apply(_stage_label)
        log_df["timestamp"] = pd.to_datetime(
            log_df["timestamp"], errors="coerce"
        ).dt.strftime("%Y-%m-%d %H:%M")
        log_df = log_df.rename(
            columns={
                "stage": _t("pdca_column_stage"),
                "note": _t("pdca_column_note"),
                "timestamp": _t("pdca_column_timestamp"),
            }
        )
        st.dataframe(log_df, use_container_width=True, hide_index=True)
    else:
        st.info(_t("pdca_empty"))

    st.markdown("#### 差異分析（標準 vs 実績）")
    variance_state = st.session_state.setdefault("sr_variance_inputs", {})
    default_actual_minutes = variance_state.get("actual_minutes", results["annual_minutes"])
    default_actual_rate = variance_state.get("actual_rate", results["required_rate"])
    var_cols = st.columns(2, gap="large")
    with var_cols[0]:
        actual_minutes = st.number_input(
            "実際稼働時間（分）",
            value=float(default_actual_minutes),
            min_value=0.0,
            step=1.0,
            help="分析対象期間の実際稼働時間。タイムカードや工程実績から入力します。",
        )
    with var_cols[1]:
        actual_rate = st.number_input(
            "実際賃率（円/分）",
            value=float(default_actual_rate),
            min_value=0.0,
            step=0.01,
            format="%.3f",
            help="実績の人件費÷実働時間で求めた実際賃率。賃率差異＝実際時間×(標準賃率−実際賃率)。",
        )

    standard_minutes = results["annual_minutes"]
    standard_rate = results["required_rate"]
    standard_labor_cost = standard_minutes * standard_rate
    actual_labor_cost = actual_minutes * actual_rate
    rate_variance = actual_minutes * (standard_rate - actual_rate)
    efficiency_variance = standard_rate * (standard_minutes - actual_minutes)
    total_variance = actual_labor_cost - standard_labor_cost

    variance_state.update(
        {
            "standard_minutes": standard_minutes,
            "standard_rate": standard_rate,
            "standard_labor_cost": standard_labor_cost,
            "actual_minutes": actual_minutes,
            "actual_rate": actual_rate,
            "actual_labor_cost": actual_labor_cost,
            "rate_variance": rate_variance,
            "efficiency_variance": efficiency_variance,
            "total_variance": total_variance,
        }
    )
    st.session_state["sr_variance_inputs"] = variance_state

    metric_cols = st.columns(2, gap="large")
    with metric_cols[0]:
        st.metric("標準時間（分）", f"{standard_minutes:.0f}")
        st.metric("標準賃率（円/分）", f"{standard_rate:.3f}")
    with metric_cols[1]:
        st.metric("標準労務費（円）", f"{standard_labor_cost:,.0f}")
        st.metric(
            "実際労務費（円）",
            f"{actual_labor_cost:,.0f}",
            delta=f"{total_variance:,.0f}",
            delta_color="inverse",
        )

    variance_df = pd.DataFrame(
        [
            {
                "指標": "賃率差異",
                "金額": rate_variance,
                "判定": classify_variance(rate_variance),
                "差異の考え方": "実際賃率との比較 (実際時間×(標準賃率−実際賃率))",
            },
            {
                "指標": "効率差異",
                "金額": efficiency_variance,
                "判定": classify_variance(efficiency_variance),
                "差異の考え方": "実際時間との比較 (標準賃率×(標準時間−実際時間))",
            },
            {
                "指標": "総差異",
                "金額": total_variance,
                "判定": classify_variance(total_variance),
                "差異の考え方": "賃率差異 + 効率差異",
            },
        ]
    )
    variance_style = variance_df.style.format({"金額": "{:+,.0f}"}).applymap(
        lambda v: "color:#1f8a5c;font-weight:700;" if v == "有利 (F)" else "color:#d64550;font-weight:700;" if v == "不利 (A)" else "",
        subset=["判定"],
    )
    st.dataframe(variance_style, use_container_width=True, hide_index=True)
    st.caption("標準労務費＝標準時間×標準賃率、賃率差異＝実際時間×(標準賃率−実際賃率)、効率差異＝標準賃率×(標準時間−実際時間)。")

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
        base_headcount = params["fulltime_workers"] + params["part1_workers"] + params["part2_workers"]
        base_part_ratio = (
            (params["part1_workers"] + params["part2_workers"]) / base_headcount * 100
            if base_headcount > 0
            else 0.0
        )
        plan_rows = []
        for sc in ai_plan_data:
            param_changes = sc.get("param_changes") or {}
            change_desc = ", ".join(f"{k}: {v}" for k, v in param_changes.items()) if param_changes else "－"
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

    st.subheader("感度分析（PDFエクスポート用）")
    static_fig = plot_sensitivity(params)
    with st.expander("固定グラフを表示", expanded=False):
        st.caption("PDF出力に含まれる感度分析の固定図です。")
        st.pyplot(static_fig)

    df_csv = pd.DataFrame(list(nodes.values()))
    df_csv["depends_on"] = df_csv["depends_on"].apply(lambda x: ",".join(x))
    csv = df_csv.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "CSVエクスポート",
        data=csv,
        file_name=f"standard_rate__{current}.csv",
        mime="text/csv",
    )

    pdf_bytes = generate_pdf(nodes, static_fig)
    st.download_button(
        "PDFエクスポート",
        data=pdf_bytes,
        file_name=f"standard_rate_summary__{current}.pdf",
        mime="application/pdf",
    )

    excel_bytes = build_excel_report(params, nodes, variance_state, variance_df)
    st.download_button(
        "Excelエクスポート",
        data=excel_bytes,
        file_name=f"standard_rate_report__{current}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

render_wizard_nav(
    current_step,
    location="bottom",
    next_disabled=should_block_next,
    next_disabled_help="必須項目を入力すると次のステップへ進めます。",
)

sync_offline_cache()
