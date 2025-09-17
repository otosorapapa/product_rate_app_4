from __future__ import annotations

import pandas as pd
import streamlit as st

from calculations import (
    DEFAULT_OPERATIONAL_INPUTS,
    calculate_product_metrics,
    compute_operational_metrics,
)
from data_loader import (
    append_quality_flags,
    load_sample_data,
    load_tabular_data,
    validate_dataframe,
    detect_anomalies,
    sanitize_dataframe,
)
from dashboard import render_dashboard
from sensitivity import render_sensitivity_form
from ui_utils import apply_page_config, render_sidebar, select_language

apply_page_config()
translator = select_language()

if "product_data" not in st.session_state:
    st.session_state["product_data"] = None
if "operational_params" not in st.session_state:
    st.session_state["operational_params"] = DEFAULT_OPERATIONAL_INPUTS.copy()
if "scenarios" not in st.session_state:
    st.session_state["scenarios"] = {}
if "current_scenario" not in st.session_state:
    st.session_state["current_scenario"] = "ベース"

current_view = render_sidebar(translator, st.session_state["current_scenario_view"] if "current_scenario_view" in st.session_state else "data")
st.session_state["current_scenario_view"] = current_view


def _update_base_scenario(df: pd.DataFrame) -> None:
    """セッションのベースシナリオを最新のデータと指標で更新する."""

    params = st.session_state["operational_params"]
    try:
        metrics = compute_operational_metrics(params)
    except ValueError as exc:
        st.error(f"前提の計算に失敗しました: {exc}")
        return
    product_metrics = calculate_product_metrics(
        df,
        break_even_rate=metrics.break_even_rate,
        required_rate=metrics.required_rate,
    )
    st.session_state["scenarios"]["ベース"] = {
        "products": product_metrics,
        "metrics": metrics,
        "params": params.copy(),
    }
    st.session_state["current_scenario"] = st.session_state.get("current_scenario", "ベース")


def render_data_input() -> None:
    st.title(translator("nav_data"))
    st.caption(translator("data_intro_caption"))

    sample_clicked = st.button(translator("sample_button"))
    uploaded_file = st.file_uploader(
        translator("upload_label"),
        type=["xlsx", "xls", "csv", "json"],
    )

    data = st.session_state.get("product_data")
    if sample_clicked:
        try:
            data = load_sample_data()
            st.session_state["product_data"] = data
            st.success("サンプルデータを読み込みました。必要に応じて編集してください。")
        except Exception as exc:
            st.error(f"サンプルデータの読み込みに失敗しました: {exc}")
    elif uploaded_file is not None:
        try:
            data = load_tabular_data(uploaded_file)
            st.session_state["product_data"] = data
            st.success("アップロードに成功しました。欠損を確認してください。")
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"データ読込中にエラーが発生しました: {exc}")

    if data is None:
        st.info(translator("no_data"))
        return

    flagged = append_quality_flags(data)
    edited = st.data_editor(
        flagged,
        num_rows="dynamic",
        use_container_width=True,
        key="product_editor",
        hide_index=True,
        column_config={
            "row_alert": st.column_config.Column(
                "品質アラート",
                help="欠損やマイナス値を検知した行は赤色でハイライトされます。",
                disabled=True,
            )
        },
    )
    cleaned_df = sanitize_dataframe(edited.drop(columns=["row_alert"], errors="ignore"))
    st.session_state["product_data"] = cleaned_df

    validation = validate_dataframe(cleaned_df)
    st.subheader(translator("quality_header"))
    if validation.issues.empty:
        st.success(translator("quality_ok"))
    else:
        st.dataframe(validation.issues, use_container_width=True)
    if not validation.summary.empty:
        st.dataframe(validation.summary, use_container_width=True)

    params = st.session_state["operational_params"].copy()
    with st.expander(translator("operational_expander"), expanded=False):
        with st.form("operational_form"):
            col1, col2 = st.columns(2)
            params["labor_cost"] = col1.number_input("労務費 (円/年)", value=float(params.get("labor_cost", 0.0)), step=100000.0)
            params["sga_cost"] = col1.number_input("販管費 (円/年)", value=float(params.get("sga_cost", 0.0)), step=100000.0)
            params["loan_repayment"] = col1.number_input("借入返済 (円/年)", value=float(params.get("loan_repayment", 0.0)), step=50000.0)
            params["tax_payment"] = col1.number_input("納税・納付 (円/年)", value=float(params.get("tax_payment", 0.0)), step=50000.0)
            params["future_business"] = col1.number_input("未来投資費 (円/年)", value=float(params.get("future_business", 0.0)), step=50000.0)
            params["fulltime_workers"] = col2.number_input("正社員数", value=float(params.get("fulltime_workers", 1.0)), step=0.5)
            params["part1_workers"] = col2.number_input("パート①人数", value=float(params.get("part1_workers", 0.0)), step=0.5)
            params["part2_workers"] = col2.number_input("パート②人数", value=float(params.get("part2_workers", 0.0)), step=0.5)
            params["part2_coefficient"] = col2.slider("パート②労働係数", min_value=0.0, max_value=1.0, value=float(params.get("part2_coefficient", 0.0)), step=0.05)
            params["working_days"] = col2.number_input("年間稼働日数", value=float(params.get("working_days", 236.0)), step=1.0)
            params["daily_hours"] = col2.number_input("1日稼働時間", value=float(params.get("daily_hours", 8.68)), step=0.25)
            params["operation_rate"] = col2.slider("操業度", min_value=0.3, max_value=1.2, value=float(params.get("operation_rate", 0.75)), step=0.05)
            submitted = st.form_submit_button("前提を保存")
        if submitted:
            st.session_state["operational_params"] = params
            st.success("前提条件を更新しました。")

    _update_base_scenario(cleaned_df)
    base_payload = st.session_state["scenarios"].get("ベース")
    if base_payload:
        metrics = base_payload["metrics"]
        st.subheader(translator("base_metrics_header"))
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric(translator("kpi_required_rate"), f"{metrics.required_rate:,.2f} 円/分")
        col_b.metric(translator("kpi_break_even"), f"{metrics.break_even_rate:,.2f} 円/分")
        col_c.metric(translator("kpi_annual_minutes"), f"{metrics.annual_minutes:,.0f} 分")
        col_d.metric(translator("kpi_net_workers"), f"{metrics.net_workers:,.2f} 人")


def render_validation_and_sensitivity() -> None:
    st.title(translator("nav_validation"))
    df = st.session_state.get("product_data")
    if df is None:
        st.info(translator("no_data"))
        return

    validation = validate_dataframe(df)
    st.subheader(translator("quality_header"))
    if validation.issues.empty:
        st.success(translator("quality_ok"))
    else:
        st.dataframe(validation.issues, use_container_width=True)

    anomalies = detect_anomalies(df)
    if anomalies.empty:
        st.success(translator("no_anomaly"))
    else:
        st.subheader(translator("anomaly_header"))
        st.dataframe(anomalies, use_container_width=True)

    updated_params, base_metrics, updated_metrics, _ = render_sensitivity_form(
        st.session_state["operational_params"], translator
    )
    st.session_state["last_sensitivity_params"] = updated_params

    default_name = "新しいシナリオ" if translator.language == "ja" else "New scenario"
    scenario_name = st.text_input(translator("scenario_name_label"), value=default_name)
    if st.button(translator("scenario_save_button")):
        product_metrics = calculate_product_metrics(
            st.session_state["product_data"],
            break_even_rate=updated_metrics.break_even_rate,
            required_rate=updated_metrics.required_rate,
        )
        st.session_state["scenarios"][scenario_name] = {
            "products": product_metrics,
            "metrics": updated_metrics,
            "params": updated_params.copy(),
        }
        st.session_state["current_scenario"] = scenario_name
        st.success(translator("save_success").format(name=scenario_name))


def render_dashboard_view() -> None:
    st.title(translator("nav_dashboard"))
    if not st.session_state["scenarios"]:
        st.info(translator("no_data"))
        return
    render_dashboard(st.session_state["current_scenario"], st.session_state["scenarios"], translator)


if current_view == "data":
    render_data_input()
elif current_view == "validation":
    render_validation_and_sensitivity()
else:
    render_dashboard_view()
