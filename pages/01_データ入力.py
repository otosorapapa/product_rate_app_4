import sys
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parents[1])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from datetime import date, datetime, timedelta
from io import BytesIO
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from data_integrations import (
    IntegrationConfig,
    SUPPORTED_ACCOUNTING_APPS,
    SUPPORTED_POS_SYSTEMS,
    auto_sync_transactions,
    apply_transaction_summary_to_products,
    extract_transaction_period,
    load_transactions_for_sync,
    summarize_transactions,
)
from utils import (
    generate_product_template,
    get_product_template_guide,
    get_template_field_anchor,
    get_template_field_info,
    infer_category_from_name,
    infer_major_customer,
    list_template_presets,
    parse_hyochin,
    parse_products,
    read_excel_safely,
    validate_product_dataframe,
)
from components import (
    apply_user_theme,
    render_glossary_popover,
    render_help_button,
    render_onboarding,
    render_page_tutorial,
    render_stepper,
    render_sidebar_nav,
)
from offline import (
    mark_restore_notice_shown,
    restore_session_state_from_cache,
    should_show_restore_notice,
    sync_offline_cache,
)


def _format_fermi_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number - round(number)) < 1e-6:
        return f"{int(round(number)):,}"
    return f"{number:,.2f}"


apply_user_theme()

render_sidebar_nav(page_key="data")

header_col, help_col = st.columns([0.78, 0.22], gap="small")
with header_col:
    st.title("① データ入力 & 取り込み")

render_help_button("data", container=help_col)

render_onboarding()
render_page_tutorial("data")
render_stepper(1)

restored_from_cache = False
if "df_products_raw" not in st.session_state:
    restored_from_cache = restore_session_state_from_cache()

if restored_from_cache and should_show_restore_notice():
    timestamp = st.session_state.get("offline_cache_timestamp")
    if timestamp:
        st.success(f"ブラウザに保存していた {timestamp} 時点のデータを復元しました。")
    else:
        st.success("ブラウザに保存していた直近のデータを復元しました。")
    st.caption("通信が不安定な環境でもサイドバーの『オフラインモード』から最新版を更新できます。")
    mark_restore_notice_shown()

template_header_col, template_help_col = st.columns([0.74, 0.26], gap="small")
with template_header_col:
    st.subheader("Excelテンプレート")
with template_help_col:
    render_glossary_popover(
        ["必要賃率", "ブレークイーブン賃率", "付加価値/分"],
        label="関連用語の解説",
        container=template_help_col,
    )

st.markdown(
    "テンプレートには必須項目の説明とサンプル値、入力単位を記載しています。"
    " 自社データを入力する前にダウンロードし、列名と単位を変更しないようご注意ください。"
    " アップロード時には必須項目の欠損や単位の誤りを自動チェックします。"
)

presets = list_template_presets()
preset_keys = [preset["key"] for preset in presets]
preset_label_map = {preset["key"]: preset["label"] for preset in presets}
default_key = st.session_state.get("template_preset_key", "general")
if default_key not in preset_label_map:
    default_key = "general"
default_index = preset_keys.index(default_key) if default_key in preset_keys else 0
selected_key = st.selectbox(
    "業種別のサンプル値を選択",
    options=preset_keys,
    index=default_index,
    key="template_preset_key",
    format_func=lambda key: preset_label_map.get(key, key),
)
selected_preset = next((preset for preset in presets if preset["key"] == selected_key), presets[0])

if selected_preset.get("description"):
    st.caption(f"想定シナリオ: {selected_preset['description']}")

fermi_records = selected_preset.get("fermi_assumptions") or []
if fermi_records:
    fermi_df = pd.DataFrame(fermi_records)
    fermi_display = fermi_df.copy()
    if "推定値" in fermi_display.columns:
        fermi_display["推定値"] = fermi_display["推定値"].apply(_format_fermi_value)
    st.dataframe(fermi_display, use_container_width=True)
    st.caption("※ 推定値は一般的なフェルミ推定です。自社の実績値で上書きしてください。")

guide_df = get_product_template_guide()
guide_display = guide_df[["Excel列名", "説明", "単位/形式", "必須", "サンプル値"]]
st.dataframe(guide_display, use_container_width=True)

with st.expander("列別の記入ポイントを詳しく見る", expanded=False):
    st.caption("テンプレート列と同じ順番で記入方法をまとめています。")
    for row in guide_df.to_dict("records"):
        anchor = row.get("アンカー")
        if anchor:
            st.markdown(f"<div id='{anchor}'></div>", unsafe_allow_html=True)
        st.markdown(f"**{row['Excel列名']}** ({row['単位/形式']})")
        st.caption(row["説明"])
        st.caption(f"サンプル値: {row['サンプル値']}")

template_bytes = generate_product_template(selected_key)
st.download_button(
    "📄 製品マスタ入力テンプレートをダウンロード",
    data=template_bytes,
    file_name=f"product_master_template_{selected_key}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.caption("※『標賃』シートにもサンプル値を用意しています。必要に応じて実績値へ置き換えてください。")

st.divider()

default_path = "data/sample.xlsx"
file = st.file_uploader("Excelをアップロード（未指定ならサンプルを使用）", type=["xlsx"])

if file is not None or "df_products_raw" not in st.session_state:
    if file is None:
        st.info("サンプルデータを使用します。")
        xls = read_excel_safely(default_path)
        st.session_state["using_sample_data"] = True
    else:
        xls = read_excel_safely(file)
        st.session_state["using_sample_data"] = False

    if xls is None:
        st.error("Excel 読込に失敗しました。ファイル形式・シート名をご確認ください。")
        st.stop()

    with st.spinner("『標賃』を解析中..."):
        calc_params, sr_params, warn1 = parse_hyochin(xls)

    with st.spinner("『R6.12』製品データを解析中..."):
        df_products, warn2 = parse_products(xls, sheet_name="R6.12")

    for w in (warn1 + warn2):
        st.warning(w)

    if st.session_state.get("using_sample_data"):
        df_products = df_products.copy()
        if "category" not in df_products.columns or df_products["category"].isna().all():
            df_products["category"] = df_products["product_name"].apply(
                infer_category_from_name
            )
        if "major_customer" not in df_products.columns or df_products["major_customer"].isna().all():
            df_products["major_customer"] = [
                infer_major_customer(no, name)
                for no, name in zip(
                    df_products.get("product_no"), df_products.get("product_name")
                )
            ]
        st.caption(
            "※ サンプルデータには製品名から推定したカテゴリーと主要顧客を自動付与しています。"
        )

        unit_info = df_products.attrs.get("column_unit_info")
        if not isinstance(unit_info, dict):
            unit_info = {}

        def _ensure_unit(column_key: str, unit_label: str) -> None:
            entry = unit_info.get(column_key)
            if not isinstance(entry, dict):
                entry = {"unit": None, "source": "サンプルデータ"}
            if not entry.get("unit"):
                entry["unit"] = unit_label
            entry.setdefault("source", "サンプルデータ")
            unit_info[column_key] = entry

        sample_unit_defaults = {
            "product_no": "コード",
            "product_name": "テキスト",
            "actual_unit_price": "円/個",
            "material_unit_cost": "円/個",
            "minutes_per_unit": "分/個",
            "daily_qty": "個/日",
        }
        for key, unit_label in sample_unit_defaults.items():
            _ensure_unit(key, unit_label)

        df_products.attrs["column_unit_info"] = unit_info

    column_unit_info = df_products.attrs.get("column_unit_info")
    if not isinstance(column_unit_info, dict):
        column_unit_info = {}

    column_labels = {
        "product_no": "製品番号",
        "product_name": "製品名",
        "actual_unit_price": "販売単価（円/個）",
        "material_unit_cost": "材料費（円/個）",
        "minutes_per_unit": "分/個",
        "daily_qty": "日産数（個/日）",
        "category": "カテゴリー",
        "major_customer": "主要顧客",
        "notes": "備考",
    }
    derived_labels = {
        "gp_per_unit": "粗利（円/個）",
        "daily_total_minutes": "日産合計時間（分）",
        "daily_va": "日次付加価値（円）",
        "va_per_min": "付加価値/分（円）",
    }
    number_formats = {
        "actual_unit_price": "%.2f",
        "material_unit_cost": "%.2f",
        "minutes_per_unit": "%.2f",
        "daily_qty": "%.0f",
        "gp_per_unit": "%.2f",
        "daily_total_minutes": "%.2f",
        "daily_va": "%.2f",
        "va_per_min": "%.2f",
    }
    derived_help = {
        "gp_per_unit": "販売単価−材料費で自動計算されます。",
        "daily_total_minutes": "分/個×日産数で自動計算されます。",
        "daily_va": "粗利×日産数で自動計算されます。",
        "va_per_min": "日次付加価値÷合計時間で自動計算されます。",
    }

    column_config: Dict[str, Any] = {}
    for col, label in column_labels.items():
        if col not in df_products.columns:
            continue
        info = get_template_field_info(label)
        help_text = info.get("説明") if info else None
        if col in {"actual_unit_price", "material_unit_cost", "minutes_per_unit", "daily_qty"}:
            column_config[col] = st.column_config.NumberColumn(
                label,
                help=help_text,
                format=number_formats.get(col),
            )
        else:
            column_config[col] = st.column_config.TextColumn(label, help=help_text)
    for col, label in derived_labels.items():
        if col not in df_products.columns:
            continue
        column_config[col] = st.column_config.NumberColumn(
            label,
            format=number_formats.get(col),
            help=derived_help.get(col),
            disabled=True,
        )

    edited_df = df_products
    with st.expander("アプリ内で直接編集・追加入力", expanded=False):
        st.caption(
            "Excelに戻らずに主要列を更新できます。数値はテンプレートと同じ単位で入力してください。"
        )
        edited_df = st.data_editor(
            df_products,
            num_rows="dynamic",
            use_container_width=True,
            key="inline_products_editor",
            column_config=column_config,
            hide_index=True,
        )
        export_buffer = BytesIO()
        export_df = edited_df.copy()
        export_df.to_excel(export_buffer, index=False, sheet_name="products")
        st.download_button(
            "編集内容をExcelでエクスポート",
            data=export_buffer.getvalue(),
            file_name="edited_products.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    if isinstance(edited_df, pd.DataFrame):
        df_products = edited_df.copy()
    else:
        df_products = pd.DataFrame(edited_df)
    df_products.attrs["column_unit_info"] = column_unit_info

    numeric_cols = [
        "actual_unit_price",
        "material_unit_cost",
        "minutes_per_unit",
        "daily_qty",
        "gp_per_unit",
        "daily_total_minutes",
        "daily_va",
        "va_per_min",
    ]
    for col in numeric_cols:
        if col in df_products.columns:
            df_products[col] = pd.to_numeric(df_products[col], errors="coerce")

    if {"actual_unit_price", "material_unit_cost"}.issubset(df_products.columns):
        df_products["gp_per_unit"] = df_products["actual_unit_price"] - df_products["material_unit_cost"]
    if {"minutes_per_unit", "daily_qty"}.issubset(df_products.columns):
        df_products["daily_total_minutes"] = df_products["minutes_per_unit"] * df_products["daily_qty"]
    if {"gp_per_unit", "daily_qty"}.issubset(df_products.columns):
        df_products["daily_va"] = df_products["gp_per_unit"] * df_products["daily_qty"]
    if {"daily_va", "daily_total_minutes"}.issubset(df_products.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            df_products["va_per_min"] = df_products["daily_va"] / df_products["daily_total_minutes"].replace(
                {0: np.nan}
            )

    errors, val_warnings, detail_df = validate_product_dataframe(df_products)
    for msg in val_warnings:
        st.warning(msg)
    for msg in errors:
        st.error(msg)

    if not detail_df.empty:
        level_map = {"エラー": "致命的", "警告": "注意"}
        display_df = detail_df.copy()
        display_df["重要度"] = display_df["レベル"].map(level_map).fillna(display_df["レベル"])
        display_df["Excel列"] = display_df["項目"].map(
            lambda label: (info := get_template_field_info(label)) and info.get("excel_column")
        )
        display_df["入力ガイド"] = display_df["項目"].map(
            lambda label: get_template_field_anchor(label) or ""
        )
        display_df = display_df.drop(columns=["レベル"])
        ordered_columns = [
            "重要度",
            "製品番号",
            "製品名",
            "項目",
            "Excel列",
            "原因/状況",
            "入力値",
            "対処方法",
            "入力ガイド",
        ]
        display_df = display_df[[col for col in ordered_columns if col in display_df.columns]]

        def _highlight_issue(row: pd.Series) -> List[str]:
            level = row.get("重要度")
            color = "#FDEAEA" if level == "致命的" else "#FFF7E1"
            return [f"background-color: {color}"] * len(row)

        def _format_anchor_cell(anchor: str) -> str:
            if not anchor:
                return "―"
            return f'<a href="#{anchor}">テンプレート列へ移動</a>'

        styled = display_df.style.apply(_highlight_issue, axis=1).format(
            {"入力ガイド": _format_anchor_cell}, escape=False
        )
        with st.expander("検知されたデータ品質アラートの詳細", expanded=bool(errors)):
            st.caption("致命的な項目は赤、注意項目は黄でハイライトしています。")
            st.markdown(styled.to_html(index=False), unsafe_allow_html=True)
            option_labels = [
                f"{row.get('重要度', '-') } | {row.get('項目', '-') } | {row.get('製品番号', '全体')}"
                for row in display_df.to_dict("records")
            ]
            if option_labels:
                with st.popover("🛠️ 修正ヒントを確認"):
                    st.caption("対象行を選ぶと原因と修正方法を表示します。")
                    selected_index = st.selectbox(
                        "対象行",
                        options=list(range(len(option_labels))),
                        format_func=lambda idx: option_labels[idx],
                    )
                    issue_row = detail_df.iloc[int(selected_index)]
                    st.markdown(
                        f"**対象製品:** {issue_row.get('製品番号', '全体')} {issue_row.get('製品名', '')}"
                    )
                    st.markdown(f"**原因/状況:** {issue_row.get('原因/状況')}")
                    st.markdown(f"**対処方法:** {issue_row.get('対処方法')}")
                    anchor = get_template_field_anchor(issue_row.get("項目", ""))
                    info = get_template_field_info(issue_row.get("項目", ""))
                    if anchor:
                        label_text = info.get("excel_column") if info else issue_row.get("項目")
                        st.markdown(f"[テンプレート『{label_text}』の説明に移動](#{anchor})")

    if errors:
        st.stop()
    elif not val_warnings:
        st.success("データ品質チェックを通過しました。")

    st.session_state["sr_params"] = sr_params
    st.session_state["df_products_raw"] = df_products
    st.session_state["calc_params"] = calc_params
else:
    sr_params = st.session_state["sr_params"]
    df_products = st.session_state["df_products_raw"]
    calc_params = st.session_state.get("calc_params", {})

if "scenarios" not in st.session_state:
    st.session_state["scenarios"] = {"ベース": sr_params.copy()}
    st.session_state["current_scenario"] = "ベース"
else:
    st.session_state["scenarios"]["ベース"] = sr_params.copy()
if "current_scenario" not in st.session_state:
    st.session_state["current_scenario"] = "ベース"

st.session_state.setdefault("external_sync_history", [])
st.session_state.setdefault("external_sync_last_summary", [])
st.session_state.setdefault("external_sync_last_transactions", None)
st.session_state.setdefault("external_sync_last_run_logs", [])

st.caption(f"適用中シナリオ: {st.session_state['current_scenario']}")

c1, c2, c3 = st.columns(3)
c1.metric("固定費計 (千円/年)", f"{calc_params.get('fixed_total', 0)/1000:,.0f}")
c2.metric("必要利益計 (千円/年)", f"{calc_params.get('required_profit_total', 0)/1000:,.0f}")
c3.metric("年間標準稼働分 (分/年)", f"{calc_params.get('annual_minutes', 0):,.0f}")

st.divider()
st.subheader("製品データ")
keyword = st.text_input("製品番号または名称で検索", "")
if keyword:
    mask = (
        df_products["product_no"].astype(str).str.contains(keyword, regex=False)
        | df_products["product_name"].astype(str).str.contains(keyword, regex=False)
    )
    df_view = df_products[mask]
else:
    df_view = df_products
st.dataframe(df_view, use_container_width=True)

with st.expander("新規製品を追加", expanded=False):
    with st.form("add_product_form"):
        col_a, col_b = st.columns(2)
        product_no = col_a.text_input("製品番号", "")
        product_name = col_b.text_input("製品名", "")
        actual_unit_price = st.number_input("実際売単価", value=0.0, step=1.0)
        material_unit_cost = st.number_input("材料原価", value=0.0, step=1.0)
        minutes_per_unit = st.number_input("分/個", value=0.0, step=0.1)
        daily_qty = st.number_input("日産数", value=0.0, step=1.0)
        col_c, col_d = st.columns(2)
        category = col_c.text_input("カテゴリー", "")
        major_customer = col_d.text_input("主要顧客", "")
        submitted = st.form_submit_button("追加")
    if submitted:
        gp_per_unit = actual_unit_price - material_unit_cost
        daily_total_minutes = minutes_per_unit * daily_qty
        daily_va = gp_per_unit * daily_qty
        va_per_min = daily_va / daily_total_minutes if daily_total_minutes else 0.0
        new_row = {
            "product_no": product_no,
            "product_name": product_name,
            "actual_unit_price": actual_unit_price,
            "material_unit_cost": material_unit_cost,
            "minutes_per_unit": minutes_per_unit,
            "daily_qty": daily_qty,
            "daily_total_minutes": daily_total_minutes,
            "gp_per_unit": gp_per_unit,
            "daily_va": daily_va,
            "va_per_min": va_per_min,
            "category": category,
            "major_customer": major_customer,
        }
        df_products = pd.concat([df_products, pd.DataFrame([new_row])], ignore_index=True)
        st.session_state["df_products_raw"] = df_products
        st.success("製品を追加しました。")
        st.rerun()

st.divider()
st.subheader("生産・販売システムとのデータ連携")
st.caption(
    "弥生会計やfreeeなどの会計ソフト、POSシステムから売上・原価データを同期し、"
    "最新の製品指標を自動反映できます。"
)

feedback = st.session_state.pop("external_sync_feedback", None)
if feedback:
    level = feedback.get("level", "info")
    message = feedback.get("message", "")
    display = getattr(st, level, st.info)
    if message:
        display(message)

latest_summary_records = st.session_state.get("external_sync_last_summary") or []
if latest_summary_records:
    latest_summary_df = pd.DataFrame(latest_summary_records)
    if not latest_summary_df.empty:
        st.markdown("**直近の同期結果（SKU別集計）**")
        st.dataframe(latest_summary_df, use_container_width=True)

latest_transactions = st.session_state.get("external_sync_last_transactions")
if isinstance(latest_transactions, pd.DataFrame) and not latest_transactions.empty:
    with st.expander("直近の取り込み明細を表示", expanded=False):
        st.dataframe(latest_transactions, use_container_width=True)

latest_run_logs = st.session_state.get("external_sync_last_run_logs") or []
if latest_run_logs:
    log_df = pd.DataFrame(latest_run_logs)
    if not log_df.empty:
        st.markdown("**取り込みシステム別ステータス**")
        st.dataframe(log_df, use_container_width=True)

tab_api, tab_csv = st.tabs(["API連携", "CSV自動取り込み"])

with tab_api:
    st.write(
        "APIトークンを登録するとサンプルAPIから指定期間の売上・原価データを取得できます。"
        " 本番環境では各システムのAPIクレデンシャルを設定してください。"
    )
    api_system_label = st.radio(
        "連携するシステムの種別",
        options=("会計ソフト", "POS"),
        horizontal=True,
        key="api_system_type",
    )
    api_source_type = "accounting" if api_system_label == "会計ソフト" else "pos"
    api_vendor_options = (
        list(SUPPORTED_ACCOUNTING_APPS.keys())
        if api_source_type == "accounting"
        else list(SUPPORTED_POS_SYSTEMS.keys())
    )
    api_vendor = st.selectbox("サービスを選択", api_vendor_options, key="api_vendor")
    api_token = st.text_input("APIトークン (任意)", value="", type="password")
    today = date.today()
    default_start = today - timedelta(days=6)
    api_period = st.date_input(
        "同期期間",
        value=(default_start, today),
        key="api_period",
    )
    if isinstance(api_period, (tuple, list)):
        api_start, api_end = api_period
    else:
        api_start = api_end = api_period
    api_frequency_label = st.selectbox(
        "同期頻度",
        options=["日次", "週次", "月次"],
        index=0 if api_source_type == "pos" else 2,
        key="api_frequency",
    )
    if st.button("APIから同期", key="api_sync_button"):
        config = IntegrationConfig(
            source_type=api_source_type,
            vendor=api_vendor,
            credential_key=api_token or None,
            schedule={"日次": "daily", "週次": "weekly", "月次": "monthly"}.get(
                api_frequency_label, "daily"
            ),
        )
        try:
            transactions = load_transactions_for_sync(
                config, start_date=api_start, end_date=api_end
            )
        except FileNotFoundError:
            st.error(
                "サンプルAPIレスポンスが見つかりません。CSV取り込みを使用してください。"
            )
            st.session_state["external_sync_last_run_logs"] = []
        except ValueError as exc:
            st.error(str(exc))
            st.session_state["external_sync_last_run_logs"] = []
        else:
            summary = summarize_transactions(transactions)
            if summary.empty:
                st.warning("同期対象データが見つかりませんでした。期間を見直してください。")
                st.session_state["external_sync_last_run_logs"] = []
            else:
                updated_df = apply_transaction_summary_to_products(
                    df_products,
                    summary,
                    vendor=api_vendor,
                    synced_at=datetime.now(),
                )
                st.session_state["df_products_raw"] = updated_df
                st.session_state["external_sync_last_summary"] = summary.to_dict(
                    "records"
                )
                st.session_state["external_sync_last_transactions"] = (
                    transactions.head(500).reset_index(drop=True)
                )
                st.session_state["external_sync_last_run_logs"] = [
                    {
                        "vendor": api_vendor,
                        "source_type": api_source_type,
                        "schedule": config.schedule,
                        "status": "success",
                        "records": int(len(transactions)),
                        "period_start": api_start.isoformat()
                        if isinstance(api_start, date)
                        else None,
                        "period_end": api_end.isoformat()
                        if isinstance(api_end, date)
                        else None,
                    }
                ]
                period_start, period_end = extract_transaction_period(transactions)
                st.session_state["external_sync_history"].append(
                    {
                        "synced_at": datetime.now().isoformat(timespec="seconds"),
                        "mode": "API",
                        "source_type": api_system_label,
                        "vendor": api_vendor,
                        "records": int(len(transactions)),
                        "updated_products": int(summary["product_no"].nunique()),
                        "period_start": period_start.isoformat()
                        if period_start
                        else None,
                        "period_end": period_end.isoformat() if period_end else None,
                        "frequency": api_frequency_label,
                    }
                )
                st.session_state["external_sync_feedback"] = {
                    "level": "success",
                    "message": f"{api_vendor}から{int(summary['product_no'].nunique())}件のSKUを同期しました。",
                }
                st.rerun()

    st.divider()
    st.markdown("### 日次自動取得")
    st.caption(
        "対象システムをまとめて指定すると、選択期間の売上・原価データを一括取得します。"
        " クラウド連携で手動入力を減らし、常に最新の実績で分析できます。"
    )

    auto_accounting_targets = st.multiselect(
        "対象の会計ソフト",
        options=list(SUPPORTED_ACCOUNTING_APPS.keys()),
        default=list(SUPPORTED_ACCOUNTING_APPS.keys()),
        key="auto_accounting_targets",
    )
    auto_pos_targets = st.multiselect(
        "対象のPOSシステム",
        options=list(SUPPORTED_POS_SYSTEMS.keys()),
        default=list(SUPPORTED_POS_SYSTEMS.keys()),
        key="auto_pos_targets",
    )
    auto_days = st.slider(
        "取得期間（日数）",
        min_value=1,
        max_value=31,
        value=7,
        help="通常は直近7日間を取得します。必要に応じて日数を調整してください。",
        key="auto_days",
    )
    if st.button("選択したシステムから自動取得", key="auto_sync_button"):
        targets: list[IntegrationConfig] = []
        for vendor in auto_accounting_targets:
            targets.append(
                IntegrationConfig(
                    source_type="accounting",
                    vendor=vendor,
                    schedule="daily",
                )
            )
        for vendor in auto_pos_targets:
            targets.append(
                IntegrationConfig(
                    source_type="pos",
                    vendor=vendor,
                    schedule="daily",
                )
            )

        if not targets:
            st.warning("自動取得するシステムを選択してください。")
        else:
            auto_end = today
            auto_start = today - timedelta(days=int(auto_days) - 1)
            try:
                result = auto_sync_transactions(
                    targets, start_date=auto_start, end_date=auto_end
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                transactions = result.transactions
                summary = result.summary
                st.session_state["external_sync_last_run_logs"] = result.logs

                if summary.empty or transactions.empty:
                    st.warning(
                        "選択した期間・システムに同期対象データが見つかりませんでした。"
                    )
                else:
                    updated_df = apply_transaction_summary_to_products(
                        df_products,
                        summary,
                        vendor="自動同期",
                        synced_at=datetime.now(),
                    )
                    st.session_state["df_products_raw"] = updated_df
                    st.session_state["external_sync_last_summary"] = summary.to_dict(
                        "records"
                    )
                    st.session_state["external_sync_last_transactions"] = (
                        transactions.head(500).reset_index(drop=True)
                    )
                    period_start, period_end = extract_transaction_period(transactions)
                    joined_vendors = ", ".join(
                        log["vendor"]
                        for log in result.logs
                        if log.get("status") == "success"
                    ) or "対象なし"
                    st.session_state["external_sync_history"].append(
                        {
                            "synced_at": datetime.now().isoformat(timespec="seconds"),
                            "mode": "AUTO",
                            "source_type": "自動連携",
                            "vendor": joined_vendors,
                            "records": int(len(transactions)),
                            "updated_products": int(
                                summary["product_no"].nunique()
                            ),
                            "period_start": period_start.isoformat()
                            if period_start
                            else None,
                            "period_end": period_end.isoformat()
                            if period_end
                            else None,
                            "frequency": "日次",
                        }
                    )
                    st.session_state["external_sync_feedback"] = {
                        "level": "success",
                        "message": (
                            f"{joined_vendors}から{int(summary['product_no'].nunique())}件のSKUを"
                            "自動更新しました。"
                        ),
                    }
                    st.rerun()

with tab_csv:
    st.write(
        "フォルダに出力されたCSVを指定すると、自動でヘッダーを解析し日次・月次の更新を取り込めます。"
    )
    csv_system_label = st.radio(
        "取り込み元の種別",
        options=("会計ソフト", "POS"),
        horizontal=True,
        key="csv_system_type",
    )
    csv_source_type = "accounting" if csv_system_label == "会計ソフト" else "pos"
    csv_vendor_options = (
        list(SUPPORTED_ACCOUNTING_APPS.keys())
        if csv_source_type == "accounting"
        else list(SUPPORTED_POS_SYSTEMS.keys())
    )
    csv_vendor = st.selectbox("取込フォーマット", csv_vendor_options, key="csv_vendor")
    csv_file = st.file_uploader(
        "売上・原価CSVをアップロード", type=["csv"], key="csv_file"
    )
    csv_frequency_label = st.selectbox(
        "更新頻度",
        options=["日次", "週次", "月次"],
        index=0 if csv_source_type == "pos" else 2,
        key="csv_frequency",
    )
    if st.button("CSVを取り込む", key="csv_sync_button", disabled=csv_file is None):
        if csv_file is None:
            st.error("CSVファイルを選択してください。")
        else:
            config = IntegrationConfig(source_type=csv_source_type, vendor=csv_vendor)
            try:
                transactions = load_transactions_for_sync(config, csv_file=csv_file)
            except ValueError as exc:
                st.error(str(exc))
                st.session_state["external_sync_last_run_logs"] = []
            else:
                summary = summarize_transactions(transactions)
                if summary.empty:
                    st.warning("CSVに同期対象データが含まれていませんでした。")
                    st.session_state["external_sync_last_run_logs"] = []
                else:
                    updated_df = apply_transaction_summary_to_products(
                        df_products,
                        summary,
                        vendor=csv_vendor,
                        synced_at=datetime.now(),
                    )
                    st.session_state["df_products_raw"] = updated_df
                    st.session_state["external_sync_last_summary"] = summary.to_dict(
                        "records"
                    )
                    st.session_state["external_sync_last_transactions"] = (
                        transactions.head(500).reset_index(drop=True)
                    )
                    st.session_state["external_sync_last_run_logs"] = [
                        {
                            "vendor": csv_vendor,
                            "source_type": csv_source_type,
                            "schedule": config.schedule,
                            "status": "success",
                            "records": int(len(transactions)),
                        }
                    ]
                    period_start, period_end = extract_transaction_period(transactions)
                    st.session_state["external_sync_history"].append(
                        {
                            "synced_at": datetime.now().isoformat(timespec="seconds"),
                            "mode": "CSV",
                            "source_type": csv_system_label,
                            "vendor": csv_vendor,
                            "records": int(len(transactions)),
                            "updated_products": int(summary["product_no"].nunique()),
                            "period_start": period_start.isoformat()
                            if period_start
                            else None,
                            "period_end": period_end.isoformat()
                            if period_end
                            else None,
                            "frequency": csv_frequency_label,
                        }
                    )
                    st.session_state["external_sync_feedback"] = {
                        "level": "success",
                        "message": f"CSVから{int(summary['product_no'].nunique())}件のSKUを更新しました。",
                    }
                    st.rerun()

history = st.session_state.get("external_sync_history", [])
if history:
    history_df = pd.DataFrame(history)
    if not history_df.empty:
        history_df = history_df.sort_values("synced_at", ascending=False)
        st.markdown("**同期履歴**")
        st.dataframe(history_df, use_container_width=True)

st.success("保存しました。上部のナビから『ダッシュボード』へ進んでください。")

sync_offline_cache()
