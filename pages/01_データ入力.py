import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from datetime import date, datetime, timedelta

import streamlit as st
import pandas as pd
from data_integrations import (
    IntegrationConfig,
    SUPPORTED_ACCOUNTING_APPS,
    SUPPORTED_POS_SYSTEMS,
    apply_transaction_summary_to_products,
    extract_transaction_period,
    load_transactions_for_sync,
    summarize_transactions,
)
from utils import (
    read_excel_safely,
    parse_hyochin,
    parse_products,
    generate_product_template,
    get_product_template_guide,
    validate_product_dataframe,
    infer_category_from_name,
    infer_major_customer,
)
from components import (
    apply_user_theme,
    render_help_button,
    render_onboarding,
    render_page_tutorial,
    render_stepper,
    render_sidebar_nav,
)

apply_user_theme()

st.title("① データ入力 & 取り込み")
render_sidebar_nav(page_key="data")
render_help_button("data")

render_onboarding()
render_page_tutorial("data")
render_stepper(1)

st.subheader("Excelテンプレート")
st.markdown(
    "テンプレートには必須項目の説明とサンプル値を記載しています。"
    " 自社データを入力する前にダウンロードし、列名と単位を変更しないようご注意ください。"
)

guide_df = get_product_template_guide()
st.table(guide_df)

template_bytes = generate_product_template()
st.download_button(
    "📄 製品マスタ入力テンプレートをダウンロード",
    data=template_bytes,
    file_name="product_master_template.xlsx",
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

    errors, val_warnings, detail_df = validate_product_dataframe(df_products)
    for msg in val_warnings:
        st.warning(msg)
    for msg in errors:
        st.error(msg)

    if not detail_df.empty:
        with st.expander("検知されたデータ品質アラートの詳細", expanded=bool(errors)):
            st.dataframe(detail_df, use_container_width=True)

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
        except ValueError as exc:
            st.error(str(exc))
        else:
            summary = summarize_transactions(transactions)
            if summary.empty:
                st.warning("同期対象データが見つかりませんでした。期間を見直してください。")
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
            else:
                summary = summarize_transactions(transactions)
                if summary.empty:
                    st.warning("CSVに同期対象データが含まれていませんでした。")
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
