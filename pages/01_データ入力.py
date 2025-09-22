import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
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

st.success("保存しました。上部のナビから『ダッシュボード』へ進んでください。")
