import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
from utils import read_excel_safely, parse_hyochin, parse_products
from components import render_stepper, render_sidebar_nav

st.title("① データ入力 & 取り込み")
render_sidebar_nav()
render_stepper(1)

default_path = "data/sample.xlsx"
file = st.file_uploader("Excelをアップロード（未指定ならサンプルを使用）", type=["xlsx"])

if file is not None or "df_products_raw" not in st.session_state:
    if file is None:
        st.info("サンプルデータを使用します。")
        xls = read_excel_safely(default_path)
    else:
        xls = read_excel_safely(file)

    if xls is None:
        st.error("Excel 読込に失敗しました。ファイル形式・シート名をご確認ください。")
        st.stop()

    with st.spinner("『標賃』を解析中..."):
        calc_params, sr_params, warn1 = parse_hyochin(xls)

    with st.spinner("『R6.12』製品データを解析中..."):
        df_products, warn2 = parse_products(xls, sheet_name="R6.12")

    for w in (warn1 + warn2):
        st.warning(w)

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
        }
        df_products = pd.concat([df_products, pd.DataFrame([new_row])], ignore_index=True)
        st.session_state["df_products_raw"] = df_products
        st.success("製品を追加しました。")
        st.rerun()

st.success("保存しました。上部のナビから『ダッシュボード』へ進んでください。")
