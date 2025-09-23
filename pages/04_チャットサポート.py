import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import re
from typing import Dict, Any, Optional

import pandas as pd
import streamlit as st

from components import (
    apply_user_theme,
    render_help_button,
    render_onboarding,
    render_page_tutorial,
    render_sidebar_nav,
)
from standard_rate_core import DEFAULT_PARAMS, compute_rates, sanitize_params
from utils import (
    compute_results,
    infer_category_from_name,
    infer_major_customer,
    parse_hyochin,
    parse_products,
    read_excel_safely,
)

_SAMPLE_PATH = "data/sample.xlsx"
_FAQ_PRESETS = [
    ("損益分岐賃率の違い", "損益分岐賃率と必要賃率の違いを教えて"),
    ("必要販売単価（例:苺大福）", "苺大福の必要販売単価はいくらですか？"),
    ("未達SKUを知りたい", "必要賃率を達成できていない製品はどれですか？"),
]


def _coerce_float(value: Any) -> float:
    try:
        if value in ("", None):
            return float("nan")
        if isinstance(value, str):
            value = value.replace(",", "")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _format_number(value: Any, unit: str = "", digits: int = 2) -> str:
    if value in (None, ""):
        return "データ未設定"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "データ未設定"
    if pd.isna(val):
        return "データ未設定"
    formatted = f"{val:,.{digits}f}"
    return f"{formatted}{unit}"


def _format_currency(value: Any, unit: str, digits: int = 0) -> str:
    return _format_number(value, unit, digits)


def _count_meets_required(df: Optional[pd.DataFrame]) -> int:
    if df is None or df.empty:
        return 0
    gap = pd.to_numeric(df.get("rate_gap_vs_required"), errors="coerce")
    if gap.notna().any():
        meets = gap >= 0
    else:
        meets_series = df.get("meets_required_rate")
        if meets_series is None:
            return 0
        if meets_series.dtype == bool:
            meets = meets_series.fillna(False)
        else:
            meets = pd.to_numeric(meets_series, errors="coerce") > 0
    return int(meets.fillna(False).sum())


def _match_product(query: str, df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    normalized = query.replace("　", " ")
    normalized_lower = normalized.lower()
    normalized_compact = normalized.replace(" ", "")
    for _, row in df.iterrows():
        name = str(row.get("product_name", "") or "").strip()
        number = str(row.get("product_no", "") or "").strip()
        if name and name.lower() != "nan" and name in normalized:
            return row
        if name and name.lower() != "nan" and name.lower() in normalized_lower:
            return row
        if number and number.lower() != "nan" and number in normalized_compact:
            return row
    tokens = [t for t in re.split(r"[\s、,。・]+", normalized) if t]
    name_series = df.get("product_name")
    number_series = df.get("product_no")
    name_str = name_series.astype(str) if name_series is not None else None
    number_str = number_series.astype(str) if number_series is not None else None
    for token in tokens:
        if name_str is not None:
            mask = name_str.str.contains(token, regex=False, na=False)
            if mask.any():
                return df[mask].iloc[0]
        if number_str is not None:
            mask = number_str.str.contains(token, regex=False, na=False)
            if mask.any():
                return df[mask].iloc[0]
    return None


def _format_rate_comparison(
    rates: Dict[str, float], df: pd.DataFrame, scenario: str
) -> str:
    fixed = _coerce_float(rates.get("fixed_total"))
    req_profit = _coerce_float(rates.get("required_profit_total"))
    annual_minutes = _coerce_float(rates.get("annual_minutes"))
    be_rate = _coerce_float(rates.get("break_even_rate"))
    req_rate = _coerce_float(rates.get("required_rate"))
    diff = req_rate - be_rate if not pd.isna(req_rate) and not pd.isna(be_rate) else float("nan")
    lines = [
        "### 損益分岐賃率と必要賃率の違い",
        f"- **損益分岐賃率**: 固定費計 ÷ 年間標準稼働分 = {_format_currency(fixed, '円/年')} ÷ {_format_number(annual_minutes, ' 分/年', 0)} = {_format_currency(be_rate, '円/分', 2)}",
        f"- **必要賃率**: (固定費計 + 必要利益計) ÷ 年間標準稼働分 = ({_format_currency(fixed, '円/年')} + {_format_currency(req_profit, '円/年')}) ÷ {_format_number(annual_minutes, ' 分/年', 0)} = {_format_currency(req_rate, '円/分', 2)}",
    ]
    if not pd.isna(diff):
        lines.append(
            f"- **差分の意味**: 必要賃率 − 損益分岐賃率 = {_format_currency(diff, '円/分', 2)} "
            f"（必要利益 {_format_currency(req_profit, '円/年')} を確保するための上乗せです）"
        )
    total = len(df) if df is not None else 0
    if total:
        lines.append(f"- 参照データ: シナリオ『{scenario}』で取り込んだ {total} SKU の前提値を使用")
    lines.append("→ 必要賃率を下回ると目標利益に届かず、損益分岐賃率を下回ると固定費も回収できません。")
    return "\n".join(lines)


def _format_required_rate_explanation(
    rates: Dict[str, float], df: pd.DataFrame, scenario: str
) -> str:
    fixed = _coerce_float(rates.get("fixed_total"))
    req_profit = _coerce_float(rates.get("required_profit_total"))
    annual_minutes = _coerce_float(rates.get("annual_minutes"))
    req_rate = _coerce_float(rates.get("required_rate"))
    total = len(df) if df is not None else 0
    meets = _count_meets_required(df)
    meet_pct = (meets / total * 100.0) if total else 0.0
    lines = [
        "### 必要賃率の算出方法",
        "- 計算式: 必要賃率 = (固定費計 + 必要利益計) ÷ 年間標準稼働分",
        f"- 直近の値: ({_format_currency(fixed, '円/年')} + {_format_currency(req_profit, '円/年')}) ÷ {_format_number(annual_minutes, ' 分/年', 0)} = {_format_currency(req_rate, '円/分', 2)}",
    ]
    if total:
        lines.append(f"- 達成状況: {total} SKU 中 {meets} SKU ({meet_pct:.1f}%) が必要賃率を満たしています。")
    lines.append("→ 必要賃率を上回らない製品は付加価値向上や価格改定を検討してください。")
    return "\n".join(lines)


def _format_break_even_explanation(
    rates: Dict[str, float], df: pd.DataFrame, scenario: str
) -> str:
    fixed = _coerce_float(rates.get("fixed_total"))
    annual_minutes = _coerce_float(rates.get("annual_minutes"))
    be_rate = _coerce_float(rates.get("break_even_rate"))
    daily_be = _coerce_float(rates.get("daily_be_va"))
    lines = [
        "### 損益分岐賃率の考え方",
        "- 計算式: 損益分岐賃率 = 固定費計 ÷ 年間標準稼働分",
        f"- 直近の値: {_format_currency(fixed, '円/年')} ÷ {_format_number(annual_minutes, ' 分/年', 0)} = {_format_currency(be_rate, '円/分', 2)}",
    ]
    if not pd.isna(daily_be):
        lines.append(f"- 1日当たり損益分岐付加価値: {_format_currency(daily_be, '円/日')}（固定費計 ÷ 年間稼働日数）")
    total = len(df) if df is not None else 0
    if total:
        lines.append(f"- 参照データ: シナリオ『{scenario}』の {total} SKU に基づく計算です。")
    lines.append("→ この賃率を下回ると固定費を回収できないため、最低ラインとして意識してください。")
    return "\n".join(lines)


def _format_product_pricing(
    row: pd.Series, rates: Dict[str, float], scenario: str
) -> str:
    req_rate = _coerce_float(rates.get("required_rate"))
    material = _coerce_float(row.get("material_unit_cost"))
    minutes = _coerce_float(row.get("minutes_per_unit"))
    required_price = _coerce_float(row.get("required_selling_price"))
    if pd.isna(required_price) and not pd.isna(material) and not pd.isna(minutes) and not pd.isna(req_rate):
        required_price = material + minutes * req_rate
    actual = _coerce_float(row.get("actual_unit_price"))
    gap_price = _coerce_float(row.get("price_gap_vs_required"))
    rate_gap = _coerce_float(row.get("rate_gap_vs_required"))
    va_per_min = _coerce_float(row.get("va_per_min"))
    category = str(row.get("category", "") or "").strip()
    customer = str(row.get("major_customer", "") or "").strip()
    name = str(row.get("product_name", "") or "").strip()
    number = str(row.get("product_no", "") or "").strip()
    label = " / ".join([part for part in [name, number] if part and part.lower() != "nan"])
    if not label:
        label = "指定の製品"
    lines = [
        f"### {label} の必要販売単価",
        "- 計算式: 必要販売単価 = 材料原価 + (分/個 × 必要賃率)",
        f"- 入力値: 材料原価 {_format_currency(material, '円/個')}、分/個 {_format_number(minutes, ' 分/個', 2)}、必要賃率 {_format_currency(req_rate, '円/分', 2)}",
        f"- 計算結果: {_format_currency(material, '円/個')} + ({_format_number(minutes, '', 2)} × {_format_currency(req_rate, '円/分', 2)}) = {_format_currency(required_price, '円/個')}",
    ]
    if not pd.isna(actual):
        lines.append(f"- 実際売単価: {_format_currency(actual, '円/個')}（ギャップ {_format_currency(gap_price, '円/個')}）")
    if not pd.isna(rate_gap):
        lines.append(
            f"- 付加価値/分: {_format_currency(va_per_min, '円/分', 2)} → 必要賃率との差 {_format_currency(rate_gap, '円/分', 2)}"
        )
    if category and category.lower() != "nan":
        if customer and customer.lower() != "nan":
            lines.append(f"- カテゴリー/主要顧客: {category} / {customer}")
        else:
            lines.append(f"- カテゴリー: {category}")
    elif customer and customer.lower() != "nan":
        lines.append(f"- 主要顧客: {customer}")
    lines.append(f"- 参照データ: シナリオ『{scenario}』で取り込んだ履歴から算出しました。")
    if not pd.isna(gap_price):
        if gap_price < 0:
            lines.append(f"→ 必要販売単価まで {abs(gap_price):,.0f}円/個 の値上げが必要です。")
        elif gap_price > 0:
            lines.append(f"→ 現在の売価は必要販売単価を {gap_price:,.0f}円/個 上回っています。")
        else:
            lines.append("→ 現在の売価は必要販売単価と一致しています。")
    return "\n".join(lines)


def _format_general_summary(
    rates: Dict[str, float], df: pd.DataFrame, scenario: str
) -> str:
    be_rate = _coerce_float(rates.get("break_even_rate"))
    req_rate = _coerce_float(rates.get("required_rate"))
    total = len(df) if df is not None else 0
    meets = _count_meets_required(df)
    not_meet = total - meets
    lines = [
        "### 主要指標サマリ",
        f"- 損益分岐賃率: {_format_currency(be_rate, '円/分', 2)}（式: 固定費計 ÷ 年間標準稼働分）",
        f"- 必要賃率: {_format_currency(req_rate, '円/分', 2)}（式: (固定費計 + 必要利益計) ÷ 年間標準稼働分）",
    ]
    if total:
        lines.append(f"- 対象データ: シナリオ『{scenario}』に {total} SKU を取り込み済み")
        lines.append(f"- 達成状況: 必要賃率達成 {meets} SKU / 未達 {not_meet} SKU")
        gap_series = pd.to_numeric(df.get("rate_gap_vs_required"), errors="coerce")
        if gap_series.notna().any():
            shortfalls = df.copy()
            shortfalls["rate_gap_vs_required"] = gap_series
            shortfalls = shortfalls[shortfalls["rate_gap_vs_required"] < 0]
            shortfalls = shortfalls.sort_values("rate_gap_vs_required").head(3)
            if not shortfalls.empty:
                lines.append("### 必要賃率未達の上位SKU")
                for _, row in shortfalls.iterrows():
                    pname = str(row.get("product_name", "") or "").strip()
                    pno = str(row.get("product_no", "") or "").strip()
                    label = " / ".join([part for part in [pname, pno] if part and part.lower() != "nan"])
                    if not label:
                        label = "SKU"
                    gap_val = _coerce_float(row.get("rate_gap_vs_required"))
                    req_price = _coerce_float(row.get("required_selling_price"))
                    lines.append(
                        f"- {label}: ギャップ {_format_currency(gap_val, '円/分', 2)}、必要販売単価 {_format_currency(req_price, '円/個')}"
                    )
    lines.append("→ 製品名や品番を指定すると、必要販売単価の内訳を詳しく算出します。")
    return "\n".join(lines)


def _generate_answer(
    question: str, df_results: pd.DataFrame, rates: Dict[str, float], scenario: str
) -> str:
    if not question:
        return "質問が空のようです。知りたい内容を入力してください。"
    normalized = question.strip()
    normalized_lower = normalized.lower()
    product_row = _match_product(normalized, df_results)
    keywords_price = any(k in normalized for k in ["価格", "単価", "いくら", "必要販売単価", "値段"]) or "price" in normalized_lower
    if ("損益" in normalized and "必要" in normalized and ("違" in normalized or "差" in normalized or "比較" in normalized)):
        return _format_rate_comparison(rates, df_results, scenario)
    if product_row is not None and keywords_price:
        return _format_product_pricing(product_row, rates, scenario)
    if product_row is not None and ("必要賃率" in normalized or "必要単価" in normalized or "付加価値" in normalized):
        return _format_product_pricing(product_row, rates, scenario)
    if "必要賃率" in normalized:
        return _format_required_rate_explanation(rates, df_results, scenario)
    if "損益分岐" in normalized:
        return _format_break_even_explanation(rates, df_results, scenario)
    if product_row is not None:
        return _format_product_pricing(product_row, rates, scenario)
    if keywords_price:
        return (
            "製品名または品番を含めると具体的な必要販売単価を計算できます。\n"
            + _format_general_summary(rates, df_results, scenario)
        )
    if "未達" in normalized or "ギャップ" in normalized:
        return _format_general_summary(rates, df_results, scenario)
    return _format_general_summary(rates, df_results, scenario)


def _build_intro_message(
    rates: Dict[str, float], df_results: pd.DataFrame, scenario: str
) -> str:
    req_rate = _coerce_float(rates.get("required_rate"))
    be_rate = _coerce_float(rates.get("break_even_rate"))
    total = len(df_results) if df_results is not None else 0
    meets = _count_meets_required(df_results)
    not_meet = total - meets
    summary_part = (
        f"現在の必要賃率は {_format_currency(req_rate, '円/分', 2)} / 損益分岐賃率は {_format_currency(be_rate, '円/分', 2)}。"
    )
    if total:
        summary_part += f" {total} SKU 中 {not_meet} SKU が必要賃率未達です。"
    else:
        summary_part += " 製品データがまだ読み込まれていません。"
    return (
        f"こんにちは！賃率チャットボットです（シナリオ『{scenario}』を参照）。"
        f"{summary_part}損益分岐賃率との違いや製品ごとの必要販売単価など、気になる点を質問してください。"
    )


def _build_signature(rates: Dict[str, float], scenario: str, df: pd.DataFrame) -> tuple:
    req = _coerce_float(rates.get("required_rate"))
    be = _coerce_float(rates.get("break_even_rate"))
    total = len(df) if df is not None else 0
    req_sig = None if pd.isna(req) else round(req, 4)
    be_sig = None if pd.isna(be) else round(be, 4)
    return (req_sig, be_sig, scenario, total)


def _bootstrap_sample_data() -> None:
    xls = read_excel_safely(_SAMPLE_PATH)
    if xls is None:
        return
    calc_params, sr_params, _ = parse_hyochin(xls)
    df_products, _ = parse_products(xls, sheet_name="R6.12")
    if df_products.empty:
        return
    df_products = df_products.copy()
    if "category" not in df_products.columns or df_products["category"].isna().all():
        df_products["category"] = df_products.get("product_name", pd.Series(dtype=str)).apply(
            infer_category_from_name
        )
    if "major_customer" not in df_products.columns or df_products["major_customer"].isna().all():
        df_products["major_customer"] = [
            infer_major_customer(no, name)
            for no, name in zip(
                df_products.get("product_no"), df_products.get("product_name")
            )
        ]
    st.session_state["df_products_raw"] = df_products
    st.session_state.setdefault("calc_params", calc_params)
    st.session_state.setdefault("sr_params", sr_params)
    st.session_state.setdefault("scenarios", {"ベース": st.session_state["sr_params"].copy()})
    st.session_state.setdefault("current_scenario", "ベース")
    st.session_state["using_sample_data"] = True
    st.session_state["chat_sample_notice"] = True


def _prepare_context() -> tuple[pd.DataFrame, Dict[str, float], str]:
    if "df_products_raw" not in st.session_state or st.session_state["df_products_raw"] is None:
        _bootstrap_sample_data()
    df_products = st.session_state.get("df_products_raw")
    if df_products is None or df_products.empty:
        return pd.DataFrame(), {}, st.session_state.get("current_scenario", "ベース")
    scenarios = st.session_state.get("scenarios") or {}
    current = st.session_state.get("current_scenario", "ベース")
    if current in scenarios:
        base_params = scenarios[current]
    else:
        base_params = st.session_state.get("sr_params", DEFAULT_PARAMS)
    params, warnings = sanitize_params(base_params)
    for msg in warnings:
        st.sidebar.warning(msg)
    st.session_state["sr_params"] = params.copy()
    if current in scenarios:
        scenarios[current] = params.copy()
        st.session_state["scenarios"] = scenarios
    nodes, rate_results = compute_rates(params)
    calc_keys = [
        "fixed_total",
        "required_profit_total",
        "annual_minutes",
        "break_even_rate",
        "required_rate",
        "daily_be_va",
        "daily_req_va",
    ]
    st.session_state["calc_params"] = {k: rate_results[k] for k in calc_keys if k in rate_results}
    df_results = compute_results(
        df_products,
        rate_results.get("break_even_rate"),
        rate_results.get("required_rate"),
    )
    return df_results, rate_results, current


apply_user_theme()
render_sidebar_nav(page_key="chat")
st.title("④ チャットボット / FAQ")
render_help_button("chat")
render_onboarding()
render_page_tutorial("chat")

if st.session_state.pop("chat_sample_notice", False):
    st.info("製品データが未設定だったためサンプル data/sample.xlsx を読み込みました。")
if st.session_state.pop("chat_reset_notice", False):
    st.success("チャット履歴をクリアしました。")

st.caption("取り込んだデータをもとに、賃率や価格に関する疑問へ即時回答します。")

df_results, rate_results, scenario_name = _prepare_context()
if df_results.empty:
    st.error("製品データが読み込まれていません。まずは『データ入力』でExcelを取り込んでください。")
    st.stop()

req_rate_val = _coerce_float(rate_results.get("required_rate"))
be_rate_val = _coerce_float(rate_results.get("break_even_rate"))
meets = _count_meets_required(df_results)
not_meet = len(df_results) - meets

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("損益分岐賃率 (円/分)", _format_number(be_rate_val, "", 2))
with m2:
    st.metric("必要賃率 (円/分)", _format_number(req_rate_val, "", 2))
with m3:
    st.metric("必要賃率未達SKU", f"{not_meet} 件", delta=f"全体 {len(df_results)} SKU")

st.caption(f"適用中シナリオ: {scenario_name}")
if st.session_state.get("using_sample_data"):
    st.caption("※ 未取り込みのためサンプルデータを参照しています。")

st.divider()

faq_cols = st.columns(len(_FAQ_PRESETS) + 1)
for col, (label, question) in zip(faq_cols, _FAQ_PRESETS):
    if col.button(label):
        st.session_state["chat_pending_question"] = question
        st.rerun()

with faq_cols[-1]:
    if st.button("会話をリセット"):
        st.session_state["chat_history"] = []
        st.session_state.pop("chat_last_signature", None)
        st.session_state.pop("chat_pending_question", None)
        st.session_state["chat_reset_notice"] = True
        st.rerun()

history = st.session_state.setdefault("chat_history", [])
signature = _build_signature(rate_results, scenario_name, df_results)
if not history:
    history.append(
        {
            "role": "assistant",
            "content": _build_intro_message(rate_results, df_results, scenario_name),
            "kind": "intro",
        }
    )
    st.session_state["chat_last_signature"] = signature
else:
    if history[0].get("kind") == "intro" and st.session_state.get("chat_last_signature") != signature:
        history[0]["content"] = _build_intro_message(rate_results, df_results, scenario_name)
        st.session_state["chat_last_signature"] = signature

pending_question = st.session_state.pop("chat_pending_question", None)
user_message = st.chat_input("賃率や価格について質問してください")

if pending_question:
    history.append({"role": "user", "content": pending_question})
    answer = _generate_answer(pending_question, df_results, rate_results, scenario_name)
    history.append({"role": "assistant", "content": answer})

if user_message:
    history.append({"role": "user", "content": user_message})
    answer = _generate_answer(user_message, df_results, rate_results, scenario_name)
    history.append({"role": "assistant", "content": answer})

for message in history:
    role = message.get("role", "assistant")
    content = message.get("content", "")
    with st.chat_message(role):
        st.markdown(content)
