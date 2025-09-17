"""標準賦率ダッシュボードの計算ロジック群."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

FERMI_DEFAULTS = {
    "fixed_total": 57_115_000.0,
    "required_profit_total": 93_798_000.0,
    "annual_minutes": 506_999.0,
}

DEFAULT_OPERATIONAL_INPUTS = {
    "labor_cost": 16_829_175.0,
    "sga_cost": 40_286_204.0,
    "loan_repayment": 4_000_000.0,
    "tax_payment": 3_797_500.0,
    "future_business": 2_000_000.0,
    "fulltime_workers": 4.0,
    "part1_workers": 2.0,
    "part2_workers": 0.0,
    "part2_coefficient": 0.0,
    "working_days": 236.0,
    "daily_hours": 8.68,
    "operation_rate": 0.75,
}


@dataclass
class OperationalMetrics:
    """人員・時間前提から導出した主要指標を格納するデータ構造."""

    fixed_total: float
    required_profit_total: float
    net_workers: float
    annual_minutes: float
    break_even_rate: float
    required_rate: float
    minutes_per_day: float
    standard_daily_minutes: float
    daily_be_va: float
    daily_req_va: float


# ------------------------------ 入力補正 ------------------------------

def sanitize_operational_inputs(params: Dict[str, float]) -> Tuple[Dict[str, float], List[str]]:
    """入力値を補正しながら警告文を返す."""

    sanitized = {**DEFAULT_OPERATIONAL_INPUTS}
    warnings: List[str] = []
    for key, default in DEFAULT_OPERATIONAL_INPUTS.items():
        raw = params.get(key, default)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            warnings.append(f"{key} は数値でないため既定値 {default:,.0f} を採用しました。")
            value = default
        if np.isnan(value):
            warnings.append(f"{key} が NaN だったため既定値に置換しました。")
            value = default
        if value < 0:
            warnings.append(f"{key} が負数だったため 0 に補正しました。")
            value = 0.0
        sanitized[key] = value

    if sanitized["working_days"] <= 0:
        warnings.append("年間稼働日数は 1 日以上でなければなりません。既定値 1 を設定しました。")
        sanitized["working_days"] = 1.0
    if sanitized["daily_hours"] <= 0:
        warnings.append("1日当たり稼働時間は 1 時間以上に補正しました。")
        sanitized["daily_hours"] = 1.0
    if sanitized["operation_rate"] <= 0:
        warnings.append("操業度が 0 以下のため 0.01 に補正しました。")
        sanitized["operation_rate"] = 0.01

    net = (
        sanitized["fulltime_workers"]
        + 0.75 * sanitized["part1_workers"]
        + sanitized["part2_coefficient"] * sanitized["part2_workers"]
    )
    if net <= 0:
        warnings.append("正味直接工員数が 0 以下のため正社員 1 名を仮定しました。")
        sanitized["fulltime_workers"] = max(1.0, sanitized["fulltime_workers"])
    return sanitized, warnings


# ------------------------------ メトリクス計算 ------------------------------

@st.cache_data(show_spinner=False)
def compute_operational_metrics(params: Dict[str, float]) -> OperationalMetrics:
    """固定費・必要利益・稼働前提から主要賦率を算出する."""

    sanitized, _ = sanitize_operational_inputs(params)
    fixed_total = sanitized["labor_cost"] + sanitized["sga_cost"]
    required_profit_total = (
        sanitized["loan_repayment"]
        + sanitized["tax_payment"]
        + sanitized["future_business"]
    )
    net_workers = (
        sanitized["fulltime_workers"]
        + 0.75 * sanitized["part1_workers"]
        + sanitized["part2_coefficient"] * sanitized["part2_workers"]
    )
    minutes_per_day = sanitized["daily_hours"] * 60.0
    standard_daily_minutes = minutes_per_day * sanitized["operation_rate"]
    annual_minutes = net_workers * sanitized["working_days"] * standard_daily_minutes
    if annual_minutes <= 0:
        raise ValueError("年間稼働時間が 0 以下のため負担率を計算できません。入力を見直してください。")
    break_even_rate = fixed_total / annual_minutes
    required_rate = (fixed_total + required_profit_total) / annual_minutes
    daily_be_va = fixed_total / sanitized["working_days"]
    daily_req_va = (fixed_total + required_profit_total) / sanitized["working_days"]
    return OperationalMetrics(
        fixed_total=fixed_total,
        required_profit_total=required_profit_total,
        net_workers=net_workers,
        annual_minutes=annual_minutes,
        break_even_rate=break_even_rate,
        required_rate=required_rate,
        minutes_per_day=minutes_per_day,
        standard_daily_minutes=standard_daily_minutes,
        daily_be_va=daily_be_va,
        daily_req_va=daily_req_va,
    )


@st.cache_data(show_spinner=False)
def calculate_product_metrics(
    df_products: pd.DataFrame,
    break_even_rate: float,
    required_rate: float,
    *,
    healthy_ratio: float = 1.1,
    caution_ratio: float = 0.95,
) -> pd.DataFrame:
    """製品別に負担率ギャップや分類ラベルを付与する."""

    df = df_products.copy()
    df["minutes_per_unit"] = pd.to_numeric(df.get("minutes_per_unit"), errors="coerce")
    df["va_per_min"] = pd.to_numeric(df.get("va_per_min"), errors="coerce")
    df["material_unit_cost"] = pd.to_numeric(df.get("material_unit_cost"), errors="coerce")
    df["actual_unit_price"] = pd.to_numeric(df.get("actual_unit_price"), errors="coerce")

    mpu = df["minutes_per_unit"].fillna(0)
    df["be_unit_value"] = mpu * break_even_rate
    df["required_unit_value"] = mpu * required_rate
    df["target_price"] = df["material_unit_cost"] + df["required_unit_value"]
    df["price_gap_vs_required"] = df["actual_unit_price"] - df["target_price"]
    df["rate_gap_vs_required"] = df["va_per_min"] - required_rate
    df["meets_required_rate"] = df["rate_gap_vs_required"] >= 0

    def classify_row(va: float) -> str:
        if required_rate <= 0 or np.isnan(required_rate) or np.isnan(va):
            return "評価保留"
        ratio = va / required_rate
        if ratio >= healthy_ratio:
            return "健康商品"
        if ratio >= caution_ratio:
            return "改善余地"
        return "至急改善"

    df["rate_class"] = df["va_per_min"].apply(classify_row)
    return df


def _estimate_roi_components(row: pd.Series, required_rate: float) -> Dict[str, float]:
    """製品単位で複数の改善アプローチによる効果を計算する内部関数."""

    actual_price = row.get("actual_unit_price", np.nan)
    material_cost = row.get("material_unit_cost", np.nan)
    va_per_min = row.get("va_per_min", np.nan)
    minutes_per_unit = row.get("minutes_per_unit", np.nan)
    daily_qty = row.get("daily_qty", np.nan)

    improvement_price = actual_price * 0.03 if not np.isnan(actual_price) else np.nan
    improvement_cost = material_cost * 0.05 if not np.isnan(material_cost) else np.nan
    improvement_cycle = (
        required_rate * minutes_per_unit * 0.08 if not np.isnan(minutes_per_unit) else np.nan
    )
    improvement_material = material_cost * 0.03 if not np.isnan(material_cost) else np.nan

    improvements = {
        "price_improvement": improvement_price,
        "cost_reduction": improvement_cost,
        "cycle_time_reduction": improvement_cycle,
        "material_improvement": improvement_material,
    }

    if not np.isnan(daily_qty):
        daily_gain = sum(v for v in improvements.values() if not np.isnan(v)) * daily_qty
    else:
        daily_gain = np.nan

    improvements["daily_gain"] = daily_gain
    return improvements


@st.cache_data(show_spinner=False)
def estimate_roi_table(df: pd.DataFrame, required_rate: float) -> pd.DataFrame:
    """ROI シミュレーション結果を表形式で返す."""

    records: List[Dict[str, float]] = []
    for _, row in df.iterrows():
        improvements = _estimate_roi_components(row, required_rate)
        monthly_gain = (
            improvements["daily_gain"] * 20 if not np.isnan(improvements["daily_gain"]) else np.nan
        )
        investment = 300_000.0
        if monthly_gain and not np.isnan(monthly_gain) and monthly_gain > 0:
            roi_months = investment / monthly_gain
        else:
            roi_months = np.nan
        record = {
            "product_no": row.get("product_no"),
            "product_name": row.get("product_name"),
            "price_improvement": improvements["price_improvement"],
            "cost_reduction": improvements["cost_reduction"],
            "cycle_time_reduction": improvements["cycle_time_reduction"],
            "material_improvement": improvements["material_improvement"],
            "monthly_gain": monthly_gain,
            "roi_months": roi_months,
        }
        records.append(record)
    roi_df = pd.DataFrame(records)
    return roi_df.sort_values(by="monthly_gain", ascending=False)


def list_parameter_impacts() -> Dict[str, List[str]]:
    """感度分析で用いる入力と影響指標の対応関係を返す."""

    return {
        "固定費": ["fixed_total", "break_even_rate", "required_rate", "daily_be_va"],
        "必要利益": ["required_profit_total", "required_rate", "daily_req_va"],
        "稼働前提": ["net_workers", "annual_minutes", "break_even_rate", "required_rate"],
    }
