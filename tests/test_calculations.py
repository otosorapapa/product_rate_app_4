import math

import pandas as pd

from calculations import (
    DEFAULT_OPERATIONAL_INPUTS,
    calculate_product_metrics,
    compute_operational_metrics,
    estimate_roi_table,
)


def test_compute_operational_metrics_matches_manual_formula():
    metrics = compute_operational_metrics(DEFAULT_OPERATIONAL_INPUTS)

    expected_fixed = DEFAULT_OPERATIONAL_INPUTS["labor_cost"] + DEFAULT_OPERATIONAL_INPUTS["sga_cost"]
    expected_profit = (
        DEFAULT_OPERATIONAL_INPUTS["loan_repayment"]
        + DEFAULT_OPERATIONAL_INPUTS["tax_payment"]
        + DEFAULT_OPERATIONAL_INPUTS["future_business"]
    )
    expected_net = (
        DEFAULT_OPERATIONAL_INPUTS["fulltime_workers"]
        + 0.75 * DEFAULT_OPERATIONAL_INPUTS["part1_workers"]
        + DEFAULT_OPERATIONAL_INPUTS["part2_coefficient"] * DEFAULT_OPERATIONAL_INPUTS["part2_workers"]
    )
    minutes_per_day = DEFAULT_OPERATIONAL_INPUTS["daily_hours"] * 60
    standard_minutes = minutes_per_day * DEFAULT_OPERATIONAL_INPUTS["operation_rate"]
    expected_annual = expected_net * DEFAULT_OPERATIONAL_INPUTS["working_days"] * standard_minutes

    assert math.isclose(metrics.fixed_total, expected_fixed, rel_tol=1e-9)
    assert math.isclose(metrics.required_profit_total, expected_profit, rel_tol=1e-9)
    assert math.isclose(metrics.net_workers, expected_net, rel_tol=1e-9)
    assert math.isclose(metrics.annual_minutes, expected_annual, rel_tol=1e-6)
    assert math.isclose(metrics.break_even_rate, expected_fixed / expected_annual, rel_tol=1e-9)
    assert math.isclose(
        metrics.required_rate,
        (expected_fixed + expected_profit) / expected_annual,
        rel_tol=1e-9,
    )


def test_calculate_product_metrics_classifies_products():
    df = pd.DataFrame(
        {
            "product_no": ["A", "B", "C"],
            "product_name": ["Alpha", "Beta", "Gamma"],
            "minutes_per_unit": [1.0, 1.5, 2.0],
            "va_per_min": [6.0, 3.8, 2.0],
            "material_unit_cost": [2.0, 2.0, 2.0],
            "actual_unit_price": [8.0, 5.0, 4.0],
        }
    )
    metrics = calculate_product_metrics(df, break_even_rate=2.0, required_rate=4.0)
    assert metrics.loc[0, "rate_class"] == "健康商品"
    assert metrics.loc[1, "rate_class"] == "改善余地"
    assert metrics.loc[2, "rate_class"] == "至急改善"
    assert math.isclose(metrics.loc[0, "price_gap_vs_required"], 8.0 - (2.0 + 4.0))


def test_estimate_roi_table_returns_sorted_values():
    df = pd.DataFrame(
        {
            "product_no": [1, 2],
            "product_name": ["A", "B"],
            "actual_unit_price": [1000.0, 800.0],
            "material_unit_cost": [300.0, 500.0],
            "minutes_per_unit": [5.0, 7.0],
            "daily_qty": [50, 20],
            "va_per_min": [8.0, 4.0],
        }
    )
    roi_df = estimate_roi_table(df, required_rate=4.0)
    assert list(roi_df.columns)[:4] == [
        "product_no",
        "product_name",
        "price_improvement",
        "cost_reduction",
    ]
    # 1番目のSKUの方が日次改善額が大きい想定
    assert roi_df.iloc[0]["product_no"] == 1
    assert roi_df["monthly_gain"].is_monotonic_decreasing
