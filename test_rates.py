import math
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = str(Path(__file__).resolve().parents[1])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from standard_rate_core import (
    DEFAULT_PARAMS,
    sanitize_params,
    compute_rates,
    build_reverse_index,
)
from rate_utils import compute_results, summarize_segment_performance


def test_compute_rates_basic():
    params = DEFAULT_PARAMS.copy()
    params, _ = sanitize_params(params)
    nodes, results = compute_rates(params)
    assert results["fixed_total"] == params["labor_cost"] + params["sga_cost"]
    assert results["required_profit_total"] == (
        params["loan_repayment"] + params["tax_payment"] + params["future_business"]
    )
    expected_annual = (
        (
            params["fulltime_workers"]
            + 0.75 * params["part1_workers"]
            + params["part2_coefficient"] * params["part2_workers"]
        )
        * params["working_days"]
        * (params["daily_hours"] * 60 * params["operation_rate"])
    )
    assert results["annual_minutes"] == expected_annual
    assert results["break_even_rate"] == results["fixed_total"] / results["annual_minutes"]
    assert results["required_rate"] == (
        results["fixed_total"] + results["required_profit_total"]
    ) / results["annual_minutes"]


def test_dependencies_and_no_cycle():
    params = DEFAULT_PARAMS.copy()
    params, _ = sanitize_params(params)
    nodes, _ = compute_rates(params)
    assert set(nodes["break_even_rate"]["depends_on"]) == {"fixed_total", "annual_minutes"}
    assert set(nodes["required_rate"]["depends_on"]) == {
        "fixed_total",
        "required_profit_total",
        "annual_minutes",
    }
    for key, node in nodes.items():
        assert key not in node["depends_on"]


def test_reverse_index_links_base_to_rates():
    params = DEFAULT_PARAMS.copy()
    params, _ = sanitize_params(params)
    nodes, _ = compute_rates(params)
    rev = build_reverse_index(nodes)
    for base in ["labor_cost", "sga_cost", "operation_rate"]:
        assert "break_even_rate" in rev[base]
        assert "required_rate" in rev[base]


def test_sanitize_params_negative():
    raw = DEFAULT_PARAMS.copy()
    raw.update(
        {
            "labor_cost": -100,
            "working_days": 0,
            "daily_hours": 0,
            "operation_rate": 0,
            "fulltime_workers": 0,
            "part1_workers": 0,
            "part2_workers": 0,
        }
    )
    params, warnings = sanitize_params(raw)
    assert params["labor_cost"] == 0
    assert params["working_days"] == 1
    assert params["daily_hours"] == 1
    assert params["operation_rate"] == 0.01
    assert params["fulltime_workers"] == 1.0
    assert warnings


def test_rates_recompute_from_params_no_excel():
    params = DEFAULT_PARAMS.copy()
    params["labor_cost"] += 12345.0
    params, _ = sanitize_params(params)
    _, res = compute_rates(params)
    fixed_total = params["labor_cost"] + params["sga_cost"]
    req_profit = params["loan_repayment"] + params["tax_payment"] + params["future_business"]
    net_workers = (
        params["fulltime_workers"]
        + 0.75 * params["part1_workers"]
        + params["part2_coefficient"] * params["part2_workers"]
    )
    minutes_per_day = params["daily_hours"] * 60
    annual_minutes = net_workers * params["working_days"] * minutes_per_day * params["operation_rate"]
    expected_be = fixed_total / annual_minutes
    expected_req = (fixed_total + req_profit) / annual_minutes
    assert math.isclose(res["break_even_rate"], expected_be, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(res["required_rate"], expected_req, rel_tol=1e-9, abs_tol=1e-9)


def test_compute_results_classification():
    df = pd.DataFrame({
        "product_no": [1, 2, 3],
        "product_name": ["A", "B", "C"],
        "va_per_min": [5.0, 3.0, 1.0],
        "material_unit_cost": [0.0, 0.0, 0.0],
        "minutes_per_unit": [1.0, 1.0, 1.0],
        "actual_unit_price": [0.0, 0.0, 0.0],
    })
    res = compute_results(
        df,
        break_even_rate=2.0,
        required_rate=4.0,
        low_ratio=0.5,
        high_ratio=1.25,
    )
    classes = res["rate_class"].tolist()
    assert classes == ["健康商品", "貧血商品", "出血商品"]


def test_summarize_segment_performance_basic():
    df = pd.DataFrame(
        {
            "product_no": ["P1", "P2", "P3", "P4"],
            "product_name": ["Cake", "Pie", "Mochi", "Daifuku"],
            "category": ["洋菓子", "洋菓子", "和菓子", "和菓子"],
            "va_per_min": [160.0, 150.0, 120.0, 130.0],
            "meets_required_rate": [True, True, False, False],
            "required_selling_price": [0.0, 0.0, 420.0, 410.0],
            "actual_unit_price": [0.0, 0.0, 360.0, 380.0],
        }
    )
    summary = summarize_segment_performance(df, required_rate=140.0, segment_col="category")
    assert list(summary["segment"]) == ["洋菓子", "和菓子"]
    western = summary[summary["segment"] == "洋菓子"].iloc[0]
    japanese = summary[summary["segment"] == "和菓子"].iloc[0]
    assert math.isclose(western["ach_rate_pct"], 100.0, rel_tol=1e-6)
    assert math.isclose(japanese["ach_rate_pct"], 0.0, rel_tol=1e-6)
    assert japanese["avg_gap"] < 0
    assert math.isclose(japanese["avg_roi_months"], 3.0, rel_tol=1e-6)
