import pandas as pd

from utils import detect_anomalies, detect_quality_issues


def test_detect_quality_issues():
    df = pd.DataFrame({
        "product_no": [1, 1, 2, 3],
        "product_name": ["A", "A", "B", "C"],
        "actual_unit_price": [100, 100, None, -50],
        "material_unit_cost": [10, 10, 20, 30],
        "minutes_per_unit": [1, 1, 2, 3],
        "daily_qty": [10, 10, 20, 30],
    })
    issues = detect_quality_issues(df)
    kinds = set(issues["type"].tolist())
    assert {"欠損", "外れ値", "重複"} == kinds
    assert (issues["type"] == "重複").sum() == 1
    assert any((issues["type"] == "欠損") & (issues["column"] == "actual_unit_price"))
    assert any((issues["type"] == "外れ値") & (issues["product_no"] == 3))


def test_detect_anomalies_highlights_extreme_values():
    df = pd.DataFrame(
        {
            "product_no": [1, 2, 3, 4, 5],
            "product_name": list("ABCDE"),
            "va_per_min": [100, 102, 98, 250, 101],
            "minutes_per_unit": [1.0, 1.1, 1.0, 5.0, 1.2],
            "actual_unit_price": [200, 205, 198, 600, 204],
            "material_unit_cost": [80, 82, 81, 90, 83],
            "daily_qty": [100, 120, 110, 15, 115],
            "rate_gap_vs_required": [5, 4.5, 5.2, -20, 4.8],
        }
    )

    anomalies = detect_anomalies(df, metrics=["va_per_min", "minutes_per_unit"])

    assert not anomalies.empty
    assert (anomalies["product_no"] == 4).any()
    target = anomalies[anomalies["product_no"] == 4].iloc[0]
    assert target["direction"] in {"high", "low"}
    assert target["severity"] > 3.5
