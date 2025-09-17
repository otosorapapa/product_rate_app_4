import pandas as pd
from utils import detect_quality_issues


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
