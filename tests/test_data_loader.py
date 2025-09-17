import pandas as pd

from data_loader import (
    REQUIRED_COLUMNS,
    detect_anomalies,
    detect_quality_issues,
    sanitize_dataframe,
    validate_dataframe,
)


def test_sanitize_dataframe_converts_numeric_columns():
    df = pd.DataFrame({
        "product_no": ["001"],
        "product_name": ["Test"],
        "actual_unit_price": ["1,200"],
        "material_unit_cost": ["300"],
        "minutes_per_unit": ["5.5"],
        "va_per_min": ["8.2"],
    })
    cleaned = sanitize_dataframe(df)
    for col in REQUIRED_COLUMNS[2:]:
        assert pd.api.types.is_numeric_dtype(cleaned[col])


def test_detect_quality_issues_identifies_missing_and_duplicates():
    df = pd.DataFrame({
        "product_no": ["A", "A", "B"],
        "product_name": ["Alpha", "Alpha", "Beta"],
        "actual_unit_price": [100.0, None, 120.0],
        "material_unit_cost": [40.0, 40.0, -10.0],
        "minutes_per_unit": [2.0, 2.0, 3.0],
        "va_per_min": [3.0, 3.0, 4.0],
    })
    issues = detect_quality_issues(df)
    kinds = set(issues["type"].tolist())
    assert {"欠損", "外れ値", "重複"} <= kinds


def test_detect_anomalies_flags_outliers():
    df = pd.DataFrame({
        "product_no": [1, 2, 3, 4, 5],
        "product_name": list("ABCDE"),
        "va_per_min": [5, 5.2, 5.1, 20, 5.3],
        "minutes_per_unit": [1, 1.1, 1.0, 5.0, 1.2],
        "actual_unit_price": [100, 102, 98, 400, 101],
        "material_unit_cost": [40, 41, 39, 80, 40],
    })
    anomalies = detect_anomalies(df, metrics=["va_per_min", "minutes_per_unit"])
    assert not anomalies.empty
    assert (anomalies["product_no"] == 4).any()


def test_validate_dataframe_returns_summary():
    df = pd.DataFrame({
        "product_no": ["A"],
        "product_name": ["Alpha"],
        "actual_unit_price": [100.0],
        "material_unit_cost": [40.0],
        "minutes_per_unit": [2.0],
        "va_per_min": [3.0],
    })
    result = validate_dataframe(df)
    assert "平均値" in result.summary.columns
    assert result.summary.loc["actual_unit_price", "平均値"] == 100.0
    assert result.issues.empty
