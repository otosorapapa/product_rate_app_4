from io import BytesIO

import pandas as pd

from utils import (
    detect_anomalies,
    detect_quality_issues,
    generate_product_template,
    parse_products,
    validate_product_dataframe,
)


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


def test_validate_product_dataframe_detects_unit_mismatch():
    df = pd.DataFrame(
        {
            "product_no": ["SKU-1"],
            "product_name": ["テスト商品"],
            "actual_unit_price": [150.0],
            "material_unit_cost": [80.0],
            "minutes_per_unit": [4.5],
            "daily_qty": [120.0],
        }
    )
    df.attrs["column_unit_info"] = {
        "product_no": {"unit": "コード", "source": "製品№"},
        "product_name": {"unit": "テキスト", "source": "製品名"},
        "actual_unit_price": {"unit": "千円/個", "source": "販売単価 (千円/個)"},
        "material_unit_cost": {"unit": "円/個", "source": "原価（材料費）"},
        "minutes_per_unit": {"unit": "分/個", "source": "リードタイム (分/個)"},
        "daily_qty": {"unit": "個/日", "source": "日産数 (個/日)"},
    }

    errors, warnings, detail = validate_product_dataframe(df)
    assert any("単位" in msg for msg in errors)
    target = detail[detail["項目"] == "販売単価（円/個）"].iloc[0]
    assert "千円/個" in str(target["原因/状況"]) or "千円/個" in str(target["入力値"])
    assert "テンプレート" in str(target["対処方法"])


def test_validate_product_dataframe_warns_when_unit_missing():
    df = pd.DataFrame(
        {
            "product_no": ["SKU-2"],
            "product_name": ["検証用"],
            "actual_unit_price": [220.0],
            "material_unit_cost": [150.0],
            "minutes_per_unit": [6.0],
            "daily_qty": [90.0],
        }
    )
    df.attrs["column_unit_info"] = {
        "product_no": {"unit": "コード", "source": "製品№"},
        "product_name": {"unit": "テキスト", "source": "製品名"},
        "actual_unit_price": {"unit": "円/個", "source": "販売単価"},
        "material_unit_cost": {"unit": "円/個", "source": "材料費"},
        "minutes_per_unit": {"unit": None, "source": "リードタイム"},
        "daily_qty": {"unit": "個/日", "source": "日産数"},
    }

    errors, warnings, detail = validate_product_dataframe(df)
    assert not errors
    assert any("リードタイム" in msg for msg in warnings)
    dataset_issue = detail[detail["項目"] == "リードタイム（分/個）"].iloc[0]
    assert "未設定" in str(dataset_issue["原因/状況"])
    assert "分/個" in str(dataset_issue["対処方法"])


def test_parse_products_retains_unit_metadata_from_template():
    template_bytes = generate_product_template()
    xls = pd.ExcelFile(BytesIO(template_bytes))
    df, warnings = parse_products(xls, sheet_name="R6.12")
    assert not warnings
    info = df.attrs.get("column_unit_info")
    assert info
    assert info["actual_unit_price"]["unit"] == "円/個"
    assert info["minutes_per_unit"]["unit"] == "分/個"


def test_generate_product_template_overrides_industry_defaults():
    template_bytes = generate_product_template("manufacturing")
    xls = pd.ExcelFile(BytesIO(template_bytes))
    hyochin_df = pd.read_excel(xls, sheet_name="標賃", header=None)

    working_days = hyochin_df.loc[hyochin_df[1] == "年間稼働日数", 3].iloc[0]
    daily_hours = hyochin_df.loc[hyochin_df[1] == "1日当り稼働時間", 3].iloc[0]
    fixed_total = hyochin_df.loc[
        hyochin_df[1].isin(["労務費", "販管費", "借入返済", "納税", "未来事業費"]), 3
    ].sum()

    assert working_days == 236
    assert round(float(daily_hours), 2) == 8.68
    assert int(round(float(fixed_total))) == 57_110_000
