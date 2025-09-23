import sys
from datetime import date, datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_integrations import (  # noqa: E402
    IntegrationConfig,
    auto_sync_transactions,
    apply_transaction_summary_to_products,
    load_transactions_for_sync,
    summarize_transactions,
)


def test_load_transactions_for_sync_filters_by_date():
    config = IntegrationConfig(source_type="accounting", vendor="弥生会計")
    df = load_transactions_for_sync(
        config,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 1),
    )
    assert not df.empty
    assert df["date"].dt.date.min() >= date(2024, 1, 1)
    assert df["date"].dt.date.max() <= date(2024, 1, 1)
    assert set(df["product_no"]) == {"P-1001", "P-1002", "P-1003"}


def test_summarize_transactions_produces_expected_metrics():
    transactions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-03-01", "2024-03-02", "2024-03-01"]),
            "product_no": ["SKU-1", "SKU-1", "SKU-2"],
            "sold_qty": [10, 5, 8],
            "sales_amount": [1000, 500, 800],
            "material_cost": [400, 200, 320],
            "work_minutes": [40, 20, 24],
            "source": ["dummy", "dummy", "dummy"],
        }
    )
    summary = summarize_transactions(transactions)
    assert set(summary["product_no"]) == {"SKU-1", "SKU-2"}
    sku1 = summary[summary["product_no"] == "SKU-1"].iloc[0]
    assert sku1["total_days"] == 2
    assert sku1["total_qty"] == 15
    assert pytest.approx(sku1["avg_unit_price"], rel=1e-6) == 100.0
    assert pytest.approx(sku1["avg_minutes_per_unit"], rel=1e-6) == 4.0
    sku2 = summary[summary["product_no"] == "SKU-2"].iloc[0]
    assert pytest.approx(sku2["avg_gp_per_unit"], rel=1e-6) == 60.0


def test_apply_transaction_summary_updates_product_master():
    products = pd.DataFrame(
        {
            "product_no": ["P-1001", "P-9999"],
            "product_name": ["苺大福", "テスト商品"],
            "actual_unit_price": [320.0, 100.0],
            "material_unit_cost": [110.0, 40.0],
            "minutes_per_unit": [4.2, 3.0],
            "daily_qty": [80.0, 10.0],
        }
    )
    summary = pd.DataFrame(
        {
            "product_no": ["P-1001"],
            "total_days": [2],
            "total_qty": [180.0],
            "total_sales_amount": [63000.0],
            "total_material_cost": [21600.0],
            "avg_unit_price": [350.0],
            "avg_material_cost": [120.0],
            "avg_gp_per_unit": [230.0],
            "avg_daily_qty": [90.0],
            "avg_minutes_per_unit": [4.5],
            "source": ["弥生会計"],
        }
    )

    updated = apply_transaction_summary_to_products(
        products,
        summary,
        vendor="弥生会計",
        synced_at=datetime(2024, 1, 31, 12, 0, 0),
    )

    target = updated[updated["product_no"] == "P-1001"].iloc[0]
    assert pytest.approx(target["actual_unit_price"], rel=1e-6) == 350.0
    assert pytest.approx(target["material_unit_cost"], rel=1e-6) == 120.0
    assert pytest.approx(target["daily_qty"], rel=1e-6) == 90.0
    assert pytest.approx(target["minutes_per_unit"], rel=1e-6) == 4.5
    expected_va_per_min = (
        (target["actual_unit_price"] - target["material_unit_cost"]) * target["daily_qty"]
    ) / (target["minutes_per_unit"] * target["daily_qty"])
    assert pytest.approx(target["va_per_min"], rel=1e-6) == expected_va_per_min
    expected_total_minutes = target["minutes_per_unit"] * target["daily_qty"]
    assert pytest.approx(target["daily_total_minutes"], rel=1e-6) == expected_total_minutes
    assert target["last_synced_source"] == "弥生会計"
    assert pd.notna(target["last_synced_at"])


def test_load_transactions_for_sync_raises_when_columns_missing():
    csv_buffer = StringIO("date,product_no,sold_qty\n2024-01-01,P-1,10\n")
    config = IntegrationConfig(source_type="accounting", vendor="弥生会計")
    with pytest.raises(ValueError):
        load_transactions_for_sync(config, csv_file=csv_buffer)


def test_auto_sync_transactions_merges_multiple_sources():
    configs = [
        IntegrationConfig(source_type="accounting", vendor="弥生会計"),
        IntegrationConfig(source_type="accounting", vendor="MFクラウド会計"),
        IntegrationConfig(source_type="pos", vendor="スマレジPOS"),
    ]

    result = auto_sync_transactions(
        configs, start_date=date(2024, 1, 1), end_date=date(2024, 1, 2)
    )

    assert not result.transactions.empty
    assert result.summary["product_no"].nunique() >= 1
    assert result.transactions["date"].dt.date.min() >= date(2024, 1, 1)
    assert result.transactions["date"].dt.date.max() <= date(2024, 1, 2)

    vendors = set(result.transactions["source"].unique())
    assert {"弥生会計", "MFクラウド会計", "スマレジPOS"}.intersection(vendors)

    assert result.logs
    for log in result.logs:
        assert log["status"] in {"success", "empty", "error"}
        assert "schedule" in log
