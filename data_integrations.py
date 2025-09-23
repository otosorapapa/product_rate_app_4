from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, IO, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


SUPPORTED_ACCOUNTING_APPS: Dict[str, str] = {
    "弥生会計": "yayoi",
    "freee会計": "freee",
    "MFクラウド会計": "mf_cloud",
}
SUPPORTED_POS_SYSTEMS: Dict[str, str] = {
    "スマレジPOS": "smaregi",
    "Square POS": "square",
}

_BASE_DIR = Path(__file__).resolve().parent
_SAMPLE_DATA_DIR = _BASE_DIR / "data" / "external"

_SAMPLE_DATA_FILES: Dict[Tuple[str, str], Path] = {
    ("accounting", vendor): _SAMPLE_DATA_DIR / f"{code}_transactions_sample.csv"
    for vendor, code in SUPPORTED_ACCOUNTING_APPS.items()
}
_SAMPLE_DATA_FILES.update(
    {
        ("pos", vendor): _SAMPLE_DATA_DIR / "pos_transactions_sample.csv"
        for vendor in SUPPORTED_POS_SYSTEMS
    }
)

_REQUIRED_COLUMNS = {"date", "product_no", "sold_qty", "sales_amount", "material_cost"}
_OPTIONAL_COLUMNS = {"work_minutes"}
_COLUMN_ORDER = [
    "date",
    "product_no",
    "sold_qty",
    "sales_amount",
    "material_cost",
    "work_minutes",
]


@dataclass
class AutoSyncResult:
    """Result bundle returned by :func:`auto_sync_transactions`."""

    transactions: pd.DataFrame
    summary: pd.DataFrame
    logs: List[Dict[str, Any]]


@dataclass
class IntegrationConfig:
    """Connection settings for external production or sales systems."""

    source_type: str
    vendor: str
    credential_key: Optional[str] = None
    data_path: Optional[Union[str, Path]] = None
    schedule: str = "daily"
    extra_params: Dict[str, Union[str, int, float]] = field(default_factory=dict)

    def resolve_sample_path(self) -> Optional[Path]:
        """Return a built-in sample CSV path for the configuration if available."""

        return _SAMPLE_DATA_FILES.get((self.source_type, self.vendor))


def _read_transactions_csv(source: Union[str, Path, IO[str], IO[bytes]]) -> pd.DataFrame:
    """Read a CSV file from path or buffer and return as DataFrame."""

    if hasattr(source, "seek"):
        source.seek(0)
    try:
        df = pd.read_csv(source, dtype=str)
    except UnicodeDecodeError:
        if hasattr(source, "seek"):
            source.seek(0)
        df = pd.read_csv(source, dtype=str, encoding="utf-8-sig")
    return df


def _normalise_transactions(df: pd.DataFrame, vendor: str) -> pd.DataFrame:
    """Validate and coerce transaction records."""

    missing = _REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            "取込データに必須列が不足しています: " + ", ".join(sorted(missing))
        )

    work = df.copy()
    for col in _REQUIRED_COLUMNS.union(_OPTIONAL_COLUMNS):
        if col not in work.columns:
            work[col] = np.nan

    columns_to_keep = [col for col in _COLUMN_ORDER if col in work.columns]
    work = work[columns_to_keep].copy()

    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date", "product_no"])
    work["product_no"] = work["product_no"].astype(str).str.strip()
    work = work[work["product_no"] != ""]

    def _to_numeric(series: pd.Series) -> pd.Series:
        text = series.astype(str).str.replace(",", "", regex=False).str.strip()
        text = text.replace({"": np.nan, "nan": np.nan})
        return pd.to_numeric(text, errors="coerce")

    for col in ["sold_qty", "sales_amount", "material_cost", "work_minutes"]:
        work[col] = _to_numeric(work[col])

    work = work.dropna(subset=["sold_qty", "sales_amount", "material_cost"])
    work = work[work["sold_qty"] > 0]
    work["source"] = vendor
    work = work.sort_values(["date", "product_no"]).reset_index(drop=True)
    return work


def load_transactions_for_sync(
    config: IntegrationConfig,
    *,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    csv_file: Optional[Union[IO[str], IO[bytes]]] = None,
) -> pd.DataFrame:
    """Load external sales/cost records for synchronisation."""

    if start_date and end_date and start_date > end_date:
        raise ValueError("開始日は終了日以前に設定してください。")

    if csv_file is not None:
        raw_df = _read_transactions_csv(csv_file)
    else:
        source_path: Optional[Path]
        if config.data_path is not None:
            source_path = Path(config.data_path)
        else:
            source_path = config.resolve_sample_path()
        if source_path is None or not source_path.exists():
            raise FileNotFoundError(
                "指定されたシステムのサンプルデータが見つかりません。CSVを指定してください。"
            )
        raw_df = _read_transactions_csv(source_path)

    transactions = _normalise_transactions(raw_df, vendor=config.vendor)

    if start_date:
        transactions = transactions[transactions["date"] >= pd.Timestamp(start_date)]
    if end_date:
        transactions = transactions[transactions["date"] <= pd.Timestamp(end_date)]

    return transactions.reset_index(drop=True)


def summarize_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily transactions into SKU-level metrics."""

    summary_columns = [
        "product_no",
        "total_days",
        "total_qty",
        "total_sales_amount",
        "total_material_cost",
        "avg_unit_price",
        "avg_material_cost",
        "avg_gp_per_unit",
        "avg_daily_qty",
        "avg_minutes_per_unit",
        "source",
    ]

    if transactions.empty:
        return pd.DataFrame(columns=summary_columns)

    grouped = (
        transactions.groupby("product_no", dropna=False)
        .agg(
            total_days=("date", lambda s: int(s.dt.normalize().nunique())),
            total_qty=("sold_qty", "sum"),
            total_sales_amount=("sales_amount", "sum"),
            total_material_cost=("material_cost", "sum"),
            total_work_minutes=("work_minutes", "sum"),
            source=("source", lambda s: s.dropna().iloc[-1] if not s.dropna().empty else None),
        )
        .reset_index()
    )

    qty = grouped["total_qty"].replace({0: np.nan})
    days = grouped["total_days"].replace({0: np.nan})

    with np.errstate(divide="ignore", invalid="ignore"):
        grouped["avg_unit_price"] = grouped["total_sales_amount"] / qty
        grouped["avg_material_cost"] = grouped["total_material_cost"] / qty
        grouped["avg_gp_per_unit"] = (
            grouped["avg_unit_price"] - grouped["avg_material_cost"]
        )
        grouped["avg_daily_qty"] = grouped["total_qty"] / days
        grouped["avg_minutes_per_unit"] = grouped["total_work_minutes"] / qty

    grouped.loc[grouped["total_work_minutes"].isna(), "avg_minutes_per_unit"] = np.nan
    grouped = grouped.drop(columns=["total_work_minutes"])

    return grouped[summary_columns].reset_index(drop=True)


def auto_sync_transactions(
    configs: Iterable[IntegrationConfig],
    *,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> AutoSyncResult:
    """Fetch and aggregate transactions from multiple external systems.

    Parameters
    ----------
    configs:
        An iterable of :class:`IntegrationConfig` describing the systems to
        poll.  Each configuration is processed independently so that
        connection errors for one vendor do not abort the remaining syncs.
    start_date, end_date:
        Optional date range filters applied to every connector.  When omitted
        the full sample range is used.

    Returns
    -------
    AutoSyncResult
        Dataclass containing the merged transaction records, their
        SKU-level summary and a per-vendor status log.
    """

    config_list = list(configs)
    if start_date and end_date and start_date > end_date:
        raise ValueError("開始日は終了以前に設定してください。")

    frames: List[pd.DataFrame] = []
    logs: List[Dict[str, Any]] = []

    for config in config_list:
        try:
            df = load_transactions_for_sync(
                config, start_date=start_date, end_date=end_date
            )
        except (FileNotFoundError, ValueError) as exc:
            logs.append(
                {
                    "vendor": config.vendor,
                    "source_type": config.source_type,
                    "schedule": config.schedule,
                    "status": "error",
                    "error": str(exc),
                    "records": 0,
                }
            )
            continue

        if df.empty:
            logs.append(
                {
                    "vendor": config.vendor,
                    "source_type": config.source_type,
                    "schedule": config.schedule,
                    "status": "empty",
                    "records": 0,
                }
            )
            continue

        df = df.copy()
        df["source_type"] = config.source_type
        frames.append(df)

        period_start, period_end = extract_transaction_period(df)
        logs.append(
            {
                "vendor": config.vendor,
                "source_type": config.source_type,
                "schedule": config.schedule,
                "status": "success",
                "records": int(len(df)),
                "period_start": period_start.isoformat() if period_start else None,
                "period_end": period_end.isoformat() if period_end else None,
            }
        )

    if frames:
        transactions = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["date", "product_no"])
            .reset_index(drop=True)
        )
    else:
        transactions = pd.DataFrame(columns=_COLUMN_ORDER + ["source", "source_type"])

    summary = summarize_transactions(transactions)
    return AutoSyncResult(transactions=transactions, summary=summary, logs=logs)


def apply_transaction_summary_to_products(
    df_products: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    vendor: str,
    synced_at: datetime,
) -> pd.DataFrame:
    """Merge aggregated metrics into the product master."""

    result = df_products.copy()
    if result.empty or summary.empty:
        if "last_synced_source" not in result.columns:
            result["last_synced_source"] = pd.Series(
                pd.NA, index=result.index, dtype="object"
            )
        else:
            result["last_synced_source"] = result["last_synced_source"].astype(
                "object"
            )
        if "last_synced_at" not in result.columns:
            result["last_synced_at"] = pd.NaT
        return result

    if "product_no" not in result.columns:
        raise ValueError("製品マスタにproduct_no列がありません。")

    if "last_synced_source" not in result.columns:
        result["last_synced_source"] = pd.Series(
            pd.NA, index=result.index, dtype="object"
        )
    else:
        result["last_synced_source"] = result["last_synced_source"].astype("object")
    if "last_synced_at" not in result.columns:
        result["last_synced_at"] = pd.NaT

    updated_indices: set[int] = set()
    summary_indexed = summary.set_index("product_no")

    for product_no, metrics in summary_indexed.iterrows():
        mask = result["product_no"].astype(str) == str(product_no)
        if not mask.any():
            continue
        updated_indices.update(result.index[mask].tolist())

        if not pd.isna(metrics.get("avg_unit_price")):
            result.loc[mask, "actual_unit_price"] = metrics["avg_unit_price"]
        if not pd.isna(metrics.get("avg_material_cost")):
            result.loc[mask, "material_unit_cost"] = metrics["avg_material_cost"]
        if not pd.isna(metrics.get("avg_gp_per_unit")):
            if "gp_per_unit" not in result.columns:
                result["gp_per_unit"] = np.nan
            result.loc[mask, "gp_per_unit"] = metrics["avg_gp_per_unit"]
        if not pd.isna(metrics.get("avg_daily_qty")):
            if "daily_qty" not in result.columns:
                result["daily_qty"] = np.nan
            result.loc[mask, "daily_qty"] = metrics["avg_daily_qty"]
        if not pd.isna(metrics.get("avg_minutes_per_unit")):
            if "minutes_per_unit" not in result.columns:
                result["minutes_per_unit"] = np.nan
            result.loc[mask, "minutes_per_unit"] = metrics["avg_minutes_per_unit"]

        result.loc[mask, "last_synced_source"] = vendor
        result.loc[mask, "last_synced_at"] = pd.Timestamp(synced_at)

    if not updated_indices:
        return result

    idx = sorted(updated_indices)
    for col in ["gp_per_unit", "daily_qty", "minutes_per_unit"]:
        if col not in result.columns:
            result[col] = np.nan

    if "daily_total_minutes" not in result.columns:
        result["daily_total_minutes"] = np.nan
    if "daily_va" not in result.columns:
        result["daily_va"] = np.nan
    if "va_per_min" not in result.columns:
        result["va_per_min"] = np.nan

    idx_slice = result.loc[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        idx_slice["gp_per_unit"] = (
            idx_slice["actual_unit_price"] - idx_slice["material_unit_cost"]
        )
        idx_slice["daily_total_minutes"] = (
            idx_slice["minutes_per_unit"] * idx_slice["daily_qty"]
        )
        idx_slice["daily_va"] = idx_slice["gp_per_unit"] * idx_slice["daily_qty"]
        idx_slice["va_per_min"] = idx_slice["daily_va"] / idx_slice["daily_total_minutes"]

    result.loc[idx, "gp_per_unit"] = idx_slice["gp_per_unit"].values
    result.loc[idx, "daily_total_minutes"] = idx_slice["daily_total_minutes"].values
    result.loc[idx, "daily_va"] = idx_slice["daily_va"].values
    result.loc[idx, "va_per_min"] = idx_slice["va_per_min"].values

    return result


def extract_transaction_period(
    transactions: pd.DataFrame,
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Return min/max timestamp of transactions for logging."""

    if transactions.empty or "date" not in transactions.columns:
        return None, None
    return (
        transactions["date"].min().to_pydatetime(),
        transactions["date"].max().to_pydatetime(),
    )
