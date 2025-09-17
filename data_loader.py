"""データ読込と品質検証ロジックを集約したモジュール."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------ 定数 ------------------------------

REQUIRED_COLUMNS: List[str] = [
    "product_no",
    "product_name",
    "actual_unit_price",
    "material_unit_cost",
    "minutes_per_unit",
    "va_per_min",
]
SUPPORTED_EXTENSIONS = {".xlsx", ".xls", ".csv", ".json"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB 制限


@dataclass
class DataValidationResult:
    """データ検証の結果を保持する単純なデータクラス."""

    issues: pd.DataFrame
    summary: pd.DataFrame


# ------------------------------ 基本関数 ------------------------------

def _ensure_file_size_ok(file_obj: Any) -> None:
    """アップロードサイズを検証し、10MBを超える場合は ValueError を投げる."""

    size: Optional[int] = None
    if hasattr(file_obj, "size"):
        size = int(getattr(file_obj, "size"))
    elif isinstance(file_obj, (str, Path)):
        path = Path(file_obj)
        if path.exists():
            size = path.stat().st_size
    if size is not None and size > MAX_FILE_SIZE:
        raise ValueError("ファイルサイズが10MBを超えています。分割してアップロードしてください。")


def _load_from_path(path: Path) -> pd.DataFrame:
    """拡張子に応じて DataFrame を読み込む内部関数."""

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"対応していないファイル形式です: {suffix}")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_json(path)


def _read_dataframe(file_obj: Any) -> pd.DataFrame:
    """ファイルオブジェクトやパスから DataFrame を作成する."""

    _ensure_file_size_ok(file_obj)
    if isinstance(file_obj, (str, Path)):
        return _load_from_path(Path(file_obj))

    if hasattr(file_obj, "name"):
        suffix = Path(file_obj.name).suffix.lower()
    else:
        suffix = ""

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError("Excel/CSV/JSON のみアップロードできます。")

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_obj)
    if suffix == ".csv":
        return pd.read_csv(file_obj)
    return pd.read_json(file_obj)


@st.cache_data(show_spinner=False)
def load_tabular_data(file_obj: Any) -> pd.DataFrame:
    """Streamlit のキャッシュを利用して DataFrame を読み込む."""

    df = _read_dataframe(file_obj)
    return sanitize_dataframe(df)


@st.cache_data(show_spinner=False)
def load_sample_data(path: str = "data/sample_data.xlsx") -> pd.DataFrame:
    """サンプルファイルを読み込み、列名を整形する."""

    return sanitize_dataframe(_load_from_path(Path(path)))


# ------------------------------ 整形処理 ------------------------------

def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """列名・データ型を整備し、欠損を NaN に統一する."""

    clean = df.copy()
    clean.columns = [str(c).strip().lower() for c in clean.columns]
    clean = clean.replace({"": np.nan, "-": np.nan})
    for col in clean.columns:
        if col in {"product_no", "product_name"}:
            continue
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    return clean


def append_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """欠損行を特定し、ハイライト用のフラグ列を付与する."""

    flagged = df.copy()
    if flagged.empty:
        flagged["row_alert"] = ""
        return flagged

    mask_missing = flagged[REQUIRED_COLUMNS].isna().any(axis=1)
    mask_negative = (
        flagged[[c for c in ["actual_unit_price", "material_unit_cost", "minutes_per_unit"] if c in flagged.columns]]
        < 0
    ).any(axis=1)

    status = []
    for miss, neg in zip(mask_missing, mask_negative):
        if miss and neg:
            status.append("⚠️ 欠損・マイナス値あり")
        elif miss:
            status.append("⚠️ 欠損あり")
        elif neg:
            status.append("⚠️ マイナス値あり")
        else:
            status.append("")
    flagged["row_alert"] = status
    return flagged


# ------------------------------ 品質チェック ------------------------------

def detect_quality_issues(df: pd.DataFrame) -> pd.DataFrame:
    """必須列の欠損・負値・重複を一覧化する."""

    issues: List[Dict[str, Any]] = []
    base_cols = [c for c in REQUIRED_COLUMNS if c in df.columns]

    for col in base_cols:
        missing_mask = df[col].isna()
        for _, row in df.loc[missing_mask].iterrows():
            issues.append(
                {
                    "product_no": row.get("product_no"),
                    "product_name": row.get("product_name"),
                    "type": "欠損",
                    "column": col,
                }
            )

    numeric_cols = [
        c
        for c in ["actual_unit_price", "material_unit_cost", "minutes_per_unit", "va_per_min"]
        if c in df.columns
    ]
    for col in numeric_cols:
        neg_mask = df[col] < 0
        for _, row in df.loc[neg_mask].iterrows():
            issues.append(
                {
                    "product_no": row.get("product_no"),
                    "product_name": row.get("product_name"),
                    "type": "外れ値",
                    "column": col,
                }
            )

    if "product_no" in df.columns:
        duplicated = df[df["product_no"].duplicated(keep=False)]["product_no"].unique()
        for sku in duplicated:
            sample_name = df[df["product_no"] == sku]["product_name"].iloc[0]
            issues.append(
                {
                    "product_no": sku,
                    "product_name": sample_name,
                    "type": "重複",
                    "column": "product_no",
                }
            )

    return pd.DataFrame(issues, columns=["product_no", "product_name", "type", "column"])


def detect_anomalies(
    df: pd.DataFrame,
    metrics: Optional[Iterable[str]] = None,
    z_threshold: float = 3.5,
) -> pd.DataFrame:
    """修正 Z スコアを用いた統計的外れ値検出を行う."""

    if metrics is None:
        metrics = ["va_per_min", "minutes_per_unit", "actual_unit_price", "material_unit_cost"]

    findings: List[Dict[str, Any]] = []
    for col in metrics:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        series = series.replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            continue
        median = float(series.median())
        mad = float(np.median(np.abs(series - median)))
        if mad == 0:
            continue
        modified_z = 0.6745 * (series - median) / mad
        mask = modified_z.abs() >= z_threshold
        if not mask.any():
            continue
        for idx in series[mask].index:
            value = series.loc[idx]
            findings.append(
                {
                    "product_no": df.loc[idx, "product_no"] if "product_no" in df.columns else idx,
                    "product_name": df.loc[idx, "product_name"] if "product_name" in df.columns else "",
                    "metric": col,
                    "value": value,
                    "direction": "high" if modified_z.loc[idx] > 0 else "low",
                    "severity": float(abs(modified_z.loc[idx])),
                    "median": median,
                }
            )

    return pd.DataFrame(findings)


def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """代表的な統計量をまとめた DataFrame を返す."""

    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.DataFrame()
    summary = numeric.describe().T
    summary = summary.rename(columns={
        "mean": "平均値",
        "std": "標準偏差",
        "min": "最小値",
        "max": "最大値",
    })
    ordered_cols = ["count", "平均値", "標準偏差", "最小値", "25%", "50%", "75%", "最大値"]
    return summary[[c for c in ordered_cols if c in summary.columns]]


def validate_dataframe(df: pd.DataFrame) -> DataValidationResult:
    """品質検証とサマリ統計を同時に実行するヘルパー."""

    return DataValidationResult(
        issues=detect_quality_issues(df),
        summary=summarize_dataframe(df),
    )
