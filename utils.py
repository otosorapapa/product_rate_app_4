from __future__ import annotations

from io import BytesIO
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Callable

# --------------- Low-level helpers ---------------
def _clean(s):
    if pd.isna(s):
        return ""
    return str(s).replace("\n", "").strip()

def find_header_row(df: pd.DataFrame, keyword: str) -> int:
    for i in range(len(df)):
        if (df.iloc[i] == keyword).any():
            return i
    return -1

def build_columns_from_two_rows(header_row: pd.Series, unit_row: pd.Series) -> List[str]:
    cols = []
    for h, u in zip(header_row, unit_row):
        h2 = _clean(h); u2 = _clean(u)
        if not h2:
            cols.append("")
        elif u2:
            cols.append(f"{h2} ({u2})")
        else:
            cols.append(h2)
    return cols

def series_or_nan(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a numeric Series or NaN Series aligned to df.index."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")

def classify_by_rate(
    va_per_min: float, required_rate: float, low_ratio: float = 0.95, high_ratio: float = 1.05
) -> str:
    """製品を賃率で分類する。

    `va_per_min / required_rate` の比率 (δ) が `high_ratio` 以上なら「健康商品」、
    `low_ratio` 以上 `high_ratio` 未満なら「貧血商品」、それ以外は「出血商品」とする。
    `required_rate` が 0 もしくは NaN の場合は分類不能として "不明" を返す。
    """
    if pd.isna(va_per_min) or required_rate in [0, None] or pd.isna(required_rate):
        return "不明"
    delta = va_per_min / required_rate
    if delta >= high_ratio:
        return "健康商品"
    if delta >= low_ratio:
        return "貧血商品"
    return "出血商品"

# --------------- Excel parsing ---------------
def read_excel_safely(path_or_bytes) -> Optional[pd.ExcelFile]:
    try:
        xls = pd.ExcelFile(path_or_bytes, engine="openpyxl")
        return xls
    except Exception:
        return None

def parse_hyochin(xls: pd.ExcelFile) -> Tuple[Dict[str, float], Dict[str, float], List[str]]:
    """『標賃』シートから諸元を抽出し、賃率を再計算"""
    from standard_rate_core import DEFAULT_PARAMS, sanitize_params, compute_rates

    warnings: List[str] = []
    try:
        df = pd.read_excel(xls, sheet_name="標賃", header=None)
    except Exception as e:
        warnings.append(f"シート『標賃』が読めません: {e}")
        return {}, DEFAULT_PARAMS.copy(), warnings

    def find_value(col1_kw: str | None = None, col2_kw: str | None = None) -> Optional[float]:
        mask = pd.Series(True, index=df.index)
        if col1_kw:
            mask &= df.iloc[:, 1].astype(str).str.contains(col1_kw, na=False)
        if col2_kw:
            mask &= df.iloc[:, 2].astype(str).str.contains(col2_kw, na=False)
        rows = df[mask]
        if rows.empty:
            return None
        row = rows.iloc[0]
        for x in row[::-1]:
            try:
                return float(x)
            except Exception:
                continue
        return None

    extracted = {
        "labor_cost": find_value("労務費"),
        "sga_cost": find_value("販管費"),
        "loan_repayment": find_value("借入返済"),
        "tax_payment": find_value("納税"),
        "future_business": find_value("未来事業費"),
        "fulltime_workers": find_value("直接工員数", "正社員"),
        "part1_workers": find_value(col2_kw="準社員➀"),
        "part2_workers": find_value(col2_kw="準社員②"),
        "working_days": find_value("年間稼働日数"),
        "daily_hours": find_value("1日当り稼働時間"),
        "operation_rate": find_value("1日当り操業度"),
    }

    # part2 coefficient (row after 準社員②)
    part2_coef = None
    rows = df[df.iloc[:, 2].astype(str).str.contains("準社員②", na=False)]
    if not rows.empty:
        idx = rows.index[0]
        for i in range(idx + 1, min(idx + 4, len(df))):
            if "労働係数" in str(df.iloc[i, 2]):
                try:
                    part2_coef = float(df.iloc[i, 3])
                except Exception:
                    pass
                break
    extracted["part2_coefficient"] = part2_coef

    sr_params: Dict[str, float] = {}
    for k, default in DEFAULT_PARAMS.items():
        v = extracted.get(k)
        if v is None:
            warnings.append(f"{k} をExcelから取得できませんでした。既定値を使用します。")
            sr_params[k] = default
        else:
            sr_params[k] = v

    sr_params, warn2 = sanitize_params(sr_params)
    warnings.extend(warn2)
    _, flat = compute_rates(sr_params)
    params = {k: flat[k] for k in [
        "fixed_total",
        "required_profit_total",
        "annual_minutes",
        "break_even_rate",
        "required_rate",
        "daily_be_va",
        "daily_req_va",
    ]}
    return params, sr_params, warnings

def parse_products(xls: pd.ExcelFile, sheet_name: str="R6.12") -> Tuple[pd.DataFrame, List[str]]:
    """『R6.12』の製品マスタを構造化"""
    warnings: List[str] = []
    try:
        raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)
    except Exception as e:
        warnings.append(f"シート『{sheet_name}』が読めません: {e}")
        return pd.DataFrame(), warnings

    hdr_row = find_header_row(raw, "製品№")
    if hdr_row < 0:
        warnings.append("『製品№』行が見つかりません。")
        return pd.DataFrame(), warnings

    header_row = raw.iloc[hdr_row]
    unit_row = raw.iloc[hdr_row+1] if hdr_row+1 < len(raw) else pd.Series(dtype=object)
    cols = build_columns_from_two_rows(header_row, unit_row)
    data = raw.iloc[hdr_row+2:].reset_index(drop=True)
    if len(cols) != data.shape[1]:
        data = data.iloc[:, :len(cols)]
    data.columns = cols
    data.columns = [c.replace("\n","") if isinstance(c,str) else c for c in data.columns]

    keep_candidates = [
        "製品№ (1)", "製品名 (大福生地)", "実際売単価", "必要販売単価", "損益分岐単価", "必要単価",
        "外注費", "原価（材料費）", "粗利 (0)", "月間製造数(個数）", "月間売上 (0)", "月間支払 (0)",
        "付加価値率", "日産製造数（個数）", "合計 (151)", "付加価値", "1分当り付加価値", "時", "分",
        "受注数当り付加価値/日 (0)", "1分当り付加価値2 (0)",
        "製品№", "製品番号", "製品番号 (コード)", "製品№ (コード)",
        "製品名", "製品名 (名称)",
        "実際売単価 (円/個)", "販売単価", "販売単価 (円/個)", "販売価格", "販売価格 (円/個)",
        "原価（材料費） (円/個)", "材料費", "材料費 (円/個)",
        "リードタイム", "リードタイム (分/個)", "タクトタイム", "タクトタイム (分/個)",
        "日産製造数（個数） (個/日)", "日産数", "日産数 (個/日)",
        "備考", "備考 (任意)",
    ]
    keep = [k for k in dict.fromkeys(keep_candidates) if k in data.columns]
    df = data[keep].copy()

    def to_float(x):
        try:
            if x in ["", None, np.nan]:
                return np.nan
            return float(str(x).replace(",", ""))
        except Exception:
            return np.nan

    text_columns = {
        "製品№ (1)", "製品名 (大福生地)", "製品№", "製品番号", "製品番号 (コード)",
        "製品№ (コード)", "製品名", "製品名 (名称)", "備考", "備考 (任意)"
    }
    for col in df.columns:
        if col not in text_columns:
            df[col] = df[col].map(to_float)

    rename_map = {
        "製品№ (1)": "product_no",
        "製品名 (大福生地)": "product_name",
        "製品№": "product_no",
        "製品番号": "product_no",
        "製品番号 (コード)": "product_no",
        "製品№ (コード)": "product_no",
        "製品名": "product_name",
        "製品名 (名称)": "product_name",
        "実際売単価": "actual_unit_price",
        "実際売単価 (円/個)": "actual_unit_price",
        "販売単価": "actual_unit_price",
        "販売単価 (円/個)": "actual_unit_price",
        "販売価格": "actual_unit_price",
        "販売価格 (円/個)": "actual_unit_price",
        "原価（材料費）": "material_unit_cost",
        "原価（材料費） (円/個)": "material_unit_cost",
        "材料費": "material_unit_cost",
        "材料費 (円/個)": "material_unit_cost",
        "リードタイム": "minutes_per_unit",
        "リードタイム (分/個)": "minutes_per_unit",
        "タクトタイム": "minutes_per_unit",
        "タクトタイム (分/個)": "minutes_per_unit",
        "日産製造数（個数）": "daily_qty",
        "日産製造数（個数） (個/日)": "daily_qty",
        "日産数": "daily_qty",
        "日産数 (個/日)": "daily_qty",
        "分": "minutes_per_unit",
        "合計 (151)": "daily_total_minutes",
        "付加価値": "daily_va",
        "1分当り付加価値": "va_per_min",
        "必要販売単価": "required_selling_price_excel",
        "損益分岐単価": "be_unit_price_excel",
        "必要単価": "req_va_unit_price_excel",
        "粗利 (0)": "gp_per_unit_excel",
        "備考": "notes",
        "備考 (任意)": "notes",
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})

    # Core fields
    df["gp_per_unit"] = df.get("actual_unit_price", np.nan) - df.get("material_unit_cost", np.nan)

    # Safe compute minutes_per_unit
    if "minutes_per_unit" not in df.columns:
        df["minutes_per_unit"] = np.nan
    numer = series_or_nan(df, "daily_total_minutes")
    denom = series_or_nan(df, "daily_qty").replace({0: np.nan})
    with np.errstate(divide='ignore', invalid='ignore'):
        computed_mpu = numer / denom
    df["minutes_per_unit"] = df["minutes_per_unit"].fillna(computed_mpu)

    # Safe compute daily_total_minutes
    if "daily_total_minutes" not in df.columns:
        df["daily_total_minutes"] = np.nan
    mpu = series_or_nan(df, "minutes_per_unit")
    qty = series_or_nan(df, "daily_qty")
    with np.errstate(divide='ignore', invalid='ignore'):
        computed_total = mpu * qty
    df["daily_total_minutes"] = df["daily_total_minutes"].fillna(computed_total)

    # daily_va
    if "daily_va" not in df.columns:
        df["daily_va"] = np.nan
    gpu = series_or_nan(df, "gp_per_unit")
    with np.errstate(divide='ignore', invalid='ignore'):
        computed_va = gpu * qty
    df["daily_va"] = df["daily_va"].fillna(computed_va)

    # va_per_min
    if "va_per_min" not in df.columns:
        df["va_per_min"] = np.nan
    total_min = series_or_nan(df, "daily_total_minutes").replace({0: np.nan})
    with np.errstate(divide='ignore', invalid='ignore'):
        computed_vapm = df["daily_va"] / total_min
    df["va_per_min"] = df["va_per_min"].fillna(computed_vapm)

    df = df[~(df.get("product_name").isna() & df.get("actual_unit_price").isna())].reset_index(drop=True)
    return df, warnings


TEMPLATE_FIELD_GUIDE: List[Dict[str, Any]] = [
    {
        "Excel列名": "製品№",
        "説明": "製品コード。半角英数字やハイフンで記入します。",
        "単位/形式": "コード",
        "サンプル値": "P-1001",
    },
    {
        "Excel列名": "製品名",
        "説明": "製品・SKUの正式名称。",
        "単位/形式": "テキスト",
        "サンプル値": "苺大福",
    },
    {
        "Excel列名": "実際売単価",
        "説明": "1個あたりの実際販売価格（税抜）。",
        "単位/形式": "円/個",
        "サンプル値": 320,
    },
    {
        "Excel列名": "原価（材料費）",
        "説明": "1個あたりの材料費。副資材も含める場合は合算してください。",
        "単位/形式": "円/個",
        "サンプル値": 120,
    },
    {
        "Excel列名": "リードタイム",
        "説明": "1個を製造するのに必要な時間。分単位で入力します。",
        "単位/形式": "分/個",
        "サンプル値": 4.5,
    },
    {
        "Excel列名": "日産製造数（個数）",
        "説明": "1日あたりの生産数量（能力値）。",
        "単位/形式": "個/日",
        "サンプル値": 800,
    },
    {
        "Excel列名": "備考",
        "説明": "任意入力欄。ライン名や補足情報などがあれば記入します。",
        "単位/形式": "任意",
        "サンプル値": "既存ラインA",
    },
]


def get_product_template_guide() -> pd.DataFrame:
    """Return a DataFrame describing the Excel input template columns."""

    return pd.DataFrame(TEMPLATE_FIELD_GUIDE)


def generate_product_template() -> bytes:
    """Generate a starter Excel workbook for product data entry."""

    guide_df = get_product_template_guide()

    product_header = [
        "製品№",
        "製品名",
        "実際売単価",
        "原価（材料費）",
        "リードタイム",
        "日産製造数（個数）",
        "備考",
    ]
    product_units = ["コード", "名称", "円/個", "円/個", "分/個", "個/日", "任意"]
    product_samples = [
        ["P-1001", "苺大福", 320, 120, 4.5, 800, "既存ラインA"],
        ["P-1002", "栗大福", 280, 110, 3.8, 600, "新規投入予定"],
    ]
    product_rows = [product_header, product_units, *product_samples]
    template_df = pd.DataFrame(product_rows)

    hyochin_columns = ["", "項目", "区分", "値", "補足"]
    hyochin_rows = [
        ["", "労務費", "", 24_000_000, ""],
        ["", "販管費", "", 12_000_000, ""],
        ["", "借入返済", "", 3_600_000, ""],
        ["", "納税", "", 3_000_000, ""],
        ["", "未来事業費", "", 1_200_000, ""],
        ["", "直接工員数", "正社員", 25, ""],
        ["", "", "準社員➀", 8, ""],
        ["", "", "準社員②", 5, ""],
        ["", "", "労働係数", 0.8, "準社員②の労働係数"],
        ["", "年間稼働日数", "", 250, ""],
        ["", "1日当り稼働時間", "", 7.5, ""],
        ["", "1日当り操業度", "", 0.85, ""],
    ]
    hyochin_df = pd.DataFrame(hyochin_rows, columns=hyochin_columns)

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        guide_df.to_excel(writer, sheet_name="入力ガイド", index=False)
        template_df.to_excel(writer, sheet_name="R6.12", header=False, index=False)
        hyochin_df.to_excel(writer, sheet_name="標賃", header=False, index=False)

    return buffer.getvalue()


def validate_product_dataframe(
    df: pd.DataFrame,
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """Validate core product master fields and surface actionable issues.

    Returns
    -------
    (errors, warnings, detail_df)
        ``errors`` と ``warnings`` は画面表示用のメッセージ文字列、
        ``detail_df`` は問題のある製品の一覧です。
    """

    errors: List[str] = []
    warnings: List[str] = []
    detail_rows: List[Dict[str, Any]] = []

    column_info: Dict[str, Dict[str, Any]] = {
        "product_no": {"label": "製品番号", "missing_level": "error"},
        "product_name": {"label": "製品名", "missing_level": "error"},
        "actual_unit_price": {
            "label": "販売単価（円/個）",
            "missing_level": "warning",
        },
        "material_unit_cost": {
            "label": "材料費（円/個）",
            "missing_level": "warning",
        },
        "minutes_per_unit": {
            "label": "リードタイム（分/個）",
            "missing_level": "warning",
        },
        "daily_qty": {
            "label": "日産数（個/日）",
            "missing_level": "warning",
        },
    }

    missing_columns = [
        meta["label"]
        for col, meta in column_info.items()
        if col not in df.columns
    ]
    for label in missing_columns:
        errors.append(
            f"{label}の列が見つかりません。公式テンプレートを使用し、列名と単位を確認してください。"
        )
    if missing_columns:
        return errors, warnings, pd.DataFrame(
            columns=["レベル", "製品番号", "製品名", "項目", "内容", "入力値"]
        )

    def resolve_detail(detail: Any, row: pd.Series) -> Any:
        if callable(detail):
            return detail(row)
        return detail

    def register_issue(
        mask: pd.Series,
        column_key: str,
        level: str,
        message: str,
        detail: Any,
        value_column: Optional[str] = None,
    ) -> None:
        count = int(mask.sum())
        if count <= 0:
            return

        label = column_info.get(column_key, {}).get("label", column_key)
        formatted = message.format(count=count, label=label)
        if level == "error":
            errors.append(formatted)
        else:
            warnings.append(formatted)

        value_col = value_column or column_key
        subset_cols = [c for c in ["product_no", "product_name", value_col] if c in df.columns]
        subset = df.loc[mask, subset_cols]
        for _, row in subset.iterrows():
            raw_value = row.get(value_col, "") if value_col in row else ""
            if pd.isna(raw_value):
                raw_value = ""
            detail_rows.append(
                {
                    "レベル": "エラー" if level == "error" else "警告",
                    "製品番号": row.get("product_no"),
                    "製品名": row.get("product_name"),
                    "項目": label,
                    "内容": resolve_detail(detail, row),
                    "入力値": raw_value,
                }
            )

    # Missing values
    for col, meta in column_info.items():
        mask = df[col].isna()
        if mask.any():
            level = meta.get("missing_level", "warning")
            register_issue(
                mask,
                col,
                level,
                "{label}が未入力の製品が{count}件あります。テンプレートのサンプルを参考に入力してください。",
                "未入力",
            )

    # Invalid or suspicious values
    value_checks = [
        {
            "column": "minutes_per_unit",
            "level": "error",
            "message": "リードタイム（分/個）が0以下の製品が{count}件あります。正しい値を入力してください。",
            "detail": "リードタイムが0以下",
            "condition": lambda s: s <= 0,
        },
        {
            "column": "actual_unit_price",
            "level": "error",
            "message": "販売単価（円/個）が0以下の製品が{count}件あります。単価を見直してください。",
            "detail": "販売単価が0以下",
            "condition": lambda s: s <= 0,
        },
        {
            "column": "material_unit_cost",
            "level": "error",
            "message": "材料費（円/個）が負の値の製品が{count}件あります。単位や入力値を確認してください。",
            "detail": "材料費が負の値",
            "condition": lambda s: s < 0,
        },
        {
            "column": "daily_qty",
            "level": "error",
            "message": "日産数（個/日）が0以下の製品が{count}件あります。生産能力を入力してください。",
            "detail": "日産数が0以下",
            "condition": lambda s: s <= 0,
        },
        {
            "column": "minutes_per_unit",
            "level": "warning",
            "message": "リードタイム（分/個）が600分を超える製品が{count}件あります。時間の単位（分）を確認してください。",
            "detail": "リードタイムが600分超",
            "condition": lambda s: s > 600,
        },
    ]

    for check in value_checks:
        col = check["column"]
        if col not in df.columns:
            continue
        mask = check["condition"](df[col].astype(float))
        register_issue(mask, col, check["level"], check["message"], check["detail"])

    # Material cost higher than selling price
    if {"actual_unit_price", "material_unit_cost"}.issubset(df.columns):
        mask = df["actual_unit_price"].notna() & df["material_unit_cost"].notna()
        mask &= df["actual_unit_price"] < df["material_unit_cost"]
        register_issue(
            mask,
            "actual_unit_price",
            "warning",
            "材料費が販売単価を上回る製品が{count}件あります。採算を確認してください。",
            lambda row: (
                f"販売単価 {row.get('actual_unit_price')} < 材料費 {row.get('material_unit_cost')}"
            ),
            value_column="actual_unit_price",
        )

    # Duplicate product numbers
    if "product_no" in df.columns:
        dup_mask = df["product_no"].duplicated(keep=False)
        register_issue(
            dup_mask,
            "product_no",
            "error",
            "製品番号が重複している行が{count}件あります。各製品に一意の番号を設定してください。",
            "製品番号が重複",
        )

    detail_df = pd.DataFrame(
        detail_rows,
        columns=["レベル", "製品番号", "製品名", "項目", "内容", "入力値"],
    )

    return errors, warnings, detail_df

# --------------- Core compute ---------------
def compute_results(
    df_products: pd.DataFrame,
    break_even_rate: float,
    required_rate: float,
    low_ratio: float = 0.95,
    high_ratio: float = 1.05,
) -> pd.DataFrame:
    df = df_products.copy()
    be_rate = 0.0 if break_even_rate is None else float(break_even_rate)
    req_rate = 0.0 if required_rate is None else float(required_rate)

    mpu = series_or_nan(df, "minutes_per_unit")
    df["be_va_unit_price"] = mpu * be_rate
    df["req_va_unit_price"] = mpu * req_rate
    df["required_selling_price"] = df.get("material_unit_cost") + df["req_va_unit_price"]
    df["price_gap_vs_required"] = df.get("actual_unit_price") - df["required_selling_price"]
    df["rate_gap_vs_required"] = df.get("va_per_min") - req_rate
    df["meets_required_rate"] = df["rate_gap_vs_required"] >= 0
    df["rate_class"] = df["va_per_min"].apply(
        lambda v: classify_by_rate(v, req_rate, low_ratio, high_ratio)
    )
    out_cols = [
        "product_no","product_name",
        "actual_unit_price","material_unit_cost",
        "minutes_per_unit","daily_qty","daily_total_minutes",
        "gp_per_unit","daily_va","va_per_min",
        "be_va_unit_price","req_va_unit_price","required_selling_price",
        "price_gap_vs_required","rate_gap_vs_required","meets_required_rate","rate_class"
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    return df[out_cols]


def detect_quality_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Detect missing values, negative outliers and duplicate SKUs.

    Parameters
    ----------
    df : pd.DataFrame
        Product master data.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: product_no, product_name, type, column.
    """
    issues: List[Dict[str, Any]] = []
    required_cols = [
        "product_no",
        "product_name",
        "actual_unit_price",
        "material_unit_cost",
        "minutes_per_unit",
        "daily_qty",
    ]

    # Missing values
    for col in required_cols:
        if col in df.columns:
            mask = df[col].isna()
            for sku, name in df.loc[mask, ["product_no", "product_name"]].itertuples(index=False):
                issues.append(
                    {
                        "product_no": sku,
                        "product_name": name,
                        "type": "欠損",
                        "column": col,
                    }
                )

    # Negative numeric outliers
    numeric_cols = [
        c
        for c in [
            "actual_unit_price",
            "material_unit_cost",
            "minutes_per_unit",
            "daily_qty",
        ]
        if c in df.columns
    ]
    for col in numeric_cols:
        mask = df[col] < 0
        for sku, name in df.loc[mask, ["product_no", "product_name"]].itertuples(index=False):
            issues.append(
                {
                    "product_no": sku,
                    "product_name": name,
                    "type": "外れ値",
                    "column": col,
                }
            )

    # Duplicate SKUs
    if "product_no" in df.columns:
        dup_skus = df[df.duplicated(subset="product_no", keep=False)]["product_no"].unique()
        for sku in dup_skus:
            name = ""
            if "product_name" in df.columns:
                name = df[df["product_no"] == sku]["product_name"].iloc[0]
            issues.append(
                {
                    "product_no": sku,
                    "product_name": name,
                    "type": "重複",
                    "column": "product_no",
                }
            )

    return pd.DataFrame(issues, columns=["product_no", "product_name", "type", "column"])


def detect_anomalies(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    z_threshold: float = 3.5,
) -> pd.DataFrame:
    """Detect statistical outliers for the specified metrics.

    The function uses the modified Z-score based on the median absolute
    deviation (MAD) to robustly flag values that are far away from the
    center of the distribution. IQR-based thresholds are also returned for
    reference so that users can compare where the value sits versus the
    typical range.

    Parameters
    ----------
    df : pd.DataFrame
        Product level dataset.
    metrics : list[str] | None
        Target metric column names. When ``None`` a sensible default set of
        monetory/throughput metrics is used.
    z_threshold : float
        Absolute modified Z-score threshold.

    Returns
    -------
    pd.DataFrame
        Columns: ``product_no``, ``product_name``, ``metric``, ``value``,
        ``direction`` (high/low), ``severity`` (abs(modified z-score)),
        ``median`` and ``iqr_bounds`` for context.
    """

    if metrics is None:
        metrics = [
            "va_per_min",
            "minutes_per_unit",
            "actual_unit_price",
            "material_unit_cost",
            "daily_qty",
            "rate_gap_vs_required",
        ]

    anomalies: List[Dict[str, Any]] = []
    for col in metrics:
        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce")
        clean = series.dropna()
        if clean.empty:
            continue

        median = float(clean.median())
        mad = float(np.median(np.abs(clean - median)))
        if mad == 0 or np.isnan(mad):
            std = float(clean.std(ddof=0))
            if std == 0 or np.isnan(std):
                continue
            z_scores = (clean - median) / std
        else:
            z_scores = 0.6745 * (clean - median) / mad

        mask = z_scores.abs() >= z_threshold
        if not mask.any():
            continue

        q1 = float(clean.quantile(0.25))
        q3 = float(clean.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        for idx, score in z_scores[mask].items():
            row = df.loc[idx]
            value = row.get(col)
            anomalies.append(
                {
                    "product_no": row.get("product_no"),
                    "product_name": row.get("product_name"),
                    "metric": col,
                    "value": float(value) if pd.notna(value) else np.nan,
                    "direction": "high" if value >= median else "low",
                    "severity": float(abs(score)),
                    "median": median,
                    "iqr_lower": float(lower),
                    "iqr_upper": float(upper),
                }
            )

    if not anomalies:
        return pd.DataFrame(
            columns=[
                "product_no",
                "product_name",
                "metric",
                "value",
                "direction",
                "severity",
                "median",
                "iqr_lower",
                "iqr_upper",
            ]
        )

    result = pd.DataFrame(anomalies)
    result.sort_values("severity", ascending=False, inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result
