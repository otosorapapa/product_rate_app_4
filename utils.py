from __future__ import annotations

from io import BytesIO
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Callable


CATEGORY_KEYWORDS = {
    "洋菓子": [
        "ケーキ",
        "タルト",
        "プリン",
        "ショコラ",
        "チョコ",
        "シュー",
        "パイ",
        "ムース",
        "マカロン",
    ],
    "和菓子": [
        "大福",
        "饅頭",
        "餅",
        "団子",
        "羊羹",
        "どら",
        "最中",
        "葛",
        "栗",
        "抹茶",
    ],
}


def split_column_and_unit(value: Any) -> Tuple[Any, Optional[str]]:
    """Split a column label into base text and unit text."""

    if not isinstance(value, str):
        return value, None

    text = value.replace("\n", "").strip()
    if not text:
        return text, None

    text = text.replace("（", "(").replace("）", ")")
    if text.endswith(")") and "(" in text:
        base, unit = text.rsplit("(", 1)
        unit = unit[:-1].strip()
        normalized = unit.replace(" ", "").replace("　", "").replace("／", "/").lower()
        unit_keywords = {"コード", "名称", "テキスト", "任意", "code", "name", "text", "optional"}
        if (
            "/" in normalized
            or "円" in unit
            or "個" in unit
            or "分" in unit
            or "日" in unit
            or normalized in unit_keywords
            or unit in unit_keywords
        ):
            return base.strip(), unit
        # treat parentheses as part of the header label (e.g., 製品№ (1))
        return text, None
    return text, None


def normalize_unit_text(unit: Optional[str]) -> Optional[str]:
    """Normalize unit strings for consistent comparison."""

    if unit in [None, ""]:
        return None

    text = str(unit)
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("／", "/")
    text = text.replace(" ", "").replace("　", "")
    text = text.lower()
    if "(" in text:
        text = text.split("(", 1)[0]
    return text


def infer_category_from_name(name: Any) -> str:
    """Infer a coarse product category from its name."""

    if name in [None, ""] or (isinstance(name, float) and pd.isna(name)):
        return "未設定"
    text = str(name)
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return category
    return "その他"


def infer_major_customer(product_no: Any, fallback: Any = None) -> str:
    """Derive a deterministic major customer label for sample data."""

    if product_no in [None, ""] and fallback in [None, ""]:
        return "未設定"
    base = str(product_no if product_no not in [None, ""] else fallback)
    customers = ["百貨店A社", "量販店B社", "通販C社"]
    checksum = sum(ord(ch) for ch in base)
    return customers[checksum % len(customers)]

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
        "カテゴリ", "カテゴリー", "主要顧客", "主要取引先", "メイン顧客",
    ]
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
        "カテゴリ": "category",
        "カテゴリー": "category",
        "主要顧客": "major_customer",
        "主要取引先": "major_customer",
        "メイン顧客": "major_customer",
    }
    candidate_keys = list(dict.fromkeys(keep_candidates))
    base_to_columns: Dict[str, List[str]] = {}
    for col in data.columns:
        if not isinstance(col, str):
            continue
        base, _ = split_column_and_unit(col)
        if isinstance(base, str):
            base_to_columns.setdefault(base, []).append(col)

    selected_columns: List[str] = []
    used_columns: set[str] = set()
    used_targets: set[str] = set()
    for candidate in candidate_keys:
        target_column = None
        if candidate in data.columns and candidate not in used_columns:
            target_column = candidate
        else:
            base_candidate, _ = split_column_and_unit(candidate)
            for actual in base_to_columns.get(base_candidate, []):
                if actual not in used_columns:
                    target_column = actual
                    break
        if target_column is None:
            continue
        base_name, _ = split_column_and_unit(target_column)
        normalized_key = rename_map.get(target_column) or rename_map.get(base_name)
        final_key = normalized_key or base_name
        if final_key in used_targets:
            continue
        selected_columns.append(target_column)
        used_columns.add(target_column)
        used_targets.add(final_key)

    df = data[selected_columns].copy()

    def to_float(x):
        try:
            if x in ["", None, np.nan]:
                return np.nan
            return float(str(x).replace(",", ""))
        except Exception:
            return np.nan

    text_columns = {
        "製品№ (1)", "製品名 (大福生地)", "製品№", "製品番号", "製品番号 (コード)",
        "製品№ (コード)", "製品名", "製品名 (名称)", "備考", "備考 (任意)",
        "カテゴリ", "カテゴリー", "主要顧客", "主要取引先", "メイン顧客",
    }
    for col in df.columns:
        if col not in text_columns:
            df[col] = df[col].map(to_float)

    column_unit_info: Dict[str, Dict[str, Any]] = {}
    original_columns = list(df.columns)
    for col in original_columns:
        base, unit = split_column_and_unit(col)
        normalized_key = rename_map.get(col) or rename_map.get(base)
        final_key = normalized_key or base
        existing = column_unit_info.get(final_key)
        if existing is None or (not existing.get("unit") and unit):
            column_unit_info[final_key] = {"unit": unit, "source": col}

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df.attrs["column_unit_info"] = column_unit_info

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
        "必須": "必須",
        "サンプル値": "P-1001",
    },
    {
        "Excel列名": "製品名",
        "説明": "製品・SKUの正式名称。",
        "単位/形式": "テキスト",
        "必須": "必須",
        "サンプル値": "苺大福",
    },
    {
        "Excel列名": "実際売単価",
        "説明": "1個あたりの実際販売価格（税抜）。",
        "単位/形式": "円/個",
        "必須": "必須",
        "サンプル値": 320,
    },
    {
        "Excel列名": "原価（材料費）",
        "説明": "1個あたりの材料費。副資材も含める場合は合算してください。",
        "単位/形式": "円/個",
        "必須": "必須",
        "サンプル値": 120,
    },
    {
        "Excel列名": "リードタイム",
        "説明": "1個を製造するのに必要な時間。分単位で入力します。",
        "単位/形式": "分/個",
        "必須": "必須",
        "サンプル値": 4.5,
    },
    {
        "Excel列名": "日産製造数（個数）",
        "説明": "1日あたりの生産数量（能力値）。",
        "単位/形式": "個/日",
        "必須": "必須",
        "サンプル値": 800,
    },
    {
        "Excel列名": "カテゴリー",
        "説明": "製品の分類（和菓子・洋菓子など）。",
        "単位/形式": "テキスト",
        "必須": "推奨",
        "サンプル値": "和菓子",
    },
    {
        "Excel列名": "主要顧客",
        "説明": "主要な販売先やチャネル。",
        "単位/形式": "テキスト",
        "必須": "推奨",
        "サンプル値": "量販店B社",
    },
    {
        "Excel列名": "備考",
        "説明": "任意入力欄。ライン名や補足情報などがあれば記入します。",
        "単位/形式": "任意",
        "必須": "任意",
        "サンプル値": "既存ラインA",
    },
]


UNIT_EXPECTATIONS: Dict[str, Dict[str, Any]] = {
    "actual_unit_price": {
        "allowed": {"円/個", "円/pcs", "yen/pcs", "yen/pc"},
        "expected_display": "円/個",
        "missing_level": "warning",
        "mismatch_level": "error",
        "missing_message": "{label}の単位行が空欄です。テンプレート2行目に『{expected}』と入力してください。",
        "mismatch_message": "{label}の単位が『{actual}』になっています。テンプレートでは『{expected}』で入力してください。",
        "missing_detail": "単位が未設定",
        "mismatch_detail": "単位が{actual}",
        "missing_action": "価格表の単位を確認し、テンプレートの単位行を{expected}に戻してください。",
        "mismatch_action": "千円やケース単価で管理している場合は円/個に換算し、テンプレートの単位行を{expected}に修正してください。",
    },
    "material_unit_cost": {
        "allowed": {"円/個", "円/pcs", "yen/pcs", "yen/pc"},
        "expected_display": "円/個",
        "missing_level": "warning",
        "mismatch_level": "error",
        "missing_message": "{label}の単位行が空欄です。テンプレート2行目に『{expected}』を入力してください。",
        "mismatch_message": "{label}の単位が『{actual}』になっています。テンプレートでは『{expected}』で入力してください。",
        "missing_detail": "単位が未設定",
        "mismatch_detail": "単位が{actual}",
        "missing_action": "購買データの単位を確認し、テンプレートの単位行を{expected}に戻してください。",
        "mismatch_action": "千円やケース単位の場合は円/個へ換算し、テンプレートの単位行を{expected}に修正してください。",
    },
    "minutes_per_unit": {
        "allowed": {"分/個", "分/pcs", "min/pcs", "minutes/pcs"},
        "expected_display": "分/個",
        "missing_level": "warning",
        "mismatch_level": "error",
        "missing_message": "{label}の単位行が空欄です。テンプレート2行目に『{expected}』を入力してください。",
        "mismatch_message": "{label}の単位が『{actual}』になっています。テンプレートでは『{expected}』で入力してください。",
        "missing_detail": "単位が未設定",
        "mismatch_detail": "単位が{actual}",
        "missing_action": "製造現場に確認し、1個あたりの時間を分単位で記録し、テンプレートの単位行を{expected}にしてください。",
        "mismatch_action": "時間(hrs)などで管理している場合は分単位に換算し、テンプレートの単位行を{expected}に戻してください。",
    },
    "daily_qty": {
        "allowed": {"個/日", "個/day", "units/day", "個/Day"},
        "expected_display": "個/日",
        "missing_level": "warning",
        "mismatch_level": "error",
        "missing_message": "{label}の単位行が空欄です。テンプレート2行目に『{expected}』を入力してください。",
        "mismatch_message": "{label}の単位が『{actual}』になっています。テンプレートでは『{expected}』で入力してください。",
        "missing_detail": "単位が未設定",
        "mismatch_detail": "単位が{actual}",
        "missing_action": "生産管理担当に確認し、1日あたりの数量となるようテンプレートの単位行を{expected}に設定してください。",
        "mismatch_action": "月間・年間数量の場合は日割りに換算し、テンプレートの単位行を{expected}に修正してください。",
    },
}


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
        "カテゴリー",
        "主要顧客",
        "備考",
    ]
    product_units = ["コード", "名称", "円/個", "円/個", "分/個", "個/日", "テキスト", "テキスト", "任意"]
    product_samples = [
        ["P-1001", "苺大福", 320, 120, 4.5, 800, "和菓子", "量販店B社", "既存ラインA"],
        ["P-1002", "栗大福", 280, 110, 3.8, 600, "和菓子", "百貨店A社", "新規投入予定"],
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

    detail_templates: Dict[str, str] = {
        "未入力": "{label}が未入力です。",
        "単位が未設定": "{label}の単位が未設定です。",
        "単位が不正": "{label}の単位が想定と異なります。",
        "製品番号が重複": "{label}が別の行と重複しています。",
        "リードタイムが0以下": "{label}が0以下の値になっています。",
        "販売単価が0以下": "{label}が0以下の値になっています。",
        "材料費が負の値": "{label}が負の値になっています。",
        "日産数が0以下": "{label}が0以下の値になっています。",
        "リードタイムが600分超": "{label}が600分を超えています。単位の桁を確認してください。",
        "リードタイムが0.1分未満": "{label}が0.1分未満です。秒単位のままになっていないか確認してください。",
        "日産数が10,000個超": "{label}が10,000個/日を超えています。月次や年間の値を入力していないか確認してください。",
        "日産数が20,000個超": "{label}が20,000個/日を超えています。日割り換算が必要な可能性があります。",
    }

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
            columns=["レベル", "製品番号", "製品名", "項目", "原因/状況", "入力値", "対処方法"]
        )

    def resolve_value(value: Any, row: pd.Series) -> str:
        if value is None:
            return ""
        try:
            if callable(value):
                value = value(row)
        except Exception:
            # If the callable expects fields that are not available we fall back to an
            # empty string rather than breaking the validation flow.
            value = ""
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()

    def ensure_sentence(text: str) -> str:
        if not text:
            return ""
        text = text.strip()
        if not text:
            return ""
        if text.endswith(("。", "！", "？", ".", "!", "?", "）", ")", "】", "]", "」")):
            return text
        return text + "。"

    def format_detail_text(label: str, detail: Any, row: pd.Series) -> str:
        resolved = resolve_value(detail, row)
        if not resolved:
            return ensure_sentence(f"{label}の値を確認してください。")
        template = detail_templates.get(resolved)
        if template:
            text = template.format(label=label)
        elif "{label}" in resolved:
            text = resolved.format(label=label)
        else:
            text = resolved
            if label and label not in text and all(ch not in text for ch in ["＝", "=", ":", "：", "<", ">", "≦", "≧"]):
                text = f"{label}：{text}"
        return ensure_sentence(text)

    def format_action_text(label: str, action: Any, row: pd.Series) -> str:
        resolved = resolve_value(action, row)
        if not resolved:
            return ensure_sentence(f"{label}の入力内容を担当部署に確認してください。")
        if "{label}" in resolved:
            resolved = resolved.format(label=label)
        return ensure_sentence(resolved)

    def register_issue(
        mask: pd.Series,
        column_key: str,
        level: str,
        message: str,
        detail: Any,
        value_column: Optional[str] = None,
        action: Any = None,
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
                    "原因/状況": format_detail_text(label, detail, row),
                    "入力値": raw_value,
                    "対処方法": format_action_text(label, action, row),
                }
            )

    def register_dataset_issue(
        column_key: str,
        level: str,
        message: str,
        detail: str,
        value: str = "",
        action: Any = None,
    ) -> None:
        label = column_info.get(column_key, {}).get("label", column_key)
        pseudo_row = pd.Series(dtype=object)
        detail_text = format_detail_text(label, detail, pseudo_row)
        action_text = format_action_text(label, action, pseudo_row)
        if level == "error":
            errors.append(message)
        else:
            warnings.append(message)
        detail_rows.append(
            {
                "レベル": "エラー" if level == "error" else "警告",
                "製品番号": "全体",
                "製品名": "",
                "項目": label,
                "原因/状況": detail_text,
                "入力値": value,
                "対処方法": action_text,
            }
        )

    column_unit_info = df.attrs.get("column_unit_info") or {}
    if not isinstance(column_unit_info, dict):
        column_unit_info = {}

    if column_unit_info:
        for col, config in UNIT_EXPECTATIONS.items():
            if col not in df.columns:
                continue
            info = column_unit_info.get(col, {})
            if not isinstance(info, dict):
                info = {"unit": info, "source": None}
            actual_unit = info.get("unit")
            normalized_actual = normalize_unit_text(actual_unit)
            allowed_raw = config.get("allowed", set())
            allowed_normalized = set()
            for candidate_unit in allowed_raw:
                normalized_candidate = normalize_unit_text(candidate_unit)
                if normalized_candidate:
                    allowed_normalized.add(normalized_candidate)
            expected_display = config.get("expected_display")
            if not expected_display:
                for candidate in allowed_raw:
                    if candidate:
                        expected_display = candidate
                        break
            label = column_info.get(col, {}).get("label", col)

            if not normalized_actual:
                message = config["missing_message"].format(
                    label=label, expected=expected_display or ""
                )
                detail = config.get("missing_detail", "単位が未設定")
                action = config.get("missing_action")
                if action:
                    action = action.format(expected=expected_display or "")
                register_dataset_issue(
                    col,
                    config.get("missing_level", "warning"),
                    message,
                    detail,
                    value=str(actual_unit or ""),
                    action=action,
                )
            elif allowed_normalized and normalized_actual not in allowed_normalized:
                actual_display = actual_unit if actual_unit not in [None, ""] else normalized_actual
                message = config["mismatch_message"].format(
                    label=label,
                    actual=actual_display,
                    expected=expected_display or "",
                )
                detail = config.get("mismatch_detail", "単位が不正").format(
                    actual=actual_display,
                    expected=expected_display or "",
                )
                action = config.get("mismatch_action")
                if action:
                    action = action.format(
                        expected=expected_display or "",
                        actual=actual_display,
                    )
                register_dataset_issue(
                    col,
                    config.get("mismatch_level", "error"),
                    message,
                    detail,
                    value=str(actual_display),
                    action=action,
                )

    missing_summary_hints = {
        "product_no": "製品マスタ台帳を確認して一意の番号を入力してください。",
        "product_name": "商品企画資料などから正式名称を入力してください。",
        "actual_unit_price": "最新の販売価格表を確認して円単位で入力してください。",
        "material_unit_cost": "原材料の見積や購買データを確認して円単位で入力してください。",
        "minutes_per_unit": "製造現場に確認して1個あたりの製造時間（分）を入力してください。",
        "daily_qty": "生産計画担当に確認して1日あたりの能力値を入力してください。",
    }
    missing_actions = {
        "product_no": "製品番号台帳を確認し、重複しないコードを入力してください。",
        "product_name": "商品仕様書に記載の正式な製品名を入力してください。",
        "actual_unit_price": "価格表や販売管理システムで最新の単価（円/個）を確認して入力してください。",
        "material_unit_cost": "購買担当またはBOMの原価を確認し、1個あたりの材料費を入力してください。",
        "minutes_per_unit": "製造ライン担当者に確認して、1個を作るのに必要な分数を記録してください。",
        "daily_qty": "生産管理担当に日産能力を確認し、1日あたりの数量で入力してください。",
    }

    # Missing values
    for col, meta in column_info.items():
        mask = df[col].isna()
        if mask.any():
            level = meta.get("missing_level", "warning")
            summary_hint = missing_summary_hints.get(
                col, "テンプレートのサンプルを参考に入力してください。"
            )
            action_hint = missing_actions.get(
                col, "担当部署に確認して数値を入力してください。"
            )
            register_issue(
                mask,
                col,
                level,
                f"{{label}}が未入力の製品が{{count}}件あります。{summary_hint}",
                "未入力",
                action=action_hint,
            )

    # Invalid or suspicious values
    value_checks = [
        {
            "column": "minutes_per_unit",
            "level": "error",
            "message": "リードタイム（分/個）が0以下の製品が{count}件あります。製造現場に確認して実績時間を分単位で入力してください。",
            "detail": "リードタイムが0以下",
            "condition": lambda s: s <= 0,
            "action": "製造ライン担当者に確認し、1個あたりの製造時間を分単位で入力してください。",
        },
        {
            "column": "actual_unit_price",
            "level": "error",
            "message": "販売単価（円/個）が0以下の製品が{count}件あります。販売価格を確認し円単位で入力してください。",
            "detail": "販売単価が0以下",
            "condition": lambda s: s <= 0,
            "action": "販売管理システムを確認し、税抜の販売単価（円/個）を入力してください。",
        },
        {
            "column": "material_unit_cost",
            "level": "error",
            "message": "材料費（円/個）が負の値の製品が{count}件あります。単位や入力値を確認してください。",
            "detail": "材料費が負の値",
            "condition": lambda s: s < 0,
            "action": "購買データを確認し、円単位の原価を正の値で入力してください。",
        },
        {
            "column": "daily_qty",
            "level": "error",
            "message": "日産数（個/日）が0以下の製品が{count}件あります。生産能力を入力してください。",
            "detail": "日産数が0以下",
            "condition": lambda s: s <= 0,
            "action": "生産管理担当に確認し、1日あたりに製造可能な数量を入力してください。",
        },
        {
            "column": "minutes_per_unit",
            "level": "warning",
            "message": "リードタイム（分/個）が600分を超える製品が{count}件あります。時間の単位（分）を確認してください。",
            "detail": "リードタイムが600分超",
            "condition": lambda s: s > 600,
            "action": "時間単位が時間(hrs)になっていないか確認し、分に換算してください。",
        },
        {
            "column": "minutes_per_unit",
            "level": "warning",
            "message": "リードタイム（分/個）が0.1分未満の製品が{count}件あります。秒単位の値を分に換算しているか確認してください。",
            "detail": "リードタイムが0.1分未満",
            "condition": lambda s: (s > 0) & (s < 0.1),
            "action": "秒で管理している場合は60で割り、分単位に換算した値を入力してください。",
        },
        {
            "column": "actual_unit_price",
            "level": "warning",
            "message": "販売単価（円/個）が1円未満の製品が{count}件あります。単位が千円になっていないか確認してください。",
            "detail": "販売単価が1円未満",
            "condition": lambda s: (s > 0) & (s < 1),
            "action": "テンプレートでは円/個で入力します。千円単位などの場合は円に換算してください。",
        },
        {
            "column": "actual_unit_price",
            "level": "warning",
            "message": "販売単価（円/個）が100,000円以上の製品が{count}件あります。桁違いで入力していないか確認してください。",
            "detail": "販売単価が高すぎる",
            "condition": lambda s: s >= 100000,
            "action": "単価が桁違いの場合は円単位に修正してください。高額品であればコメント欄に根拠を記載してください。",
        },
        {
            "column": "material_unit_cost",
            "level": "warning",
            "message": "材料費（円/個）が1円未満の製品が{count}件あります。単位が千円になっていないか確認してください。",
            "detail": "材料費が1円未満",
            "condition": lambda s: (s > 0) & (s < 1),
            "action": "テンプレートでは円/個で入力します。千円単位などの場合は円に換算してください。",
        },
        {
            "column": "material_unit_cost",
            "level": "warning",
            "message": "材料費（円/個）が100,000円以上の製品が{count}件あります。桁違いで入力していないか確認してください。",
            "detail": "材料費が高すぎる",
            "condition": lambda s: s >= 100000,
            "action": "材料費の桁を確認し、円単位に修正してください。高額原価の場合は備考に理由を記載してください。",
        },
        {
            "column": "daily_qty",
            "level": "warning",
            "message": "日産数（個/日）が20,000個を超える製品が{count}件あります。月間数量を入力していないか確認してください。",
            "detail": "日産数が20,000個超",
            "condition": lambda s: s > 20000,
            "action": "日産数は1日あたりの数量です。月間・年間の実績を入力している場合は日割りに換算してください。",
        },
    ]

    for check in value_checks:
        col = check["column"]
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        mask = check["condition"](series)
        register_issue(
            mask,
            col,
            check["level"],
            check["message"],
            check["detail"],
            action=check.get("action"),
        )

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
            action="価格改定や原価削減の対応要否を確認し、必要に応じて標準単価を見直してください。",
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
            action="重複している行の担当部署に確認し、ユニークな製品番号に修正してください。",
        )

    detail_df = pd.DataFrame(
        detail_rows,
        columns=["レベル", "製品番号", "製品名", "項目", "原因/状況", "入力値", "対処方法"],
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
        "product_no","product_name","category","major_customer",
        "actual_unit_price","material_unit_cost",
        "minutes_per_unit","daily_qty","daily_total_minutes",
        "gp_per_unit","daily_va","va_per_min",
        "be_va_unit_price","req_va_unit_price","required_selling_price",
        "price_gap_vs_required","rate_gap_vs_required","meets_required_rate","rate_class"
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    return df[out_cols]


def summarize_segment_performance(
    df: pd.DataFrame, required_rate: float, segment_col: str
) -> pd.DataFrame:
    """Aggregate KPI metrics by the specified segment column."""

    if df is None or df.empty or segment_col not in df.columns:
        return pd.DataFrame(
            columns=[
                "segment",
                "sku_count",
                "ach_rate_pct",
                "avg_va_per_min",
                "avg_gap",
                "avg_roi_months",
            ]
        )

    req_rate = 0.0 if required_rate is None else float(required_rate)
    work = df.copy()
    seg_series = work[segment_col].copy()
    seg_series = seg_series.fillna("未設定")
    seg_series = seg_series.astype(str)
    seg_series = seg_series.str.strip().replace({"": "未設定", "nan": "未設定"})
    work[segment_col] = seg_series

    work["va_per_min"] = pd.to_numeric(work.get("va_per_min"), errors="coerce")
    if "meets_required_rate" in work.columns:
        meets_series = work["meets_required_rate"]
        if meets_series.dtype != bool:
            work["meets_required_rate"] = (
                pd.to_numeric(meets_series, errors="coerce").fillna(0) > 0
            )
    else:
        work["meets_required_rate"] = work["va_per_min"] >= req_rate

    work["gap_to_required"] = work["va_per_min"] - req_rate

    required_price = pd.to_numeric(
        work.get("required_selling_price"), errors="coerce"
    )
    actual_price = pd.to_numeric(work.get("actual_unit_price"), errors="coerce")
    price_improve = required_price - actual_price
    price_improve = price_improve.where(price_improve > 0)
    gap_positive = req_rate - work["va_per_min"]
    gap_positive = gap_positive.where(gap_positive > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        roi_months = price_improve / gap_positive
    roi_months = roi_months.replace([np.inf, -np.inf], np.nan)
    work["roi_months"] = roi_months

    summary = (
        work.groupby(segment_col, dropna=False)
        .agg(
            sku_count=(
                "product_no",
                lambda s: int(s.dropna().nunique() or s.notna().sum()),
            ),
            ach_rate=("meets_required_rate", "mean"),
            avg_va_per_min=("va_per_min", "mean"),
            avg_gap=("gap_to_required", "mean"),
            avg_roi_months=(
                "roi_months",
                lambda s: float(s.dropna().mean()) if not s.dropna().empty else np.nan,
            ),
        )
        .reset_index()
    )

    if summary.empty:
        return pd.DataFrame(
            columns=[
                "segment",
                "sku_count",
                "ach_rate_pct",
                "avg_va_per_min",
                "avg_gap",
                "avg_roi_months",
            ]
        )

    summary = summary.rename(columns={segment_col: "segment"})
    summary["ach_rate_pct"] = summary["ach_rate"].astype(float) * 100.0
    summary = summary.drop(columns=["ach_rate"])
    summary = summary.sort_values("avg_gap", ascending=False).reset_index(drop=True)
    return summary


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
