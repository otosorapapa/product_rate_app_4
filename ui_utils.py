"""UI の共通部品と国際化ヘルパーをまとめたモジュール."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import streamlit as st

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "ja": {
        "app_title": "標準賦率ダッシュボード",
        "nav_header": "ナビゲーション",
        "nav_data": "① データ入力",
        "nav_validation": "② 検証・感度分析",
        "nav_dashboard": "③ ダッシュボード",
        "language_label": "表示言語",
        "sensitivity_inputs": "感度分析シナリオ設定",
        "dashboard_title": "シナリオ別ダッシュボード",
        "no_data": "データを読み込んでください",
        "sample_button": "サンプルデータを使う",
        "upload_label": "Excel / CSV / JSON ファイルをアップロード",
        "quality_header": "データ品質チェック",
        "data_intro_caption": "固定費・人員前提と製品データを登録し、次のステップへ進みます。",
        "operational_expander": "固定費・人員前提を入力",
        "base_metrics_header": "ベースシナリオ計算結果",
        "quality_ok": "重大な品質問題は検出されませんでした。",
        "anomaly_header": "統計的に目立つSKU",
        "no_anomaly": "統計的な外れ値は検出されませんでした。",
        "scenario_name_label": "シナリオ名",
        "scenario_save_button": "シナリオとして保存",
        "kpi_required_rate": "必要負担率 (円/分)",
        "kpi_break_even": "損益分岐負担率 (円/分)",
        "kpi_avg_va": "付加価値/分 平均",
        "kpi_meet_ratio": "達成SKU比率",
        "kpi_annual_minutes": "年間標準稼働時間(分)",
        "kpi_net_workers": "正味直接工員数(人)",
        "roi_header": "ROI シミュレーション上位 SKU",
        "ai_commentary": "AI コメント生成",
        "pdf_button": "PDF出力",
        "scatter_title": "加工時間と付加価値の散布図",
        "scatter_x": "加工時間(分/個)",
        "scatter_y": "付加価値(円/分)",
        "bar_gap_title": "必要負担率との差",
        "bar_va_title": "付加価値/分 平均",
        "hist_va_title": "付加価値/分",
        "hist_mpu_title": "加工時間(分/個)",
        "save_success": "シナリオ『{name}』を保存しました。",
        "sensitivity_fixed_slider": "固定費の調整(%)",
        "sensitivity_profit_slider": "必要利益の調整(%)",
        "sensitivity_net_workers": "正味直接工員数",
        "sensitivity_operation_rate": "操業度(%)",
        "sensitivity_working_days": "年間稼働日数",
        "sensitivity_daily_hours": "1日稼働時間",
        "sensitivity_submit": "シミュレーションを更新",
        "sensitivity_mapping_title": "感度と影響指標の対応",
        "waterfall_title": "必要負担率の感度分析",
    },
    "en": {
        "app_title": "Standard Rate Dashboard",
        "nav_header": "Navigation",
        "nav_data": "1. Data Input",
        "nav_validation": "2. Validation & Sensitivity",
        "nav_dashboard": "3. Dashboard",
        "language_label": "Language",
        "sensitivity_inputs": "Sensitivity Scenario Settings",
        "dashboard_title": "Scenario Dashboard",
        "no_data": "Please load your data",
        "sample_button": "Load sample data",
        "upload_label": "Upload Excel / CSV / JSON",
        "quality_header": "Data Quality Checks",
        "data_intro_caption": "Register cost assumptions and product data, then proceed to the next step.",
        "operational_expander": "Edit cost and workforce assumptions",
        "base_metrics_header": "Base scenario results",
        "quality_ok": "No critical quality issues were found.",
        "anomaly_header": "Statistically significant SKUs",
        "no_anomaly": "No statistical outliers detected.",
        "scenario_name_label": "Scenario name",
        "scenario_save_button": "Save as scenario",
        "kpi_required_rate": "Required burden rate (JPY/min)",
        "kpi_break_even": "Break-even burden rate (JPY/min)",
        "kpi_avg_va": "Average VA per min",
        "kpi_meet_ratio": "Share meeting target",
        "kpi_annual_minutes": "Annual standard operating minutes",
        "kpi_net_workers": "Net direct workers",
        "roi_header": "Top ROI opportunities",
        "ai_commentary": "AI commentary",
        "pdf_button": "Export PDF",
        "scatter_title": "Process time vs. value-add",
        "scatter_x": "Process time (min/unit)",
        "scatter_y": "Value-add (JPY/min)",
        "bar_gap_title": "Gap vs. required rate",
        "bar_va_title": "Average VA per min",
        "hist_va_title": "Value-add (JPY/min)",
        "hist_mpu_title": "Process time (min/unit)",
        "save_success": "Scenario '{name}' has been saved.",
        "sensitivity_fixed_slider": "Adjust fixed cost (%)",
        "sensitivity_profit_slider": "Adjust required profit (%)",
        "sensitivity_net_workers": "Net direct workers",
        "sensitivity_operation_rate": "Operating rate (%)",
        "sensitivity_working_days": "Working days per year",
        "sensitivity_daily_hours": "Daily operating hours",
        "sensitivity_submit": "Run simulation",
        "sensitivity_mapping_title": "Sensitivity inputs vs. metrics",
        "waterfall_title": "Sensitivity of required rate",
    },
}


@dataclass
class Translator:
    """辞書ベースの簡易翻訳クラス."""

    language: str

    def __call__(self, key: str) -> str:
        return TRANSLATIONS.get(self.language, {}).get(
            key, TRANSLATIONS["ja"].get(key, key)
        )


def apply_page_config() -> None:
    """ページ設定とフォント指定をまとめて実行する."""

    st.set_page_config(page_title="標準賦率ダッシュボード", layout="wide")
    st.markdown(
        """
        <style>
        :root {
            --primary-color: #2F4F6E;
            --accent-color: #4F6D7A;
            --bg-color: #F4F7FB;
            --card-bg: #FFFFFF;
        }
        html, body, [class*="css"]  {
            font-family: 'Noto Sans JP', sans-serif;
        }
        .stApp {
            background-color: var(--bg-color);
        }
        section[data-testid="stSidebar"] {
            background-color: #E9EEF5;
        }
        .card-box {
            background-color: var(--card-bg);
            border-radius: 16px;
            padding: 16px 20px;
            box-shadow: 0px 10px 24px rgba(47,79,110,0.1);
        }
        button[kind="primary"] {
            background-color: var(--primary-color);
            color: white;
            border-radius: 12px;
            transition: all 0.3s ease;
        }
        button[kind="primary"]:hover {
            background-color: #16324F;
            transform: translateY(-1px);
        }
        [data-testid="stDataFrameCell"] div:has(span:contains("⚠️")) {
            background-color: rgba(255, 204, 204, 0.4) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def select_language(default: str = "ja") -> Translator:
    """サイドバーで言語を選択し Translator を返す."""

    lang = st.sidebar.selectbox(
        TRANSLATIONS[default]["language_label"],
        options=list(TRANSLATIONS.keys()),
        format_func=lambda x: "日本語" if x == "ja" else "English",
    )
    return Translator(lang)


def render_sidebar(translator: Translator, current: str) -> str:
    """サイドバーのナビゲーションメニューを描画し、選択肢を返す."""

    st.sidebar.header(translator("nav_header"))
    options = [
        ("data", translator("nav_data")),
        ("validation", translator("nav_validation")),
        ("dashboard", translator("nav_dashboard")),
    ]
    ids = [oid for oid, _ in options]
    if current not in ids:
        current = ids[0]
    index = ids.index(current)
    choice = st.sidebar.radio(
        "menu",
        options=ids,
        index=index,
        format_func=lambda x: dict(options)[x],
        label_visibility="collapsed",
    )
    return choice
