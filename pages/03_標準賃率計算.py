import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from components import (
    render_onboarding,
    render_page_tutorial,
    render_stepper,
    render_sidebar_nav,
)
import os
from openai import OpenAI


def _explain_standard_rate(results: dict[str, float]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OpenAI APIキーが設定されていません。"
    client = OpenAI(api_key=api_key)
    prompt = (
        "以下の標準賃率計算結果を分かりやすく説明してください。"
        "専門用語を避けて日本語で100字程度にまとめてください。\n"
        + "\n".join(f"{k}: {v}" for k, v in results.items())
    )
    try:
        resp = client.responses.create(model="gpt-4o-mini", input=prompt)
        return resp.output_text.strip()
    except Exception as exc:
        return f"AI説明の生成に失敗しました: {exc}"

from standard_rate_core import (
    DEFAULT_PARAMS,
    sanitize_params,
    compute_rates,
    build_reverse_index,
    plot_sensitivity,
    generate_pdf,
)

st.title("③ 標準賃率 計算/感度分析")
render_sidebar_nav(page_key="standard_rate")

render_onboarding()
render_page_tutorial("standard_rate")
render_stepper(4)

st.markdown(
    """
    <style>
    .sr-section {
        background: linear-gradient(145deg, rgba(38, 46, 74, 0.88), rgba(18, 24, 42, 0.92));
        border-radius: 22px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 24px 48px rgba(8, 12, 28, 0.45);
    }
    .sr-section h4 {
        color: #f4f6ff;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sr-section p, .sr-section .sr-helper {
        color: rgba(238, 241, 255, 0.75);
        margin-bottom: 0.6rem;
    }
    .sr-section div[data-baseweb="input"] > input,
    .sr-section textarea,
    .sr-section select,
    .sr-section input[type="number"],
    .sr-section input[type="text"] {
        background-color: rgba(12, 17, 32, 0.85) !important;
        color: #f5f7ff !important;
        border-radius: 12px;
        border: 1px solid rgba(132, 146, 255, 0.35);
        font-weight: 600;
    }
    .sr-section label {
        color: #f0f2ff !important;
        font-weight: 600 !important;
    }
    .sr-section .stSlider > div > div > div[data-testid="stTickBar"] {
        background-color: rgba(132, 146, 255, 0.35);
    }
    .sr-section .stSlider > div > div > div > div {
        background: linear-gradient(90deg, rgba(132, 146, 255, 0.9), rgba(111, 180, 255, 0.9));
    }
    .sr-section .stSlider [data-testid="stThumbValue"] > div {
        color: #0a1024 !important;
        font-weight: 700;
    }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(32, 42, 68, 0.8), rgba(18, 24, 42, 0.9));
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1.2rem;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05), 0 18px 36px rgba(10, 15, 30, 0.55);
    }
    div[data-testid="metric-container"] label {
        color: rgba(226, 232, 255, 0.8) !important;
        font-weight: 600;
    }
    div[data-testid="metric-container"] > div:nth-child(2) {
        color: #f6f8ff !important;
        font-weight: 700;
    }
    .sr-metric-caption {
        margin-top: -0.6rem;
        font-size: 0.76rem;
        color: rgba(226, 232, 255, 0.68);
    }
    .sr-highlight {
        background: rgba(120, 150, 255, 0.12);
        border-radius: 16px;
        border: 1px solid rgba(120, 150, 255, 0.25);
        padding: 0.8rem 1rem;
        color: rgba(235, 238, 255, 0.8);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

scenarios = st.session_state.setdefault("scenarios", {"ベース": st.session_state.get("sr_params", DEFAULT_PARAMS)})
current = st.session_state.setdefault("current_scenario", "ベース")
st.caption(f"適用中シナリオ: {current}")

params = scenarios.get(current, st.session_state.get("sr_params", DEFAULT_PARAMS)).copy()

st.sidebar.header("シナリオ")
names = list(scenarios.keys())
selected = st.sidebar.selectbox("シナリオ選択", names, index=names.index(current))
if selected != current:
    st.session_state["current_scenario"] = selected
    st.session_state["sr_params"] = scenarios[selected].copy()
    st.rerun()

new_name = st.sidebar.text_input("新規シナリオ名", "")
if st.sidebar.button("追加") and new_name:
    scenarios[new_name] = params.copy()
    st.session_state["current_scenario"] = new_name
    st.session_state["sr_params"] = params.copy()
    st.rerun()

if current != "ベース" and st.sidebar.button("削除"):
    del scenarios[current]
    st.session_state["current_scenario"] = "ベース"
    st.session_state["sr_params"] = scenarios["ベース"].copy()
    st.rerun()

st.markdown(
    """
    <div class="sr-highlight">
        標準賃率は「工場を動かすのに最低限必要な売上単価」です。下記の前提を変えると、主要な賃率指標が即座にアップデートされます。
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### 入力条件")
st.caption("数字を直接入力すると即時に再計算され、影響を受ける指標が項目ごとに表示されます。")

placeholders = {}

section_container = st.container()
section_container.markdown('<div class="sr-section">', unsafe_allow_html=True)
with section_container.container():
    st.markdown("#### A) 必要固定費（円/年）")
    st.caption("製造を維持するために必ず発生する年間コストです。")
    fixed_col1, fixed_col2 = st.columns(2, gap="large")
    with fixed_col1:
        params["labor_cost"] = st.number_input(
            "労務費（製造担当の年間人件費）",
            value=float(params["labor_cost"]),
            step=1000.0,
            format="%.0f",
            min_value=0.0,
            help="工場で働くメンバーに支払う給与・賞与・社会保険料などの合計です。",
        )
        placeholders["labor_cost"] = st.empty()
    with fixed_col2:
        params["sga_cost"] = st.number_input(
            "販管費（共通管理コスト）",
            value=float(params["sga_cost"]),
            step=1000.0,
            format="%.0f",
            min_value=0.0,
            help="製造以外に必要な管理部門・営業部門の共通費用を含みます。",
        )
        placeholders["sga_cost"] = st.empty()
section_container.markdown("</div>", unsafe_allow_html=True)

section_container = st.container()
section_container.markdown('<div class="sr-section">', unsafe_allow_html=True)
with section_container.container():
    st.markdown("#### B) 必要利益（円/年）")
    st.caption("事業を健全に続けるために確保したい利益水準です。")
    profit_col1, profit_col2, profit_col3 = st.columns(3, gap="large")
    with profit_col1:
        params["loan_repayment"] = st.number_input(
            "借入返済（年間返済額）",
            value=float(params["loan_repayment"]),
            step=1000.0,
            format="%.0f",
            min_value=0.0,
            help="銀行等への年間返済額。資金繰り計画に沿って設定してください。",
        )
        placeholders["loan_repayment"] = st.empty()
    with profit_col2:
        params["tax_payment"] = st.number_input(
            "納税・納付（年間見込額）",
            value=float(params["tax_payment"]),
            step=1000.0,
            format="%.0f",
            min_value=0.0,
            help="法人税や社会保険料など、利益確保後に支払う義務がある金額です。",
        )
        placeholders["tax_payment"] = st.empty()
    with profit_col3:
        params["future_business"] = st.number_input(
            "未来事業費（投資原資）",
            value=float(params["future_business"]),
            step=1000.0,
            format="%.0f",
            min_value=0.0,
            help="新設備や人材育成など、将来に向けて確保したい資金を入力します。",
        )
        placeholders["future_business"] = st.empty()
section_container.markdown("</div>", unsafe_allow_html=True)

section_container = st.container()
section_container.markdown('<div class="sr-section">', unsafe_allow_html=True)
with section_container.container():
    st.markdown("#### C) 人員・稼働前提")
    st.caption("工場の人員体制と稼働日数/時間の想定です。『稼働係数』はフルタイム勤務を1.00とした場合の働き方の比率を表します。")
    staff_col1, staff_col2, staff_col3 = st.columns(3, gap="large")
    with staff_col1:
        params["fulltime_workers"] = st.number_input(
            "正社員の人数（常勤）",
            value=float(params["fulltime_workers"]),
            step=0.5,
            format="%.2f",
            min_value=0.0,
            help="常勤で働く正社員人数。1人あたり稼働係数は1.00です。",
        )
        placeholders["fulltime_workers"] = st.empty()
    with staff_col2:
        params["part1_workers"] = st.number_input(
            "準社員Aの人数（短時間勤務）",
            value=float(params["part1_workers"]),
            step=0.5,
            format="%.2f",
            min_value=0.0,
            help="短時間勤務の準社員人数。想定稼働係数は0.75です。",
        )
        placeholders["part1_workers"] = st.empty()
    with staff_col3:
        params["part2_workers"] = st.number_input(
            "準社員Bの人数（柔軟シフト）",
            value=float(params["part2_workers"]),
            step=0.5,
            format="%.2f",
            min_value=0.0,
            help="シフトが柔軟な準社員人数。稼働係数は下のスライダーで調整します。",
        )
        placeholders["part2_workers"] = st.empty()

    coeff_col, days_col, hours_col, rate_col = st.columns(4, gap="large")
    with coeff_col:
        params["part2_coefficient"] = st.slider(
            "準社員Bの稼働係数",
            value=float(params["part2_coefficient"]),
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            help="フルタイムを1.00とした場合の働きぶりの割合です。0.60なら6割稼働を意味します。",
        )
        placeholders["part2_coefficient"] = st.empty()
    with days_col:
        params["working_days"] = st.number_input(
            "年間稼働日数",
            value=float(params["working_days"]),
            step=1.0,
            format="%.0f",
            min_value=1.0,
            help="1年間で工場を動かす日数です（休日や定期点検日を除きます）。",
        )
        placeholders["working_days"] = st.empty()
    with hours_col:
        params["daily_hours"] = st.number_input(
            "1日あたりの稼働時間",
            value=float(params["daily_hours"]),
            step=0.1,
            format="%.2f",
            min_value=0.1,
            help="工場を動かす時間帯の合計です。たとえば8時間稼働なら『8.0』を入力します。",
        )
        placeholders["daily_hours"] = st.empty()
    with rate_col:
        params["operation_rate"] = st.slider(
            "1日の稼働率",
            value=float(params["operation_rate"]),
            min_value=0.5,
            max_value=1.0,
            step=0.01,
            help="実働時間のうち、生産に充てられる割合です。0.85なら85%の時間が有効稼働になります。",
        )
        placeholders["operation_rate"] = st.empty()
section_container.markdown("</div>", unsafe_allow_html=True)

params, warn_list = sanitize_params(params)
for w in warn_list:
    st.sidebar.warning(w)
st.session_state["sr_params"] = params
scenarios[current] = params
st.session_state["scenarios"] = scenarios

nodes, results = compute_rates(params)
reverse_index = build_reverse_index(nodes)
for k, ph in placeholders.items():
    affected = ", ".join(reverse_index.get(k, []))
    if affected:
        ph.caption(f"この入力が影響する指標: {affected}")

c1, c2, c3, c4 = st.columns(4, gap="large")
with c1:
    st.metric("損益分岐賃率（円/分）", f"{results['break_even_rate']:.3f}")
    st.caption("売上単価がこの水準を上回ると、固定費を回収して黒字化します。")
with c2:
    st.metric("必要賃率（円/分）", f"{results['required_rate']:.3f}")
    st.caption("借入返済や将来投資を含め、目標利益を確保するための最低単価です。")
with c3:
    st.metric("年間標準稼働時間（分）", f"{results['annual_minutes']:.0f}")
    st.caption("人員構成と稼働率から算出した、年間で確保できる生産可能時間です。")
with c4:
    st.metric("正味直接工員数合計", f"{results['net_workers']:.2f}")
    st.caption("稼働係数を考慮した実働ベースの生産要員数です。")

st.subheader("AIによる説明")
st.caption("計算結果を経営者目線の文章で要約します。専門用語が多いと感じたらこちらを活用してください。")
if st.button("結果をAIで説明"):
    with st.spinner("生成中..."):
        st.session_state["sr_ai_comment"] = _explain_standard_rate(results)
st.write(st.session_state.get("sr_ai_comment", ""))

_, wf_col = st.columns([3, 1])
with wf_col:
    with st.expander("必要賃率ウォーターフォール", expanded=False):
        prev_params = st.session_state.get("prev_month_params")
        if prev_params is not None:
            _, prev_res = compute_rates(prev_params)
            f_prev = prev_res["fixed_total"]
            p_prev = prev_res["required_profit_total"]
            m_prev = prev_res["annual_minutes"]
            r_prev = prev_res["required_rate"]
            f_cur = results["fixed_total"]
            p_cur = results["required_profit_total"]
            m_cur = results["annual_minutes"]
            r_cur = results["required_rate"]
            diff_fixed = (f_cur - f_prev) / m_prev
            diff_profit = (p_cur - p_prev) / m_prev
            diff_minutes = r_cur - r_prev - diff_fixed - diff_profit
            wf_fig = go.Figure(
                go.Waterfall(
                    x=["前月必要賃率", "固定費差分", "必要利益差分", "年間稼働分差分", "当月必要賃率"],
                    measure=["absolute", "relative", "relative", "relative", "total"],
                    y=[r_prev, diff_fixed, diff_profit, diff_minutes, r_cur],
                    increasing={"marker": {"color": "#D55E00"}},
                    decreasing={"marker": {"color": "#009E73"}},
                    totals={"marker": {"color": "#0072B2"}},
                )
            )
            wf_fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(wf_fig, use_container_width=True)
            comp_table = pd.DataFrame(
                {
                    "項目": ["固定費計", "必要利益計", "年間標準稼働分", "必要賃率"],
                    "前月": [f_prev, p_prev, m_prev, r_prev],
                    "当月": [f_cur, p_cur, m_cur, r_cur],
                }
            )
            comp_table["差額"] = comp_table["当月"] - comp_table["前月"]
            styled = comp_table.style.applymap(
                lambda v: "color:red" if v > 0 else "color:blue", subset=["差額"]
            )
            st.dataframe(styled, use_container_width=True)
        else:
            st.info("前月データがありません。")

st.subheader("ブレークダウン")
st.caption("各指標の計算式と、どの入力が影響しているかを一覧で確認できます。")
cat_map = {
    "fixed_total": "固定費",
    "required_profit_total": "必要利益",
    "net_workers": "工数前提",
    "minutes_per_day": "工数前提",
    "standard_daily_minutes": "工数前提",
    "annual_minutes": "工数前提",
    "break_even_rate": "賃率",
    "required_rate": "賃率",
    "daily_be_va": "付加価値",
    "daily_req_va": "付加価値",
}
df_break = pd.DataFrame(
    [
        (
            cat_map.get(n["key"], ""),
            n["label"],
            n["value"],
            n.get("unit", ""),
            n["formula"],
            ", ".join(n["depends_on"]),
        )
        for n in nodes.values()
    ],
    columns=["区分", "項目", "値", "単位", "式", "依存要素"],
)
st.dataframe(df_break, use_container_width=True)

st.subheader("感度分析")
st.caption("主要な入力を増減させたときに賃率がどのように変わるかを可視化します。")
fig = plot_sensitivity(params)
st.pyplot(fig)

df_csv = pd.DataFrame(list(nodes.values()))
df_csv["depends_on"] = df_csv["depends_on"].apply(lambda x: ",".join(x))
csv = df_csv.to_csv(index=False, encoding="utf-8-sig")
st.download_button("CSVエクスポート", data=csv, file_name=f"standard_rate__{current}.csv", mime="text/csv")

pdf_bytes = generate_pdf(nodes, fig)
st.download_button("PDFエクスポート", data=pdf_bytes, file_name=f"standard_rate_summary__{current}.pdf", mime="application/pdf")
