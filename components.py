from typing import Optional

import streamlit as st

_STYLE_SESSION_KEY = "_mckinsey_style_injected"

_MCKINSEY_STYLE = """
<style>
:root {
    --mck-deep-blue: #0B1F3A;
    --mck-steel-blue: #1C3D63;
    --mck-accent: #00A3E0;
    --mck-soft-blue: #E7EEF7;
    --mck-page-bg: #F5F7FA;
    --mck-text: #1F2937;
}

.stApp {
    background-color: var(--mck-page-bg);
    color: var(--mck-text);
    font-family: "Helvetica Neue", "Noto Sans JP", sans-serif;
}

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4 {
    color: var(--mck-deep-blue);
    font-weight: 600;
    letter-spacing: 0.02em;
}

.stApp p,
.stApp label,
.stApp span,
.stApp li {
    color: #324157;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

.stApp a {
    color: var(--mck-accent);
}

.mck-header-card {
    display: flex;
    align-items: center;
    gap: 1.6rem;
    padding: 2rem 2.4rem;
    margin-bottom: 1.8rem;
    border-radius: 20px;
    background: linear-gradient(130deg, var(--mck-deep-blue) 0%, #13294B 55%, var(--mck-steel-blue) 100%);
    color: #FFFFFF;
    box-shadow: 0 28px 52px rgba(7, 23, 43, 0.35);
}

.mck-header-icon {
    width: 68px;
    height: 68px;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.15);
    display: grid;
    place-items: center;
    font-size: 2rem;
    font-weight: 600;
}

.mck-header-card h1 {
    margin: 0;
    color: #FFFFFF;
    font-size: 1.9rem;
}

.mck-header-subtitle {
    margin: 0.3rem 0 0;
    font-size: 1.05rem;
    color: rgba(255, 255, 255, 0.88);
}

.mck-card {
    background: #FFFFFF;
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    border: 1px solid rgba(15, 37, 62, 0.08);
    box-shadow: 0 26px 54px rgba(15, 23, 42, 0.08);
    margin-bottom: 1.5rem;
}

.mck-card h3 {
    margin-top: 0;
    margin-bottom: 0.6rem;
}

.mck-feature-card {
    background: #FFFFFF;
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    border: 1px solid rgba(12, 34, 64, 0.08);
    box-shadow: 0 24px 48px rgba(15, 23, 42, 0.09);
    display: flex;
    flex-direction: column;
    gap: 1.1rem;
    min-height: 220px;
    position: relative;
    overflow: hidden;
}

.mck-feature-icon {
    width: 58px;
    height: 58px;
    border-radius: 16px;
    background: rgba(0, 163, 224, 0.14);
    color: var(--mck-accent);
    display: grid;
    place-items: center;
    font-size: 1.6rem;
}

.mck-feature-card h4 {
    margin: 0;
    font-size: 1.15rem;
    color: var(--mck-deep-blue);
}

.mck-feature-card p {
    margin: 0;
    color: #4B5C73;
    line-height: 1.55;
    font-size: 0.95rem;
}

a[data-testid="stPageLink"],
a[data-testid="stPageLink-NavLink"] {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.55rem 1.15rem;
    border-radius: 999px;
    text-decoration: none;
    font-weight: 600;
    margin-top: 0.75rem;
    background: rgba(0, 163, 224, 0.12);
    color: var(--mck-deep-blue);
    transition: all 0.2s ease-in-out;
}

a[data-testid="stPageLink"]:hover,
a[data-testid="stPageLink-NavLink"]:hover {
    background: linear-gradient(120deg, rgba(0, 163, 224, 0.18), rgba(11, 31, 58, 0.18));
    color: var(--mck-deep-blue);
    transform: translateY(-1px);
}

.mck-stepper {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 0.6rem;
    margin: 1.6rem 0 1rem;
    position: relative;
}

.mck-stepper .step {
    text-align: center;
    position: relative;
    padding: 0.4rem 0.6rem 0.6rem;
    display: flex;
    flex-direction: column;
    gap: 0.55rem;
    align-items: center;
}

.mck-stepper .step::before {
    content: "";
    position: absolute;
    top: 19px;
    left: -50%;
    width: 100%;
    height: 2px;
    background: #D5DEEA;
    z-index: 0;
}

.mck-stepper .step:first-child::before {
    display: none;
}

.mck-stepper .circle {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: grid;
    place-items: center;
    font-weight: 600;
    font-size: 1rem;
    box-shadow: 0 12px 20px rgba(15, 23, 42, 0.12);
    z-index: 1;
}

.mck-stepper .label {
    font-size: 0.9rem;
    color: #4A5D75;
    font-weight: 500;
}

.mck-stepper .step.done::before {
    background: linear-gradient(90deg, var(--mck-deep-blue), var(--mck-accent));
}

.mck-stepper .step.done .circle {
    background: var(--mck-deep-blue);
    color: #FFFFFF;
}

.mck-stepper .step.active::before {
    background: linear-gradient(90deg, var(--mck-deep-blue), var(--mck-accent));
}

.mck-stepper .step.active .circle {
    background: var(--mck-accent);
    color: #FFFFFF;
}

.mck-stepper .step.upcoming .circle {
    background: #FFFFFF;
    border: 2px solid #CBD5E1;
    color: #7A889C;
}

div[data-testid="stMetric"] {
    background: #FFFFFF;
    border-radius: 16px;
    border: 1px solid rgba(15, 37, 62, 0.08);
    box-shadow: 0 26px 52px rgba(15, 23, 42, 0.08);
    padding: 1.2rem 1.4rem;
}

div[data-testid="stMetric"] label {
    text-transform: uppercase;
    color: #74859D;
    font-size: 0.75rem;
    letter-spacing: 0.06em;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--mck-deep-blue);
    font-size: 1.5rem;
    font-weight: 700;
}

div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    font-weight: 600;
}

div[data-testid="stDataFrame"] {
    border-radius: 18px;
    border: 1px solid rgba(15, 37, 62, 0.08);
    box-shadow: 0 24px 50px rgba(15, 23, 42, 0.08);
    overflow: hidden;
}

details {
    border-radius: 16px !important;
    border: 1px solid rgba(15, 37, 62, 0.08) !important;
    padding: 0.75rem 1rem !important;
    background: #FFFFFF !important;
    box-shadow: 0 18px 36px rgba(15, 23, 42, 0.08) !important;
}

details[open] {
    border-color: rgba(0, 163, 224, 0.3) !important;
}

details summary {
    font-weight: 600;
    color: var(--mck-deep-blue);
}

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(120deg, var(--mck-deep-blue), var(--mck-steel-blue));
    color: #FFFFFF;
    border-radius: 12px;
    border: none;
    font-weight: 600;
    padding: 0.6rem 1.3rem;
    box-shadow: 0 20px 32px rgba(15, 37, 62, 0.24);
    transition: all 0.2s ease-in-out;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: linear-gradient(120deg, var(--mck-deep-blue), var(--mck-accent));
    box-shadow: 0 22px 36px rgba(15, 37, 62, 0.28);
    transform: translateY(-1px);
}

[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid rgba(15, 37, 62, 0.08);
    padding: 1.6rem 1.2rem 2rem;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--mck-deep-blue);
}

[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stFormSubmitButton > button {
    width: 100%;
}

.mck-sidebar-header {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    margin-bottom: 1.4rem;
}

.mck-sidebar-header span {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--mck-deep-blue);
    letter-spacing: 0.05em;
}

.mck-sidebar-header p {
    margin: 0;
    color: #5B6C82;
    font-size: 0.85rem;
}

.mck-sidebar-footer {
    margin-top: 2rem;
    font-size: 0.75rem;
    color: #8391A6;
}

.mck-cta-banner {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1rem 1.3rem;
    border-radius: 16px;
    background: rgba(0, 163, 224, 0.08);
    border: 1px solid rgba(0, 163, 224, 0.18);
    color: var(--mck-deep-blue);
}

.mck-cta-banner strong {
    font-size: 0.95rem;
}

.mck-cta-banner span {
    font-size: 1.4rem;
}

@media (max-width: 992px) {
    .block-container {
        padding: 1.4rem 1rem 2.5rem;
    }

    .mck-header-card {
        flex-direction: column;
        align-items: flex-start;
        text-align: left;
    }

    .mck-header-icon {
        width: 56px;
        height: 56px;
        font-size: 1.6rem;
    }

    .mck-feature-card {
        min-height: auto;
    }
}
</style>
"""


def inject_mckinsey_style() -> None:
    """Inject bespoke CSS to give the app a McKinsey-inspired aesthetic."""

    if st.session_state.get(_STYLE_SESSION_KEY):
        return
    st.markdown(_MCKINSEY_STYLE, unsafe_allow_html=True)
    st.session_state[_STYLE_SESSION_KEY] = True


def render_page_header(title: str, subtitle: str = "", icon: Optional[str] = None) -> None:
    """Render a hero-style header block used across pages."""

    inject_mckinsey_style()
    icon_html = f"<div class='mck-header-icon'>{icon}</div>" if icon else ""
    subtitle_html = f"<p class='mck-header-subtitle'>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="mck-header-card">
            {icon_html}
            <div class="mck-header-copy">
                <h1>{title}</h1>
                {subtitle_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stepper(current_step: int) -> None:
    """Render a stylised progress stepper for the workflow."""

    inject_mckinsey_style()
    steps = ["ホーム", "取り込み", "自動検証", "結果サマリ", "ダッシュボード"]
    step_markup = []
    for idx, label in enumerate(steps):
        if idx < current_step:
            status = "done"
        elif idx == current_step:
            status = "active"
        else:
            status = "upcoming"
        step_markup.append(
            f"""
            <div class="step {status}">
                <div class="circle">{idx + 1}</div>
                <div class="label">{label}</div>
            </div>
            """
        )
    st.markdown(f"<div class='mck-stepper'>{''.join(step_markup)}</div>", unsafe_allow_html=True)


def render_sidebar_nav() -> None:
    """Render sidebar navigation links across pages with refined styling."""

    inject_mckinsey_style()
    with st.sidebar:
        st.markdown(
            """
            <div class="mck-sidebar-header">
                <span>ナビゲーション</span>
                <p>分析したいステップを選択してください。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.page_link("app.py", label="ホーム", icon="🏠")
        st.page_link("pages/01_データ入力.py", label="① データ入力", icon="📥")
        st.page_link("pages/02_ダッシュボード.py", label="② ダッシュボード", icon="📊")
        st.page_link("pages/03_標準賃率計算.py", label="③ 標準賃率計算", icon="🧮")
        st.markdown(
            """
            <div class="mck-sidebar-footer">Powered by 製品賃率ダッシュボード</div>
            """,
            unsafe_allow_html=True,
        )
