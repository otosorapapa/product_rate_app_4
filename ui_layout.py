import streamlit as st


def sidebar_steps():
    st.sidebar.header("メニュー")
    use_new = st.sidebar.toggle("新UIを試す", value=True, help="不具合時はOFFで旧UIに戻せます")
    step = st.sidebar.radio("ステップ", ["① データ", "② 設定", "③ 結果", "④ 書き出し"], index=0)
    return use_new, step
