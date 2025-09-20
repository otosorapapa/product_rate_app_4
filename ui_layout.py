from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any, Callable, Optional, TypeVar

import streamlit as st

T = TypeVar("T")

_HELP_MARKDOWN = """
**30秒で使い方**
1. 「①データ」でテンプレCSVをダウンロード → 記入してアップロード
2. 「②設定」で分析条件を指定
3. 「③結果」でKPI/グラフ/テーブルを確認
4. 「④書き出し」でCSV/XLSXを保存

**よくあるエラー**
- 文字コード: UTF-8 推奨
- 日付列: YYYY-MM-DD 形式
- 金額列: 数値のみ
"""


def _get_streamlit(st_module: Optional[Any] = None) -> Any:
    """Return an active Streamlit-like module for UI rendering."""

    module = st_module if st_module is not None else st
    if module is None:  # pragma: no cover - defensive guard for tests
        raise RuntimeError("Streamlit is required to render this component.")
    return module


def sidebar_steps():
    st.sidebar.header("メニュー")
    use_new = st.sidebar.toggle("新UIを試す", value=True, help="不具合時はOFFで旧UIに戻せます")
    step = st.sidebar.radio("ステップ", ["① データ", "② 設定", "③ 結果", "④ 書き出し"], index=0)
    return use_new, step


def help_panel(st_module: Optional[Any] = None) -> None:
    """Render a reusable help popover describing the app flow."""

    ui = _get_streamlit(st_module)
    popover = getattr(ui, "popover", None)
    if callable(popover):
        container = popover("❓ヘルプ")
    else:  # pragma: no cover - fallback for older Streamlit versions used in tests
        container = getattr(ui, "expander")("❓ヘルプ", expanded=False)
    with container:
        ui.markdown(_HELP_MARKDOWN)


def run_with_feedback(
    fn: Callable[..., T], *args: Any, st_module: Optional[Any] = None, sleep_fn: Optional[Callable[[float], None]] = None, **kwargs: Any
) -> Optional[T]:
    """Execute ``fn`` while surfacing progress, spinner feedback and errors.

    Parameters
    ----------
    fn:
        Callable to execute.  Its return value is forwarded to the caller.
    *args, **kwargs:
        Positional and keyword arguments passed to ``fn``.
    st_module:
        Optional Streamlit-like module allowing tests to supply a stub.
    sleep_fn:
        Optional sleep function used to simulate progress updates.  Defaults
        to :func:`time.sleep`.

    Returns
    -------
    Optional[T]
        The result of ``fn`` or ``None`` when an exception occurs.
    """

    ui = _get_streamlit(st_module)
    sleeper = sleep_fn if sleep_fn is not None else time.sleep

    progress = ui.progress(0, text="分析を開始します…")
    spinner_ctx = getattr(ui, "spinner", None)
    spinner = spinner_ctx("分析中…") if callable(spinner_ctx) else nullcontext()
    cm = spinner if hasattr(spinner, "__enter__") else nullcontext()

    try:
        with cm:
            try:
                for p in range(0, 101, 10):
                    sleeper(0.05)
                    progress.progress(p, text=f"進捗 {p}%")
                result: T = fn(*args, **kwargs)
                return result
            except Exception as exc:  # noqa: BLE001 - surfacing detailed feedback to the UI
                status_fn = getattr(ui, "status", None)
                if callable(status_fn):
                    status_fn(
                        "エラーが発生しました。入力形式としきい値をご確認ください。",
                        state="error",
                    )
                else:
                    ui.error("エラーが発生しました。入力形式としきい値をご確認ください。")
                expander = getattr(ui, "expander", None)
                if callable(expander):
                    with expander("技術ログ（担当者向け）"):
                        ui.exception(exc)
                else:  # pragma: no cover - minimal fallback
                    ui.exception(exc)
                return None
    finally:
        empty_fn = getattr(progress, "empty", None)
        if callable(empty_fn):
            empty_fn()

