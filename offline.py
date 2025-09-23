from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

try:  # pragma: no cover - streamlit is optional during unit tests
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - allows running tests without Streamlit
    st = None  # type: ignore

try:  # pragma: no cover - the JS bridge is optional for tests
    from streamlit_js_eval import streamlit_js_eval
except ModuleNotFoundError:  # pragma: no cover - degrade gracefully when not installed
    streamlit_js_eval = None  # type: ignore

_STORAGE_KEY = "rate_app_cache_v1"
_FLAG_KEY = "offline_cache_enabled"
_AVAILABLE_KEY = "_offline_cache_available"
_TIMESTAMP_KEY = "offline_cache_timestamp"
_NOTICE_KEY = "_offline_restore_notified"
_JS_KEY_PREFIX = "offline_cache"


def _call_js(expression: str, suffix: str) -> Optional[Any]:
    """Execute ``expression`` in the browser via :mod:`streamlit_js_eval`."""

    if streamlit_js_eval is None:  # pragma: no cover - JS bridge unavailable during tests
        return None
    key = f"{_JS_KEY_PREFIX}_{suffix}"
    try:
        return streamlit_js_eval(js_expressions=expression, key=key)
    except Exception:  # pragma: no cover - runtime JS errors should not crash the app
        return None


def ensure_offline_state_defaults() -> None:
    """Initialise the session state keys used by the offline helpers."""

    if st is None:  # pragma: no cover - streamlit not imported in tests
        return
    st.session_state.setdefault(_FLAG_KEY, True)
    st.session_state.setdefault(_AVAILABLE_KEY, False)
    st.session_state.setdefault(_TIMESTAMP_KEY, None)
    st.session_state.setdefault(_NOTICE_KEY, False)


def _build_payload() -> Optional[Dict[str, Any]]:
    """Serialise the current session data into a JSON-friendly payload."""

    if st is None:
        return None
    df = st.session_state.get("df_products_raw")
    if df is None or not isinstance(df, pd.DataFrame):
        return None

    products = df.where(pd.notnull(df), None).to_dict(orient="records")
    timestamp = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    payload: Dict[str, Any] = {
        "timestamp": timestamp,
        "sr_params": st.session_state.get("sr_params", {}),
        "calc_params": st.session_state.get("calc_params", {}),
        "scenarios": st.session_state.get("scenarios", {}),
        "current_scenario": st.session_state.get("current_scenario"),
        "using_sample_data": st.session_state.get("using_sample_data", False),
        "products": products,
    }
    return payload


def load_offline_cache() -> Optional[Dict[str, Any]]:
    """Return the cached payload stored in the browser, if any."""

    if st is None:
        return None
    ensure_offline_state_defaults()
    if streamlit_js_eval is None:  # pragma: no cover - offline mode unavailable during tests
        st.session_state[_AVAILABLE_KEY] = False
        return None

    raw = _call_js(f"window.localStorage.getItem('{_STORAGE_KEY}')", "get")
    if not raw:
        st.session_state[_AVAILABLE_KEY] = False
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        st.session_state[_AVAILABLE_KEY] = False
        return None
    if isinstance(payload, dict):
        st.session_state[_AVAILABLE_KEY] = True
    return payload


def restore_session_state_from_cache() -> bool:
    """Populate :mod:`streamlit` session state from the offline cache."""

    if st is None:
        return False
    ensure_offline_state_defaults()
    if st.session_state.get("df_products_raw") is not None:
        return False

    payload = load_offline_cache()
    if not payload:
        return False

    try:
        records = payload.get("products", [])
        df_products = pd.DataFrame(records)
    except Exception:
        return False

    st.session_state["df_products_raw"] = df_products
    st.session_state["sr_params"] = payload.get("sr_params", {})
    st.session_state["calc_params"] = payload.get("calc_params", {})
    scenarios = payload.get("scenarios")
    if isinstance(scenarios, dict):
        st.session_state["scenarios"] = scenarios
    current = payload.get("current_scenario")
    if current:
        st.session_state["current_scenario"] = current
    st.session_state["using_sample_data"] = payload.get("using_sample_data", False)
    timestamp = payload.get("timestamp")
    if isinstance(timestamp, str):
        st.session_state[_TIMESTAMP_KEY] = timestamp
    else:
        st.session_state[_TIMESTAMP_KEY] = None
    st.session_state[_AVAILABLE_KEY] = True
    st.session_state[_NOTICE_KEY] = False
    return True


def sync_offline_cache() -> None:
    """Persist the latest session snapshot to the browser cache."""

    if st is None:
        return
    ensure_offline_state_defaults()
    if not st.session_state.get(_FLAG_KEY, False):
        return
    if streamlit_js_eval is None:  # pragma: no cover - JS bridge unavailable during tests
        return

    payload = _build_payload()
    if payload is None:
        return

    payload_json = json.dumps(payload, ensure_ascii=False)
    _call_js(
        f"window.localStorage.setItem('{_STORAGE_KEY}', JSON.stringify({payload_json}))",
        "set",
    )
    st.session_state[_AVAILABLE_KEY] = True
    st.session_state[_TIMESTAMP_KEY] = payload["timestamp"]


def clear_offline_cache() -> None:
    """Delete the stored snapshot from the browser."""

    if st is None:
        return
    ensure_offline_state_defaults()
    if streamlit_js_eval is not None:
        _call_js(f"window.localStorage.removeItem('{_STORAGE_KEY}')", "clear")
    st.session_state[_AVAILABLE_KEY] = False
    st.session_state[_TIMESTAMP_KEY] = None


def render_offline_controls() -> None:
    """Display sidebar controls for managing offline caching."""

    if st is None:
        return
    ensure_offline_state_defaults()

    container = st.sidebar
    container.subheader("オフラインモード")
    container.caption(
        "通信が不安定な環境でも閲覧できるよう、ブラウザに最新データを保存します。"
    )

    js_ready = streamlit_js_eval is not None
    enabled = container.toggle(
        "端末に最新版をキャッシュ",
        key=_FLAG_KEY,
        help="ONにすると取り込んだExcelやシナリオ設定をブラウザに保存します。",
        disabled=not js_ready,
    )
    if not js_ready:
        container.caption("ブラウザ連携ライブラリが利用できないため保存機能は無効です。")
        return

    timestamp = st.session_state.get(_TIMESTAMP_KEY)
    available = st.session_state.get(_AVAILABLE_KEY, False)
    if enabled and available and timestamp:
        container.caption(f"最終保存: {timestamp}")
    elif enabled and not available:
        container.caption("データを読み込むと自動的に保存されます。")
    else:
        container.caption("OFFにすると端末には保存されません。")

    if container.button(
        "保存済みデータを削除",
        key="offline_cache_clear",
        use_container_width=True,
        disabled=not available,
        help="端末に保存したデータを削除します。",
    ):
        clear_offline_cache()
        toast = getattr(st, "toast", None)
        if callable(toast):
            toast("ローカルキャッシュを削除しました。")
        else:
            container.success("ローカルキャッシュを削除しました。")


def should_show_restore_notice() -> bool:
    """Return ``True`` when the offline restore notice should be displayed."""

    if st is None:
        return False
    ensure_offline_state_defaults()
    return not bool(st.session_state.get(_NOTICE_KEY))


def mark_restore_notice_shown() -> None:
    """Record that the offline restore notice has been shown this session."""

    if st is None:
        return
    ensure_offline_state_defaults()
    st.session_state[_NOTICE_KEY] = True
