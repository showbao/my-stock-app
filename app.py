
import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import re
import json
import urllib.request
import urllib.parse
import html
import ssl
from datetime import datetime, date, timedelta

# =========================
# 頁面設定
# =========================
st.set_page_config(page_title="投資儀表板", layout="wide")

DASHBOARD_CSS = """

<style>
/* 版面留白 */
.block-container { padding-top: 1.1rem; padding-bottom: 2.0rem; }

/* Sidebar 不要太擠 */
section[data-testid="stSidebar"] .block-container { padding-top: 1.0rem; }

/* Tabs 更好按（手機） */
button[data-baseweb="tab"] {
  padding-top: 10px !important;
  padding-bottom: 10px !important;
  font-size: 16px !important;
}

/* 主要標題區（儀表板頂部） */
.db-hero {
  padding: 14px 16px;
  border-radius: 14px;
  background: linear-gradient(135deg, #0B1220 0%, #111B2E 55%, #0B1220 100%);
  color: #E5E7EB;
  border: 1px solid rgba(255,255,255,0.08);
  margin-bottom: 10px;
}
.db-hero-title { font-size: 14px; color: rgba(229,231,235,0.75); margin-bottom: 4px; }
.db-hero-value { font-size: 36px; font-weight: 750; line-height: 1.1; letter-spacing: 0.2px; font-variant-numeric: tabular-nums; }
.db-hero-sub { font-size: 12px; color: rgba(229,231,235,0.60); margin-top: 6px; }

/* KPI 卡片 */
.kpi-card {
  padding: 12px 14px;
  border-radius: 14px;
  border: 1px solid rgba(15,23,42,0.10);
  background: #FFFFFF;
  box-shadow: 0 1px 12px rgba(15,23,42,0.05);
  margin-bottom: 10px;
}
.kpi-title { font-size: 13px; color: #6B7280; margin-bottom: 6px; }
.kpi-value { font-size: 26px; font-weight: 750; line-height: 1.15; color: #111827; font-variant-numeric: tabular-nums; }
.kpi-sub { font-size: 12px; color: #9CA3AF; margin-top: 4px; }

/* 小提示字 */
.small-muted { color: #6B7280; font-size: 12px; }

/* 表單區塊 */
.form-card {
  padding: 14px 14px;
  border-radius: 14px;
  border: 1px solid rgba(15,23,42,0.10);
  background: #FFFFFF;
  box-shadow: 0 1px 12px rgba(15,23,42,0.05);
}

/* 按鈕整行寬更像 App */
div.stButton > button { border-radius: 12px; }

/* Alert 圓角 */
div[data-testid="stAlert"] { border-radius: 12px; }

/* 正負顏色（不用只靠顏色也看得懂：仍保留 + / -） */
.profit-pos { color: #2E7D32; }
.profit-neg { color: #C62828; }
.profit-zero { color: #111827; }

/* 小表格外觀（更像儀表板區塊） */
.table-wrap {
  padding: 12px 14px;
  border-radius: 14px;
  border: 1px solid rgba(15,23,42,0.10);
  background: #FFFFFF;
  box-shadow: 0 1px 12px rgba(15,23,42,0.05);
  margin-top: 8px;
}

.kpi-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 10px;
  margin-top: 10px;
}
@media (min-width: 900px) {
  .kpi-grid { grid-template-columns: 1fr 1fr; }
}

/* 完全隱藏 Sidebar（左邊欄）與收合按鈕 */
section[data-testid="stSidebar"] { display: none; }
div[data-testid="stSidebarNav"] { display: none; }
button[data-testid="collapsedControl"] { display: none; }

</style>
"""
st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

# =========================
# 連線 Google Sheet
# =========================
@st.cache_resource
def connect_sheet():
    creds_info = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(
        creds_info,
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    client = gspread.authorize(creds)
    return client.open_by_key(st.secrets["SHEET_ID"])

sheet = connect_sheet()

def ensure_worksheet(ws_name: str, headers: list[str]):
    try:
        ws = sheet.worksheet(ws_name)
        return ws
    except Exception:
        ws = sheet.add_worksheet(title=ws_name, rows=1000, cols=max(10, len(headers) + 2))
        ws.append_row(headers)
        return ws

def safe_get_records(ws_name: str, default_headers: list[str]) -> pd.DataFrame:
    ws = ensure_worksheet(ws_name, default_headers)
    rows = ws.get_all_records()
    return pd.DataFrame(rows)

def load_data():
    transactions = safe_get_records(
        "transactions",
        ["id", "date", "action", "symbol", "currency", "fx_rate", "qty", "price", "amount_original", "amount_twd"]
    )
    assets = safe_get_records(
        "assets",
        ["symbol", "name", "asset_type", "quote_source", "quote_code", "nav_code", "currency", "price", "updated_at", "strategy", "enabled"]
    )
    allowed = safe_get_records("settings_allowed_emails", ["email"])
    return transactions, assets, allowed

transactions_df, assets_df, allowed_df = load_data()

# =========================
# Google 登入
# =========================
def login_screen():
    st.caption("請使用 Google 登入")
    if st.button("用 Google 登入", use_container_width=True):
        st.login()

if not st.user.is_logged_in:
    login_screen()
    st.stop()

user_email = (st.user.email or "").strip().lower()
allowed_emails = set(
    allowed_df.get("email", pd.Series(dtype=str))
    .astype(str).str.strip().str.lower().tolist()
)

if user_email not in allowed_emails:
    st.error("無權限")
    st.button("登出", on_click=st.logout, use_container_width=True)
    st.stop()

# =========================
# 頂部工具列（取代側邊欄）
# =========================
top_l, top_r = st.columns([7, 3])
with top_l:
    st.markdown("## 投資儀表板")
with top_r:
    st.caption(f"登入：{user_email}")
    st.button("登出", on_click=st.logout, use_container_width=True)

# =========================
# 工具函式
# =========================
def clean_symbol(symbol, asset_type=""):
    s = str(symbol).strip()
    at = str(asset_type).strip().lower()
    if at == "stock":
        s = s.upper()
    if at == "fund":
        return s
    return s

def symbol_key(symbol: str) -> str:
    return str(symbol).strip().upper()

def to_number(x):
    if x is None:
        return 0.0
    s = str(x).strip()
    if s == "":
        return 0.0
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return 0.0

def parse_date_series(series):
    s = series.astype(str).str.strip()
    s = s.str.replace("/", "-", regex=False).str.replace(".", "-", regex=False)
    dt = pd.to_datetime(s, errors="coerce", format=None)
    return dt.dt.date

def parse_dt_any(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None

def normalize_assets_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy() if df is not None else pd.DataFrame()
    required = ["symbol", "name", "asset_type", "quote_source", "quote_code", "nav_code", "currency", "price", "updated_at", "strategy", "enabled"]
    for c in required:
        if c not in out.columns:
            out[c] = ""
    if out.empty:
        return out
    out["symbol"] = out["symbol"].astype(str).str.strip()
    out["symbol_key"] = out["symbol"].astype(str).apply(symbol_key)
    out["name"] = out["name"].astype(str).str.strip()
    out["asset_type"] = out["asset_type"].astype(str).str.strip().str.lower()
    out["quote_source"] = out["quote_source"].astype(str).str.strip().str.lower()
    out["quote_code"] = out["quote_code"].astype(str).str.strip()
    out["currency"] = out["currency"].astype(str).str.strip().str.upper()
    out["strategy"] = out["strategy"].astype(str).str.strip()
    out["enabled"] = out["enabled"].astype(str).str.strip().str.upper()
    out["price_num"] = out["price"].apply(to_number)
    return out

assets_df = normalize_assets_df(assets_df)

def get_asset_row(symbol: str):
    if assets_df is None or assets_df.empty:
        return None
    key = symbol_key(symbol)
    rows = assets_df[assets_df["symbol_key"] == key]
    if rows.empty:
        return None
    return rows.iloc[-1].to_dict()

def get_default_strategy(symbol: str) -> str:
    row = get_asset_row(symbol)
    if not row:
        return ""
    return str(row.get("strategy", "")).strip()

def get_asset_currency(symbol: str) -> str:
    row = get_asset_row(symbol)
    if not row:
        return ""
    return str(row.get("currency", "")).strip().upper()

def get_asset_type(symbol: str) -> str:
    row = get_asset_row(symbol)
    if not row:
        return ""
    return str(row.get("asset_type", "")).strip().lower()

def update_asset_master(symbol: str, asset_type: str = "", currency: str = "", strategy: str = "", name: str = "", quote_source: str = "", quote_code: str = "", enabled: str = "Y"):
    ws = ensure_worksheet("assets", ["symbol", "name", "asset_type", "quote_source", "quote_code", "nav_code", "currency", "price", "updated_at", "strategy", "enabled"])
    rows = ws.get_all_values()
    if not rows:
        ws.append_row(["symbol", "name", "asset_type", "quote_source", "quote_code", "nav_code", "currency", "price", "updated_at", "strategy", "enabled"])
        rows = ws.get_all_values()

    header = rows[0]
    header_idx = {h: i for i, h in enumerate(header)}

    required = ["symbol", "name", "asset_type", "quote_source", "quote_code", "nav_code", "currency", "price", "updated_at", "strategy", "enabled"]
    missing = [c for c in required if c not in header_idx]
    if missing:
        raise ValueError("assets 表欄位不足，需包含：" + ",".join(required))

    key = symbol_key(symbol)
    target_row = None
    existing = None
    for i in range(1, len(rows)):
        r = rows[i]
        sym = str(r[header_idx["symbol"]]).strip() if len(r) > header_idx["symbol"] else ""
        if symbol_key(sym) == key:
            target_row = i + 1
            existing = {col: (r[idx] if len(r) > idx else "") for col, idx in header_idx.items()}
            break

    if existing is None:
        existing = {col: "" for col in header}

    values = existing.copy()
    values["symbol"] = str(symbol).strip()
    if name != "":
        values["name"] = str(name).strip()
    elif values.get("name", "") == "":
        values["name"] = str(symbol).strip()
    if asset_type != "":
        values["asset_type"] = str(asset_type).strip().lower()
    if quote_source != "":
        values["quote_source"] = str(quote_source).strip().lower()
    if quote_code != "":
        values["quote_code"] = str(quote_code).strip()
    elif values.get("quote_code", "") == "":
        values["quote_code"] = str(symbol).strip()
    if currency != "":
        values["currency"] = str(currency).strip().upper()
    if strategy != "":
        values["strategy"] = str(strategy).strip()
    if enabled != "":
        values["enabled"] = str(enabled).strip().upper()
    if values.get("enabled", "") == "":
        values["enabled"] = "Y"

    out_row = [values.get(col, "") for col in header]
    if target_row is None:
        ws.append_row(out_row)
    else:
        end_col = chr(ord("A") + len(header) - 1)
        ws.update(f"A{target_row}:{end_col}{target_row}", [out_row])

def enrich_transactions_with_assets(df_tx: pd.DataFrame, assets_df: pd.DataFrame) -> pd.DataFrame:
    df = df_tx.copy() if df_tx is not None else pd.DataFrame()
    if df.empty:
        for c in ["symbol", "strategy_effective", "asset_type_effective", "currency_effective", "price_current", "name_effective", "enabled_effective"]:
            if c not in df.columns:
                df[c] = ""
        return df

    # 讓這個函式可重複呼叫：若傳進來的是已 enrich 過的資料，先把上次加上的欄位移除，避免 merge 衝突
    derived_cols = [
        "symbol_key",
        "asset_name_master", "asset_type_master", "currency_master", "strategy_master",
        "enabled_master", "price_master", "updated_at_master", "quote_source_master", "quote_code_master", "nav_code_master",
        "strategy_effective", "asset_type_effective", "currency_effective", "price_current",
        "name_effective", "enabled_effective",
    ]
    drop_cols = [c for c in derived_cols if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    if "symbol" not in df.columns:
        df["symbol"] = ""
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["symbol_key"] = df["symbol"].astype(str).apply(symbol_key)

    a = assets_df.copy() if assets_df is not None else pd.DataFrame()
    if a.empty:
        df["strategy_effective"] = df.get("strategy", "").astype(str).str.strip() if "strategy" in df.columns else ""
        df["asset_type_effective"] = df.get("asset_type", "").astype(str).str.strip().str.lower() if "asset_type" in df.columns else ""
        df["currency_effective"] = df.get("currency", "").astype(str).str.strip().str.upper() if "currency" in df.columns else ""
        df["price_current"] = 0.0
        df["name_effective"] = df["symbol"]
        df["enabled_effective"] = "Y"
        return df

    keep = a[["symbol_key", "name", "asset_type", "currency", "strategy", "enabled", "price_num", "updated_at", "quote_source", "quote_code", "nav_code"]].copy()
    keep = keep.rename(columns={
        "name": "asset_name_master",
        "asset_type": "asset_type_master",
        "currency": "currency_master",
        "strategy": "strategy_master",
        "enabled": "enabled_master",
        "price_num": "price_master",
        "updated_at": "updated_at_master",
        "quote_source": "quote_source_master",
        "quote_code": "quote_code_master",
        "nav_code": "nav_code_master",
    })
    df = df.merge(keep, on="symbol_key", how="left")

    # 防呆：即使 assets 欄位不完整、或合併後沒有某些欄位，也不要直接報錯
    master_defaults = {
        "asset_name_master": "",
        "asset_type_master": "",
        "currency_master": "",
        "strategy_master": "",
        "enabled_master": "Y",
        "price_master": 0.0,
        "updated_at_master": "",
        "quote_source_master": "",
        "quote_code_master": "",
        "nav_code_master": "",
    }
    for c, default in master_defaults.items():
        if c not in df.columns:
            df[c] = default

    tx_strategy = df["strategy"].astype(str).str.strip() if "strategy" in df.columns else pd.Series([""] * len(df), index=df.index)
    tx_asset_type = df["asset_type"].astype(str).str.strip().str.lower() if "asset_type" in df.columns else pd.Series([""] * len(df), index=df.index)
    tx_currency = df["currency"].astype(str).str.strip().str.upper() if "currency" in df.columns else pd.Series([""] * len(df), index=df.index)

    df["strategy_effective"] = df["strategy_master"].fillna("").astype(str).str.strip()
    df.loc[df["strategy_effective"] == "", "strategy_effective"] = tx_strategy
    df.loc[df["strategy_effective"] == "", "strategy_effective"] = "未分類"

    df["asset_type_effective"] = df["asset_type_master"].fillna("").astype(str).str.strip().str.lower()
    df.loc[df["asset_type_effective"] == "", "asset_type_effective"] = tx_asset_type

    df["currency_effective"] = df["currency_master"].fillna("").astype(str).str.strip().str.upper()
    df.loc[df["currency_effective"] == "", "currency_effective"] = tx_currency

    df["price_current"] = df["price_master"].fillna(0).apply(to_number)
    df["name_effective"] = df["asset_name_master"].fillna("").astype(str).str.strip()
    df.loc[df["name_effective"] == "", "name_effective"] = df["symbol"]

    df["enabled_effective"] = df["enabled_master"].fillna("").astype(str).str.strip().str.upper()
    df.loc[df["enabled_effective"] == "", "enabled_effective"] = "Y"
    return df

enriched_tx_df = enrich_transactions_with_assets(transactions_df, assets_df)


def get_current_qty(transactions_df, symbol, asset_type=""):
    df = enrich_transactions_with_assets(transactions_df, assets_df)
    if df.empty:
        return 0.0
    target = df[df["symbol_key"] == symbol_key(symbol)]
    qty = 0.0
    for _, row in target.iterrows():
        action = str(row.get("action", "")).strip().lower()
        q = to_number(row.get("qty", 0))
        if action in ["buy", "initial"]:
            qty += q
        elif action == "sell":
            qty -= q
    return qty

ACTION_ORDER = {
    "initial": 0,
    "buy": 1,
    "dividend": 2,
    "sell": 3,
}

ERROR_MESSAGES = {
    "multi_initial": "資料異常：同一標的不可有多筆 initial",
    "sell_qty": "資料異常：賣出數量不可小於等於 0",
    "sell_over": "資料異常：賣出時可用持股不足",
    "neg_qty": "資料異常：剩餘持股小於 0",
    "usd_missing": "資料異常：缺少 USD_TWD 匯率",
    "no_data": "查無交易紀錄",
    "need_initial": "資料異常，請檢查該標的是否漏填 initial 交易",
}

def empty_position_state():
    return {
        "remaining_qty": 0.0,
        "remaining_cost": 0.0,
        "avg_cost": 0.0,
        "realized_profit": 0.0,
        "dividend_income": 0.0,
        "buy_amount": 0.0,
        "sell_amount": 0.0,
        "is_valid": True,
        "error_message": "",
    }

def invalidate_state(state: dict, message: str):
    state["is_valid"] = False
    state["error_message"] = message
    return state

def normalize_action(v):
    return str(v or "").strip().lower()

def get_symbol_tx(df_tx: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df_tx is None or df_tx.empty:
        return pd.DataFrame()
    df = df_tx.copy()
    df["row_seq"] = range(len(df))
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df = df[df["symbol"] == str(symbol).strip()].copy()
    if df.empty:
        return df
    df["date_norm"] = parse_date_series(df["date"])
    df["action_norm"] = df["action"].map(normalize_action)
    df["action_order"] = df["action_norm"].map(lambda x: ACTION_ORDER.get(x, 99))
    df["id_num"] = pd.to_numeric(df.get("id"), errors="coerce").fillna(10**12)
    df = df.sort_values(["date_norm", "action_order", "id_num", "row_seq"], ascending=[True, True, True, True])
    return df

def detect_multiple_initial(tx: pd.DataFrame) -> str:
    if tx is None or tx.empty:
        return ""
    cnt = int(tx["action"].map(normalize_action).eq("initial").sum())
    return ERROR_MESSAGES["multi_initial"] if cnt > 1 else ""

def apply_tx_to_state(state: dict, row: dict) -> dict:
    if not state["is_valid"]:
        return state
    action = normalize_action(row.get("action"))
    qty = to_number(row.get("qty", 0))
    amount_twd = to_number(row.get("amount_twd", 0))

    if action in ["buy", "initial"]:
        state["remaining_qty"] += qty
        state["remaining_cost"] += amount_twd
        state["buy_amount"] += amount_twd
    elif action == "sell":
        if qty <= 0:
            return invalidate_state(state, ERROR_MESSAGES["sell_qty"])
        if state["remaining_qty"] <= 0 or qty > state["remaining_qty"]:
            return invalidate_state(state, ERROR_MESSAGES["sell_over"])
        avg_cost = state["remaining_cost"] / state["remaining_qty"] if state["remaining_qty"] > 0 else 0.0
        sell_cost = qty * avg_cost
        state["remaining_qty"] -= qty
        state["remaining_cost"] -= sell_cost
        state["sell_amount"] += amount_twd
        state["realized_profit"] += (amount_twd - sell_cost)
        if state["remaining_qty"] < 0:
            return invalidate_state(state, ERROR_MESSAGES["neg_qty"])
        if abs(state["remaining_qty"]) < 1e-9:
            state["remaining_qty"] = 0.0
            state["remaining_cost"] = 0.0
    elif action == "dividend":
        state["dividend_income"] += amount_twd

    state["avg_cost"] = state["remaining_cost"] / state["remaining_qty"] if state["remaining_qty"] > 0 else 0.0
    return state

def get_market_value_twd(asset_row: dict, qty: float, usd_twd_row: dict | None):
    price_now = to_number((asset_row or {}).get("price_num", 0))
    currency = str((asset_row or {}).get("currency", "")).strip().upper()
    market_value = qty * price_now
    if currency == "USD":
        usd_twd = to_number((usd_twd_row or {}).get("price_num", 0))
        if usd_twd <= 0:
            return None, ERROR_MESSAGES["usd_missing"]
        market_value *= usd_twd
    return market_value, ""

def calc_symbol_full_history(df_tx: pd.DataFrame, symbol: str, asset_row: dict | None = None, usd_twd_row: dict | None = None) -> dict:
    tx = get_symbol_tx(df_tx, symbol)
    asset_row = asset_row or get_asset_row(symbol) or {}
    usd_twd_row = usd_twd_row or get_asset_row("USD_TWD")
    if tx.empty:
        return {
            "symbol": symbol,
            "has_data": False,
            "is_valid": True,
            "error_message": ERROR_MESSAGES["no_data"],
            "strategy": str(asset_row.get("strategy", "")).strip() or "未分類",
            "asset_type": str(asset_row.get("asset_type", "")).strip().lower(),
        }
    multi_initial_error = detect_multiple_initial(tx)
    if multi_initial_error:
        return {
            "symbol": symbol,
            "has_data": True,
            "is_valid": False,
            "error_message": multi_initial_error,
            "strategy": str(asset_row.get("strategy", "")).strip() or "未分類",
            "asset_type": str(asset_row.get("asset_type", "")).strip().lower(),
        }

    state = empty_position_state()
    for _, row in tx.iterrows():
        state = apply_tx_to_state(state, row.to_dict())
        if not state["is_valid"]:
            break

    market_value, mv_error = get_market_value_twd(asset_row, state["remaining_qty"], usd_twd_row)
    if mv_error and state["is_valid"]:
        state = invalidate_state(state, mv_error)

    unrealized_profit = None
    total_profit = None
    total_return_pct = None
    yoc = None
    recovery_rate = None
    if state["is_valid"] and market_value is not None:
        unrealized_profit = market_value - state["remaining_cost"]
        total_profit = state["realized_profit"] + state["dividend_income"] + unrealized_profit
        total_return_pct = (total_profit / state["buy_amount"] * 100) if state["buy_amount"] > 0 else 0.0
        yoc = (state["dividend_income"] / state["remaining_cost"] * 100) if state["remaining_cost"] > 0 else 0.0
        recovery_rate = ((state["dividend_income"] + state["realized_profit"]) / state["buy_amount"] * 100) if state["buy_amount"] > 0 else 0.0

    return {
        "symbol": symbol,
        "has_data": True,
        "strategy": str(asset_row.get("strategy", "")).strip() or "未分類",
        "asset_type": str(asset_row.get("asset_type", "")).strip().lower(),
        "remaining_qty": state["remaining_qty"],
        "remaining_cost": state["remaining_cost"],
        "market_value": market_value if state["is_valid"] else None,
        "realized_profit": state["realized_profit"] if state["is_valid"] else None,
        "unrealized_profit": unrealized_profit,
        "dividend_income": state["dividend_income"] if state["is_valid"] else None,
        "buy_amount": state["buy_amount"],
        "sell_amount": state["sell_amount"],
        "total_profit": total_profit,
        "total_return_pct": total_return_pct,
        "yoc": yoc,
        "recovery_rate": recovery_rate,
        "is_valid": state["is_valid"],
        "error_message": state["error_message"],
    }

def calc_symbol_in_range(df_tx: pd.DataFrame, symbol: str, start_d, end_d, asset_row: dict | None = None, usd_twd_row: dict | None = None) -> dict:
    tx = get_symbol_tx(df_tx, symbol)
    asset_row = asset_row or get_asset_row(symbol) or {}
    usd_twd_row = usd_twd_row or get_asset_row("USD_TWD")
    base = {
        "symbol": symbol,
        "strategy": str(asset_row.get("strategy", "")).strip() or "未分類",
        "asset_type": str(asset_row.get("asset_type", "")).strip().lower(),
    }
    if tx.empty:
        return {**base, "has_data": False, "is_valid": True, "error_message": ERROR_MESSAGES["no_data"]}
    multi_initial_error = detect_multiple_initial(tx)
    if multi_initial_error:
        return {**base, "has_data": True, "is_valid": False, "error_message": multi_initial_error}

    before_df = tx[tx["date_norm"] < start_d].copy()
    range_df = tx[(tx["date_norm"] >= start_d) & (tx["date_norm"] <= end_d)].copy()

    start_state = empty_position_state()
    for _, row in before_df.iterrows():
        start_state = apply_tx_to_state(start_state, row.to_dict())
        if not start_state["is_valid"]:
            break
    if not start_state["is_valid"]:
        return {**base, "has_data": not range_df.empty, "is_valid": False, "error_message": start_state["error_message"]}

    end_state = dict(start_state)
    buy_in_range = 0.0
    sell_in_range = 0.0
    dividend_in_range = 0.0
    for _, row in range_df.iterrows():
        action = normalize_action(row.get("action"))
        amount_twd = to_number(row.get("amount_twd", 0))
        if action in ["buy", "initial"]:
            buy_in_range += amount_twd
        elif action == "sell":
            sell_in_range += amount_twd
        elif action == "dividend":
            dividend_in_range += amount_twd
        end_state = apply_tx_to_state(end_state, row.to_dict())
        if not end_state["is_valid"]:
            break
    if not end_state["is_valid"]:
        return {**base, "has_data": not range_df.empty, "is_valid": False, "error_message": end_state["error_message"]}

    market_value_start, start_err = get_market_value_twd(asset_row, start_state["remaining_qty"], usd_twd_row)
    market_value_end, end_err = get_market_value_twd(asset_row, end_state["remaining_qty"], usd_twd_row)
    if start_err or end_err:
        return {**base, "has_data": not range_df.empty, "is_valid": False, "error_message": start_err or end_err}

    unrealized_profit = (market_value_end - end_state["remaining_cost"]) if market_value_end is not None else None
    period_profit = (market_value_end - market_value_start) + (sell_in_range - buy_in_range + dividend_in_range)
    base_amount = market_value_start + buy_in_range
    period_return_pct = (period_profit / base_amount * 100) if base_amount > 0 else 0.0

    return {
        **base,
        "has_data": not range_df.empty,
        "is_valid": True,
        "error_message": "",
        "start_qty": start_state["remaining_qty"],
        "start_cost": start_state["remaining_cost"],
        "start_market_value": market_value_start,
        "end_qty": end_state["remaining_qty"],
        "end_cost": end_state["remaining_cost"],
        "end_market_value": market_value_end,
        "buy_in_range": buy_in_range,
        "sell_in_range": sell_in_range,
        "dividend_in_range": dividend_in_range,
        "realized_profit": end_state["realized_profit"] - start_state["realized_profit"],
        "unrealized_profit": unrealized_profit,
        "period_profit": period_profit,
        "period_return_pct": period_return_pct,
    }

def build_symbol_results(df_tx: pd.DataFrame, asset_type_filter: str | None = None, symbols: list[str] | None = None, strategy: str | None = None):
    symbols = symbols or []
    df = enrich_transactions_with_assets(df_tx, assets_df).copy() if df_tx is not None else pd.DataFrame()
    if df.empty:
        return []
    if asset_type_filter:
        df = df[df["asset_type_effective"] == asset_type_filter].copy()
    if symbols:
        df["symbol"] = df["symbol"].astype(str).str.strip()
        df = df[df["symbol"].isin([str(x).strip() for x in symbols])].copy()
    if strategy and strategy != "全部":
        df["strategy_effective"] = df["strategy_effective"].astype(str).str.strip()
        df = df[df["strategy_effective"] == strategy].copy()
    if df.empty:
        return []
    rows = []
    usd_twd_row = get_asset_row("USD_TWD")
    for sym in sorted(df["symbol"].astype(str).str.strip().unique().tolist()):
        rows.append(calc_symbol_full_history(df, sym, get_asset_row(sym), usd_twd_row))
    return rows

def summarize_valid_only(result_list: list[dict]) -> dict:
    valid_items = [x for x in result_list if x.get("has_data") and x.get("is_valid")]
    invalid_items = [x for x in result_list if x.get("has_data") and not x.get("is_valid")]
    total_market_value = sum(to_number(x.get("market_value", 0)) for x in valid_items)
    total_cost = sum(to_number(x.get("remaining_cost", 0)) for x in valid_items)
    total_dividend = sum(to_number(x.get("dividend_income", 0)) for x in valid_items)
    total_unrealized = sum(to_number(x.get("unrealized_profit", 0)) for x in valid_items if x.get("unrealized_profit") is not None)
    total_realized = sum(to_number(x.get("realized_profit", 0)) for x in valid_items if x.get("realized_profit") is not None)
    total_profit = total_realized + total_unrealized + total_dividend
    total_return_pct = (total_profit / total_cost * 100) if total_cost > 0 else 0.0
    return {
        "valid_count": len(valid_items),
        "invalid_count": len(invalid_items),
        "market_value": total_market_value,
        "remaining_cost": total_cost,
        "dividend_income": total_dividend,
        "unrealized_profit": total_unrealized,
        "realized_profit": total_realized,
        "total_profit": total_profit,
        "total_return_pct": total_return_pct,
    }

def calculate_metrics(df, assets_df):
    results = build_symbol_results(df)
    s = summarize_valid_only(results)
    return s["remaining_cost"], s["market_value"], s["dividend_income"], s["total_profit"], s["total_return_pct"]

def strategy_summary(df_tx: pd.DataFrame, assets_df: pd.DataFrame) -> pd.DataFrame:
    results = build_symbol_results(df_tx)
    if not results:
        return pd.DataFrame(columns=["策略", "持有成本(TWD)", "目前市值(TWD)", "總報酬率", "異常筆數"])
    rows = []
    by_strategy = {}
    for r in results:
        strat = str(r.get("strategy", "")).strip() or "未分類"
        by_strategy.setdefault(strat, []).append(r)
    for strat, items in by_strategy.items():
        s = summarize_valid_only(items)
        rows.append({
            "策略": strat,
            "持有成本(TWD)": s["remaining_cost"],
            "目前市值(TWD)": s["market_value"],
            "總報酬率": s["total_return_pct"],
            "異常筆數": s["invalid_count"],
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["策略", "持有成本(TWD)", "目前市值(TWD)", "總報酬率", "異常筆數"])
    out = out.sort_values("目前市值(TWD)", ascending=False)
    out["持有成本(TWD)"] = out["持有成本(TWD)"].map(lambda x: f"{x:,.0f}")
    out["目前市值(TWD)"] = out["目前市值(TWD)"].map(lambda x: f"{x:,.0f}")
    out["總報酬率"] = out["總報酬率"].map(lambda x: f"{x:.2f}%")
    return out

def top_holdings_table(df_tx: pd.DataFrame, assets_df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    results = [x for x in build_symbol_results(df_tx) if x.get("has_data") and x.get("is_valid") and to_number(x.get("remaining_qty", 0)) > 0]
    if not results:
        return pd.DataFrame(columns=["代碼", "策略", "持有數量", "市值(TWD)", "未實現損益"])
    rows = []
    total_mv = sum(to_number(x.get("market_value", 0)) for x in results)
    for r in results:
        rows.append({
            "代碼": r.get("symbol", ""),
            "策略": r.get("strategy", "未分類"),
            "持有數量": to_number(r.get("remaining_qty", 0)),
            "市值(TWD)": to_number(r.get("market_value", 0)),
            "未實現損益": to_number(r.get("unrealized_profit", 0)),
            "佔比": (to_number(r.get("market_value", 0)) / total_mv) if total_mv > 0 else 0.0,
        })
    out = pd.DataFrame(rows).sort_values("市值(TWD)", ascending=False).head(top_n)
    out["持有數量"] = out["持有數量"].map(lambda x: f"{x:,.4f}".rstrip("0").rstrip("."))
    out["市值(TWD)"] = out["市值(TWD)"].map(lambda x: f"{x:,.0f}")
    out["未實現損益"] = out["未實現損益"].map(lambda x: f"{x:,.0f}")
    out["佔比"] = out["佔比"].map(lambda x: f"{x*100:.1f}%")
    return out[["代碼", "策略", "持有數量", "市值(TWD)", "未實現損益", "佔比"]]

def format_symbol_label(symbol: str) -> str:
    row = get_asset_row(symbol) or {}
    name = str(row.get("name", "")).strip()
    return f"{name} / {symbol}" if name else symbol

def compact_symbol_card(result: dict):
    title = format_symbol_label(result.get("symbol", ""))
    strategy = result.get("strategy", "未分類")
    market_value = fmt_money(result.get("market_value")) if result.get("market_value") is not None else "—"
    profit_html = fmt_signed_money_html(result.get("total_profit")) if result.get("total_profit") is not None else "—"
    st.markdown(f"### {title}")
    st.caption(f"策略：{strategy}")
    cols = st.columns(2)
    with cols[0]:
        st.metric("市值(TWD)", market_value)
    with cols[1]:
        st.markdown(kpi_card_html("總報酬", profit_html), unsafe_allow_html=True)

def detail_symbol_card(result: dict, key_prefix: str = "detail"):
    title = format_symbol_label(result.get("symbol", ""))
    asset_type = result.get("asset_type", "")
    strategy = result.get("strategy", "未分類")
    with st.container(border=True):
        st.markdown(f"### {title}")
        st.caption(f"資產類型：{asset_type or '未分類'} ｜ 策略：{strategy}")
        if not result.get("has_data"):
            st.info(ERROR_MESSAGES["no_data"])
            return
        if not result.get("is_valid"):
            st.error(result.get("error_message") or ERROR_MESSAGES["need_initial"])
            return
        c1, c2 = st.columns(2)
        with c1:
            st.metric("持股", f"{to_number(result.get('remaining_qty', 0)):,.4f}".rstrip("0").rstrip("."))
            st.metric("持有成本(TWD)", fmt_money(result.get("remaining_cost")))
            st.metric("市值(TWD)", fmt_money(result.get("market_value")))
        with c2:
            st.metric("已實現損益", fmt_money(result.get("realized_profit")))
            st.metric("未實現損益", fmt_money(result.get("unrealized_profit")))
            st.metric("累計股息", fmt_money(result.get("dividend_income")))
        st.markdown("#### 指標")
        if "存股" in str(strategy):
            k1, k2, k3 = st.columns(3)
            k1.metric("成本殖利率", f"{to_number(result.get('yoc', 0)):.2f}%")
            k2.metric("資產回收率", f"{to_number(result.get('recovery_rate', 0)):.2f}%")
            k3.metric("未實現損益", fmt_money(result.get("unrealized_profit")))
        else:
            k1, k2, k3 = st.columns(3)
            k1.metric("總報酬", fmt_money(result.get("total_profit")))
            k2.metric("總報酬率", f"{to_number(result.get('total_return_pct', 0)):.2f}%")
            k3.metric("已實現損益", fmt_money(result.get("realized_profit")))

def summarize_range_results(results: list[dict]) -> list[dict]:
    buckets = {}
    for r in results:
        strat = str(r.get("strategy", "")).strip() or "未分類"
        key = "波段" if "波段" in strat else "存股" if "存股" in strat else strat
        buckets.setdefault(key, []).append(r)
    cards = []
    for key, items in buckets.items():
        valid = [x for x in items if x.get("has_data") and x.get("is_valid")]
        invalid = [x for x in items if x.get("has_data") and not x.get("is_valid")]
        if not valid and not invalid:
            continue
        start_mv = sum(to_number(x.get("start_market_value", 0)) for x in valid)
        end_mv = sum(to_number(x.get("end_market_value", 0)) for x in valid)
        buy_amt = sum(to_number(x.get("buy_in_range", 0)) for x in valid)
        sell_amt = sum(to_number(x.get("sell_in_range", 0)) for x in valid)
        dividend_amt = sum(to_number(x.get("dividend_in_range", 0)) for x in valid)
        realized = sum(to_number(x.get("realized_profit", 0)) for x in valid)
        unrealized = sum(to_number(x.get("unrealized_profit", 0)) for x in valid if x.get("unrealized_profit") is not None)
        period_profit = sum(to_number(x.get("period_profit", 0)) for x in valid)
        base_amount = start_mv + buy_amt
        period_return_pct = (period_profit / base_amount * 100) if base_amount > 0 else 0.0
        cards.append({
            "strategy_bucket": key,
            "symbol_count": len(valid),
            "invalid_count": len(invalid),
            "start_market_value": start_mv,
            "end_market_value": end_mv,
            "buy_in_range": buy_amt,
            "sell_in_range": sell_amt,
            "dividend_in_range": dividend_amt,
            "realized_profit": realized,
            "unrealized_profit": unrealized,
            "period_profit": period_profit,
            "period_return_pct": period_return_pct,
            "errors": [x.get("error_message", "") for x in invalid],
        })
    order = {"波段": 0, "存股": 1}
    return sorted(cards, key=lambda x: order.get(x["strategy_bucket"], 99))

def render_range_strategy_card(card: dict):
    with st.container(border=True):
        st.markdown(f"### {card['strategy_bucket']} 策略結果")
        st.caption(f"納入標的：{card['symbol_count']} 支")
        if card.get("invalid_count"):
            st.warning(f"有 {card['invalid_count']} 支標的資料異常，未納入加總")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("區間損益", fmt_money(card.get("period_profit")))
            st.metric("區間報酬率", f"{to_number(card.get('period_return_pct', 0)):.2f}%")
            st.metric("區間配息", fmt_money(card.get("dividend_in_range")))
        with c2:
            st.metric("已實現損益", fmt_money(card.get("realized_profit")))
            st.metric("未實現損益", fmt_money(card.get("unrealized_profit")))
            st.metric("區間結束市值", fmt_money(card.get("end_market_value")))

# =========================
# 顯示用（儀表板風）
# =========================
def fmt_money(n):
    try:
        return f"{float(n):,.0f}"
    except Exception:
        return "—"

def fmt_signed_money_html(n):
    try:
        n = float(n)
        sign = "+" if n > 0 else ""
        cls = "profit-pos" if n > 0 else "profit-neg" if n < 0 else "profit-zero"
        return f"<span class='{cls}'>{sign}{n:,.0f}</span>"
    except Exception:
        return "—"

def fmt_signed_pct_html(n):
    try:
        n = float(n)
        sign = "+" if n > 0 else ""
        cls = "profit-pos" if n > 0 else "profit-neg" if n < 0 else "profit-zero"
        return f"<span class='{cls}'>{sign}{n:.2f}%</span>"
    except Exception:
        return "—"

def kpi_card_html(title, value_text, sub_text=""):
    sub = f'<div class="kpi-sub">{sub_text}</div>' if sub_text else ""
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-title">{title}</div>'
        f'<div class="kpi-value">{value_text}</div>'
        f'{sub}'
        f'</div>'
    )

def hero_card(title, value_text, sub_text=""):
    st.markdown(
        f"""
        <div class="db-hero">
          <div class="db-hero-title">{title}</div>
          <div class="db-hero-value">{value_text}</div>
          {f'<div class="db-hero-sub">{sub_text}</div>' if sub_text else ''}
        </div>
        """,
        unsafe_allow_html=True
    )

def get_usd_twd_info(assets_df):
    row = get_asset_row("USD_TWD")
    if not row:
        return None, None
    fx = to_number(row.get("price_num"))
    updated_at = str(row.get("updated_at", "")).strip()
    return fx, (updated_at if updated_at else None)

def get_prices_updated_at(assets_df):
    if assets_df is None or assets_df.empty or "updated_at" not in assets_df.columns:
        return None
    s = assets_df["updated_at"].astype(str).str.strip()
    s = s[s != ""]
    if s.empty:
        return None
    return s.max()

# =========================
# 抓價
# =========================


def decode_response_bytes(data: bytes, content_type: str = "") -> str:
    """盡量正確解碼中文基金網站頁面。MoneyDJ 常見 cp950 / big5。"""
    ct = str(content_type or "").lower()

    # 先依 header 指定 charset
    m = re.search(r"charset=([\w\-]+)", ct)
    candidates = []
    if m:
        candidates.append(m.group(1).strip())

    # 常見中文網站編碼依序嘗試
    candidates += [
        "utf-8",
        "utf-8-sig",
        "cp950",
        "big5",
        "big5hkscs",
        "latin1",
    ]

    tried = set()
    for enc in candidates:
        enc_l = enc.lower()
        if enc_l in tried:
            continue
        tried.add(enc_l)
        try:
            txt = data.decode(enc)
            # 若解出來有常見亂碼，再繼續試別的
            if txt.count("�") > 5:
                continue
            return txt
        except Exception:
            continue

    return data.decode("utf-8", errors="ignore")
def http_get_json(url: str, timeout: int = 10):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    return json.loads(raw)

def http_get_text(url: str, timeout: int = 10):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/csv,text/plain,*/*",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        return decode_response_bytes(raw, resp.headers.get("Content-Type", ""))


def http_get_text_unverified(url: str, timeout: int = 10):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/html,text/plain,*/*",
            "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
        },
        method="GET",
    )
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        raw = resp.read()
        return decode_response_bytes(raw, resp.headers.get("Content-Type", ""))



def strip_html_text(raw: str) -> str:
    s = html.unescape(str(raw or ""))
    s = re.sub(r"<script[\s\S]*?</script>", " ", s, flags=re.I)
    s = re.sub(r"<style[\s\S]*?</style>", " ", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_nav_from_html(raw_html: str):
    html_text = str(raw_html or "")
    plain = strip_html_text(html_text)
    patterns = [
        r'最新淨值[^0-9-]{0,20}([0-9]+(?:\.[0-9]+)?)',
        r'基金淨值[^0-9-]{0,20}([0-9]+(?:\.[0-9]+)?)',
        r'淨值日期[^0-9]{0,20}[0-9]{4}[/-][0-9]{1,2}[/-][0-9]{1,2}[^0-9]{0,30}淨值[^0-9-]{0,20}([0-9]+(?:\.[0-9]+)?)',
        r'"(?:latest)?nav"\s*[:=]\s*"?([0-9]+(?:\.[0-9]+)?)"?',
        r'"price"\s*[:=]\s*"?([0-9]+(?:\.[0-9]+)?)"?',
    ]
    for pat in patterns:
        m = re.search(pat, html_text, flags=re.I) or re.search(pat, plain, flags=re.I)
        if m:
            try:
                price = float(m.group(1))
                if price > 0:
                    return price
            except Exception:
                pass
    nums = re.findall(r'([0-9]+(?:\.[0-9]{2,6})?)', plain)
    for n in nums:
        try:
            price = float(n)
            if 0 < price < 100000:
                return price
        except Exception:
            continue
    raise ValueError("網頁中找不到淨值")


def fetch_nav_from_url(url: str, verify_ssl: bool = True):
    if verify_ssl:
        raw = http_get_text(url, timeout=15)
    else:
        raw = http_get_text_unverified(url, timeout=15)
    return parse_nav_from_html(raw)


def parse_moneydj_nav(raw_html: str):
    html_text = str(raw_html or "")
    plain = strip_html_text(html_text)

    # 1) 最優先：抓「最新淨值」表頭後面那一列的日期 + 淨值
    patterns = [
        r'最新淨值[\s\S]{0,200}?(?:20\d{2}[/-]\d{1,2}[/-]\d{1,2}|\d{2}/\d{2})\s*[,，\s]\s*([0-9]+\.[0-9]{2,6})',
        r'最新值\s*([0-9]+\.[0-9]{2,6})',
        r'最新淨值[^0-9]{0,20}([0-9]+\.[0-9]{2,6})',
        r'淨值[^0-9]{0,20}(?:20\d{2}[/-]\d{1,2}[/-]\d{1,2}|\d{2}/\d{2})\s*[,，\s]\s*([0-9]+\.[0-9]{2,6})',
    ]
    for pat in patterns:
        m = re.search(pat, html_text, flags=re.I) or re.search(pat, plain, flags=re.I)
        if m:
            try:
                price = float(m.group(1))
                if 0 < price < 100000:
                    return price
            except Exception:
                pass

    # 2) 次優先：在「最新淨值」附近找日期後第一個小數，不接受像 2026 這種整數年份
    anchor_pos = plain.find('最新淨值')
    if anchor_pos == -1:
        anchor_pos = plain.find('最新值')
    if anchor_pos != -1:
        segment = plain[anchor_pos:anchor_pos + 300]
        m = re.search(r'(?:20\d{2}[/-]\d{1,2}[/-]\d{1,2}|\d{2}/\d{2})\s*[,，\s]\s*([0-9]+\.[0-9]{2,6})', segment)
        if m:
            price = float(m.group(1))
            if 0 < price < 100000:
                return price

    # 3) 再退一步：找「近30日淨值 / 日期, 淨值」區塊的第一筆小數
    for marker in ['近30日淨值', '日期, 淨值', '淨值.']:
        pos = plain.find(marker)
        if pos != -1:
            segment = plain[pos:pos + 400]
            m = re.search(r'(?:20\d{2}[/-]\d{1,2}[/-]\d{1,2}|\d{2}/\d{2})\s*[,，\s]\s*([0-9]+\.[0-9]{2,6})', segment)
            if m:
                price = float(m.group(1))
                if 0 < price < 100000:
                    return price

    raise ValueError('MoneyDJ 頁面中找不到正確淨值')


def fetch_moneydj_nav(nav_code: str):
    code = str(nav_code or "").strip()
    if code == "":
        raise ValueError("MoneyDJ 代碼空白")

    if code.startswith("http://") or code.startswith("https://"):
        base_urls = [code]
    else:
        q = urllib.parse.quote(code)
        base_urls = [
            f"https://www.moneydj.com/funddj/ya/yp010000.djhtm?a={q}",
            f"https://www.moneydj.com/funddj/ya/yp010001.djhtm?a={q}",
        ]

    # MoneyDJ 的 &topc= 版本較接近純文字摘要，優先使用，較不容易抓錯到年份或其他數字
    urls = []
    for u in base_urls:
        if "topc=" not in u:
            sep = "&" if "?" in u else "?"
            urls.append(f"{u}{sep}topc=")
        urls.append(u)

    last_err = None
    for url in urls:
        try:
            raw = http_get_text_unverified(url, timeout=15)
            plain = strip_html_text(raw)

            # 優先抓 MoneyDJ 常見摘要格式
            patterns = [
                r'淨值日期\s*[,，\s]+最新淨值[\s\S]{0,80}?(?:20\d{2}[/-]\d{1,2}[/-]\d{1,2}|\d{2}/\d{2})\s*[,，\s]+([0-9]+(?:\.[0-9]{2,6})?)',
                r'最新值\s*([0-9]+(?:\.[0-9]{2,6})?)\s*(?:台幣|美元|人民幣|歐元)',
                r'最新淨值[^0-9]{0,40}(?:20\d{2}[/-]\d{1,2}[/-]\d{1,2})\s*[,，]\s*([0-9]+(?:\.[0-9]{2,6})?)',
                r'最新淨值[\s\S]{0,120}?([0-9]+(?:\.[0-9]{2,6})?)\s*(?:台幣|美元|人民幣|歐元)',
                r'日期\s*[,，]\s*淨值[\s\S]{0,80}?(?:\d{2}/\d{2}|20\d{2}/\d{2}/\d{2})\s*[,，]\s*([0-9]+(?:\.[0-9]{2,6})?)',
            ]
            for pat in patterns:
                m = re.search(pat, plain, flags=re.I)
                if m:
                    price = float(m.group(1))
                    if 0 < price < 100000:
                        return price

            # 若摘要格式沒抓到，再交給原本解析器
            return parse_moneydj_nav(raw)
        except Exception as e:
            last_err = e
            continue
    raise ValueError(f"MoneyDJ 抓不到：{last_err}")


def fetch_fundrich_nav(nav_code: str):
    code = str(nav_code or "").strip()
    if code == "":
        raise ValueError("FundRich 代碼空白")
    if code.startswith("http://") or code.startswith("https://"):
        urls = [code]
    else:
        urls = [
            f"https://www.fundrich.com.tw/fund/detail/{urllib.parse.quote(code)}",
            f"https://www.fundrich.com.tw/fund/{urllib.parse.quote(code)}",
        ]
    last_err = None
    for url in urls:
        try:
            return fetch_nav_from_url(url, verify_ssl=False)
        except Exception as e:
            last_err = e
            continue
    raise ValueError(f"FundRich 抓不到：{last_err}")


def fetch_fund_price_for_asset(asset_row: dict):
    sym = str(asset_row.get("symbol", "")).strip()
    quote_source = str(asset_row.get("quote_source", "")).strip().lower()
    quote_code = str(asset_row.get("quote_code", "")).strip()
    nav_code = str(asset_row.get("nav_code", "")).strip()
    currency = str(asset_row.get("currency", "")).strip().upper()

    last_err = None

    if quote_source == "manual":
        return None, None

    # 1) Yahoo：有正式基金代碼時優先
    if quote_source in ["", "yahoo", "auto", "fund_auto"] and quote_code:
        try:
            p, ccy = fetch_yahoo_chart_last_price(quote_code)
            return p, (ccy or currency or "USD")
        except Exception as e:
            last_err = e

    # 2) MoneyDJ：可用完整網址，或只填基金代碼
    if quote_source in ["moneydj", "auto", "fund_auto"] and nav_code:
        try:
            p = fetch_moneydj_nav(nav_code)
            return p, (currency or "USD")
        except Exception as e:
            last_err = e

    # 3) FundRich：可用完整網址，或只填基金代碼
    if quote_source in ["fundrich", "auto", "fund_auto"] and nav_code:
        try:
            p = fetch_fundrich_nav(nav_code)
            return p, (currency or "USD")
        except Exception as e:
            last_err = e

    # 4) 若 nav_code 是網址，最後再用一般網頁解析試一次
    if nav_code.startswith("http://") or nav_code.startswith("https://"):
        try:
            verify_ssl = ("moneydj.com" not in nav_code.lower() and "fundrich.com.tw" not in nav_code.lower())
            p = fetch_nav_from_url(nav_code, verify_ssl=verify_ssl)
            return p, (currency or "USD")
        except Exception as e:
            last_err = e

    # 5) 若 nav_code 看起來像 Yahoo 基金代碼，再試一次 Yahoo
    if nav_code and not nav_code.startswith("http://") and not nav_code.startswith("https://") and quote_source not in ["moneydj", "fundrich"]:
        try:
            p, ccy = fetch_yahoo_chart_last_price(nav_code)
            return p, (ccy or currency or "USD")
        except Exception as e:
            last_err = e

    raise ValueError(f"抓不到：{last_err}")


def fetch_usd_twd_open_er_api():
    url = "https://open.er-api.com/v6/latest/USD"
    j = http_get_json(url, timeout=10)
    rates = j.get("rates", {}) if isinstance(j, dict) else {}
    rate = rates.get("TWD", None)
    if rate is None:
        raise ValueError("open.er-api 回傳缺少 TWD")
    return float(rate)

def fetch_usd_twd_fawaz():
    url = "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd/twd.json"
    j = http_get_json(url, timeout=10)
    rate = j.get("twd", None) if isinstance(j, dict) else None
    if rate is None:
        raise ValueError("fawaz 回傳缺少 twd")
    return float(rate)

def fetch_usd_twd():
    last_err = None
    for fn in (fetch_usd_twd_open_er_api, fetch_usd_twd_fawaz):
        try:
            r = fn()
            if r > 0:
                return r
        except Exception as e:
            last_err = e
            continue
    raise ValueError(f"USD_TWD 抓取失敗：{last_err}")

def fetch_us_stock_price_stooq(symbol: str):
    s = str(symbol).strip().upper()
    if "." not in s:
        s = f"{s}.US"
    url = f"https://stooq.com/q/l/?s={urllib.parse.quote(s.lower())}&f=sd2t2ohlcv&h&e=csv"
    txt = http_get_text(url, timeout=10).strip()
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError("Stooq 回傳格式不正確")
    parts = [p.strip() for p in lines[1].split(",")]
    if len(parts) < 8:
        raise ValueError("Stooq 回傳欄位不足")
    close = to_number(parts[6])
    if close <= 0:
        raise ValueError("Stooq 回傳價格不合法")
    return float(close)

def fetch_yahoo_chart_last_price(ticker: str):
    t = str(ticker).strip()
    if t == "":
        raise ValueError("ticker 空白")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(t)}?interval=1d&range=5d"
    j = http_get_json(url, timeout=12)

    chart = j.get("chart", {})
    err = chart.get("error")
    if err:
        raise ValueError(err.get("description") or "Yahoo 回傳 error")

    res = chart.get("result") or []
    if not res:
        raise ValueError("Yahoo 查不到資料")

    r0 = res[0]
    meta = r0.get("meta", {}) or {}
    currency = str(meta.get("currency", "")).strip().upper()

    ind = (r0.get("indicators", {}) or {})
    quote = (ind.get("quote") or [])
    if not quote:
        raise ValueError("Yahoo 缺少 quote")
    closes = (quote[0].get("close") or [])
    last = None
    for v in reversed(closes):
        if v is None:
            continue
        last = v
        break
    if last is None:
        raise ValueError("Yahoo close 都是空")

    price = float(last)
    if price <= 0:
        raise ValueError("Yahoo 價格不合法")
    return price, currency

def normalize_yahoo_ticker(symbol: str) -> str:
    s = str(symbol).strip().upper()
    if re.fullmatch(r"\d{4}", s):
        return s + ".TW"
    if s.endswith(".TW") or s.endswith(".TWO") or s.endswith(".US"):
        return s
    return s

def fetch_price_for_asset(asset_row: dict):
    sym = str(asset_row.get("symbol", "")).strip()
    asset_type = str(asset_row.get("asset_type", "")).strip().lower()
    quote_source = str(asset_row.get("quote_source", "")).strip().lower()
    quote_code = str(asset_row.get("quote_code", "")).strip()
    currency = str(asset_row.get("currency", "")).strip().upper()

    if symbol_key(sym) == "USD_TWD" or asset_type == "system":
        return fetch_usd_twd(), "TWD"

    if asset_type == "fund":
        return fetch_fund_price_for_asset(asset_row)

    code = quote_code if quote_code else sym

    if quote_source == "manual":
        return None, None

    if quote_source == "stooq":
        p = fetch_us_stock_price_stooq(code)
        return p, (currency or "USD")

    if quote_source == "yahoo" or quote_source == "":
        try:
            t = normalize_yahoo_ticker(code)
            p, ccy = fetch_yahoo_chart_last_price(t)
            if ccy == "":
                if t.endswith(".TW") or t.endswith(".TWO"):
                    ccy = "TWD"
                elif t.endswith(".US"):
                    ccy = "USD"
            return p, (ccy or currency or "TWD")
        except Exception:
            if asset_type == "stock":
                p = fetch_us_stock_price_stooq(code)
                return p, (currency or "USD")
            raise

    # 其他來源未定義時，股票預設仍先走 Yahoo
    t = normalize_yahoo_ticker(code)
    p, ccy = fetch_yahoo_chart_last_price(t)
    return p, (ccy or currency or "TWD")


def write_price_to_assets(symbol: str, price: float, currency: str):
    ws = ensure_worksheet("assets", ["symbol", "name", "asset_type", "quote_source", "quote_code", "nav_code", "currency", "price", "updated_at", "strategy", "enabled"])
    rows = ws.get_all_values()
    header = rows[0]
    idx = {h: i for i, h in enumerate(header)}
    required = ["symbol", "currency", "price", "updated_at"]
    if any(c not in idx for c in required):
        raise ValueError("assets 表欄位不足，需包含 symbol/currency/price/updated_at")

    key = symbol_key(symbol)
    target_row = None
    for r_i in range(1, len(rows)):
        r = rows[r_i]
        sym = str(r[idx["symbol"]]).strip() if len(r) > idx["symbol"] else ""
        if symbol_key(sym) == key:
            target_row = r_i + 1
            break

    if target_row is None:
        raise ValueError(f"assets 找不到商品：{symbol}")

    row = rows[target_row - 1]
    while len(row) < len(header):
        row.append("")
    row[idx["price"]] = str(float(price))
    row[idx["currency"]] = str(currency).strip().upper()
    row[idx["updated_at"]] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    end_col = chr(ord("A") + len(header) - 1)
    ws.update(f"A{target_row}:{end_col}{target_row}", [row])

def get_last_prices_updated_dt(assets_df: pd.DataFrame):
    if assets_df is None or assets_df.empty or "updated_at" not in assets_df.columns:
        return None
    dts = assets_df["updated_at"].apply(parse_dt_any).dropna()
    if dts.empty:
        return None
    return max(dts.tolist())

def can_refresh_prices(assets_df: pd.DataFrame, cooldown_seconds: int = 60, grace_seconds: int = 3):
    last_dt = get_last_prices_updated_dt(assets_df)
    if last_dt is None:
        return True, 0
    elapsed = (datetime.now() - last_dt).total_seconds()
    threshold = max(0, cooldown_seconds - grace_seconds)
    if elapsed < threshold:
        return False, int(threshold - elapsed)
    return True, 0

def refresh_prices(assets_df: pd.DataFrame):
    if assets_df is None or assets_df.empty:
        return [], ["assets 表為空"]

    a = assets_df.copy()
    a = a[a["enabled"].fillna("").astype(str).str.upper().isin(["", "Y"])]
    a = a[a["asset_type"].isin(["stock", "fund", "system"])]
    ok_list = []
    fail_list = []

    for _, row in a.iterrows():
        sym = str(row.get("symbol", "")).strip()
        if sym == "":
            continue
        try:
            price, ccy = fetch_price_for_asset(row.to_dict())
            if price is None:
                fail_list.append(f"{sym}（抓不到）")
                continue
            write_price_to_assets(sym, price, ccy or row.get("currency", ""))
            ok_list.append(sym)
        except Exception as e:
            fail_list.append(f"{sym}（{str(e)}）")
    return ok_list, fail_list

def apply_filters(df: pd.DataFrame, start_d: date | None, end_d: date | None, symbols: list[str], strategy: str):
    if df is None or df.empty:
        return df

    out = df.copy()
    if "date" in out.columns:
        out["_d"] = parse_date_series(out["date"])
        if start_d:
            out = out[out["_d"] >= start_d]
        if end_d:
            out = out[out["_d"] <= end_d]

    if symbols:
        out["symbol"] = out["symbol"].astype(str).str.strip()
        out = out[out["symbol"].isin(symbols)]

    if strategy and strategy != "全部":
        if "strategy_effective" not in out.columns:
            out = enrich_transactions_with_assets(out, assets_df)
        out["strategy_effective"] = out["strategy_effective"].astype(str).str.strip()
        out = out[out["strategy_effective"] == strategy]

    return out.drop(columns=[c for c in ["_d"] if c in out.columns], errors="ignore")

def strategy_summary(df_tx: pd.DataFrame, assets_df: pd.DataFrame) -> pd.DataFrame:
    if df_tx is None or df_tx.empty:
        return pd.DataFrame(columns=["策略", "總投入(TWD)", "目前市值(TWD)", "報酬率"])

    df = enrich_transactions_with_assets(df_tx, assets_df).copy()
    rows = []
    for strat, g in df.groupby("strategy_effective"):
        m = calculate_metrics(g, assets_df)
        if m is None:
            continue
        invest, value, divi, profit, rate = m
        rows.append({
            "策略": strat,
            "總投入(TWD)": invest,
            "目前市值(TWD)": value,
            "報酬率": rate
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["策略", "總投入(TWD)", "目前市值(TWD)", "報酬率"])

    out = out.sort_values("目前市值(TWD)", ascending=False)
    out["總投入(TWD)"] = out["總投入(TWD)"].map(lambda x: f"{x:,.0f}")
    out["目前市值(TWD)"] = out["目前市值(TWD)"].map(lambda x: f"{x:,.0f}")
    out["報酬率"] = out["報酬率"].map(lambda x: f"{x:.2f}%")
    return out

def append_transaction_dynamic(row_dict: dict):
    ws = sheet.worksheet("transactions")
    values = ws.get_all_values()
    if not values:
        headers = ["id", "date", "action", "symbol", "currency", "fx_rate", "qty", "price", "amount_original", "amount_twd"]
        ws.append_row(headers)
        values = ws.get_all_values()
    header = values[0]
    out_row = []
    for col in header:
        out_row.append(row_dict.get(col, ""))
    ws.append_row(out_row)

# =========================
# 頁面（用 Tabs 當導覽：不使用 Sidebar）
# =========================

page_dash, page_search, page_tx = st.tabs(["Dashboard", "搜尋頁", "交易明細"])

# =========================
# Dashboard
# =========================
with page_dash:
    st.markdown(
        """
        <a href="?add=1" class="fab">+</a>
        <style>
          .fab{
            position: fixed;
            right: 18px;
            bottom: 18px;
            width: 56px;
            height: 56px;
            border-radius: 50%;
            background: #111B2E;
            color: #fff !important;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 34px;
            text-decoration: none;
            box-shadow: 0 10px 24px rgba(0,0,0,0.22);
            z-index: 9999;
          }
          .fab:active { transform: scale(0.98); }
        </style>
        """,
        unsafe_allow_html=True
    )

    def qp_get(key: str, default: str = "") -> str:
        v = st.query_params.get(key, default)
        if isinstance(v, list):
            return v[0] if v else default
        return v

    open_add = qp_get("add", "0") == "1"

    if open_add:
        @st.dialog("新增交易")
        def add_trade_dialog():
            if st.button("關閉", use_container_width=True):
                st.query_params.clear()
                st.rerun()

            st.session_state.setdefault("processing", False)
            st.session_state.setdefault("btn_label", "送出")

            st.session_state.setdefault("in_action", "buy")
            st.session_state.setdefault("in_asset_type", "stock")
            st.session_state.setdefault("in_symbol", "")
            st.session_state.setdefault("in_strategy", "")
            st.session_state.setdefault("in_currency", "TWD")
            st.session_state.setdefault("in_qty", 0.0)
            st.session_state.setdefault("in_price", 0.0)
            st.session_state.setdefault("in_date", date.today())
            st.session_state.setdefault("in_fx_rate", 1.0)
            st.session_state.setdefault("in_amount_twd", 0.0)
            st.session_state.setdefault("strategy_user_edited", False)
            st.session_state.setdefault("asset_exists", False)
            st.session_state.setdefault("asset_name_hint", "")

            def mark_dirty():
                st.session_state.btn_label = "送出"

            def on_symbol_change():
                st.session_state.btn_label = "送出"
                raw_symbol = st.session_state.get("in_symbol", "")
                sel_asset_type = st.session_state.get("in_asset_type", "stock")
                sym = clean_symbol(raw_symbol, sel_asset_type)
                row = get_asset_row(sym)
                st.session_state.asset_exists = row is not None
                if row:
                    st.session_state.in_asset_type = str(row.get("asset_type", st.session_state.get("in_asset_type", "stock")) or "stock")
                    st.session_state.in_currency = str(row.get("currency", st.session_state.get("in_currency", "TWD")) or "TWD")
                    st.session_state.asset_name_hint = str(row.get("name", "")).strip()
                    if not st.session_state.strategy_user_edited or str(st.session_state.get("in_strategy", "")).strip() == "":
                        st.session_state.in_strategy = str(row.get("strategy", "")).strip()
                else:
                    st.session_state.asset_name_hint = ""
                    if not st.session_state.strategy_user_edited:
                        st.session_state.in_strategy = ""

            def on_strategy_change():
                st.session_state.btn_label = "送出"
                st.session_state.strategy_user_edited = True

            action = st.selectbox("交易類型", ["buy", "sell", "dividend", "initial"], key="in_action", on_change=mark_dirty)
            asset_type = st.selectbox("資產類型", ["stock", "fund"], key="in_asset_type", on_change=on_symbol_change)
            symbol_input = st.text_input("代號", key="in_symbol", on_change=on_symbol_change)
            strategy = st.text_input("策略（商品主檔）", key="in_strategy", on_change=on_strategy_change)
            tx_date = st.date_input("日期", key="in_date", on_change=mark_dirty)

            symbol_norm = clean_symbol(symbol_input, asset_type)
            existing_asset = get_asset_row(symbol_norm)
            if existing_asset:
                st.caption(f"已找到商品主檔：{existing_asset.get('name', symbol_norm)}")
            else:
                st.caption("若商品主檔尚未存在，送出後會自動補一筆到 assets。")

            currency = st.selectbox("幣別", ["TWD", "USD"], key="in_currency", on_change=mark_dirty)
            qty = st.number_input("數量", min_value=0.0, key="in_qty", on_change=mark_dirty)
            price = st.number_input("單價", min_value=0.0, key="in_price", on_change=mark_dirty)

            if currency == "TWD":
                st.session_state.in_fx_rate = 1.0
                fx_rate = st.number_input("匯率（USD_TWD）", value=1.0, disabled=True)
            else:
                fx_rate = st.number_input("匯率（USD_TWD）", min_value=0.0, key="in_fx_rate", on_change=mark_dirty)

            sync_default = st.checkbox("同步更新商品主檔的策略", value=True)
            amount_guess = to_number(qty) * to_number(price) * (to_number(fx_rate) if currency == "USD" else 1.0)
            amount_twd_input = st.number_input("最終台幣金額（amount_twd）", min_value=0.0, value=float(amount_guess if st.session_state.get("in_amount_twd", 0.0) == 0 else st.session_state.get("in_amount_twd", 0.0)), step=1.0, key="in_amount_twd", on_change=mark_dirty)
            st.caption(f"參考試算：{to_number(qty):,.4f} × {to_number(price):,.4f} × {(to_number(fx_rate) if currency == 'USD' else 1.0):,.4f} = {amount_guess:,.2f} TWD")
            st.caption("amount_twd 以你手動輸入的最終台幣金額為準；系統只做提醒，不做強制驗證。")
            st.divider()

            btn_text = "🔄 寫入中..." if st.session_state.processing else st.session_state.btn_label
            clicked = st.button(btn_text, disabled=st.session_state.processing, use_container_width=True)
            status_area = st.empty()

            if clicked:
                st.session_state.processing = True
                st.session_state.btn_label = "🔄 寫入中..."
                st.rerun()

            if st.session_state.processing:
                try:
                    symbol = clean_symbol(symbol_input, asset_type)
                    if symbol == "":
                        st.error("代號不可空白")
                        raise ValueError("代號不可空白")

                    if currency == "USD" and to_number(fx_rate) == 0:
                        st.error("USD 必須填匯率")
                        raise ValueError("USD 必須填匯率")

                    fx_used = to_number(fx_rate) if currency == "USD" else 1.0
                    amount_original = to_number(qty) * to_number(price)
                    amount_twd = to_number(amount_twd_input)

                    same = enriched_tx_df[enriched_tx_df["symbol_key"] == symbol_key(symbol)] if not enriched_tx_df.empty else pd.DataFrame()

                    if action == "initial":
                        if to_number(qty) <= 0:
                            st.error("initial 數量必須大於 0")
                            raise ValueError("initial 數量必須大於 0")

                        if not same.empty:
                            same_dates = pd.to_datetime(
                                same["date"].astype(str).str.strip().str.replace("/", "-").str.replace(".", "-"),
                                errors="coerce"
                            ).dropna().dt.date
                            if len(same_dates) > 0 and any(d <= tx_date for d in same_dates.tolist()):
                                st.error("initial 日期必須早於此資產的其他交易日期")
                                raise ValueError("initial 日期不合法")
                    else:
                        if action == "sell":
                            current_qty = get_current_qty(transactions_df, symbol, asset_type)
                            if to_number(qty) > current_qty:
                                st.error(f"賣出數量({qty}) 超過目前持有({current_qty})，不允許送出")
                                raise ValueError("賣出超過持有")

                    row = get_asset_row(symbol)
                    final_asset_type = str(row.get("asset_type", asset_type)).strip().lower() if row else asset_type
                    final_currency = str(row.get("currency", currency)).strip().upper() if row else currency
                    final_strategy = str(row.get("strategy", strategy)).strip() if row and str(strategy).strip() == "" else str(strategy).strip()

                    tx_row = {
                        "id": str(datetime.now().timestamp()),
                        "date": str(tx_date),
                        "action": action,
                        "symbol": symbol,
                        "currency": final_currency,
                        "fx_rate": fx_used,
                        "qty": to_number(qty),
                        "price": to_number(price),
                        "amount_original": amount_original,
                        "amount_twd": amount_twd,
                        "asset_type": final_asset_type,
                        "strategy": final_strategy,
                    }

                    append_transaction_dynamic(tx_row)

                    if sync_default:
                        default_quote_source = "yahoo" if final_asset_type == "stock" else "manual"
                        default_quote_code = symbol if final_asset_type == "stock" else ""
                        update_asset_master(
                            symbol=symbol,
                            asset_type=final_asset_type,
                            currency=final_currency,
                            strategy=final_strategy,
                            name=(row.get("name") if row else symbol),
                            quote_source=(row.get("quote_source") if row else default_quote_source),
                            quote_code=(row.get("quote_code") if row else default_quote_code),
                            enabled=(row.get("enabled") if row else "Y"),
                        )

                    saved_at = datetime.now().strftime("%H:%M")
                    st.session_state.btn_label = f"✅ 已存檔 ({saved_at})"
                    status_area.caption(st.session_state.btn_label)

                    st.session_state.in_symbol = ""
                    st.session_state.in_qty = 0.0
                    st.session_state.in_price = 0.0
                    st.session_state.in_amount_twd = 0.0
                    st.session_state.strategy_user_edited = False

                    st.query_params.clear()
                    with st.spinner("正在重新同步資料…"):
                        st.cache_resource.clear()
                    st.rerun()

                except Exception:
                    st.session_state.btn_label = "送出"
                    status_area.caption("已取消送出（請修正欄位後再送出）")
                finally:
                    st.session_state.processing = False

        add_trade_dialog()

    st.session_state.setdefault("dash_symbols", [])
    st.session_state.setdefault("dash_strategy", "全部")

    all_symbols = sorted(enriched_tx_df.get("symbol", pd.Series(dtype=str)).astype(str).str.strip().replace("", pd.NA).dropna().unique().tolist())
    dynamic_strategies = sorted([s for s in assets_df.get("strategy", pd.Series(dtype=str)).astype(str).str.strip().unique().tolist() if s])
    all_strategies = ["全部"] + dynamic_strategies
    if "未分類" not in all_strategies:
        all_strategies.append("未分類")

    f1, f2 = st.columns([7, 3])
    with f1:
        st.multiselect("Symbol（可多選）", options=all_symbols, key="dash_symbols")
        st.selectbox("Strategy", options=all_strategies, key="dash_strategy")
        if st.button("清除篩選", use_container_width=True):
            st.session_state.dash_symbols = []
            st.session_state.dash_strategy = "全部"
            st.rerun()
    with f2:
        ok_refresh, remain = can_refresh_prices(assets_df, cooldown_seconds=60, grace_seconds=3)
        btn_label = "刷新價格" if ok_refresh else f"刷新價格（{remain}s）"
        clicked = st.button(btn_label, disabled=(not ok_refresh), use_container_width=True)
        if clicked:
            with st.spinner("正在刷新價格…"):
                ok_list, fail_list = refresh_prices(assets_df)
            st.session_state["refresh_ok_list"] = ok_list
            st.session_state["refresh_fail_list"] = fail_list
            with st.spinner("正在重新同步資料…"):
                st.cache_resource.clear()
            st.rerun()

    if st.session_state.get("refresh_ok_list") is not None or st.session_state.get("refresh_fail_list") is not None:
        ok_list = st.session_state.get("refresh_ok_list") or []
        fail_list = st.session_state.get("refresh_fail_list") or []
        if ok_list:
            st.success("已更新：" + "、".join(ok_list[:8]) + ("…" if len(ok_list) > 8 else ""))
        if fail_list:
            st.warning("未更新：" + "、".join(fail_list[:6]) + ("…" if len(fail_list) > 6 else ""))
        st.session_state["refresh_ok_list"] = None
        st.session_state["refresh_fail_list"] = None

    fx, fx_updated = get_usd_twd_info(assets_df)
    prices_updated = get_prices_updated_at(assets_df)
    selected_symbols = st.session_state.get("dash_symbols", [])
    selected_strategy = st.session_state.get("dash_strategy", "全部")

    all_results = build_symbol_results(transactions_df, symbols=selected_symbols, strategy=selected_strategy)
    stock_results = build_symbol_results(transactions_df, asset_type_filter="stock", symbols=selected_symbols, strategy=selected_strategy)
    fund_results = build_symbol_results(transactions_df, asset_type_filter="fund", symbols=selected_symbols, strategy=selected_strategy)

    def render_dashboard_section(title: str, result_list: list[dict]):
        st.markdown(f"## {title}")
        summary = summarize_valid_only(result_list)
        sub_parts = []
        if prices_updated:
            sub_parts.append(f"價格更新：{prices_updated}")
        if fx is not None:
            fx_part = f"USD_TWD：{fx:.4f}"
            if fx_updated:
                fx_part += f"（{fx_updated}）"
            sub_parts.append(fx_part)
        hero_card(f"{title}｜目前市值（TWD）", fmt_money(summary["market_value"]), " ｜ ".join(sub_parts))
        if summary["invalid_count"]:
            st.warning(f"有 {summary['invalid_count']} 支標的資料異常，未納入總覽統計")
        kpis_html = "".join([
            kpi_card_html("持有成本", fmt_money(summary["remaining_cost"])),
            kpi_card_html("累計股息", fmt_money(summary["dividend_income"])),
            kpi_card_html("總報酬", fmt_signed_money_html(summary["total_profit"])),
            kpi_card_html("總報酬率", fmt_signed_pct_html(summary["total_return_pct"])),
        ])
        st.markdown(f'<div class="kpi-grid">{kpis_html}</div>', unsafe_allow_html=True)

        strat_df = strategy_summary(pd.DataFrame([{ "symbol": r.get("symbol") } for r in []]), assets_df)
        rows = []
        grouped = {}
        for r in result_list:
            grouped.setdefault(r.get("strategy", "未分類") or "未分類", []).append(r)
        for strat, items in grouped.items():
            ss = summarize_valid_only(items)
            rows.append({
                "策略": strat,
                "目前市值(TWD)": f"{ss['market_value']:,.0f}",
                "總報酬率": f"{ss['total_return_pct']:.2f}%",
                "異常筆數": ss["invalid_count"],
            })
        strat_df = pd.DataFrame(rows)
        st.markdown("#### 策略摘要")
        st.dataframe(strat_df if not strat_df.empty else pd.DataFrame(columns=["策略", "目前市值(TWD)", "總報酬率", "異常筆數"]), use_container_width=True, hide_index=True)
        st.markdown("#### 市值前 3 名")
        top_rows = []
        valid_items = [x for x in result_list if x.get("has_data") and x.get("is_valid") and to_number(x.get("remaining_qty", 0)) > 0]
        total_mv = sum(to_number(x.get("market_value", 0)) for x in valid_items)
        for r in sorted(valid_items, key=lambda x: to_number(x.get("market_value", 0)), reverse=True)[:3]:
            top_rows.append({
                "代碼": r.get("symbol", ""),
                "策略": r.get("strategy", "未分類"),
                "市值(TWD)": f"{to_number(r.get('market_value', 0)):,.0f}",
                "佔比": f"{(to_number(r.get('market_value', 0))/total_mv*100):.1f}%" if total_mv > 0 else "0.0%",
            })
        st.dataframe(pd.DataFrame(top_rows) if top_rows else pd.DataFrame(columns=["代碼", "策略", "市值(TWD)", "佔比"]), use_container_width=True, hide_index=True)

    render_dashboard_section("全部", all_results)
    st.divider()
    render_dashboard_section("股票", stock_results)
    st.divider()
    render_dashboard_section("基金", fund_results)

# =========================
# 搜尋頁
# =========================
with page_search:
    st.markdown("### 搜尋頁")
    st.caption("預設先查單一標的；日期區間會先回推起始日前歷史交易，再算區間結果。")

    search_symbols = sorted(enriched_tx_df.get("symbol", pd.Series(dtype=str)).astype(str).str.strip().replace("", pd.NA).dropna().unique().tolist())
    mode = st.radio("查詢模式", ["單一標的", "日期區間", "多標的"], horizontal=True)

    if mode == "單一標的":
        default_idx = 0 if search_symbols else None
        symbol = st.selectbox("選擇標的", options=search_symbols, index=default_idx if default_idx is not None else 0)
        if st.button("查詢單一標的", use_container_width=True, disabled=(len(search_symbols) == 0)):
            result = calc_symbol_full_history(transactions_df, symbol, get_asset_row(symbol), get_asset_row("USD_TWD"))
            detail_symbol_card(result, key_prefix=f"single_{symbol}")

    elif mode == "日期區間":
        d1, d2 = st.columns(2)
        with d1:
            start_d = st.date_input("開始日", value=date.today() - timedelta(days=90), key="range_start")
        with d2:
            end_d = st.date_input("結束日", value=date.today(), key="range_end")
        selected = st.multiselect("限定標的（可不選，代表全部）", options=search_symbols, key="range_symbols")
        st.caption("系統會先建立期初剩餘持股與期初成本，再計算區間內買進、賣出、配息。")
        if st.button("查詢日期區間", use_container_width=True):
            targets = selected if selected else search_symbols
            range_results = [calc_symbol_in_range(transactions_df, sym, start_d, end_d, get_asset_row(sym), get_asset_row("USD_TWD")) for sym in targets]
            cards = summarize_range_results(range_results)
            if not cards:
                st.info("查無符合條件的交易紀錄")
            else:
                for card in cards:
                    render_range_strategy_card(card)

    else:
        picks = st.multiselect("最多勾選 3 支", options=search_symbols, max_selections=3, key="multi_symbols")
        if st.button("查詢多標的", use_container_width=True):
            if not picks:
                st.warning("請先至少選 1 支標的")
            else:
                st.session_state.setdefault("multi_expand", {})
                for sym in picks:
                    result = calc_symbol_full_history(transactions_df, sym, get_asset_row(sym), get_asset_row("USD_TWD"))
                    with st.container(border=True):
                        compact_symbol_card(result)
                        exp_key = f"multi_expand_{sym}"
                        current = st.session_state.get(exp_key, False)
                        btn_label = "收起詳情" if current else "查看詳情"
                        if st.button(btn_label, key=f"btn_{sym}", use_container_width=True):
                            st.session_state[exp_key] = not current
                            st.rerun()
                        if st.session_state.get(exp_key, False):
                            detail_symbol_card(result, key_prefix=f"multi_{sym}")

# =========================
# 交易明細
# =========================
with page_tx:
    st.markdown("### 交易明細")

    st.session_state.setdefault("tx_end", date.today())
    st.session_state.setdefault("tx_days", 30)
    st.session_state.setdefault("tx_symbols", [])
    st.session_state.setdefault("tx_strategy", "全部")
    st.session_state.setdefault("tx_offset", 0)
    st.session_state.setdefault("tx_last_sig", "")

    a, b = st.columns(2)
    with a:
        tx_end = st.date_input("結束日期", key="tx_end")
    with b:
        tx_days = st.selectbox("預設區間", options=[30, 60, 90], index=0, key="tx_days")

    tx_symbols_list = sorted(
        enriched_tx_df.get("symbol", pd.Series(dtype=str)).astype(str).str.strip().replace("", pd.NA).dropna().unique().tolist()
    )
    tx_strategies = ["全部"] + sorted([s for s in enriched_tx_df.get("strategy_effective", pd.Series(dtype=str)).astype(str).str.strip().replace("", pd.NA).dropna().unique().tolist()])
    if "未分類" not in tx_strategies:
        tx_strategies.append("未分類")

    st.multiselect("Symbol（可多選）", options=tx_symbols_list, key="tx_symbols")
    st.selectbox("Strategy", options=tx_strategies, key="tx_strategy")

    start_d = st.session_state.tx_end - timedelta(days=st.session_state.tx_days * (st.session_state.tx_offset + 1))
    end_d = st.session_state.tx_end

    sig = f"{tx_end}|{tx_days}|{','.join(st.session_state.tx_symbols)}|{st.session_state.tx_strategy}"
    if st.session_state.tx_last_sig == "":
        st.session_state.tx_last_sig = sig
    elif sig != st.session_state.tx_last_sig:
        st.session_state.tx_last_sig = sig
        st.session_state.tx_offset = 0
        st.rerun()

    df_tx = enriched_tx_df.copy()
    if not df_tx.empty:
        df_tx["date_norm"] = parse_date_series(df_tx["date"])
        df_tx = df_tx.sort_values("date_norm", ascending=False)

    filtered = apply_filters(df_tx, start_d, end_d, st.session_state.tx_symbols, st.session_state.tx_strategy)
    cols = ["date", "symbol", "action", "qty", "price", "amount_twd"]
    existing_cols = [c for c in cols if c in filtered.columns]
    show_df = filtered[existing_cols].copy() if not filtered.empty else pd.DataFrame(columns=existing_cols)

    st.caption(f"顯示區間：{start_d} ~ {end_d}（依結束日期往前延伸）")
    st.dataframe(show_df, use_container_width=True, hide_index=True)

    if st.button("載入更多", use_container_width=True):
        st.session_state.tx_offset += 1
        st.rerun()
