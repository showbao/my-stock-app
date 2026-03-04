import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import re
from datetime import datetime, date, timedelta
import requests

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

/* KPI 佈局：手機一欄，桌機兩欄 */

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

def load_data():
    transactions = pd.DataFrame(sheet.worksheet("transactions").get_all_records())
    prices = pd.DataFrame(sheet.worksheet("prices").get_all_records())
    allowed = pd.DataFrame(sheet.worksheet("settings_allowed_emails").get_all_records())
    return transactions, prices, allowed

def load_symbol_strategy():
    # 需要你先在 Google Sheet 建立 worksheet：symbol_strategy
    ws = sheet.worksheet("symbol_strategy")
    df = pd.DataFrame(ws.get_all_records())
    return df

transactions_df, prices_df, allowed_df = load_data()

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
def qp_get(key: str, default: str = "") -> str:
    v = st.query_params.get(key, default)
    if isinstance(v, list):
        return v[0] if v else default
    return v

def qp_set(**kwargs):
    for k, v in kwargs.items():
        st.query_params[k] = str(v)

def qp_clear(keys=None):
    if keys is None:
        st.query_params.clear()
    else:
        for k in keys:
            if k in st.query_params:
                del st.query_params[k]

current_page = qp_get("page", "dashboard")  # dashboard / tx
open_add = qp_get("add", "0") == "1"

top_l, top_r = st.columns([7, 3])
with top_l:
    st.markdown("## 投資儀表板")
with top_r:
    st.caption(f"登入：{user_email}")
    st.button("登出", on_click=st.logout, use_container_width=True)

# =========================
# 工具函式（保留你原本邏輯）
# =========================
def clean_symbol(symbol, asset_type):
    symbol = str(symbol).strip().upper()
    # 允許 . ，避免 0050.TW 變 0050TW
    symbol = re.sub(r"[^A-Z0-9_.]", "", symbol)
    if asset_type == "fund" and not symbol.startswith("F_"):
        symbol = "F_" + symbol
    return symbol

def to_number(x):
    if x is None:
        return 0.0
    s = str(x).strip()
    if s == "":
        return 0.0
    s = s.replace(",", "")
    try:
        return float(s)
    except:
        return 0.0

def parse_date_series(series):
    s = series.astype(str).str.strip()
    s = s.str.replace("/", "-", regex=False).str.replace(".", "-", regex=False)
    dt = pd.to_datetime(s, errors="coerce", format=None)
    return dt.dt.date

def get_current_qty(transactions_df, symbol, asset_type):
    df = transactions_df.copy()
    df["symbol_norm"] = df["symbol"].astype(str).apply(lambda x: clean_symbol(x, asset_type))
    target = df[df["symbol_norm"] == symbol]
    if target.empty:
        return 0.0

    qty = 0.0
    for _, row in target.iterrows():
        action = str(row.get("action", "")).strip().lower()
        q = to_number(row.get("qty", 0))
        if action in ["buy", "initial"]:
            qty += q
        elif action == "sell":
            qty -= q
    return qty

def calculate_metrics(df, prices_df):
    if df.empty:
        return 0, 0, 0, 0, 0

    result = {}
    for _, row in df.iterrows():
        symbol = row["symbol"]
        action = row["action"]
        qty = to_number(row.get("qty"))
        amount_twd = to_number(row.get("amount_twd"))

        if symbol not in result:
            result[symbol] = {"qty": 0, "cost": 0, "dividend": 0}

        if action in ["buy", "initial"]:
            result[symbol]["qty"] += qty
            result[symbol]["cost"] += amount_twd
        elif action == "sell":
            result[symbol]["qty"] -= qty
            result[symbol]["cost"] -= amount_twd
        elif action == "dividend":
            result[symbol]["dividend"] += amount_twd

    total_invest = 0.0
    total_value = 0.0
    total_dividend = 0.0

    for symbol, data in result.items():
        if data["qty"] < 0:
            return None

        price_row = prices_df[prices_df["symbol"] == symbol]
        if not price_row.empty:
            price = to_number(price_row.iloc[0].get("price"))
            currency = str(price_row.iloc[0].get("currency", "")).strip().upper()
            value = data["qty"] * price

            if currency == "USD":
                fx_row = prices_df[prices_df["symbol"] == "USD_TWD"]
                fx = to_number(fx_row.iloc[0].get("price")) if not fx_row.empty else 0.0
                value *= fx
        else:
            value = 0.0

        total_invest += data["cost"]
        total_value += value
        total_dividend += data["dividend"]

    total_profit = total_value + total_dividend - total_invest
    rate = (total_profit / total_invest * 100) if total_invest != 0 else 0.0
    return total_invest, total_value, total_dividend, total_profit, rate

def top_holdings_table(df_tx: pd.DataFrame, prices_df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """回傳持有市值前 N 名（只顯示用）"""
    if df_tx is None or df_tx.empty:
        return pd.DataFrame(columns=["代碼", "持有數量", "價格", "幣別", "市值(TWD)"])

    tx = df_tx.copy()
    tx["qty"] = tx.get("qty", 0).apply(to_number)
    tx["action"] = tx.get("action", "").astype(str).str.strip().str.lower()
    tx["symbol"] = tx.get("symbol", "").astype(str)

    # 計算目前持有 qty（buy/initial 加，sell 減）
    pos = {}
    for _, r in tx.iterrows():
        sym = r["symbol"]
        act = r["action"]
        q = float(r["qty"])
        if sym not in pos:
            pos[sym] = 0.0
        if act in ["buy", "initial"]:
            pos[sym] += q
        elif act == "sell":
            pos[sym] -= q

    # 取得 USD_TWD
    fx_row = prices_df[prices_df["symbol"] == "USD_TWD"] if prices_df is not None and not prices_df.empty else pd.DataFrame()
    usd_twd = to_number(fx_row.iloc[0].get("price")) if not fx_row.empty else 0.0

    rows = []
    for sym, q in pos.items():
        if q <= 0:
            continue

        p_row = prices_df[prices_df["symbol"] == sym] if prices_df is not None and not prices_df.empty else pd.DataFrame()
        if p_row.empty:
            price = 0.0
            ccy = ""
        else:
            price = to_number(p_row.iloc[0].get("price"))
            ccy = str(p_row.iloc[0].get("currency", "")).strip().upper()

        mv = q * price
        if ccy == "USD":
            mv *= usd_twd

        rows.append({
            "代碼": sym,
            "持有數量": q,
            "價格": price,
            "幣別": ccy,
            "市值(TWD)": mv
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["代碼", "持有數量", "價格", "幣別", "市值(TWD)", "佔比"])

    total_mv = float(out["市值(TWD)"].sum()) if "市值(TWD)" in out.columns else 0.0
    out = out.sort_values("市值(TWD)", ascending=False).head(top_n)

    if total_mv > 0:
        out["佔比"] = out["市值(TWD)"].map(lambda v: v / total_mv)
    else:
        out["佔比"] = 0.0

    out["持有數量"] = out["持有數量"].map(lambda x: f"{x:,.4f}".rstrip("0").rstrip("."))
    out["價格"] = out["價格"].map(lambda x: f"{x:,.4f}".rstrip("0").rstrip("."))
    out["市值(TWD)"] = out["市值(TWD)"].map(lambda x: f"{x:,.0f}")
    out["佔比"] = out["佔比"].map(lambda x: f"{x*100:.1f}%")
    out = out[["代碼", "持有數量", "價格", "幣別", "市值(TWD)", "佔比"]]
    return out

# =========================
# 顯示用（儀表板風）
# =========================
def fmt_money(n):
    try:
        return f"{float(n):,.0f}"
    except:
        return "—"

def fmt_signed_money_html(n):
    try:
        n = float(n)
        sign = "+" if n > 0 else ""
        cls = "profit-pos" if n > 0 else "profit-neg" if n < 0 else "profit-zero"
        return f"<span class='{cls}'>{sign}{n:,.0f}</span>"
    except:
        return "—"

def fmt_signed_pct_html(n):
    try:
        n = float(n)
        sign = "+" if n > 0 else ""
        cls = "profit-pos" if n > 0 else "profit-neg" if n < 0 else "profit-zero"
        return f"<span class='{cls}'>{sign}{n:.2f}%</span>"
    except:
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

def get_usd_twd_info(prices_df):
    fx_row = prices_df[prices_df["symbol"] == "USD_TWD"]
    if fx_row.empty:
        return None, None
    fx = to_number(fx_row.iloc[0].get("price"))
    updated_at = fx_row.iloc[0].get("updated_at", "")
    updated_at = str(updated_at).strip() if updated_at is not None else ""
    return fx, (updated_at if updated_at else None)

def get_prices_updated_at(prices_df):
    if "updated_at" not in prices_df.columns or prices_df.empty:
        return None
    s = prices_df["updated_at"].astype(str).str.strip()
    s = s[s != ""]
    if s.empty:
        return None
    return s.max()

# =========================
# 新增：symbol_strategy（讀 / 寫）
# =========================
def normalize_strategy(s: str) -> str:
    s = (s or "").strip()
    if s == "":
        return "未分類"
    return s

def get_symbol_strategy_map():
    try:
        df = load_symbol_strategy()
    except Exception:
        return {}
    if df is None or df.empty:
        return {}
    df = df.copy()
    if "symbol" not in df.columns or "strategy" not in df.columns:
        return {}
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["strategy"] = df["strategy"].astype(str).apply(normalize_strategy)
    m = {}
    for _, r in df.iterrows():
        sym = r.get("symbol", "").strip().upper()
        strat = normalize_strategy(r.get("strategy", "未分類"))
        if sym != "":
            m[sym] = strat
    return m

def upsert_symbol_strategy(symbol: str, strategy: str):
    ws = sheet.worksheet("symbol_strategy")
    symbol = str(symbol).strip().upper()
    strategy = normalize_strategy(strategy)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df = pd.DataFrame(ws.get_all_records())
    if df.empty:
        ws.append_row(["symbol", "strategy", "updated_at"])
        ws.append_row([symbol, strategy, now])
        return

    if "symbol" not in df.columns:
        # 表頭不對，直接讓錯誤浮出（避免默默寫錯）
        raise ValueError("symbol_strategy 表缺少 symbol 欄位")

    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    hits = df.index[df["symbol"] == symbol].tolist()

    if hits:
        row_idx = hits[0] + 2  # +2：因為 gsheet 第 1 列是表頭，而且 DataFrame 是 0-based
        ws.update(f"B{row_idx}", strategy)
        ws.update(f"C{row_idx}", now)
    else:
        ws.append_row([symbol, strategy, now])

# =========================
# 新增：價格刷新（60 秒冷卻）
# =========================
def yahoo_quote(symbols):
    """
    用 Yahoo Finance quote API 抓報價（不需要 yfinance 套件）
    回傳 dict: symbol -> price(float)
    """
    if not symbols:
        return {}

    url = "https://query1.finance.yahoo.com/v7/finance/quote"
    params = {"symbols": ",".join(symbols)}
    r = requests.get(url, params=params, timeout=12)
    r.raise_for_status()
    data = r.json()
    out = {}
    results = data.get("quoteResponse", {}).get("result", [])
    for it in results:
        sym = it.get("symbol")
        price = it.get("regularMarketPrice")
        if sym is not None and price is not None:
            out[sym] = float(price)
    return out

def can_refresh(prices_df):
    """
    用 prices 表中最大的 updated_at 判斷冷卻。
    加 3 秒寬限（避免倒數剛到邊界被拒絕）。
    """
    last = get_prices_updated_at(prices_df)
    if not last:
        return True, 0

    # 嘗試 parse
    try:
        t = pd.to_datetime(str(last), errors="coerce")
        if pd.isna(t):
            return True, 0
        last_ts = t.to_pydatetime().timestamp()
    except Exception:
        return True, 0

    now_ts = datetime.now().timestamp()
    elapsed = now_ts - last_ts

    cooldown = 60.0
    grace = 3.0
    # elapsed >= 60 - 3 視為允許（寬限）
    if elapsed >= (cooldown - grace):
        return True, 0
    remain = int((cooldown - grace) - elapsed)
    return False, remain

def refresh_prices(transactions_df, prices_df):
    """
    刷新：股票 / 基金 / USD_TWD
    - 交易表出現過的 symbol（stock/fund）都刷新
    - USD_TWD 用 Yahoo: TWD=X
    """
    ws = sheet.worksheet("prices")

    # 交易出現過的 symbols
    tx = transactions_df.copy()
    if tx.empty:
        symbols_tx = []
    else:
        tx["symbol"] = tx.get("symbol", "").astype(str).str.strip()
        tx["asset_type"] = tx.get("asset_type", "").astype(str).str.strip().str.lower()
        symbols_tx = sorted(list(set([s for s in tx["symbol"].tolist() if s])))

    # 準備 Yahoo symbols
    yahoo_symbols = []
    sym_to_yahoo = {}
    for s in symbols_tx:
        # fund 可能帶 F_ 前綴：抓價時拿掉 F_（只是一個簡單映射）
        y = s
        if s.startswith("F_"):
            y = s[2:]
        sym_to_yahoo[s] = y
        yahoo_symbols.append(y)

    # USD_TWD
    yahoo_symbols.append("TWD=X")

    quotes = yahoo_quote(sorted(list(set(yahoo_symbols))))

    # 組出要寫回 prices 的 rows
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 先抓現有 prices 成 DataFrame（用來找 asset_type/currency）
    p = prices_df.copy() if prices_df is not None else pd.DataFrame()
    if p.empty:
        p = pd.DataFrame(columns=["symbol", "asset_type", "price", "currency", "updated_at"])

    def find_row(symbol):
        rr = p[p["symbol"] == symbol]
        if rr.empty:
            return None
        return rr.iloc[0]

    # 更新/插入：交易 symbols
    for s in symbols_tx:
        y = sym_to_yahoo.get(s, s)
        price = quotes.get(y)
        if price is None:
            continue

        rr = find_row(s)
        if rr is None:
            # 交易表沒有 currency 欄位可判斷價格幣別，先用簡單規則：
            # - 結尾 .TW 先當 TWD
            # - 其他先當 USD（你可自行在 prices 手動改）
            ccy = "TWD" if s.endswith(".TW") or s.endswith(".TWO") else "USD"
            at = "fund" if s.startswith("F_") else "stock"
            ws.append_row([s, at, price, ccy, now])
        else:
            # 更新 price + updated_at，其他欄位不動
            # 找到 row index：用 gspread 比較穩
            try:
                cell = ws.find(s)
                row_idx = cell.row
                ws.update(f"C{row_idx}", price)       # price
                ws.update(f"E{row_idx}", now)         # updated_at
            except Exception:
                pass

    # 更新 USD_TWD（symbol=USD_TWD, asset_type=system）
    usd_twd = quotes.get("TWD=X")
    if usd_twd is not None:
        rr = find_row("USD_TWD")
        if rr is None:
            ws.append_row(["USD_TWD", "system", usd_twd, "TWD", now])
        else:
            try:
                cell = ws.find("USD_TWD")
                row_idx = cell.row
                ws.update(f"C{row_idx}", usd_twd)
                ws.update(f"E{row_idx}", now)
            except Exception:
                pass

# =========================
# UI：右下角 FAB（保留）
# =========================
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

# =========================
# 新增：頁面切換按鈕（不使用 Sidebar）
# =========================
nav_l, nav_r = st.columns([1, 1])
with nav_l:
    if st.button("📊 儀表板", use_container_width=True):
        qp_set(page="dashboard")
        qp_clear(keys=["add"])
        st.rerun()
with nav_r:
    if st.button("📜 交易明細", use_container_width=True):
        qp_set(page="tx")
        qp_clear(keys=["add"])
        st.rerun()

# =========================
# 新增交易 Dialog（保留既有 + 增量：symbol_strategy 自動帶入）
# =========================
symbol_strategy_map = get_symbol_strategy_map()

def guess_strategy_for_symbol(symbol_norm: str) -> str:
    symbol_norm = str(symbol_norm).strip().upper()
    return symbol_strategy_map.get(symbol_norm, "未分類")

if open_add:
    @st.dialog("新增交易")
    def add_trade_dialog():
        # 關閉（不存檔）
        if st.button("關閉", use_container_width=True):
            qp_clear(keys=["add"])
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

        # 新增：是否同步更新 symbol_strategy
        st.session_state.setdefault("sync_symbol_strategy", False)
        st.session_state.setdefault("sync_confirm", False)

        def mark_dirty():
            st.session_state.btn_label = "送出"

        def on_symbol_change():
            # 只在 strategy 目前是空白時自動帶入（避免覆蓋使用者已輸入的內容）
            sym = clean_symbol(st.session_state.in_symbol, st.session_state.in_asset_type)
            if (st.session_state.in_strategy or "").strip() == "":
                st.session_state.in_strategy = guess_strategy_for_symbol(sym)
            mark_dirty()

        def on_strategy_change():
            # 使用者手動改 strategy 後，才顯示同步選項（規格 v3.3 的精神）
            st.session_state.sync_symbol_strategy = False
            st.session_state.sync_confirm = False
            mark_dirty()

        action = st.selectbox(
            "交易類型", ["buy", "sell", "dividend", "initial"],
            key="in_action", on_change=mark_dirty
        )
        asset_type = st.selectbox(
            "資產類型", ["stock", "fund"],
            key="in_asset_type", on_change=mark_dirty
        )
        symbol_input = st.text_input("代號", key="in_symbol", on_change=on_symbol_change)
        strategy = st.text_input("策略", key="in_strategy", on_change=on_strategy_change)
        tx_date = st.date_input("日期", key="in_date", on_change=mark_dirty)

        currency = st.selectbox(
            "幣別", ["TWD", "USD"],
            key="in_currency", on_change=mark_dirty
        )

        qty = st.number_input("數量", min_value=0.0, key="in_qty", on_change=mark_dirty)
        price = st.number_input("單價", min_value=0.0, key="in_price", on_change=mark_dirty)

        if currency == "TWD":
            st.session_state.in_fx_rate = 1.0
            fx_rate = st.number_input("匯率（USD_TWD）", value=1.0, disabled=True)
        else:
            fx_rate = st.number_input("匯率（USD_TWD）", min_value=0.0, key="in_fx_rate", on_change=mark_dirty)

        # 新增：同步更新預設策略（僅在使用者手動改 strategy 後顯示）
        sym_norm_preview = clean_symbol(symbol_input, asset_type)
        default_strat = guess_strategy_for_symbol(sym_norm_preview) if sym_norm_preview else "未分類"
        user_strat = normalize_strategy(strategy)

        if sym_norm_preview and user_strat != normalize_strategy(default_strat):
            st.session_state.sync_symbol_strategy = st.checkbox(
                "同步更新此 symbol 的預設策略",
                value=st.session_state.sync_symbol_strategy
            )
            if st.session_state.sync_symbol_strategy:
                st.warning(f"警告：這會改變「{sym_norm_preview}」在所有頁面的預設分類。")
                st.session_state.sync_confirm = st.checkbox(
                    "我了解並確認要同步更新",
                    value=st.session_state.sync_confirm
                )

        st.caption("TWD 會自動使用 1.0；USD 請填 USD_TWD 匯率。")
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
                amount_twd = amount_original * fx_used

                # initial 規則（保留）
                if action == "initial":
                    if to_number(qty) <= 0:
                        st.error("initial 數量必須大於 0")
                        raise ValueError("initial 數量必須大於 0")

                    same = transactions_df[
                        transactions_df["symbol"].astype(str).apply(lambda x: clean_symbol(x, asset_type)) == symbol
                    ]

                    if not same.empty:
                        same_dates = pd.to_datetime(
                            same["date"].astype(str).str.strip().str.replace("/", "-").str.replace(".", "-"),
                            errors="coerce"
                        ).dropna().dt.date

                        if len(same_dates) == 0:
                            st.error("舊資料的 date 格式無法判斷，請先把 transactions 的 date 全部改成 YYYY-MM-DD")
                            raise ValueError("date 格式無法判斷")

                        if any(d <= tx_date for d in same_dates.tolist()):
                            st.error("initial 日期必須早於此資產的其他交易日期")
                            raise ValueError("initial 日期不合法")

                else:
                    if abs(to_number(qty) * to_number(price) * fx_used - amount_twd) > 1:
                        st.error("金額驗證錯誤")
                        raise ValueError("金額驗證錯誤")

                    if action == "sell":
                        current_qty = get_current_qty(transactions_df, symbol, asset_type)
                        if to_number(qty) > current_qty:
                            st.error(f"賣出數量({qty}) 超過目前持有({current_qty})，不允許送出")
                            raise ValueError("賣出超過持有")

                new_row = [
                    str(datetime.now().timestamp()),
                    str(tx_date),
                    action,
                    asset_type,
                    symbol,
                    user_strat,   # ✅ 寫入使用者目前的 strategy
                    currency,
                    fx_used,
                    to_number(qty),
                    to_number(price),
                    amount_original,
                    amount_twd
                ]

                sheet.worksheet("transactions").append_row(new_row)

                # ✅ 勾選同步更新預設策略：寫入 symbol_strategy
                if st.session_state.sync_symbol_strategy:
                    if not st.session_state.sync_confirm:
                        st.error("你勾選了同步更新，但尚未確認。")
                        raise ValueError("未確認同步更新")
                    upsert_symbol_strategy(symbol, user_strat)

                saved_at = datetime.now().strftime("%H:%M")
                st.session_state.btn_label = f"✅ 已存檔 ({saved_at})"
                status_area.caption(st.session_state.btn_label)

                st.session_state.in_symbol = ""
                st.session_state.in_qty = 0.0
                st.session_state.in_price = 0.0
                st.session_state.sync_symbol_strategy = False
                st.session_state.sync_confirm = False

                qp_clear(keys=["add"])
                st.rerun()

            except Exception:
                st.session_state.btn_label = "送出"
                status_area.caption("已取消送出（請修正欄位後再送出）")

            finally:
                st.session_state.processing = False

    add_trade_dialog()

# =========================
# 新增：Dashboard 篩選 / 刷新按鈕（不改既有 CSS，只新增區塊）
# =========================
def apply_filters(df: pd.DataFrame, start_d, end_d, symbols_sel, strategy_sel):
    out = df.copy()
    if out.empty:
        return out

    d = pd.to_datetime(out["date"].astype(str).str.strip().str.replace("/", "-").str.replace(".", "-"), errors="coerce").dt.date
    out = out.assign(_d=d)

    if start_d:
        out = out[out["_d"].notna() & (out["_d"] >= start_d)]
    if end_d:
        out = out[out["_d"].notna() & (out["_d"] <= end_d)]

    # symbol 多選
    if symbols_sel:
        out["symbol_norm"] = out["symbol"].astype(str).str.strip().str.upper()
        sel_norm = set([str(s).strip().upper() for s in symbols_sel])
        out = out[out["symbol_norm"].isin(sel_norm)]

    # strategy
    if strategy_sel and strategy_sel != "全部":
        out["strategy_norm"] = out.get("strategy", "").astype(str).apply(normalize_strategy)
        out = out[out["strategy_norm"] == strategy_sel]

    return out.drop(columns=[c for c in ["_d", "symbol_norm", "strategy_norm"] if c in out.columns], errors="ignore")

def strategy_analysis_block(df_tx: pd.DataFrame, prices_df: pd.DataFrame):
    """
    顯示：策略 / 總投入 / 市值 / 報酬率
    """
    if df_tx is None or df_tx.empty:
        st.info("目前沒有資料可分析。")
        return

    tx = df_tx.copy()
    tx["strategy"] = tx.get("strategy", "").astype(str).apply(normalize_strategy)
    strategies = sorted(list(set(tx["strategy"].tolist())))

    rows = []
    for strat in strategies:
        sub = tx[tx["strategy"] == strat]
        metrics = calculate_metrics(sub, prices_df)
        if metrics is None:
            invest, value, divi, profit, rate = None, None, None, None, None
        else:
            invest, value, divi, profit, rate = metrics
        rows.append({
            "策略": strat,
            "總投入": invest if invest is not None else None,
            "市值": value if value is not None else None,
            "報酬率": rate if rate is not None else None
        })

    out = pd.DataFrame(rows)
    if out.empty:
        st.info("目前沒有資料可分析。")
        return

    # 顯示用格式（不改原本 KPI/hero，只是新增表格）
    out_disp = out.copy()
    out_disp["總投入"] = out_disp["總投入"].map(lambda x: "—" if pd.isna(x) else f"{float(x):,.0f}")
    out_disp["市值"] = out_disp["市值"].map(lambda x: "—" if pd.isna(x) else f"{float(x):,.0f}")
    out_disp["報酬率"] = out_disp["報酬率"].map(lambda x: "—" if pd.isna(x) else f"{float(x):.2f}%")

    st.markdown("#### 策略績效")
    st.dataframe(out_disp, use_container_width=True, hide_index=True)

# =========================
# Dashboard（保留原 Tabs / Hero / KPI / Top3）
# + 增量：篩選、刷新、策略績效
# =========================
if current_page == "dashboard":

    # --- 篩選 + 刷新區塊（新增，不動原 KPI/Tab 結構） ---
    with st.container():
        st.markdown('<div class="form-card">', unsafe_allow_html=True)

        # 預設：最近 30 天（符合你交易列表預設；Dashboard 也先給一個合理預設）
        st.session_state.setdefault("f_start", date.today() - timedelta(days=30))
        st.session_state.setdefault("f_end", date.today())
        st.session_state.setdefault("f_symbols", [])
        st.session_state.setdefault("f_strategy", "全部")

        f1, f2 = st.columns(2)
        with f1:
            start_d = st.date_input("開始日期", key="f_start")
        with f2:
            end_d = st.date_input("結束日期", key="f_end")

        # Symbol 多選（從交易表出現過的 symbol 取）
        all_symbols = sorted(list(set(
            transactions_df.get("symbol", pd.Series(dtype=str)).astype(str).str.strip().str.upper().tolist()
        )))
        symbols_sel = st.multiselect("Symbol（可多選）", options=all_symbols, key="f_symbols")

        strategy_sel = st.selectbox("Strategy", ["全部", "存股", "波段", "未分類"], key="f_strategy")

        # 刷新按鈕（60 秒冷卻）
        ok, remain = can_refresh(prices_df)
        refresh_cols = st.columns([1, 2])
        with refresh_cols[0]:
            if st.button("🔄 刷新價格", disabled=not ok, use_container_width=True):
                with st.spinner("正在更新價格…"):
                    refresh_prices(transactions_df, prices_df)
                # 重新讀一次資料（不改架構，只是讓畫面同步）
                transactions_df, prices_df, allowed_df = load_data()
                st.rerun()
        with refresh_cols[1]:
            if not ok:
                st.caption(f"冷卻中：{remain} 秒後可再刷新")
            else:
                st.caption("可刷新（60 秒冷卻）")

        st.markdown("</div>", unsafe_allow_html=True)

    # 套用篩選後的 df
    filtered_all = apply_filters(
        transactions_df[transactions_df["asset_type"].isin(["stock", "fund"])],
        st.session_state.f_start, st.session_state.f_end,
        st.session_state.f_symbols, st.session_state.f_strategy
    )

    stock_df = filtered_all[filtered_all["asset_type"] == "stock"]
    fund_df = filtered_all[filtered_all["asset_type"] == "fund"]
    all_df = filtered_all

    fx, fx_updated = get_usd_twd_info(prices_df)
    prices_updated = get_prices_updated_at(prices_df)

    tab_all, tab_stock, tab_fund = st.tabs(["全部", "股票", "基金"])

    all_metrics = calculate_metrics(all_df, prices_df)
    stock_metrics = calculate_metrics(stock_df, prices_df)
    fund_metrics = calculate_metrics(fund_df, prices_df)

    def render_dashboard(metrics, label, df_for_table):
        if metrics is None:
            st.error("資料異常：出現負持股")
            hero_card("目前市值（TWD）", "—", "請先確認交易資料是否正確")
            st.markdown(
                f'<div class="kpi-grid">{"".join([kpi_card_html("總投入","—"), kpi_card_html("已領息","—"), kpi_card_html("總報酬","—"), kpi_card_html("總報酬率","—")])}</div>',
                unsafe_allow_html=True
            )
            return

        invest, value, divi, profit, rate = metrics

        sub_parts = []
        if prices_updated:
            sub_parts.append(f"價格更新：{prices_updated}")
        if fx is not None:
            fx_part = f"USD_TWD：{fx:.4f}"
            if fx_updated:
                fx_part += f"（{fx_updated}）"
            sub_parts.append(fx_part)

        hero_sub = " ｜ ".join(sub_parts) if sub_parts else ""
        hero_card(f"{label}｜目前市值（TWD）", fmt_money(value), hero_sub)

        kpis_html = "".join([
            kpi_card_html("總投入", fmt_money(invest)),
            kpi_card_html("已領息", fmt_money(divi)),
            kpi_card_html("總報酬", fmt_signed_money_html(profit)),
            kpi_card_html("總報酬率", fmt_signed_pct_html(rate)),
        ])
        st.markdown(f'<div class="kpi-grid">{kpis_html}</div>', unsafe_allow_html=True)

        st.markdown("#### 持有前三名（市值）")
        top3 = top_holdings_table(df_for_table, prices_df, top_n=3)
        st.dataframe(top3, use_container_width=True, hide_index=True)

    with tab_all:
        render_dashboard(all_metrics, "全部", all_df)
        strategy_analysis_block(all_df, prices_df)

    with tab_stock:
        render_dashboard(stock_metrics, "股票", stock_df)
        strategy_analysis_block(stock_df, prices_df)

    with tab_fund:
        render_dashboard(fund_metrics, "基金", fund_df)
        strategy_analysis_block(fund_df, prices_df)

# =========================
# 交易明細頁（新增頁面：最近 30 天 + 載入更多）
# =========================
if current_page == "tx":

    st.markdown("### 交易明細")

    # 預設最近 30 天
    st.session_state.setdefault("tx_days", 30)
    st.session_state.setdefault("tx_page_size", 50)

    # 顯示目前範圍
    days = int(st.session_state.tx_days)
    start_d = date.today() - timedelta(days=days)
    end_d = date.today()

    # 讀取並過濾日期
    tx = transactions_df.copy()
    if tx.empty:
        st.info("目前沒有交易資料。")
    else:
        tx["_d"] = pd.to_datetime(
            tx["date"].astype(str).str.strip().str.replace("/", "-").str.replace(".", "-"),
            errors="coerce"
        ).dt.date

        tx = tx[tx["_d"].notna() & (tx["_d"] >= start_d) & (tx["_d"] <= end_d)]

        # 排序 date desc
        tx["_dt"] = pd.to_datetime(tx["date"].astype(str), errors="coerce")
        tx = tx.sort_values("_dt", ascending=False)

        # 顯示欄位
        show_cols = ["date", "symbol", "action", "qty", "price", "amount_twd"]
        for c in show_cols:
            if c not in tx.columns:
                tx[c] = ""

        # 分頁顯示（載入更多）
        limit = int(st.session_state.tx_page_size)
        tx_show = tx.head(limit).copy()

        # 格式化
        tx_show["qty"] = tx_show["qty"].apply(to_number).map(lambda x: f"{x:,.4f}".rstrip("0").rstrip("."))
        tx_show["price"] = tx_show["price"].apply(to_number).map(lambda x: f"{x:,.4f}".rstrip("0").rstrip("."))
        tx_show["amount_twd"] = tx_show["amount_twd"].apply(to_number).map(lambda x: f"{x:,.0f}")

        st.caption(f"顯示最近 {days} 天（{start_d} ～ {end_d}），排序：date desc")
        st.dataframe(tx_show[show_cols], use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("載入更多（同區間增加筆數）", use_container_width=True):
                st.session_state.tx_page_size += 50
                st.rerun()
        with c2:
            if st.button("往前再多 30 天", use_container_width=True):
                st.session_state.tx_days += 30
                # 依定稿：範圍變動應重置載入狀態
                st.session_state.tx_page_size = 50
                st.rerun()
