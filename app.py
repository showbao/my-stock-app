import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import re
import json
import urllib.request
import urllib.parse
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
    """若工作表不存在就建立（新增功能：不會動到既有表）"""
    try:
        ws = sheet.worksheet(ws_name)
        return ws
    except Exception:
        ws = sheet.add_worksheet(title=ws_name, rows=1000, cols=max(10, len(headers) + 2))
        ws.append_row(headers)
        return ws

def load_data():
    transactions = pd.DataFrame(sheet.worksheet("transactions").get_all_records())
    prices = pd.DataFrame(sheet.worksheet("prices").get_all_records())
    allowed = pd.DataFrame(sheet.worksheet("settings_allowed_emails").get_all_records())

    # 新增：symbol_strategy 表（不存在就建立）
    try:
        symbol_strategy = pd.DataFrame(sheet.worksheet("symbol_strategy").get_all_records())
    except Exception:
        ensure_worksheet("symbol_strategy", ["symbol", "strategy", "updated_at"])
        symbol_strategy = pd.DataFrame(sheet.worksheet("symbol_strategy").get_all_records())

    return transactions, prices, allowed, symbol_strategy

transactions_df, prices_df, allowed_df, symbol_strategy_df = load_data()

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

def parse_dt_any(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    # 常見：YYYY-MM-DD HH:mm:ss / ISO 8601
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None

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

    # 佔比用「全部持有」做分母（不是只有 top_n）
    total_mv = float(out["市值(TWD)"].sum()) if "市值(TWD)" in out.columns else 0.0

    out = out.sort_values("市值(TWD)", ascending=False).head(top_n)

    if total_mv > 0:
        out["佔比"] = out["市值(TWD)"].map(lambda v: v / total_mv)
    else:
        out["佔比"] = 0.0

    # 顯示用格式（字串化，讓表格更乾淨）
    out["持有數量"] = out["持有數量"].map(lambda x: f"{x:,.4f}".rstrip("0").rstrip("."))
    out["價格"] = out["價格"].map(lambda x: f"{x:,.4f}".rstrip("0").rstrip("."))
    out["市值(TWD)"] = out["市值(TWD)"].map(lambda x: f"{x:,.0f}")
    out["佔比"] = out["佔比"].map(lambda x: f"{x*100:.1f}%")

    # 欄位順序
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

def fmt_signed_money(n):
    try:
        n = float(n)
        sign = "+" if n > 0 else ""
        return f"{sign}{n:,.0f}"
    except:
        return "—"

def fmt_signed_pct(n):
    try:
        n = float(n)
        sign = "+" if n > 0 else ""
        return f"{sign}{n:.2f}%"
    except:
        return "—"

def fmt_signed_money_html(n):
    """回傳帶顏色的金額字串（仍保留 + / - 符號）"""
    try:
        n = float(n)
        sign = "+" if n > 0 else ""
        cls = "profit-pos" if n > 0 else "profit-neg" if n < 0 else "profit-zero"
        return f"<span class='{cls}'>{sign}{n:,.0f}</span>"
    except:
        return "—"

def fmt_signed_pct_html(n):
    """回傳帶顏色的百分比字串（仍保留 + / - 符號）"""
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
# 新增：策略對照（symbol → strategy）
# =========================
def get_default_strategy(symbol: str) -> str:
    if symbol_strategy_df is None or symbol_strategy_df.empty:
        return ""
    df = symbol_strategy_df.copy()
    df["symbol"] = df.get("symbol", "").astype(str).str.strip().str.upper()
    row = df[df["symbol"] == str(symbol).strip().upper()]
    if row.empty:
        return ""
    return str(row.iloc[-1].get("strategy", "")).strip()

def upsert_symbol_strategy(symbol: str, strategy: str):
    ws = ensure_worksheet("symbol_strategy", ["symbol", "strategy", "updated_at"])
    symbol_u = str(symbol).strip().upper()
    strategy_s = str(strategy).strip()
    now_s = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 先找是否已有同 symbol（用整欄掃描，資料量小才合理）
    try:
        col = ws.col_values(1)  # A 欄 symbol
        idx = None
        for i, v in enumerate(col):
            if i == 0:
                continue  # header
            if str(v).strip().upper() == symbol_u:
                idx = i + 1  # worksheet 是 1-based row
        if idx is not None:
            ws.update(f"A{idx}:C{idx}", [[symbol_u, strategy_s, now_s]])
        else:
            ws.append_row([symbol_u, strategy_s, now_s])
    except Exception:
        # 寫入失敗就丟出去，讓上層顯示錯誤
        raise

# =========================
# 新增：抓價（盡量穩定、可替代）
# =========================
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
        return resp.read().decode("utf-8", errors="ignore")

def fetch_usd_twd_open_er_api():
    # 不需要 API key：open.er-api.com
    # 回傳格式：{ "rates": { "TWD": 31.xx, ... }, ... }
    url = "https://open.er-api.com/v6/latest/USD"
    j = http_get_json(url, timeout=10)
    rates = j.get("rates", {}) if isinstance(j, dict) else {}
    rate = rates.get("TWD", None)
    if rate is None:
        raise ValueError("open.er-api 回傳缺少 TWD")
    return float(rate)

def fetch_usd_twd_fawaz():
    # 不需要 API key：@fawazahmed0/currency-api（走 jsDelivr CDN）
    # 回傳格式：{ "twd": 31.xx, ... }
    url = "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd/twd.json"
    j = http_get_json(url, timeout=10)
    rate = j.get("twd", None) if isinstance(j, dict) else None
    if rate is None:
        raise ValueError("fawaz 回傳缺少 twd")
    return float(rate)

def fetch_usd_twd():
    # 依序嘗試（避免單一來源掛掉）
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

def fetch_tw_stock_price_mis(symbol: str):
    """
    台股：用 TWSE MIS 介面抓盤中/最新價
    symbol 允許：'2330' / '2330.TW' / '0050.TW'
    """
    s = str(symbol).strip().upper()
    s = s.replace(".TW", "").replace(".TWO", "")
    s = re.sub(r"[^0-9A-Z]", "", s)

    # 嘗試同時查上市/上櫃
    ex_ch = f"tse_{s}.tw|otc_{s}.tw"
    url = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp?" + urllib.parse.urlencode(
        {"ex_ch": ex_ch, "json": "1", "delay": "0"}
    )
    j = http_get_json(url, timeout=10)

    arr = j.get("msgArray", []) if isinstance(j, dict) else []
    if not arr:
        raise ValueError("台股來源查不到資料")

    r0 = arr[0]
    # z = 當前成交價；若 '-' 用 y(昨收) 或 o(開)
    z = str(r0.get("z", "")).strip()
    if z in ("", "-", "0", "0.0"):
        z = str(r0.get("y", "")).strip()
    price = to_number(z)
    if price <= 0:
        raise ValueError("台股來源回傳價格不合法")
    return float(price)

def fetch_us_stock_price_stooq(symbol: str):
    """
    Stooq quote：https://stooq.com/q/l/?s=aapl.us&f=sd2t2ohlcv&h&e=csv
    需要 Stooq 代碼：AAPL.US（不是 AAPL）
    """
    s = str(symbol).strip().upper()
    # 常見：AAPL -> AAPL.US（先以 US 為預設）
    if "." not in s:
        s = f"{s}.US"
    url = f"https://stooq.com/q/l/?s={urllib.parse.quote(s.lower())}&f=sd2t2ohlcv&h&e=csv"
    txt = http_get_text(url, timeout=10).strip()
    # 期待兩行：header + data
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError("Stooq 回傳格式不正確")
    parts = [p.strip() for p in lines[1].split(",")]
    # header: Symbol,Date,Time,Open,High,Low,Close,Volume
    if len(parts) < 8:
        raise ValueError("Stooq 回傳欄位不足")
    close = to_number(parts[6])
    if close <= 0:
        raise ValueError("Stooq 回傳價格不合法")
    return float(close)

def fetch_price(symbol: str, asset_type: str):
    """
    依資產類型抓價（新增功能）
    - stock: 台股走 TWSE MIS；其他走 Stooq（US 預設）
    - fund: 目前先走 Stooq/台股（若代碼像台股）；抓不到就回 None
    """
    s = str(symbol).strip().upper()

    if s == "USD_TWD":
        return fetch_usd_twd(), "TWD"

    if asset_type == "stock":
        if s.endswith(".TW") or re.fullmatch(r"\d{4}", s) or re.fullmatch(r"\d{4}\.TW", s):
            p = fetch_tw_stock_price_mis(s)
            return p, "TWD"
        # 其他先走 Stooq（多半 US）
        p = fetch_us_stock_price_stooq(s)
        return p, "USD"

    # fund：先用台股方式（若長得像台股/ETF），否則用 Stooq 試試
    if asset_type == "fund":
        ss = s
        if ss.startswith("F_"):
            ss = ss[2:]
        try:
            if ss.endswith(".TW") or re.fullmatch(r"\d{4}", ss) or re.fullmatch(r"\d{4}\.TW", ss):
                p = fetch_tw_stock_price_mis(ss)
                return p, "TWD"
        except Exception:
            pass
        try:
            p = fetch_us_stock_price_stooq(ss)
            return p, "USD"
        except Exception:
            return None, None

    return None, None

def write_price_to_sheet(symbol: str, asset_type: str, price: float, currency: str):
    ws = sheet.worksheet("prices")
    now_s = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sym = str(symbol).strip().upper()
    at = str(asset_type).strip().lower()
    ccy = str(currency).strip().upper()

    # 找是否已有該列（symbol + asset_type）
    rows = ws.get_all_values()
    if not rows:
        ws.append_row(["symbol", "asset_type", "price", "currency", "updated_at"])
        rows = ws.get_all_values()

    header = rows[0]
    # 欄位索引
    def idx(name):
        try:
            return header.index(name)
        except Exception:
            return None

    i_sym = idx("symbol")
    i_at = idx("asset_type")
    i_price = idx("price")
    i_ccy = idx("currency")
    i_uat = idx("updated_at")

    # 若 header 缺欄位，直接不處理（避免破壞既有表）
    if None in (i_sym, i_at, i_price, i_ccy, i_uat):
        raise ValueError("prices 表欄位不足，需包含 symbol/asset_type/price/currency/updated_at")

    target_row = None
    for r_i in range(1, len(rows)):
        r = rows[r_i]
        if len(r) <= max(i_sym, i_at):
            continue
        if str(r[i_sym]).strip().upper() == sym and str(r[i_at]).strip().lower() == at:
            target_row = r_i + 1  # worksheet row number
            break

    values = [sym, at, float(price), ccy, now_s]

    if target_row is None:
        ws.append_row(values)
    else:
        # A:E 固定更新
        ws.update(f"A{target_row}:E{target_row}", [values])

def get_last_prices_updated_dt(prices_df: pd.DataFrame):
    if prices_df is None or prices_df.empty or "updated_at" not in prices_df.columns:
        return None
    dts = prices_df["updated_at"].apply(parse_dt_any).dropna()
    if dts.empty:
        return None
    return max(dts.tolist())

def can_refresh_prices(prices_df: pd.DataFrame, cooldown_seconds: int = 60, grace_seconds: int = 3):
    last_dt = get_last_prices_updated_dt(prices_df)
    if last_dt is None:
        return True, 0
    elapsed = (datetime.now() - last_dt).total_seconds()
    # 規則：冷卻 60 秒 + 寬限 3 秒 → 小於 57 秒才拒絕
    threshold = max(0, cooldown_seconds - grace_seconds)
    if elapsed < threshold:
        return False, int(threshold - elapsed)
    return True, 0

def refresh_prices(transactions_df: pd.DataFrame, prices_df: pd.DataFrame):
    # 收集需要刷新 symbol（含 USD_TWD）
    tx = transactions_df.copy() if transactions_df is not None else pd.DataFrame()
    if not tx.empty:
        tx["symbol"] = tx.get("symbol", "").astype(str).str.strip()
        tx["asset_type"] = tx.get("asset_type", "").astype(str).str.strip().str.lower()
        # 只抓 stock/fund
        tx = tx[tx["asset_type"].isin(["stock", "fund"])]
    symbols = []
    for _, r in tx.iterrows():
        sym = str(r.get("symbol", "")).strip()
        at = str(r.get("asset_type", "")).strip().lower()
        if sym == "":
            continue
        symbols.append((sym, at))

    # 去重（以 symbol+asset_type）
    seen = set()
    uniq = []
    for sym, at in symbols:
        key = (sym.upper(), at)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((sym, at))

    # 一定要更新 USD_TWD（asset_type=system）
    uniq.append(("USD_TWD", "system"))

    ok_list = []
    fail_list = []

    for sym, at in uniq:
        try:
            if sym == "USD_TWD":
                p, ccy = fetch_price("USD_TWD", "system")
                write_price_to_sheet("USD_TWD", "system", p, "TWD")
                ok_list.append("USD_TWD")
                continue

            p, ccy = fetch_price(sym, at)
            if p is None:
                fail_list.append(f"{sym}（抓不到）")
                continue
            write_price_to_sheet(sym, at, p, ccy)
            ok_list.append(sym)
        except Exception as e:
            fail_list.append(f"{sym}（{str(e)}）")

    return ok_list, fail_list

# =========================
# 交易過濾（Dashboard + 明細共用）
# =========================
def apply_filters(df: pd.DataFrame, start_d: date | None, end_d: date | None, symbols: list[str], strategy: str):
    if df is None or df.empty:
        return df

    out = df.copy()

    # 日期
    if "date" in out.columns:
        out["_d"] = parse_date_series(out["date"])
        if start_d:
            out = out[out["_d"] >= start_d]
        if end_d:
            out = out[out["_d"] <= end_d]

    # Symbol
    if symbols:
        out["symbol"] = out["symbol"].astype(str).str.strip()
        out = out[out["symbol"].isin(symbols)]

    # Strategy
    if strategy and strategy != "全部":
        out["strategy"] = out.get("strategy", "").astype(str).str.strip()
        out = out[out["strategy"] == strategy]

    return out.drop(columns=[c for c in ["_d"] if c in out.columns], errors="ignore")

def strategy_summary(df_tx: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    """策略績效（簡化版）：用同一份計算邏輯算每個策略的投入/市值/報酬率"""
    if df_tx is None or df_tx.empty:
        return pd.DataFrame(columns=["策略", "總投入(TWD)", "目前市值(TWD)", "報酬率"])

    df = df_tx.copy()
    df["strategy"] = df.get("strategy", "").astype(str).str.strip()
    df.loc[df["strategy"] == "", "strategy"] = "未分類"

    rows = []
    for strat, g in df.groupby("strategy"):
        m = calculate_metrics(g, prices_df)
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

# =========================
# 頁面（用 Tabs 當導覽：不使用 Sidebar）
# =========================
page_dash, page_tx = st.tabs(["Dashboard", "交易明細"])

# =========================
# Dashboard
# =========================
with page_dash:

    # --- 右下角懸浮「+」按鈕：開啟新增交易彈窗 ---
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
            # 關閉（不存檔）
            if st.button("關閉", use_container_width=True):
                st.query_params.clear()
                st.rerun()

            st.session_state.setdefault("processing", False)
            st.session_state.setdefault("btn_label", "送出")

            # 這些 key 讓我們能「成功後局部清空」
            st.session_state.setdefault("in_action", "buy")
            st.session_state.setdefault("in_asset_type", "stock")
            st.session_state.setdefault("in_symbol", "")
            st.session_state.setdefault("in_strategy", "")
            st.session_state.setdefault("in_currency", "TWD")
            st.session_state.setdefault("in_qty", 0.0)
            st.session_state.setdefault("in_price", 0.0)
            st.session_state.setdefault("in_date", date.today())
            st.session_state.setdefault("in_fx_rate", 1.0)

            # 新增：使用者是否手動改過 strategy（避免每次都被自動覆蓋）
            st.session_state.setdefault("strategy_user_edited", False)
            st.session_state.setdefault("auto_default_strategy", "")

            def mark_dirty():
                st.session_state.btn_label = "送出"

            def on_symbol_change():
                st.session_state.btn_label = "送出"
                st.session_state.strategy_user_edited = False

                asset_type = st.session_state.get("in_asset_type", "stock")
                sym_raw = st.session_state.get("in_symbol", "")
                sym = clean_symbol(sym_raw, asset_type)

                default_strat = get_default_strategy(sym)
                st.session_state.auto_default_strategy = default_strat

                # 只有在「使用者還沒改過」或 strategy 空白時，才自動帶入
                if (not st.session_state.strategy_user_edited) and (str(st.session_state.get("in_strategy", "")).strip() == ""):
                    st.session_state.in_strategy = default_strat

            def on_strategy_change():
                st.session_state.btn_label = "送出"
                st.session_state.strategy_user_edited = True

            action = st.selectbox(
                "交易類型", ["buy", "sell", "dividend", "initial"],
                key="in_action", on_change=mark_dirty
            )
            asset_type = st.selectbox(
                "資產類型", ["stock", "fund"],
                key="in_asset_type", on_change=on_symbol_change  # 資產類型變更也要重抓預設策略
            )
            symbol_input = st.text_input("代號", key="in_symbol", on_change=on_symbol_change)
            strategy = st.text_input("策略", key="in_strategy", on_change=on_strategy_change)
            tx_date = st.date_input("日期", key="in_date", on_change=mark_dirty)

            # 若使用者改了策略（而且跟預設不同），才顯示「同步更新」
            symbol_norm = clean_symbol(symbol_input, asset_type)
            auto_default = str(st.session_state.get("auto_default_strategy", "")).strip()
            current_strategy = str(strategy).strip()

            show_sync = (symbol_norm != "") and (current_strategy != "") and (current_strategy != auto_default)

            sync_default = False
            if show_sync:
                st.warning(f"你目前把【{symbol_norm}】的策略改成：{current_strategy}\n\n若你希望以後新增交易都自動帶這個策略，可勾選下面同步。")
                sync_default = st.checkbox("同步更新此 symbol 的預設策略（會影響所有頁面）", value=False)

            currency = st.selectbox(
                "幣別", ["TWD", "USD"],
                key="in_currency", on_change=mark_dirty
            )

            qty = st.number_input("數量", min_value=0.0, key="in_qty", on_change=mark_dirty)
            price = st.number_input("單價", min_value=0.0, key="in_price", on_change=mark_dirty)

            # fx_rate：TWD 時 disabled + 預填 1.0
            if currency == "TWD":
                st.session_state.in_fx_rate = 1.0
                fx_rate = st.number_input("匯率（USD_TWD）", value=1.0, disabled=True)
            else:
                fx_rate = st.number_input("匯率（USD_TWD）", min_value=0.0, key="in_fx_rate", on_change=mark_dirty)

            st.caption("TWD 會自動使用 1.0；USD 請填 USD_TWD 匯率。")
            st.divider()

            # 二級確認：只有勾了「同步更新預設策略」才出現
            confirm_sync = True
            if sync_default:
                st.error(f"⚠️ 警告：這會改變「{symbol_norm}」在所有頁面的預設分類。")
                confirm_sync = st.checkbox("我了解，仍要同步更新", value=False)

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

                    # 代號必填
                    if symbol == "":
                        st.error("代號不可空白")
                        raise ValueError("代號不可空白")

                    # USD 必須填匯率（包含 initial）
                    if currency == "USD" and to_number(fx_rate) == 0:
                        st.error("USD 必須填匯率")
                        raise ValueError("USD 必須填匯率")

                    fx_used = to_number(fx_rate) if currency == "USD" else 1.0

                    # 計算金額
                    amount_original = to_number(qty) * to_number(price)
                    amount_twd = amount_original * fx_used

                    # initial 規則
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
                        # 金額驗證容差 ±1 元（USD 用換算後）
                        if abs(to_number(qty) * to_number(price) * fx_used - amount_twd) > 1:
                            st.error("金額驗證錯誤")
                            raise ValueError("金額驗證錯誤")

                        # sell 超過持有：擋下
                        if action == "sell":
                            current_qty = get_current_qty(transactions_df, symbol, asset_type)
                            if to_number(qty) > current_qty:
                                st.error(f"賣出數量({qty}) 超過目前持有({current_qty})，不允許送出")
                                raise ValueError("賣出超過持有")

                    # 若勾選同步更新，但未二級確認，直接擋下
                    if sync_default and not confirm_sync:
                        st.error("你勾選了同步更新，但尚未二級確認")
                        raise ValueError("同步更新未確認")

                    new_row = [
                        str(datetime.now().timestamp()),
                        str(tx_date),
                        action,
                        asset_type,
                        symbol,
                        strategy,
                        currency,
                        fx_used,
                        to_number(qty),
                        to_number(price),
                        amount_original,
                        amount_twd
                    ]

                    sheet.worksheet("transactions").append_row(new_row)

                    # 新增：同步更新預設策略
                    if sync_default and confirm_sync:
                        upsert_symbol_strategy(symbol, strategy)

                    saved_at = datetime.now().strftime("%H:%M")
                    st.session_state.btn_label = f"✅ 已存檔 ({saved_at})"
                    status_area.caption(st.session_state.btn_label)

                    # 局部清空：清 symbol/qty/price（保留 date/asset_type/strategy/currency）
                    st.session_state.in_symbol = ""
                    st.session_state.in_qty = 0.0
                    st.session_state.in_price = 0.0
                    st.session_state.strategy_user_edited = False
                    st.session_state.auto_default_strategy = ""

                    # 存檔後：關閉彈窗（把 add=1 拿掉）
                    st.query_params.clear()

                    # 寫入成功後才清快取 + loading
                    with st.spinner("正在重新同步資料…"):
                        st.cache_resource.clear()
                    st.rerun()

                except Exception:
                    st.session_state.btn_label = "送出"
                    status_area.caption("已取消送出（請修正欄位後再送出）")

                finally:
                    st.session_state.processing = False

        add_trade_dialog()

    # =========================
    # Dashboard 篩選 + 刷新
    # =========================
    st.session_state.setdefault("dash_start", None)
    st.session_state.setdefault("dash_end", None)
    st.session_state.setdefault("dash_symbols", [])
    st.session_state.setdefault("dash_strategy", "全部")

    # 供選擇的 symbol（從交易表）
    all_symbols = sorted(
        transactions_df.get("symbol", pd.Series(dtype=str)).astype(str).str.strip().replace("", pd.NA).dropna().unique().tolist()
    )
    all_strategies = ["全部", "存股", "波段", "未分類"]

    f1, f2 = st.columns([7, 3])
    with f1:
        c1, c2 = st.columns(2)
        with c1:
            st.date_input("開始日期", key="dash_start")
        with c2:
            st.date_input("結束日期", key="dash_end")

        st.multiselect("Symbol（可多選）", options=all_symbols, key="dash_symbols")
        st.selectbox("Strategy", options=all_strategies, key="dash_strategy")

        if st.button("清除篩選", use_container_width=True):
            st.session_state.dash_start = None
            st.session_state.dash_end = None
            st.session_state.dash_symbols = []
            st.session_state.dash_strategy = "全部"
            st.rerun()

    with f2:
        # 冷卻判斷（全域：用 prices.updated_at）
        ok_refresh, remain = can_refresh_prices(prices_df, cooldown_seconds=60, grace_seconds=3)
        btn_label = "刷新價格" if ok_refresh else f"刷新價格（{remain}s）"
        clicked = st.button(btn_label, disabled=(not ok_refresh), use_container_width=True)

        if clicked:
            with st.spinner("正在刷新價格…"):
                ok_list, fail_list = refresh_prices(transactions_df, prices_df)

            # 把結果先存起來，避免 st.rerun() 把訊息洗掉
            st.session_state["refresh_ok_list"] = ok_list
            st.session_state["refresh_fail_list"] = fail_list

            # 寫入成功後才清快取 + loading
            with st.spinner("正在重新同步資料…"):
                st.cache_resource.clear()

            st.rerun()

    # 刷新結果（顯示一次後清掉）
    if st.session_state.get("refresh_ok_list") is not None or st.session_state.get("refresh_fail_list") is not None:
        ok_list = st.session_state.get("refresh_ok_list") or []
        fail_list = st.session_state.get("refresh_fail_list") or []

        if ok_list:
            st.success("已更新：" + "、".join(ok_list[:8]) + ("…" if len(ok_list) > 8 else ""))
        if fail_list:
            st.warning("未更新：" + "、".join(fail_list[:6]) + ("…" if len(fail_list) > 6 else ""))

        st.session_state["refresh_ok_list"] = None
        st.session_state["refresh_fail_list"] = None

    # 取匯率/更新時間（用剛載入的 prices_df）
    fx, fx_updated = get_usd_twd_info(prices_df)
    prices_updated = get_prices_updated_at(prices_df)

    # Dashboard 內部 Tabs：全部/股票/基金（保留原本架構）
    tab_all, tab_stock, tab_fund = st.tabs(["全部", "股票", "基金"])

    # 先依資產類型分
    stock_df = transactions_df[transactions_df["asset_type"] == "stock"]
    fund_df = transactions_df[transactions_df["asset_type"] == "fund"]
    all_df = transactions_df[transactions_df["asset_type"].isin(["stock", "fund"])]

    # 新增：套用篩選
    dash_filtered_all = apply_filters(
        all_df,
        st.session_state.get("dash_start"),
        st.session_state.get("dash_end"),
        st.session_state.get("dash_symbols", []),
        st.session_state.get("dash_strategy", "全部"),
    )
    dash_filtered_stock = apply_filters(
        stock_df,
        st.session_state.get("dash_start"),
        st.session_state.get("dash_end"),
        st.session_state.get("dash_symbols", []),
        st.session_state.get("dash_strategy", "全部"),
    )
    dash_filtered_fund = apply_filters(
        fund_df,
        st.session_state.get("dash_start"),
        st.session_state.get("dash_end"),
        st.session_state.get("dash_symbols", []),
        st.session_state.get("dash_strategy", "全部"),
    )

    all_metrics = calculate_metrics(dash_filtered_all, prices_df)
    stock_metrics = calculate_metrics(dash_filtered_stock, prices_df)
    fund_metrics = calculate_metrics(dash_filtered_fund, prices_df)

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

        # Hero：市值
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

        # KPI
        kpis_html = "".join([
            kpi_card_html("總投入", fmt_money(invest)),
            kpi_card_html("已領息", fmt_money(divi)),
            kpi_card_html("總報酬", fmt_signed_money_html(profit)),
            kpi_card_html("總報酬率", fmt_signed_pct_html(rate)),
        ])
        st.markdown(f'<div class="kpi-grid">{kpis_html}</div>', unsafe_allow_html=True)

        # 新增：策略績效
        st.markdown("#### 策略績效")
        strat_table = strategy_summary(df_for_table, prices_df)
        st.dataframe(strat_table, use_container_width=True, hide_index=True)

        # 持有前三名（市值）
        st.markdown("#### 持有前三名（市值）")
        top3 = top_holdings_table(df_for_table, prices_df, top_n=3)
        st.dataframe(top3, use_container_width=True, hide_index=True)

    with tab_all:
        render_dashboard(all_metrics, "全部", dash_filtered_all)

    with tab_stock:
        render_dashboard(stock_metrics, "股票", dash_filtered_stock)

    with tab_fund:
        render_dashboard(fund_metrics, "基金", dash_filtered_fund)

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

    # 基本篩選
    a, b = st.columns(2)
    with a:
        tx_end = st.date_input("結束日期", key="tx_end")
    with b:
        tx_days = st.selectbox("預設區間", options=[30, 60, 90], index=0, key="tx_days")

    st.multiselect("Symbol（可多選）", options=sorted(
        transactions_df.get("symbol", pd.Series(dtype=str)).astype(str).str.strip().replace("", pd.NA).dropna().unique().tolist()
    ), key="tx_symbols")
    st.selectbox("Strategy", options=["全部", "存股", "波段", "未分類"], key="tx_strategy")

    # 依 offset 往前延伸（每次載入更多，多往前 tx_days 天）
    start_d = st.session_state.tx_end - timedelta(days=st.session_state.tx_days * (st.session_state.tx_offset + 1))
    end_d = st.session_state.tx_end

    # offset 重置規則：只要 date_range / symbol / strategy 任何變動就歸零 + rerun
    sig = f"{tx_end}|{tx_days}|{','.join(st.session_state.tx_symbols)}|{st.session_state.tx_strategy}"
    if st.session_state.tx_last_sig == "":
        st.session_state.tx_last_sig = sig
    elif sig != st.session_state.tx_last_sig:
        st.session_state.tx_last_sig = sig
        st.session_state.tx_offset = 0
        st.rerun()

    df_tx = transactions_df.copy()
    if not df_tx.empty:
        df_tx["date_norm"] = parse_date_series(df_tx["date"])
        df_tx = df_tx.sort_values("date_norm", ascending=False)

    filtered = apply_filters(df_tx, start_d, end_d, st.session_state.tx_symbols, st.session_state.tx_strategy)

    # 顯示欄位（依規格）
    cols = ["date", "symbol", "action", "qty", "price", "amount_twd"]
    existing_cols = [c for c in cols if c in filtered.columns]
    show_df = filtered[existing_cols].copy() if not filtered.empty else pd.DataFrame(columns=existing_cols)

    st.caption(f"顯示區間：{start_d} ~ {end_d}（依結束日期往前延伸）")
    st.dataframe(show_df, use_container_width=True, hide_index=True)

    # 載入更多
    if st.button("載入更多", use_container_width=True):
        st.session_state.tx_offset += 1
        st.rerun()
