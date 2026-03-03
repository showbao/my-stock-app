import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import re
from datetime import datetime, date

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
.db-hero-value { font-size: 34px; font-weight: 700; line-height: 1.1; }
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
.kpi-value { font-size: 26px; font-weight: 700; line-height: 1.15; color: #111827; }
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
.kpi-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 10px;
  margin-top: 10px;
}
@media (min-width: 900px) {
  .kpi-grid { grid-template-columns: 1fr 1fr; }
}
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

transactions_df, prices_df, allowed_df = load_data()

# =========================
# Google 登入
# =========================
def login_screen():
    st.markdown("## 投資儀表板")
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

st.sidebar.success(f"登入：{user_email}")
st.sidebar.button("登出", on_click=st.logout, use_container_width=True)

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
        return pd.DataFrame(columns=["代碼", "持有數量", "價格", "幣別", "市值(TWD)"])

    out = out.sort_values("市值(TWD)", ascending=False).head(top_n)

    # 顯示用格式
    out["持有數量"] = out["持有數量"].map(lambda x: f"{x:,.4f}".rstrip("0").rstrip("."))
    out["價格"] = out["價格"].map(lambda x: f"{x:,.4f}".rstrip("0").rstrip("."))
    out["市值(TWD)"] = out["市值(TWD)"].map(lambda x: f"{x:,.0f}")

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

def kpi_card(title, value_text, sub_text=""):
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value">{value_text}</div>
          {f'<div class="kpi-sub">{sub_text}</div>' if sub_text else ''}
        </div>
        """,
        unsafe_allow_html=True
    )

# KPI 以 grid 方式排版時使用：回傳 HTML 字串（不直接 st.markdown）
def kpi_card_html(title, value_text, sub_text=""):
    # 注意：markdown 會把「前面有空格的行」當成程式碼區塊。
    # 所以這裡刻意不加任何前置空白，避免 HTML 被顯示成文字。
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
    # 用 prices 表中最大的 updated_at 當「最後更新」
    if "updated_at" not in prices_df.columns or prices_df.empty:
        return None
    s = prices_df["updated_at"].astype(str).str.strip()
    s = s[s != ""]
    if s.empty:
        return None
    # 直接取最大字串（常見格式 YYYY-MM-DD HH:mm:ss 會可用）
    return s.max()

# =========================
# 頁面
# =========================
page = st.sidebar.radio("選單", ["首頁", "新增交易"])

# =========================
# 首頁（金融儀表板風）
# =========================
if page == "首頁":
    st.markdown("## 投資儀表板")

    # 取匯率/更新時間
    fx, fx_updated = get_usd_twd_info(prices_df)
    prices_updated = get_prices_updated_at(prices_df)

    # Tabs：全部/股票/基金
    tab_all, tab_stock, tab_fund = st.tabs(["全部", "股票", "基金"])

    stock_df = transactions_df[transactions_df["asset_type"] == "stock"]
    fund_df = transactions_df[transactions_df["asset_type"] == "fund"]
    all_df = transactions_df[transactions_df["asset_type"].isin(["stock", "fund"])]

    all_metrics = calculate_metrics(all_df, prices_df)
    stock_metrics = calculate_metrics(stock_df, prices_df)
    fund_metrics = calculate_metrics(fund_df, prices_df)

    def render_dashboard(metrics, label, df_for_table):


        if metrics is None:


            st.error("資料異常：出現負持股")


            hero_card("目前市值（TWD）", "—", "請先確認交易資料是否正確")


            st.markdown(f'<div class="kpi-grid">{"".join([kpi_card_html("總投入","—"), kpi_card_html("已領息","—"), kpi_card_html("總報酬","—"), kpi_card_html("總報酬率","—")])}</div>', unsafe_allow_html=True)


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



        # KPI：手機一欄、桌機兩欄


        kpis_html = "".join([


            kpi_card_html("總投入", fmt_money(invest)),


            kpi_card_html("已領息", fmt_money(divi)),


            kpi_card_html("總報酬", fmt_signed_money(profit)),


            kpi_card_html("總報酬率", fmt_signed_pct(rate)),


        ])


        st.markdown(f'<div class="kpi-grid">{kpis_html}</div>', unsafe_allow_html=True)



        # 持有前三名（市值）


        st.markdown("#### 持有前三名（市值）")


        top3 = top_holdings_table(df_for_table, prices_df, top_n=3)


        st.dataframe(top3, use_container_width=True, hide_index=True)

    with tab_all:
        render_dashboard(all_metrics, "全部", all_df)

    with tab_stock:
        render_dashboard(stock_metrics, "股票", stock_df)

    with tab_fund:
        render_dashboard(fund_metrics, "基金", fund_df)

# =========================
# 新增交易（儀表板風 + 防連點 + 回彈 + 局部清空）
# =========================
if page == "新增交易":
    st.markdown("## 新增交易")
    st.caption("送出時按鈕會鎖住，避免重複寫入。")

    # 外框卡片（看起來像表單區）
    st.markdown('<div class="form-card">', unsafe_allow_html=True)

    # --- 狀態初始化 ---
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

    def mark_dirty():
        # 任何欄位變更：按鈕立刻回到「送出」
        st.session_state.btn_label = "送出"

    # --- 欄位順序：像券商輸入單 ---
    action = st.selectbox(
        "交易類型", ["buy", "sell", "dividend", "initial"],
        key="in_action", on_change=mark_dirty
    )
    asset_type = st.selectbox(
        "資產類型", ["stock", "fund"],
        key="in_asset_type", on_change=mark_dirty
    )
    symbol_input = st.text_input("代號", key="in_symbol", on_change=mark_dirty)
    strategy = st.text_input("策略", key="in_strategy", on_change=mark_dirty)
    tx_date = st.date_input("日期", key="in_date", on_change=mark_dirty)

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

    st.markdown("<div class='small-muted'>提示：TWD 會自動使用 1.0；USD 請填 USD_TWD 匯率。</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)  # form-card end

    st.divider()

    # --- 按鈕狀態（防連點）---
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

            saved_at = datetime.now().strftime("%H:%M")
            st.session_state.btn_label = f"✅ 已存檔 ({saved_at})"
            status_area.caption(st.session_state.btn_label)

            # 局部清空：清 symbol/qty/price（保留 date/asset_type/strategy/currency）
            st.session_state.in_symbol = ""
            st.session_state.in_qty = 0.0
            st.session_state.in_price = 0.0

        except Exception:
            # 上面已用 st.error 顯示原因
            st.session_state.btn_label = "送出"
            status_area.caption("已取消送出（請修正欄位後再送出）")

        finally:
            st.session_state.processing = False
            st.rerun()
