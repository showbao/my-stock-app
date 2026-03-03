import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import re
from datetime import datetime, date

# =========================
# 頁面設定（柔和淺灰專業風）
# =========================
st.set_page_config(page_title="投資記錄｜專業版", layout="wide")

SOFT_PRO_CSS = """
<style>
/* 整體留白 */
.block-container { padding-top: 1.1rem; padding-bottom: 2.0rem; }

/* 背景（柔和淺灰） */
html, body, [data-testid="stAppViewContainer"] {
  background: #F5F7FB;
}

/* Sidebar 乾淨 */
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] { padding-top: 0.5rem; }
section[data-testid="stSidebar"] .block-container { padding-top: 1.0rem; }

/* Tabs 更好按（手機） */
button[data-baseweb="tab"] {
  padding-top: 10px !important;
  padding-bottom: 10px !important;
  font-size: 16px !important;
}

/* 標題區（柔和） */
.soft-hero {
  padding: 14px 16px;
  border-radius: 16px;
  background: #FFFFFF;
  border: 1px solid rgba(15,23,42,0.10);
  box-shadow: 0 1px 14px rgba(15,23,42,0.06);
  margin-bottom: 10px;
}
.soft-hero-title { font-size: 13px; color: #6B7280; margin-bottom: 4px; }
.soft-hero-value { font-size: 32px; font-weight: 750; color: #111827; line-height: 1.1; }
.soft-hero-sub { font-size: 12px; color: #9CA3AF; margin-top: 6px; }

/* KPI 卡片（白卡 + 柔陰影） */
.kpi-card {
  padding: 12px 14px;
  border-radius: 16px;
  background: #FFFFFF;
  border: 1px solid rgba(15,23,42,0.10);
  box-shadow: 0 1px 14px rgba(15,23,42,0.06);
  margin-bottom: 10px;
}
.kpi-title { font-size: 13px; color: #6B7280; margin-bottom: 6px; }
.kpi-value { font-size: 26px; font-weight: 750; line-height: 1.15; color: #111827; }
.kpi-sub { font-size: 12px; color: #9CA3AF; margin-top: 4px; }

/* 小灰字 */
.small-muted { color: #6B7280; font-size: 12px; }

/* 表單卡片 */
.form-card {
  padding: 14px 14px;
  border-radius: 16px;
  background: #FFFFFF;
  border: 1px solid rgba(15,23,42,0.10);
  box-shadow: 0 1px 14px rgba(15,23,42,0.06);
}

/* 按鈕更像 App */
div.stButton > button {
  border-radius: 14px;
  padding: 0.75rem 1rem;
}

/* Alert 圓角 */
div[data-testid="stAlert"] { border-radius: 14px; }

/* 輸入欄位垂直間距縮一點 */
div[data-testid="stVerticalBlock"] > div { margin-bottom: 0.35rem; }
</style>
"""
st.markdown(SOFT_PRO_CSS, unsafe_allow_html=True)

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
    st.markdown("## 投資記錄")
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
# 工具函式（保留核心規則）
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

# =========================
# 顯示用（柔和專業）
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

def hero_block(title, value_text, sub_text=""):
    st.markdown(
        f"""
        <div class="soft-hero">
          <div class="soft-hero-title">{title}</div>
          <div class="soft-hero-value">{value_text}</div>
          {f'<div class="soft-hero-sub">{sub_text}</div>' if sub_text else ''}
        </div>
        """,
        unsafe_allow_html=True
    )

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
# 頁面
# =========================
page = st.sidebar.radio("選單", ["首頁", "新增交易"])

# =========================
# 首頁（柔和淺灰專業風）
# =========================
if page == "首頁":
    st.markdown("## 投資總覽")

    fx, fx_updated = get_usd_twd_info(prices_df)
    prices_updated = get_prices_updated_at(prices_df)

    tab_all, tab_stock, tab_fund = st.tabs(["全部", "股票", "基金"])

    stock_df = transactions_df[transactions_df["asset_type"] == "stock"]
    fund_df = transactions_df[transactions_df["asset_type"] == "fund"]
    all_df = transactions_df[transactions_df["asset_type"].isin(["stock", "fund"])]

    all_metrics = calculate_metrics(all_df, prices_df)
    stock_metrics = calculate_metrics(stock_df, prices_df)
    fund_metrics = calculate_metrics(fund_df, prices_df)

    def render_metrics(metrics, label):
        if metrics is None:
            st.error("資料異常：出現負持股")
            hero_block(f"{label}｜目前市值（TWD）", "—", "請先確認交易資料是否正確")
            kpi_card("總投入", "—")
            kpi_card("已領息", "—")
            kpi_card("總報酬", "—")
            kpi_card("總報酬率", "—")
            return

        invest, value, divi, profit, rate = metrics

        # 上方總覽（市值）
        info_parts = []
        if prices_updated:
            info_parts.append(f"價格更新：{prices_updated}")
        if fx is not None:
            fx_part = f"USD_TWD：{fx:.4f}"
            if fx_updated:
                fx_part += f"（{fx_updated}）"
            info_parts.append(fx_part)

        hero_sub = " ｜ ".join(info_parts) if info_parts else ""
        hero_block(f"{label}｜目前市值（TWD）", fmt_money(value), hero_sub)

        # KPI 卡片（單欄，手機穩）
        kpi_card("總投入", fmt_money(invest))
        kpi_card("已領息", fmt_money(divi))
        kpi_card("總報酬", fmt_signed_money(profit))
        kpi_card("總報酬率", fmt_signed_pct(rate))

    with tab_all:
        render_metrics(all_metrics, "全部")

    with tab_stock:
        render_metrics(stock_metrics, "股票")

    with tab_fund:
        render_metrics(fund_metrics, "基金")

# =========================
# 新增交易（柔和淺灰專業風 + 防連點 + 回彈 + 局部清空）
# =========================
if page == "新增交易":
    st.markdown("## 新增交易")
    st.caption("送出時按鈕會鎖住，避免重複寫入。")

    st.markdown('<div class="form-card">', unsafe_allow_html=True)

    # --- 狀態初始化 ---
    st.session_state.setdefault("processing", False)
    st.session_state.setdefault("btn_label", "送出")

    # key：用於局部清空
    st.session_state.setdefault("in_action", "buy")
    st.session_state.setdefault("in_asset_type", "stock")
    st.session_state.setdefault("in_symbol", "")
    st.session_state.setdefault("in_strategy", "未分類")
    st.session_state.setdefault("in_currency", "TWD")
    st.session_state.setdefault("in_qty", 0.0)
    st.session_state.setdefault("in_price", 0.0)
    st.session_state.setdefault("in_date", date.today())
    st.session_state.setdefault("in_fx_rate", 1.0)

    def mark_dirty():
        # 任一輸入變更：按鈕立刻回到「送出」
        st.session_state.btn_label = "送出"

    # 欄位順序（手機順）
    action = st.selectbox("交易類型", ["buy", "sell", "dividend", "initial"], key="in_action", on_change=mark_dirty)
    asset_type = st.selectbox("資產類型", ["stock", "fund"], key="in_asset_type", on_change=mark_dirty)
    symbol_input = st.text_input("代號", key="in_symbol", on_change=mark_dirty)
    strategy = st.selectbox("策略", ["未分類", "長期", "波段", "定期定額"], key="in_strategy", on_change=mark_dirty)
    tx_date = st.date_input("日期", key="in_date", on_change=mark_dirty)
    currency = st.selectbox("幣別", ["TWD", "USD"], key="in_currency", on_change=mark_dirty)

    qty = st.number_input("數量", min_value=0.0, key="in_qty", on_change=mark_dirty)
    price = st.number_input("單價", min_value=0.0, key="in_price", on_change=mark_dirty)

    # fx_rate：TWD disabled + 預填 1.0
    if currency == "TWD":
        st.session_state.in_fx_rate = 1.0
        fx_rate = st.number_input("匯率（USD_TWD）", value=1.0, disabled=True)
    else:
        fx_rate = st.number_input("匯率（USD_TWD）", min_value=0.0, key="in_fx_rate", on_change=mark_dirty)

    st.markdown("<div class='small-muted'>TWD 會自動使用 1.0；USD 請填 USD_TWD 匯率。</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)  # form-card end

    st.divider()

    # --- 按鈕（防連點）---
    btn_text = "🔄 寫入中..." if st.session_state.processing else st.session_state.btn_label
    clicked = st.button(btn_text, disabled=st.session_state.processing, use_container_width=True)
    status_area = st.empty()

    if clicked:
        st.session_state.processing = True
        st.session_state.btn_label = "🔄 寫入中..."
        st.rerun()

    # 真正寫入（processing=True 時）
    if st.session_state.processing:
        try:
            symbol = clean_symbol(symbol_input, asset_type)

            if symbol == "":
                st.error("代號不可空白")
                raise ValueError("代號不可空白")

            # USD 必須填匯率（含 initial）
            if currency == "USD" and to_number(fx_rate) == 0:
                st.error("USD 必須填匯率")
                raise ValueError("USD 必須填匯率")

            fx_used = to_number(fx_rate) if currency == "USD" else 1.0

            amount_original = to_number(qty) * to_number(price)
            amount_twd = amount_original * fx_used

            # initial 日期規則
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

                    # 新增 initial：日期晚於或等於任何既有日期 -> 擋下
                    if any(d <= tx_date for d in same_dates.tolist()):
                        st.error("initial 日期必須早於此資產的其他交易日期")
                        raise ValueError("initial 日期不合法")

            else:
                # 金額驗證容差 ±1
                if abs(to_number(qty) * to_number(price) * fx_used - amount_twd) > 1:
                    st.error("金額驗證錯誤")
                    raise ValueError("金額驗證錯誤")

                # sell 超過持有擋下
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

            # 局部清空：只清 symbol/qty/price/（保留 date/asset_type/strategy/currency）
            st.session_state.in_symbol = ""
            st.session_state.in_qty = 0.0
            st.session_state.in_price = 0.0

        except Exception:
            st.session_state.btn_label = "送出"
            status_area.caption("已取消送出（請修正欄位後再送出）")

        finally:
            st.session_state.processing = False
            st.rerun()
