import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import json
import re
from datetime import datetime

st.set_page_config(page_title="投資MVP", layout="wide")

# =========================
# 連線 Google Sheet
# =========================

@st.cache_resource
def connect_sheet():
    # ✅ 你的 Secrets 是拆欄位：[gcp_service_account]
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
    st.title("投資記錄 MVP")
    st.write("請使用 Google 登入")
    if st.button("用 Google 登入"):
        st.login()

if not st.user.is_logged_in:
    login_screen()
    st.stop()

user_email = (st.user.email or "").strip().lower()

allowed_emails = set(
    allowed_df.get("email", pd.Series(dtype=str))
    .astype(str)
    .str.strip()
    .str.lower()
    .tolist()
)

if user_email not in allowed_emails:
    st.error("無權限")
    st.button("登出", on_click=st.logout)
    st.stop()

st.sidebar.success(f"登入：{user_email}")
st.sidebar.button("登出", on_click=st.logout)

# =========================
# 工具函式
# =========================

def clean_symbol(symbol, asset_type):
    symbol = symbol.strip().upper()
    symbol = re.sub(r"[^A-Z0-9_]", "", symbol)
    if asset_type == "fund" and not symbol.startswith("F_"):
        symbol = "F_" + symbol
    return symbol

def calculate_metrics(df, prices_df):
    if df.empty:
        return 0, 0, 0, 0, 0

    result = {}

    for _, row in df.iterrows():
        symbol = row["symbol"]
        action = row["action"]
        qty = float(row["qty"])
        amount_twd = float(row["amount_twd"])

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

    total_invest = 0
    total_value = 0
    total_dividend = 0

    for symbol, data in result.items():
        if data["qty"] < 0:
            return None

        price_row = prices_df[prices_df["symbol"] == symbol]
        if not price_row.empty:
            price = float(price_row.iloc[0]["price"])
            currency = price_row.iloc[0]["currency"]
            value = data["qty"] * price

            if currency == "USD":
                fx_row = prices_df[prices_df["symbol"] == "USD_TWD"]
                fx = float(fx_row.iloc[0]["price"])
                value *= fx
        else:
            value = 0

        total_invest += data["cost"]
        total_value += value
        total_dividend += data["dividend"]

    total_profit = total_value + total_dividend - total_invest
    rate = (total_profit / total_invest * 100) if total_invest != 0 else 0

    return total_invest, total_value, total_dividend, total_profit, rate

# =========================
# 頁面
# =========================

page = st.sidebar.radio("選單", ["首頁", "新增交易"])

# =========================
# 首頁
# =========================

if page == "首頁":

    tab_all, tab_stock, tab_fund = st.tabs(["全部", "股票", "基金"])

    stock_df = transactions_df[transactions_df["asset_type"] == "stock"]
    fund_df = transactions_df[transactions_df["asset_type"] == "fund"]
    all_df = transactions_df[
        transactions_df["asset_type"].isin(["stock", "fund"])
    ]

    all_metrics = calculate_metrics(all_df, prices_df)
    stock_metrics = calculate_metrics(stock_df, prices_df)
    fund_metrics = calculate_metrics(fund_df, prices_df)

    def show_metrics(metrics):
        if metrics is None:
            st.error("資料異常：出現負持股")
            st.write("總投入：—")
            st.write("目前市值：—")
            st.write("已領息：—")
            st.write("總報酬：—")
            st.write("總報酬率：—")
        else:
            invest, value, divi, profit, rate = metrics
            st.write(f"總投入：{round(invest,2)}")
            st.write(f"目前市值：{round(value,2)}")
            st.write(f"已領息：{round(divi,2)}")
            st.write(f"總報酬：{round(profit,2)}")
            st.write(f"總報酬率：{round(rate,2)}%")

    with tab_all:
        show_metrics(all_metrics)

    with tab_stock:
        show_metrics(stock_metrics)

    with tab_fund:
        show_metrics(fund_metrics)

# =========================
# 新增交易
# =========================

if page == "新增交易":

    action = st.selectbox("交易類型", ["buy", "sell", "dividend", "initial"])
    asset_type = st.selectbox("資產類型", ["stock", "fund"])
    symbol = st.text_input("代號")
    strategy = st.text_input("策略")
    currency = st.selectbox("幣別", ["TWD", "USD"])
    qty = st.number_input("數量", min_value=0.0)
    price = st.number_input("單價", min_value=0.0)
    tx_date = st.date_input("日期")

    fx_rate = st.number_input("匯率", min_value=0.0) if currency == "USD" else 1.0

    if st.button("送出"):

        symbol = clean_symbol(symbol, asset_type)

        if action == "initial":

            amount_twd = qty * price
            if qty <= 0:
                st.error("initial 數量必須大於 0")
                st.stop()

            # initial 日期檢查
            same = transactions_df[
                transactions_df["symbol"].astype(str).str.upper() == symbol
            ]
            if not same.empty:
                same_dates = pd.to_datetime(
                    same["date"], errors="coerce"
                ).dropna().dt.date

                if any(d <= tx_date for d in same_dates):
                    st.error("initial 日期必須早於其他交易")
                    st.stop()

        else:
            if currency == "USD" and fx_rate == 0:
                st.error("USD 必須填匯率")
                st.stop()

            amount_original = qty * price
            amount_twd = amount_original * fx_rate

            if abs(qty * price * fx_rate - amount_twd) > 1:
                st.error("金額驗證錯誤")
                st.stop()

        new_row = [
            str(datetime.now().timestamp()),
            str(tx_date),
            action,
            asset_type,
            symbol,
            strategy,
            currency,
            fx_rate,
            qty,
            price,
            qty * price,
            amount_twd
        ]

        sheet.worksheet("transactions").append_row(new_row)
        st.success("新增成功")
        st.rerun()
