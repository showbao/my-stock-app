import streamlit as st
import pandas as pd
from datetime import datetime

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="My Stock App",
    layout="wide"
)

st.title("📊 My Investment Dashboard")

# -----------------------------
# MOCK DATA (若你已有 Google Sheet 載入可替換)
# -----------------------------

if "transactions_df" not in st.session_state:

    data = [
        ["2024-01-01","initial","AAPL",10,150,45000],
        ["2024-02-01","buy","AAPL",5,200,30000],
        ["2024-03-01","sell","AAPL",10,250,75000],
        ["2024-04-01","dividend","AAPL",0,0,2000],
    ]

    st.session_state.transactions_df = pd.DataFrame(
        data,
        columns=["date","action","symbol","qty","price","amount_twd"]
    )

transactions_df = st.session_state.transactions_df.copy()
transactions_df["date"] = pd.to_datetime(transactions_df["date"])

# -----------------------------
# 投資計算引擎
# -----------------------------

def get_position_at(df, target_date, symbol):

    df = df[df["symbol"] == symbol]
    df = df[df["date"] < target_date]
    df = df.sort_values("date")

    total_qty = 0
    total_cost = 0

    for _,row in df.iterrows():

        action = row["action"]
        qty = row["qty"]
        amount = row["amount_twd"]

        if action in ["buy","initial"]:
            total_qty += qty
            total_cost += amount

        elif action == "sell":

            if total_qty == 0:
                continue

            avg_cost = total_cost / total_qty
            sell_cost = qty * avg_cost

            total_qty -= qty
            total_cost -= sell_cost

    avg_cost = 0
    if total_qty > 0:
        avg_cost = total_cost / total_qty

    return {
        "qty": total_qty,
        "cost": total_cost,
        "avg_cost": avg_cost
    }


def calculate_range_performance(df, symbol, start_date, end_date):

    initial_state = get_position_at(df, start_date, symbol)

    initial_qty = initial_state["qty"]
    initial_cost = initial_state["cost"]

    range_df = df[
        (df["symbol"] == symbol) &
        (df["date"] >= start_date) &
        (df["date"] <= end_date)
    ].sort_values("date")

    buy_amount = 0
    sell_amount = 0
    dividend_amount = 0

    current_qty = initial_qty
    current_cost = initial_cost

    for _,row in range_df.iterrows():

        action = row["action"]
        qty = row["qty"]
        amount = row["amount_twd"]

        if action == "buy":
            current_qty += qty
            current_cost += amount
            buy_amount += amount

        elif action == "sell":

            if current_qty > 0:

                avg_cost = current_cost / current_qty
                sell_cost = qty * avg_cost

                current_qty -= qty
                current_cost -= sell_cost

            sell_amount += amount

        elif action == "dividend":
            dividend_amount += amount

    market_price = 200
    market_value = current_qty * market_price

    range_profit = (
        (market_value - initial_cost)
        + sell_amount
        - buy_amount
        + dividend_amount
    )

    base = initial_cost + buy_amount

    roi = 0
    if base != 0:
        roi = range_profit / base

    return {
        "ending_qty": current_qty,
        "ending_cost": current_cost,
        "market_value": market_value,
        "range_profit": range_profit,
        "roi": roi
    }

# -----------------------------
# TABS
# -----------------------------

tab1, tab2 = st.tabs([
    "📈 Dashboard",
    "🔎 投資搜尋"
])

# -----------------------------
# DASHBOARD
# -----------------------------

with tab1:

    st.subheader("交易紀錄")

    st.dataframe(transactions_df)

# -----------------------------
# SEARCH PAGE
# -----------------------------

with tab2:

    st.subheader("投資績效搜尋")

    symbols = transactions_df["symbol"].unique()

    symbol = st.selectbox("選擇標的", symbols)

    col1,col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "開始日期",
            value=datetime(2024,1,1)
        )

    with col2:
        end_date = st.date_input(
            "結束日期",
            value=datetime.today()
        )

    if st.button("計算績效"):

        result = calculate_range_performance(
            transactions_df,
            symbol,
            pd.to_datetime(start_date),
            pd.to_datetime(end_date)
        )

        st.divider()

        col1,col2,col3,col4 = st.columns(4)

        col1.metric(
            "期末持股",
            result["ending_qty"]
        )

        col2.metric(
            "市值",
            f"{result['market_value']:,.0f}"
        )

        col3.metric(
            "區間損益",
            f"{result['range_profit']:,.0f}"
        )

        col4.metric(
            "ROI",
            f"{result['roi']*100:.2f}%"
        )

# -----------------------------
# 新增交易 (示例)
# -----------------------------

st.divider()
st.subheader("新增交易")

qty = st.number_input("數量",value=1)
price = st.number_input("價格",value=100)
fx_rate = st.number_input("匯率",value=1.0)

amount_twd = st.number_input("金額 (TWD)",value=100)

try:

    hint = qty * price * fx_rate

    st.caption(
        f"💡 參考試算：{qty} × {price} × {fx_rate} = {hint:,.2f}"
    )

except:
    pass
