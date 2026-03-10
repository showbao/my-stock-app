import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Investment System", layout="wide")

st.title("投資系統 MVP")

# -----------------------------
# Mock data loader
# -----------------------------

@st.cache_data
def load_data():

    try:
        transactions = pd.read_csv("transactions.csv")
        assets = pd.read_csv("assets.csv")
    except:

        transactions = pd.DataFrame([
            ["2024-01-01","initial","AAPL",10,150,45000],
            ["2024-02-01","buy","AAPL",5,200,30000],
            ["2024-03-01","sell","AAPL",10,250,75000],
        ], columns=[
            "date","action","symbol","qty","price","amount_twd"
        ])

        assets = pd.DataFrame([
            ["AAPL",250,"swing","stock"]
        ], columns=[
            "symbol","price","strategy","type"
        ])

    transactions["date"] = pd.to_datetime(transactions["date"])

    return transactions, assets


transactions_df, assets_df = load_data()

# -----------------------------
# Investment Engine
# -----------------------------

def prepare_symbol_transactions(df, symbol):

    df = df[df["symbol"] == symbol].copy()

    df["qty"] = df["qty"].fillna(0).astype(float)
    df["amount_twd"] = df["amount_twd"].fillna(0).astype(float)

    df = df.sort_values("date")

    return df


def get_position_at(df, target_date):

    df = df[df["date"] < target_date]

    total_qty = 0.0
    total_cost = 0.0

    for _, row in df.iterrows():

        action = row["action"]
        qty = row["qty"]
        amount = row["amount_twd"]

        avg_cost = 0 if total_qty == 0 else total_cost / total_qty

        if action in ["initial","buy"]:

            total_qty += qty
            total_cost += amount

        elif action == "sell":

            if total_qty < qty:
                st.warning("資料異常：sell 超過持股")
                return None

            sell_cost = qty * avg_cost

            total_qty -= qty
            total_cost -= sell_cost

        elif action == "dividend":
            pass

        if total_qty <= 1e-8:
            total_qty = 0
            total_cost = 0

    avg_cost = 0 if total_qty == 0 else total_cost / total_qty

    return {
        "qty": total_qty,
        "cost": total_cost,
        "avg_cost": avg_cost
    }


def calculate_range_performance(df,start_date,end_date,price_end):

    initial_state = get_position_at(df,start_date)

    if initial_state is None:
        return None

    qty = initial_state["qty"]
    cost = initial_state["cost"]

    realized_profit = 0
    dividend_income = 0
    buy_total = 0

    df_range = df[
        (df["date"] >= start_date) &
        (df["date"] <= end_date)
    ]

    for _, row in df_range.iterrows():

        action = row["action"]
        q = row["qty"]
        amount = row["amount_twd"]

        avg_cost = 0 if qty == 0 else cost / qty

        if action in ["buy","initial"]:

            qty += q
            cost += amount
            buy_total += amount

        elif action == "sell":

            if qty < q:
                st.warning("資料異常：sell 超過持股")
                return None

            sell_cost = q * avg_cost

            qty -= q
            cost -= sell_cost

            realized_profit += amount - sell_cost

        elif action == "dividend":

            dividend_income += amount

        if qty <= 1e-8:
            qty = 0
            cost = 0

    market_value = qty * price_end

    unrealized_profit = market_value - cost

    range_profit = (
        realized_profit +
        unrealized_profit +
        dividend_income
    )

    denominator = initial_state["cost"] + buy_total

    roi = None

    if denominator > 0:
        roi = range_profit / denominator

    return {
        "initial_qty":initial_state["qty"],
        "initial_cost":initial_state["cost"],
        "ending_qty":qty,
        "ending_cost":cost,
        "market_value":market_value,
        "realized_profit":realized_profit,
        "unrealized_profit":unrealized_profit,
        "dividend":dividend_income,
        "range_profit":range_profit,
        "roi":roi
    }


# -----------------------------
# Search UI
# -----------------------------

st.header("投資查詢")

symbol = st.selectbox(
    "選擇標的",
    assets_df["symbol"].unique()
)

col1, col2 = st.columns(2)

start_date = col1.date_input("開始日期")
end_date = col2.date_input("結束日期")

if st.button("計算績效"):

    df_symbol = prepare_symbol_transactions(
        transactions_df,
        symbol
    )

    price = assets_df.loc[
        assets_df["symbol"] == symbol,
        "price"
    ].iloc[0]

    result = calculate_range_performance(
        df_symbol,
        pd.to_datetime(start_date),
        pd.to_datetime(end_date),
        price
    )

    if result:

        st.subheader("區間績效")

        c1,c2,c3 = st.columns(3)

        c1.metric(
            "期末持股",
            result["ending_qty"]
        )

        c2.metric(
            "期末成本",
            round(result["ending_cost"],2)
        )

        c3.metric(
            "市值",
            round(result["market_value"],2)
        )

        st.metric(
            "區間損益",
            round(result["range_profit"],2)
        )

        if result["roi"] is not None:

            st.metric(
                "ROI",
                f"{result['roi']*100:.2f}%"
            )

        else:

            st.metric("ROI","N/A")

# -----------------------------
# Debug view
# -----------------------------

with st.expander("查看資料"):

    st.write("Transactions")
    st.dataframe(transactions_df)

    st.write("Assets")
    st.dataframe(assets_df)
