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

# ======================================================
# 新增：投資計算引擎 (MVP Engine) － 不影響原有系統
# ======================================================

def _engine_prepare_symbol(df, symbol):
    try:
        df = df[df["symbol"] == symbol].copy()
    except Exception:
        return None

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "qty" in df.columns:
        df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0)

    if "amount_twd" in df.columns:
        df["amount_twd"] = pd.to_numeric(df["amount_twd"], errors="coerce").fillna(0.0)

    df = df.dropna(subset=["date"])
    df = df.sort_values("date")
    return df


def _engine_get_position_at(df, target_date):
    """
    計算 target_date 之前的持股狀態
    """
    df = df[df["date"] < target_date].copy()

    total_qty = 0.0
    total_cost = 0.0

    for _, row in df.iterrows():
        action = str(row.get("action", "")).strip().lower()
        qty = float(row.get("qty", 0) or 0)
        amount = float(row.get("amount_twd", 0) or 0)

        avg_cost = 0.0 if total_qty == 0 else total_cost / total_qty

        if action in ["initial", "buy"]:
            total_qty += qty
            total_cost += amount

        elif action == "sell":
            if total_qty < qty:
                return {
                    "is_error": True,
                    "message": "資料異常：sell 超過持股，請檢查 initial 或歷史交易紀錄"
                }

            sell_cost = qty * avg_cost
            total_qty -= qty
            total_cost -= sell_cost

        elif action == "dividend":
            # dividend 不改持股、不改成本
            pass

        if total_qty <= 1e-8:
            total_qty = 0.0
            total_cost = 0.0

    avg_cost = 0.0 if total_qty == 0 else total_cost / total_qty

    return {
        "is_error": False,
        "qty": total_qty,
        "cost": total_cost,
        "avg_cost": avg_cost
    }


def _engine_range_performance(df, start_date, end_date, price_end):
    """
    計算指定區間的績效
    """
    initial_state = _engine_get_position_at(df, start_date)

    if initial_state is None or initial_state.get("is_error"):
        return initial_state

    qty = float(initial_state["qty"])
    cost = float(initial_state["cost"])

    realized_profit = 0.0
    dividend_income = 0.0
    buy_total = 0.0
    sell_total = 0.0

    df_range = df[
        (df["date"] >= start_date) &
        (df["date"] <= end_date)
    ].copy()

    for _, row in df_range.iterrows():
        action = str(row.get("action", "")).strip().lower()
        q = float(row.get("qty", 0) or 0)
        amount = float(row.get("amount_twd", 0) or 0)

        avg_cost = 0.0 if qty == 0 else cost / qty

        if action in ["buy", "initial"]:
            qty += q
            cost += amount
            buy_total += amount

        elif action == "sell":
            if qty < q:
                return {
                    "is_error": True,
                    "message": "資料異常：sell 超過持股，請檢查 initial 或歷史交易紀錄"
                }

            sell_cost = q * avg_cost
            qty -= q
            cost -= sell_cost

            realized_profit += amount - sell_cost
            sell_total += amount

        elif action == "dividend":
            dividend_income += amount

        if qty <= 1e-8:
            qty = 0.0
            cost = 0.0

    market_value = qty * float(price_end or 0)
    unrealized_profit = market_value - cost
    range_profit = realized_profit + unrealized_profit + dividend_income

    denominator = float(initial_state["cost"]) + buy_total
    roi = None if denominator <= 0 else range_profit / denominator

    return {
        "is_error": False,
        "initial_qty": float(initial_state["qty"]),
        "initial_cost": float(initial_state["cost"]),
        "ending_qty": qty,
        "ending_cost": cost,
        "market_value": market_value,
        "realized_profit": realized_profit,
        "unrealized_profit": unrealized_profit,
        "dividend": dividend_income,
        "buy_total": buy_total,
        "sell_total": sell_total,
        "range_profit": range_profit,
        "roi": roi
    }


def _engine_find_latest_price(symbol, assets_df, symbol_col="symbol", price_col="price"):
    """
    從 assets 主檔抓目前價格
    """
    try:
        row = assets_df[assets_df[symbol_col] == symbol]
        if row.empty:
            return 0.0
        price = pd.to_numeric(row.iloc[0][price_col], errors="coerce")
        return 0.0 if pd.isna(price) else float(price)
    except Exception:
        return 0.0


def _engine_find_strategy(symbol, assets_df, symbol_col="symbol", strategy_col="strategy"):
    """
    從 assets 主檔抓策略
    """
    try:
        row = assets_df[assets_df[symbol_col] == symbol]
        if row.empty:
            return "未分類"
        strategy = str(row.iloc[0].get(strategy_col, "未分類")).strip()
        return strategy if strategy else "未分類"
    except Exception:
        return "未分類"


def _engine_metric_card(title, value, subtitle=None):
    sub_html = f'<div class="kpi-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True
    )


# ======================================================
# 新增：投資績效搜尋 (MVP) － 掛在原系統下方
# ======================================================

st.divider()
st.markdown("## 投資績效搜尋（MVP）")
st.caption("此區塊不會改動原本 Dashboard，只是在 0309app.py 下方追加新的查詢功能。")

try:
    # enriched_tx 與 assets_df 依 0309app.py 現有變數命名為準
    _search_tx = None
    _search_assets = None

    if "enriched_tx" in globals():
        _search_tx = enriched_tx.copy()
    elif "tx_df" in globals():
        _search_tx = tx_df.copy()
    elif "transactions_df" in globals():
        _search_tx = transactions_df.copy()

    if "assets_df" in globals():
        _search_assets = assets_df.copy()
    elif "assets" in globals():
        _search_assets = assets.copy()

    if _search_tx is None or _search_assets is None:
        st.info("尚未偵測到交易資料或資產主檔，請先確認 Google Sheet 載入成功。")
    else:
        # 欄位兼容處理
        tx_col_map = {}
        asset_col_map = {}

        # transactions 常見欄位
        tx_columns_lower = {c.lower(): c for c in _search_tx.columns}
        asset_columns_lower = {c.lower(): c for c in _search_assets.columns}

        tx_symbol_col = tx_columns_lower.get("symbol", "symbol")
        tx_date_col = tx_columns_lower.get("date", "date")
        tx_action_col = tx_columns_lower.get("action", "action")
        tx_qty_col = tx_columns_lower.get("qty", "qty")
        tx_amount_col = tx_columns_lower.get("amount_twd", "amount_twd")

        asset_symbol_col = asset_columns_lower.get("symbol", "symbol")
        asset_price_col = asset_columns_lower.get("price", "price")
        asset_strategy_col = asset_columns_lower.get("strategy", "strategy")
        asset_name_col = asset_columns_lower.get("name", asset_columns_lower.get("asset_name", asset_symbol_col))

        # 建立 engine 專用欄位
        _engine_tx = _search_tx.copy()

        if tx_symbol_col != "symbol":
            _engine_tx["symbol"] = _engine_tx[tx_symbol_col]
        if tx_date_col != "date":
            _engine_tx["date"] = _engine_tx[tx_date_col]
        if tx_action_col != "action":
            _engine_tx["action"] = _engine_tx[tx_action_col]
        if tx_qty_col != "qty":
            _engine_tx["qty"] = _engine_tx[tx_qty_col]
        if tx_amount_col != "amount_twd":
            _engine_tx["amount_twd"] = _engine_tx[tx_amount_col]

        # symbol list
        symbols = sorted([str(s) for s in _engine_tx["symbol"].dropna().unique().tolist()])

        mode = st.radio(
            "查詢模式",
            ["單一標的", "日期區間"],
            horizontal=True
        )

        if mode == "單一標的":
            col_a, col_b, col_c = st.columns([2, 1, 1])

            with col_a:
                selected_symbol = st.selectbox("選擇標的", symbols, key="mvp_symbol_single")

            with col_b:
                start_date = st.date_input("開始日期", value=date.today().replace(month=1, day=1), key="mvp_single_start")

            with col_c:
                end_date = st.date_input("結束日期", value=date.today(), key="mvp_single_end")

            if st.button("查詢單一標的", key="mvp_btn_single", use_container_width=True):
                df_symbol = _engine_prepare_symbol(_engine_tx, selected_symbol)

                if df_symbol is None or df_symbol.empty:
                    st.warning("這個標的沒有交易資料。")
                else:
                    latest_price = _engine_find_latest_price(
                        selected_symbol,
                        _search_assets,
                        symbol_col=asset_symbol_col,
                        price_col=asset_price_col
                    )

                    strategy = _engine_find_strategy(
                        selected_symbol,
                        _search_assets,
                        symbol_col=asset_symbol_col,
                        strategy_col=asset_strategy_col
                    )

                    result = _engine_range_performance(
                        df_symbol,
                        pd.to_datetime(start_date),
                        pd.to_datetime(end_date),
                        latest_price
                    )

                    try:
                        row = _search_assets[_search_assets[asset_symbol_col] == selected_symbol]
                        asset_name = str(row.iloc[0][asset_name_col]) if not row.empty else selected_symbol
                    except Exception:
                        asset_name = selected_symbol

                    st.markdown(f"### {asset_name}（{selected_symbol}）")
                    st.caption(f"策略：{strategy}")

                    if result is None or result.get("is_error"):
                        st.warning(
                            result.get("message", "資料異常，請檢查該標的是否漏填 initial 交易")
                            if isinstance(result, dict) else
                            "資料異常，請檢查該標的是否漏填 initial 交易"
                        )
                    else:
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            _engine_metric_card("期末持股", f'{result["ending_qty"]:.4f}'.rstrip("0").rstrip("."))
                        with c2:
                            _engine_metric_card("期末成本", f'{result["ending_cost"]:,.0f}')
                        with c3:
                            _engine_metric_card("市值", f'{result["market_value"]:,.0f}')

                        c4, c5, c6 = st.columns(3)
                        with c4:
                            _engine_metric_card("已實現損益", f'{result["realized_profit"]:,.0f}')
                        with c5:
                            _engine_metric_card("未實現損益", f'{result["unrealized_profit"]:,.0f}')
                        with c6:
                            _engine_metric_card("區間損益", f'{result["range_profit"]:,.0f}')

                        roi_text = "N/A" if result["roi"] is None else f'{result["roi"]*100:.2f}%'
                        c7, c8, c9 = st.columns(3)
                        with c7:
                            _engine_metric_card("ROI", roi_text)
                        with c8:
                            _engine_metric_card("區間買進", f'{result["buy_total"]:,.0f}')
                        with c9:
                            _engine_metric_card("區間配息", f'{result["dividend"]:,.0f}')

        else:
            col_a, col_b = st.columns(2)
            with col_a:
                range_start = st.date_input("開始日期", value=date.today().replace(month=1, day=1), key="mvp_range_start")
            with col_b:
                range_end = st.date_input("結束日期", value=date.today(), key="mvp_range_end")

            if st.button("查詢日期區間", key="mvp_btn_range", use_container_width=True):
                # 先依 assets.strategy 分 symbol
                _asset_view = _search_assets.copy()
                if asset_symbol_col != "symbol":
                    _asset_view["symbol"] = _asset_view[asset_symbol_col]
                if asset_strategy_col != "strategy":
                    _asset_view["strategy"] = _asset_view[asset_strategy_col]
                if asset_price_col != "price":
                    _asset_view["price"] = _asset_view[asset_price_col]

                swing_symbols = _asset_view[_asset_view["strategy"].astype(str).str.contains("波段|swing", case=False, na=False)]["symbol"].dropna().tolist()
                long_symbols = _asset_view[_asset_view["strategy"].astype(str).str.contains("存股|long", case=False, na=False)]["symbol"].dropna().tolist()

                def _sum_strategy(symbol_list):
                    summary = {
                        "count": 0,
                        "ending_cost": 0.0,
                        "market_value": 0.0,
                        "realized_profit": 0.0,
                        "unrealized_profit": 0.0,
                        "dividend": 0.0,
                        "range_profit": 0.0,
                        "buy_total": 0.0,
                        "error_symbols": []
                    }

                    for sym in symbol_list:
                        df_sym = _engine_prepare_symbol(_engine_tx, sym)
                        if df_sym is None or df_sym.empty:
                            continue

                        px = _engine_find_latest_price(sym, _search_assets, symbol_col=asset_symbol_col, price_col=asset_price_col)
                        res = _engine_range_performance(
                            df_sym,
                            pd.to_datetime(range_start),
                            pd.to_datetime(range_end),
                            px
                        )

                        if res is None or res.get("is_error"):
                            summary["error_symbols"].append(sym)
                            continue

                        summary["count"] += 1
                        summary["ending_cost"] += float(res["ending_cost"])
                        summary["market_value"] += float(res["market_value"])
                        summary["realized_profit"] += float(res["realized_profit"])
                        summary["unrealized_profit"] += float(res["unrealized_profit"])
                        summary["dividend"] += float(res["dividend"])
                        summary["range_profit"] += float(res["range_profit"])
                        summary["buy_total"] += float(res["buy_total"])

                    denominator = summary["ending_cost"] + summary["buy_total"]
                    summary["roi"] = None if denominator <= 0 else summary["range_profit"] / denominator
                    return summary

                swing_res = _sum_strategy(swing_symbols)
                long_res = _sum_strategy(long_symbols)

                st.markdown("### 日期區間結果")
                st.caption("系統會先回推查詢起始日前的歷史交易，建立期初剩餘持股與期初成本，再計算本區間結果。")

                c_left, c_right = st.columns(2)

                with c_left:
                    st.markdown("#### 波段")
                    _engine_metric_card("涉及標的數", swing_res["count"])
                    _engine_metric_card("區間損益", f'{swing_res["range_profit"]:,.0f}')
                    _engine_metric_card("已實現損益", f'{swing_res["realized_profit"]:,.0f}')
                    _engine_metric_card("未實現損益", f'{swing_res["unrealized_profit"]:,.0f}')
                    _engine_metric_card("區間配息", f'{swing_res["dividend"]:,.0f}')
                    _engine_metric_card("ROI", "N/A" if swing_res["roi"] is None else f'{swing_res["roi"]*100:.2f}%')

                    if swing_res["error_symbols"]:
                        st.warning("波段資料異常標的：" + "、".join(swing_res["error_symbols"]))

                with c_right:
                    st.markdown("#### 存股")
                    _engine_metric_card("涉及標的數", long_res["count"])
                    _engine_metric_card("區間損益", f'{long_res["range_profit"]:,.0f}')
                    _engine_metric_card("已實現損益", f'{long_res["realized_profit"]:,.0f}')
                    _engine_metric_card("未實現損益", f'{long_res["unrealized_profit"]:,.0f}')
                    _engine_metric_card("區間配息", f'{long_res["dividend"]:,.0f}')
                    _engine_metric_card("ROI", "N/A" if long_res["roi"] is None else f'{long_res["roi"]*100:.2f}%')

                    if long_res["error_symbols"]:
                        st.warning("存股資料異常標的：" + "、".join(long_res["error_symbols"]))

except Exception as e:
    st.warning(f"投資績效搜尋模組初始化失敗：{e}")
