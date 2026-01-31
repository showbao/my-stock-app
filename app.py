# Version: v11.0 (Pure Dashboard - No AI)
# CTOSignature: Removed all Generative AI logic, API calls, and History tracking. Kept core metrics, charts (layered), inventory split, and single ticker view.
import streamlit as st
import pandas as pd
import yfinance as yf
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, date, timedelta
import numpy as np
from scipy import optimize
import altair as alt
import time

# ==========================================
# 1. 系統設定與連線
# ==========================================
st.set_page_config(page_title="投資戰情室 v11.0", layout="wide")

ws_records = None
ws_funds = None

@st.cache_resource
def connect_google_sheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    try:
        if "gcp_service_account" in st.secrets:
            creds_dict = st.secrets["gcp_service_account"]
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        else:
            creds = ServiceAccountCredentials.from_json_keyfile_name("secrets.json", scope)
            
        client = gspread.authorize(creds)
        sheet = client.open("Investment_Tracker")
        return sheet
    except Exception as e:
        return None 

sh = connect_google_sheet()

if sh:
    try: ws_records = sh.worksheet("Records")
    except: st.error("❌ 找不到工作表 'Records'"); st.stop()
    try: ws_funds = sh.worksheet("Fund_Updates")
    except: st.error("❌ 找不到工作表 'Fund_Updates'"); st.stop()
else:
    st.error("❌ Google Sheet 連線失敗"); st.stop()

# ==========================================
# 2. 核心邏輯函數 (Data & Math)
# ==========================================

@st.cache_data(ttl=3600) 
def get_usd_twd_rate():
    try:
        ticker = yf.Ticker("TWD=X")
        hist = ticker.history(period="1d")
        if not hist.empty: return hist['Close'].iloc[-1]
        return 32.0
    except: return 32.0

def fix_ticker_suffix(ticker):
    ticker = str(ticker).strip().upper()
    if not ticker.isdigit(): return ticker 
    try_tw = f"{ticker}.TW"
    if not yf.Ticker(try_tw).history(period="1d").empty: return try_tw
    try_two = f"{ticker}.TWO"
    if not yf.Ticker(try_two).history(period="1d").empty: return try_two
    return ticker 

@st.cache_data(ttl=600)
def get_stock_data(ticker):
    try:
        real_ticker = fix_ticker_suffix(ticker)
        stock = yf.Ticker(real_ticker)
        hist = stock.history(period='1mo', auto_adjust=True)
        if not hist.empty:
            return hist['Close'].iloc[-1], (np.log(hist['Close']/hist['Close'].shift(1)).std()*np.sqrt(252)*100 if len(hist)>1 else 0)
        return 0.0, 0.0
    except: return 0.0, 0.0

def normalize_data(df):
    if df.empty: return df
    act_map = {'Buy': '買入', 'Sell': '賣出', 'Dividend': '領息', 'Split': '分割', 'Buy (Buy)': '買入', 'Sell (Sell)': '賣出'}
    type_map = {'Stock': '股票', 'Fund': '基金'}
    if 'Action' in df.columns: df['Action'] = df['Action'].replace(act_map)
    if 'Type' in df.columns: df['Type'] = df['Type'].replace(type_map)
    return df

def load_data():
    try:
        records_data = ws_records.get_all_records()
        df = pd.DataFrame(records_data)
        df['RowIndex'] = range(2, len(df) + 2)
    except: return pd.DataFrame(), pd.DataFrame(), 32.0

    try:
        funds_data = ws_funds.get_all_records()
        df_funds = pd.DataFrame(funds_data)
        if not df_funds.empty:
            df_funds.columns = [c.strip() for c in df_funds.columns]
    except: df_funds = pd.DataFrame()
    
    if df.empty: return df, df_funds, 32.0
    
    numeric_cols = ['Price', 'Shares', 'Fee', 'Total_Amount']
    for col in numeric_cols:
        if df[col].dtype == object: df[col] = df[col].astype(str).str.replace(',','').str.replace('$','')
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df = normalize_data(df)
    # We still keep this to avoid KeyError if the column exists in Sheet
    if 'AI_Review' not in df.columns: df['AI_Review'] = ""
    current_usd_rate = get_usd_twd_rate()
    return df, df_funds, current_usd_rate

def xirr(transactions):
    if not transactions: return None
    dates = [t[0] for t in transactions]
    amounts = [t[1] for t in transactions]
    if min(amounts) >= 0 or max(amounts) <= 0: return None
    def xnpv(rate, amounts, dates):
        if rate <= -1.0: return float('inf')
        d0 = dates[0]
        return sum([a / (1.0 + rate)**((d - d0).days / 365.0) for a, d in zip(amounts, dates)])
    try: return optimize.newton(lambda r: xnpv(r, amounts, dates), 0.1)
    except: return None

def calculate_portfolio(df, df_funds, current_usd_rate):
    portfolio = {}
    trade_log = [] 
    df = df.sort_values('Date')
    for _, row in df.iterrows():
        ticker = row['Ticker']; action = row['Action']; qty = row['Shares']
        amount = abs(row['Total_Amount']) 
        date_txn = row['Date']
        typ = row['Type']; strat = str(row['Strategy'])
        rid = row.get('RowIndex', -1)
        review = row.get('AI_Review', '')

        if ticker not in portfolio:
            portfolio[ticker] = {'shares': 0, 'total_cost': 0, 'dividend_collected': 0, 'type': typ, 'strategy': strat}
        p = portfolio[ticker]; p['strategy'] = strat 

        if action == '買入':
            p['shares'] += qty; p['total_cost'] += amount
        elif action == '賣出':
            if p['shares'] > 0:
                pct_sold = qty / p['shares']
                cost_of_sold_shares = p['total_cost'] * pct_sold
                pnl = amount - cost_of_sold_shares 
                p['shares'] -= qty; p['total_cost'] -= cost_of_sold_shares
                sell_price = (amount/qty) if qty>0 else 0
                trade_log.append({
                    'Date': date_txn, 'Ticker': ticker, 'Strategy': p['strategy'], 'Type': p['type'], 
                    'PnL': pnl, 'SellAmount': amount, 'SellPrice': sell_price, 'CostBasis': cost_of_sold_shares,
                    'RowIndex': rid, 'AI_Review': review
                })
                if p['shares'] <= 0.001: p['shares'] = 0; p['total_cost'] = 0
        elif action == '領息': p['dividend_collected'] += amount
        elif action == '分割': p['shares'] += qty
            
    results = []
    for ticker, data in portfolio.items():
        current_price = 0; market_value = 0
        if data['shares'] > 0.001:
            if data['type'] == '基金':
                if not df_funds.empty and ticker in df_funds['Ticker'].values:
                    fund_row = df_funds[df_funds['Ticker'] == ticker].iloc[0]
                    price_col = 'Price' if 'Price' in df_funds.columns else 'Net_Value' if 'Net_Value' in df_funds.columns else df_funds.columns[1]
                    net_val = pd.to_numeric(fund_row[price_col], errors='coerce')
                    currency = 'USD'
                    if 'Currency' in df_funds.columns: currency = fund_row['Currency']
                    current_price = net_val if currency == 'TWD' else net_val * current_usd_rate
                if current_price == 0 or pd.isna(current_price): current_price, _ = get_stock_data(ticker)
            else:
                current_price, _ = get_stock_data(ticker)
            
            if current_price == 0 or pd.isna(current_price):
                current_price = data['total_cost'] / data['shares'] if data['shares'] > 0 else 0
                
            market_value = current_price * data['shares']
            unrealized_pl = market_value - data['total_cost']
            roi_total = ((unrealized_pl + data['dividend_collected']) / data['total_cost'] * 100) if data['total_cost'] > 0 else 0
            results.append({
                "代號": ticker, "種類": data['type'], "策略": data['strategy'], "庫存": data['shares'], "平均成本": round(data['total_cost'] / data['shares'], 2),
                "市價": round(current_price, 2), "庫存現值": round(market_value, 0), "帳面損益": round(unrealized_pl, 0),
                "已領股息": round(data['dividend_collected'], 0), "含息總報%": round(roi_total, 2), "總成本": round(data['total_cost'], 0),
                "成本殖利率%": ((data['dividend_collected']/data['total_cost']*100) if data['total_cost']>0 else 0)
            })
    pf_df = pd.DataFrame(results)
    if not pf_df.empty:
        total_mv = pf_df['庫存現值'].sum()
        pf_df['佔比%'] = (pf_df['庫存現值'] / total_mv * 100).round(1) if total_mv > 0 else 0.0
    return pf_df, pd.DataFrame(trade_log)

def analyze_period_advanced(df, start_date, end_date, selected_tickers, current_portfolio_df, trade_log_df, strategy_filter=None):
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    if selected_tickers: mask = mask & (df['Ticker'].isin(selected_tickers))
    if strategy_filter: mask = mask & (df['Strategy'].str.contains(strategy_filter, na=False))
    period_df = df[mask].copy()
    if period_df.empty: return None, pd.DataFrame(), pd.DataFrame()

    total_dividend = period_df[period_df['Action'] == '領息']['Total_Amount'].abs().sum()
    total_buy = period_df[period_df['Action'] == '買入']['Total_Amount'].abs().sum()
    
    ending_inventory_value = 0; total_cost_basis = 0
    if end_date >= datetime.now().date() and not current_portfolio_df.empty:
        target_inv = current_portfolio_df
        if selected_tickers: target_inv = target_inv[target_inv['代號'].isin(selected_tickers)]
        if strategy_filter: target_inv = target_inv[target_inv['策略'].str.contains(strategy_filter, na=False)]
        ending_inventory_value = target_inv['庫存現值'].sum()
        total_cost_basis = target_inv['總成本'].sum()

    total_unrealized = ending_inventory_value - total_cost_basis
    realized_pnl_period = 0; win_rate = 0; realized_roi = 0; trade_count = 0
    if not trade_log_df.empty:
        t_mask = (trade_log_df['Date'] >= start_date) & (trade_log_df['Date'] <= end_date)
        if selected_tickers: t_mask = t_mask & (trade_log_df['Ticker'].isin(selected_tickers))
        if strategy_filter: t_mask = t_mask & (trade_log_df['Strategy'].str.contains(strategy_filter, na=False))
        period_trades = trade_log_df[t_mask]
        
        if not period_trades.empty:
            realized_pnl_period = period_trades['PnL'].sum()
            trade_count = len(period_trades)
            wins = period_trades[period_trades['PnL'] > 0]
            if trade_count > 0: win_rate = (len(wins) / trade_count) * 100
            total_sold_cost = period_trades['CostBasis'].sum()
            if total_sold_cost > 0: realized_roi = (realized_pnl_period / total_sold_cost) * 100

    total_profit = realized_pnl_period + total_unrealized + total_dividend
    
    cash_flows = []
    for _, row in period_df.iterrows():
        d = row['Date']; amt = row['Total_Amount']; act = row['Action']
        if act == '買入': cash_flows.append((d, -abs(amt)))
        elif act in ['賣出', '領息']: cash_flows.append((d, abs(amt)))
    if ending_inventory_value > 0: cash_flows.append((end_date, ending_inventory_value))
    
    xirr_val = None
    try:
        xirr_val = xirr(cash_flows)
        if xirr_val: 
            xirr_val *= 100
            if xirr_val > 10000 or xirr_val < -10000: xirr_val = None
    except: xirr_val = None

    yoc_period = (total_dividend / total_cost_basis * 100) if total_cost_basis > 0 else 0
    payback_progress = (total_dividend / total_buy * 100) if total_buy > 0 else 0

    summary = {
        "累積總損益": total_profit, "已領股息": total_dividend, "已實現損益": realized_pnl_period,
        "未實現損益": total_unrealized, "勝率%": win_rate, "XIRR%": xirr_val, 
        "YoC%": yoc_period, "回本率%": payback_progress, "庫存現值": ending_inventory_value, "總成本": total_cost_basis,
        "投資報酬率%": realized_roi, "總交易次數": trade_count
    }
    return summary, period_df, pd.DataFrame()

# ==========================================
# 4. 圖表與數據顯示
# ==========================================
def render_allocation_charts(full_portfolio_df):
    if full_portfolio_df.empty: return
    st.markdown("#### 🥧 資產配置")
    base = alt.Chart(full_portfolio_df).encode(theta=alt.Theta("庫存現值", stack=True))
    pie = base.mark_arc(outerRadius=120, innerRadius=60).encode(
        color=alt.Color("代號", title="投資標的", sort=alt.EncodingSortField(field="庫存現值", order="descending")),
        order=alt.Order("庫存現值", sort="descending"),
        tooltip=["代號", "庫存現值", "佔比%", "策略", "種類"]
    ).interactive()
    st.altair_chart(pie, use_container_width=True)

def render_global_monthly_pnl_colored(trade_log_df, df_records):
    pnl_df = pd.DataFrame()
    if not trade_log_df.empty:
        pnl_df = trade_log_df[['Date', 'PnL', 'Type']].copy()
        pnl_df['Date'] = pd.to_datetime(pnl_df['Date'])
        pnl_df['Month'] = pnl_df['Date'].dt.strftime('%Y-%m')
    div_df = df_records[df_records['Action'] == '領息'][['Date', 'Total_Amount', 'Type']].copy()
    if not div_df.empty:
        div_df['Date'] = pd.to_datetime(div_df['Date'])
        div_df['Month'] = div_df['Date'].dt.strftime('%Y-%m')
        div_df = div_df.rename(columns={'Total_Amount': 'PnL'})
    combined = pd.concat([pnl_df, div_df], ignore_index=True)
    if combined.empty: return
    combined['Type'] = combined['Type'].fillna('股票') 
    combined = combined.sort_values('Month')
    grouped = combined.groupby(['Month', 'Type'])['PnL'].sum().reset_index()
    grouped['Date'] = pd.to_datetime(grouped['Month'])
    grouped = grouped.sort_values('Date')
    grouped['Cumulative_PnL'] = grouped.groupby('Type')['PnL'].cumsum()
    
    st.markdown("#### 🌊 財富堆疊圖")
    chart = alt.Chart(grouped).mark_area(opacity=0.7).encode(
        x=alt.X('Date:T', timeUnit='yearmonth', title='月份'),
        y=alt.Y('Cumulative_PnL:Q', title='累積已實現獲利 ($)', stack=True), 
        color=alt.Color('Type:N', title='資產種類'),
        tooltip=['Date', 'Type', 'Cumulative_PnL']
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

def render_metrics_cards(summary, mode):
    if not summary: return
    if mode == "swing": 
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("累積總損益", f"${summary['累積總損益']:,.0f}")
        k2.metric("已領股息", f"${summary['已領股息']:,.0f}")
        k3.metric("已實現損益", f"${summary['已實現損益']:,.0f}")
        k4.metric("未實現損益", f"${summary['未實現損益']:,.0f}")
        k5, k6, k7, k8 = st.columns(4)
        k5.metric("交易勝率", f"{summary['勝率%']:.1f}%")
        k6.metric("投資報酬率", f"{summary['投資報酬率%']:.2f}%")
        k7.empty(); k8.empty()
    elif mode == "dividend":
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("累積總損益", f"${summary['累積總損益']:,.0f}")
        k2.metric("已領股息", f"${summary['已領股息']:,.0f}")
        k3.metric("已實現損益", f"${summary['已實現損益']:,.0f}")
        k4.metric("未實現損益", f"${summary['未實現損益']:,.0f}")
        k5, k6, k7, k8 = st.columns(4)
        xirr_display = f"{summary['XIRR%']:.2f}%" if summary['XIRR%'] is not None else "N/A"
        k5.metric("年化報酬率", xirr_display)
        k6.metric("成本殖利率 (YoC)", f"{summary['YoC%']:.2f}%")
        k7.metric("回本率", f"{summary['回本率%']:.1f}%")
        k8.empty()
    else: 
        g1, g2, g3, g4, g5 = st.columns(5)
        g1.metric("累積總損益", f"${summary['累積總損益']:,.0f}")
        g2.metric("已領股息", f"${summary['已領股息']:,.0f}")
        g3.metric("已實現損益", f"${summary['已實現損益']:,.0f}")
        g4.metric("未實現損益", f"${summary['未實現損益']:,.0f}")
        xirr_display = f"{summary['XIRR%']:.2f}%" if summary['XIRR%'] is not None else "N/A"
        g5.metric("年化報酬率", xirr_display)

def render_chart_swing(trade_log_df):
    if not trade_log_df.empty:
        trade_log_df = trade_log_df.sort_values('Date')
        trade_log_df['cumsum_PnL'] = trade_log_df['PnL'].cumsum()
        
        line = alt.Chart(trade_log_df).mark_line(color='purple').encode(
            x='Date:T', y=alt.Y('cumsum_PnL:Q', title='累積已實現損益'), 
            tooltip=['Date', 'cumsum_PnL']
        )
        
        points = alt.Chart(trade_log_df).mark_circle(size=80).encode(
            x='Date:T', y='PnL:Q', 
            color=alt.condition(alt.datum.PnL > 0, alt.value("red"), alt.value("green")),
            tooltip=['Date', 'Ticker', 'PnL', 'Action']
        )
        
        st.altair_chart((line + points).interactive(), use_container_width=True)
    else:
        st.info("區間內無已實現交易")

def render_chart_dividend_monthly(period_df):
    div_df = period_df[period_df['Action'] == '領息'].copy()
    if not div_df.empty:
        div_df['Date'] = pd.to_datetime(div_df['Date'])
        chart = alt.Chart(div_df).mark_bar().encode(
            x=alt.X('Date:T', timeUnit='yearmonth', title='月份'), 
            y=alt.Y('Total_Amount:Q', title='股息金額'),
            color=alt.Color('Ticker:N', title='投資標的')
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
    else: st.info("區間內無領息紀錄")

def render_inventory_management(full_portfolio_df, df_records, key_prefix):
    st.markdown("### 📦 庫存管理與交易登錄")
    
    stocks_pf = full_portfolio_df[full_portfolio_df['種類'] == '股票']
    if not stocks_pf.empty:
        st.markdown("#### 📈 股票庫存")
        s_cost = stocks_pf['總成本'].sum(); s_pl = stocks_pf['帳面損益'].sum()
        s_roi = ((s_pl + stocks_pf['已領股息'].sum()) / s_cost * 100) if s_cost > 0 else 0
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("股票總現值", f"${stocks_pf['庫存現值'].sum():,.0f}")
        s2.metric("股票總成本", f"${s_cost:,.0f}")
        s3.metric("股票帳面損益", f"${s_pl:,.0f}", delta_color="normal")
        s4.metric("股票總報酬率", f"{s_roi:.2f}%")
        cols_show = ["代號", "佔比%", "庫存", "平均成本", "市價", "庫存現值", "帳面損益", "含息總報%", "策略"]
        st.dataframe(stocks_pf[cols_show], use_container_width=True, hide_index=True)
        st.divider()

    funds_pf = full_portfolio_df[full_portfolio_df['種類'] == '基金']
    if not funds_pf.empty:
        st.markdown("#### 🛡️ 基金庫存")
        f1, f2, f3, f4 = st.columns(4)
        f_cost = funds_pf['總成本'].sum(); f_pl = funds_pf['帳面損益'].sum()
        f_roi = (f_pl / f_cost * 100) if f_cost > 0 else 0
        f1.metric("基金總現值", f"${funds_pf['庫存現值'].sum():,.0f}")
        f2.metric("基金總投入", f"${f_cost:,.0f}")
        f3.metric("基金帳面損益", f"${f_pl:,.0f}", delta_color="normal")
        f4.metric("基金總報酬率", f"{f_roi:.2f}%")
        cols_show = ["代號", "佔比%", "庫存", "平均成本", "市價", "庫存現值", "帳面損益", "含息總報%", "策略"]
        st.dataframe(funds_pf[cols_show], use_container_width=True, hide_index=True)
        st.divider()
    
    if full_portfolio_df.empty: st.info("尚無庫存資料，請新增交易。")

    c1, c2 = st.columns([2, 1])
    with c1:
        with st.form(f"trans_form_{key_prefix}", clear_on_submit=True):
            st.markdown("**➕ 新增交易**")
            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
            with r1c1: d_date = st.date_input("日期")
            with r1c2: d_ticker = st.text_input("代號").upper()
            with r1c3: d_type = st.selectbox("種類", ["股票", "基金"])
            with r1c4: d_action = st.selectbox("動作", ["買入", "賣出", "領息", "分割"])
            r2c1, r2c2, r2c3, r2c4 = st.columns(4)
            with r2c1: d_strat = st.multiselect("策略", ["存股", "波段"], default=["存股"])
            with r2c2: d_price = st.number_input("單價", min_value=0.0, format="%.2f")
            with r2c3: d_shares = st.number_input("股數", step=100.0)
            with r2c4: d_fee = st.number_input("手續費 (0自動算)", min_value=0)
            r3c1, r3c2 = st.columns([3, 1])
            with r3c1: d_total = st.number_input("總金額 (0自動算)", step=1000.0); d_note = st.text_input("備註")
            with r3c2: st.write(""); submitted = st.form_submit_button("送出交易", use_container_width=True)
            if submitted and d_ticker:
                db_strat = ",".join(d_strat)
                if d_fee == 0 and d_action in ["買入", "賣出"]: d_fee = int(d_price * d_shares * 0.001425)
                if d_total == 0:
                    basic = d_price * d_shares
                    if d_action == "買入": d_total = basic + d_fee
                    elif d_action == "賣出": d_total = basic - d_fee - int(basic*0.003)
                ws_records.append_row([str(d_date), d_ticker, d_type, db_strat, d_action, d_price, d_shares, d_fee, d_total, d_note])
                st.success(f"已儲存 {d_ticker}"); st.cache_data.clear()
    with c2:
        with st.form(f"fund_form_{key_prefix}", clear_on_submit=True):
            st.markdown("**💵 更新基金淨值**")
            f_tick = st.text_input("基金代號").upper()
            f_val = st.number_input("最新淨值", min_value=0.0, format="%.4f")
            f_curr = st.selectbox("幣別", ["USD", "TWD"])
            st.write(""); st.write(""); st.write("")
            if st.form_submit_button("更新", use_container_width=True):
                try:
                    cell = ws_funds.find(f_tick)
                    ws_funds.update_cell(cell.row, 2, f_val)
                    ws_funds.update_cell(cell.row, 3, str(datetime.now().date()))
                except: ws_funds.append_row([f_tick, f_val, str(datetime.now().date()), f_curr])
                st.success("更新成功"); st.cache_data.clear()

# ==========================================
# 5. 主程式佈局
# ==========================================
st.title("📊 投資戰情室 v11.0 (No AI)")

df, df_funds, usd_rate = load_data()
if df.empty: st.warning("目前無任何交易紀錄"); st.stop()

all_tickers = df['Ticker'].unique().tolist()
full_portfolio_df, trade_log_df = calculate_portfolio(df, df_funds, usd_rate)

st.markdown("#### 🔍 篩選條件")
f1, f2, f3 = st.columns([1, 1, 2])
with f1:
    min_date = df['Date'].min(); max_date = date.today()
    analysis_start = st.date_input("開始日期", value=min_date, min_value=min_date, max_value=max_date)
with f2: analysis_end = st.date_input("結束日期", value=max_date, min_value=min_date, max_value=max_date)
with f3: selected_tickers = st.multiselect("投資標的", all_tickers, default=None)

st.divider()

total_summary = None

if not selected_tickers:
    t_all, t_swing, t_div = st.tabs(["🌍 全總覽", "⚡ 波段儀表板", "💰 存股儀表板"])
    
    global_sum, _, _ = analyze_period_advanced(df, analysis_start, analysis_end, None, full_portfolio_df, trade_log_df, None)

    with t_all:
        if global_sum: render_metrics_cards(global_sum, "general")
        st.write(""); g1, g2 = st.columns([1, 2])
        if global_sum:
            with g1: render_allocation_charts(full_portfolio_df)
            with g2: render_global_monthly_pnl_colored(trade_log_df, df)
        st.divider(); render_inventory_management(full_portfolio_df, df, "overview")

    with t_swing:
        swing_sum, _, swing_log_df = analyze_period_advanced(df, analysis_start, analysis_end, None, full_portfolio_df, trade_log_df, strategy_filter="波段")
        if swing_sum: render_metrics_cards(swing_sum, "swing")
        st.markdown("##### 📈 交易損益曲線"); render_chart_swing(swing_log_df)
        st.divider()
        if not swing_log_df.empty:
            swing_log_df = swing_log_df.sort_values('Date', ascending=False)
            st.dataframe(swing_log_df[['Date', 'Ticker', 'Action', 'PnL', 'AI_Review']], use_container_width=True, hide_index=True)

    with t_div:
        div_sum, div_period_df, _ = analyze_period_advanced(df, analysis_start, analysis_end, None, full_portfolio_df, trade_log_df, strategy_filter="存股")
        if div_sum: render_metrics_cards(div_sum, "dividend")
        st.markdown("##### 💰 股息累積圖"); render_chart_dividend_monthly(df[df['Action']=='領息'])

else:
    ticker_tabs = st.tabs([f"🔍 {t}" for t in selected_tickers])
    for i, ticker in enumerate(selected_tickers):
        with ticker_tabs[i]:
            ticker_df = df[df['Ticker'] == ticker]
            strategies_used = ticker_df['Strategy'].unique().tolist()
            combined_strategies = ",".join([str(s) for s in strategies_used])
            has_swing = "波段" in combined_strategies
            has_div = "存股" in combined_strategies
            
            if not (has_swing or has_div):
                st.warning(f"⚠️ {ticker} 尚未設定明確策略")
            else:
                if has_swing:
                    with st.expander("⚡ 策略分析：波段", expanded=True):
                        s_sum, _, s_log = analyze_period_advanced(df, analysis_start, analysis_end, [ticker], full_portfolio_df, trade_log_df, "波段")
                        if s_sum: render_metrics_cards(s_sum, "swing")
                        render_chart_swing(s_log)
                if has_div:
                    with st.expander("💰 策略分析：存股", expanded=True):
                        d_sum, d_period, _ = analyze_period_advanced(df, analysis_start, analysis_end, [ticker], full_portfolio_df, trade_log_df, "存股")
                        if d_sum: render_metrics_cards(d_sum, "dividend")
                        render_chart_dividend_monthly(d_period)
            
            st.divider()
            render_inventory_management(full_portfolio_df[full_portfolio_df['代號']==ticker], df, f"tick_{i}")
