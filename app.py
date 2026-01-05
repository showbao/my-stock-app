# Version: v1.1
# CTOSignature: Cloud Deployment Version with Secure Secrets
import streamlit as st
import pandas as pd
import yfinance as yf
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import plotly.express as px

# ==========================================
# 1. 系統設定與連線 (System Config)
# ==========================================
st.set_page_config(page_title="投資追蹤指揮中心", layout="wide", initial_sidebar_state="expanded")

# 初始化 Google Sheets 連線 (雲端安全版)
@st.cache_resource
def connect_google_sheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    try:
        # v1.1 修改：從 Streamlit Secrets 讀取金鑰，而非讀取本地檔案
        # 這能防止金鑰上傳到 GitHub 導致外洩
        if "gcp_service_account" in st.secrets:
            creds_dict = st.secrets["gcp_service_account"]
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        else:
            # 本地備用方案 (如果您之後想在電腦跑)
            creds = ServiceAccountCredentials.from_json_keyfile_name("secrets.json", scope)
            
        client = gspread.authorize(creds)
        # 嘗試開啟試算表
        sheet = client.open("Investment_Tracker")
        return sheet
    except Exception as e:
        st.error(f"連線失敗！請檢查 Streamlit Secrets 設定。\n錯誤訊息: {e}")
        st.stop()

sh = connect_google_sheet()
ws_records = sh.worksheet("Records")
ws_funds = sh.worksheet("Fund_Updates")

# ==========================================
# 2. 核心邏輯函數 (Core Functions)
# ==========================================

# 抓取 USD/TWD 即時匯率
@st.cache_data(ttl=3600) 
def get_usd_twd_rate():
    try:
        ticker = yf.Ticker("TWD=X")
        hist = ticker.history(period="1d")
        if not hist.empty:
            return hist['Close'].iloc[-1]
        return 32.0
    except:
        return 32.0

# 抓取股票即時股價
@st.cache_data(ttl=600)
def get_stock_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        todays_data = stock.history(period='1d')
        if not todays_data.empty:
            return todays_data['Close'].iloc[-1]
        return 0.0
    except:
        return 0.0

# 載入並處理資料
def load_data():
    records_data = ws_records.get_all_records()
    df = pd.DataFrame(records_data)
    funds_data = ws_funds.get_all_records()
    df_funds = pd.DataFrame(funds_data)
    
    if df.empty:
        return df, df_funds, 32.0
        
    numeric_cols = ['Price', 'Shares', 'Fee', 'Total_Amount']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    current_usd_rate = get_usd_twd_rate()
    return df, df_funds, current_usd_rate

# 計算平均成本與庫存
def calculate_portfolio(df, df_funds, current_usd_rate):
    portfolio = {}
    df = df.sort_values('Date')
    
    for _, row in df.iterrows():
        ticker = row['Ticker']
        action = row['Action']
        qty = row['Shares']
        amount = row['Total_Amount']
        typ = row['Type']
        
        if ticker not in portfolio:
            portfolio[ticker] = {
                'shares': 0, 'total_cost': 0, 'realized_pl': 0, 
                'dividend_collected': 0, 'type': typ, 'strategy': row['Strategy']
            }
            
        p = portfolio[ticker]
        
        if action == 'Buy':
            p['shares'] += qty
            p['total_cost'] += amount
        elif action == 'Sell':
            if p['shares'] > 0:
                avg_cost = p['total_cost'] / p['shares']
                cost_of_sold_shares = avg_cost * qty
                p['realized_pl'] += (amount - cost_of_sold_shares)
                p['total_cost'] -= cost_of_sold_shares
                p['shares'] -= qty
        elif action == 'Dividend':
            p['dividend_collected'] += amount
            
    results = []
    for ticker, data in portfolio.items():
        current_price = 0
        market_value = 0
        
        if data['shares'] > 0:
            if data['type'] == 'Stock':
                current_price = get_stock_price(ticker)
                market_value = current_price * data['shares']
            elif data['type'] == 'Fund':
                if not df_funds.empty and ticker in df_funds['Ticker'].values:
                    usd_net = df_funds[df_funds['Ticker'] == ticker]['Net_Value_USD'].values[0]
                    current_price = usd_net * current_usd_rate
                    market_value = data['shares'] * usd_net * current_usd_rate
            
            avg_cost = data['total_cost'] / data['shares'] if data['shares'] > 0 else 0
            unrealized_pl = market_value - data['total_cost']
            roi = (unrealized_pl / data['total_cost'] * 100) if data['total_cost'] > 0 else 0
            fill_status = "✅" if current_price >= avg_cost else "🔻"
            
            results.append({
                "代號": ticker,
                "策略": data['strategy'],
                "庫存": data['shares'],
                "平均成本": round(avg_cost, 2),
                "目前市價(TWD)": round(current_price, 2),
                "總市值": round(market_value, 0),
                "未實現損益": round(unrealized_pl, 0),
                "報酬率%": round(roi, 2),
                "已領股息": round(data['dividend_collected'], 0),
                "已實現損益": round(data['realized_pl'], 0),
                "狀態": fill_status
            })
            
    return pd.DataFrame(results)

# ==========================================
# 3. 前端介面 (UI)
# ==========================================
with st.sidebar:
    st.header("📝 交易輸入")
    with st.form("entry_form"):
        date = st.date_input("日期")
        ticker = st.text_input("代號 (如 2330.TW)", value="").upper()
        typ = st.selectbox("種類", ["Stock", "Fund"])
        strategy = st.selectbox("策略", ["Dividend (存股)", "Swing (波段)"])
        action = st.selectbox("動作", ["Buy", "Sell", "Dividend"])
        price = st.number_input("單價/淨值", min_value=0.0, format="%.2f")
        shares = st.number_input("股數/單位", min_value=0.0, format="%.2f")
        fee = st.number_input("手續費 (TWD)", min_value=0, value=0)
        total_amt = st.number_input("總金額 (TWD)", min_value=0.0, format="%.2f")
        note = st.text_input("備註")
        submitted = st.form_submit_button("送出紀錄")
        
        if submitted:
            new_row = [str(date), ticker, typ, strategy.split()[0], action, price, shares, fee, total_amt, note]
            ws_records.append_row(new_row)
            st.success("已儲存！請重新整理頁面。")
            st.cache_data.clear()

    st.divider()
    st.header("💵 基金淨值更新")
    with st.form("fund_update_form"):
        f_ticker = st.text_input("基金代號").upper()
        f_net_val = st.number_input("最新淨值 (USD)", min_value=0.0, format="%.4f")
        f_submitted = st.form_submit_button("更新淨值")
        
        if f_submitted:
            try:
                cell = ws_funds.find(f_ticker)
                ws_funds.update_cell(cell.row, 2, f_net_val)
                ws_funds.update_cell(cell.row, 3, str(datetime.now().date()))
            except:
                ws_funds.append_row([f_ticker, f_net_val, str(datetime.now().date())])
            st.success(f"{f_ticker} 淨值已更新！")
            st.cache_data.clear()

st.title("📊 全能投資追蹤器 v1.1 (Cloud)")
df, df_funds, usd_rate = load_data()

if df.empty:
    st.info("目前沒有交易紀錄，請從側邊欄輸入第一筆交易。")
else:
    portfolio_df = calculate_portfolio(df, df_funds, usd_rate)
    if not portfolio_df.empty:
        total_market_value = portfolio_df['總市值'].sum()
        total_unrealized = portfolio_df['未實現損益'].sum()
        total_dividend = portfolio_df['已領股息'].sum()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("總市值 (TWD)", f"${total_market_value:,.0f}")
        col2.metric("未實現損益", f"${total_unrealized:,.0f}", delta_color="normal")
        col3.metric("今年已領股息", f"${total_dividend:,.0f}")
        col4.metric("目前 USD/TWD 匯率", f"{usd_rate:.2f}")
        
        st.subheader("🎯 資產策略分析")
        tab1, tab2 = st.tabs(["💰 現金流資產 (存股+基金)", "🚀 資本利得資產 (波段)"])
        
        with tab1:
            st.caption("目標：累積股數與配息")
            div_assets = portfolio_df[portfolio_df['策略'] == 'Dividend']
            if not div_assets.empty:
                st.dataframe(div_assets, use_container_width=True)
            else:
                st.write("尚無存股資產")
        with tab2:
            st.caption("目標：賺取價差")
            swing_assets = portfolio_df[portfolio_df['策略'] == 'Swing']
            if not swing_assets.empty:
                st.dataframe(swing_assets, use_container_width=True)
            else:
                st.write("尚無波段資產")
            
        with st.expander("查看原始交易紀錄"):
            st.dataframe(df)
    else:
        st.write("目前沒有持倉。")
