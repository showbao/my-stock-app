# Version: v1.5
# CTOSignature: Dashboard Integration & Advanced Metrics (YoC, Volatility)
import streamlit as st
import pandas as pd
import yfinance as yf
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, date, timedelta
import numpy as np # 新增：用於計算波動率

# ==========================================
# 1. 系統設定與連線
# ==========================================
st.set_page_config(page_title="投資追蹤指揮中心", layout="wide", initial_sidebar_state="expanded")

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
        st.error(f"連線失敗！請檢查 Streamlit Secrets 設定。\n錯誤訊息: {e}")
        st.stop()

sh = connect_google_sheet()
ws_records = sh.worksheet("Records")
ws_funds = sh.worksheet("Fund_Updates")

# ==========================================
# 2. 核心邏輯函數
# ==========================================

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

@st.cache_data(ttl=600)
def get_stock_data(ticker):
    """
    同時抓取「現價」與「歷史數據(算波動率用)」
    """
    try:
        stock = yf.Ticker(ticker)
        # 抓取 1 個月資料來算波動
        hist = stock.history(period='1mo')
        
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            # 計算波動率 (年化標準差)
            # Log Return -> Std Dev -> Annualize
            if len(hist) > 1:
                log_ret = np.log(hist['Close'] / hist['Close'].shift(1))
                volatility = log_ret.std() * np.sqrt(252) * 100 # 轉為百分比
            else:
                volatility = 0.0
            
            return current_price, volatility
        return 0.0, 0.0
    except:
        return 0.0, 0.0

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
    
    df['Date'] = pd.to_datetime(df['Date']).dt.date
        
    current_usd_rate = get_usd_twd_rate()
    return df, df_funds, current_usd_rate

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
        volatility = 0
        
        if data['shares'] > 0:
            # 1. 取得現價與波動率
            if data['type'] == 'Stock':
                current_price, volatility = get_stock_data(ticker)
                market_value = current_price * data['shares']
            elif data['type'] == 'Fund':
                if not df_funds.empty and ticker in df_funds['Ticker'].values:
                    usd_net = df_funds[df_funds['Ticker'] == ticker]['Net_Value_USD'].values[0]
                    current_price = usd_net * current_usd_rate
                    market_value = data['shares'] * usd_net * current_usd_rate
                    volatility = 0 # 基金暫不支援自動波動率
            
            # 2. 基礎計算
            avg_cost = data['total_cost'] / data['shares'] if data['shares'] > 0 else 0
            unrealized_pl = market_value - data['total_cost']
            
            # 3. 進階指標
            # 成本殖利率 (YoC) = 累積領到的股息 / 目前持有成本 (或總投入)
            # 這裡定義為：累積領到的股息 / (目前平均成本 * 股數) -> 即個人持有的現金回報率
            yield_on_cost = (data['dividend_collected'] / data['total_cost'] * 100) if data['total_cost'] > 0 else 0
            
            roi_price = (unrealized_pl / data['total_cost'] * 100) if data['total_cost'] > 0 else 0
            total_gain = unrealized_pl + data['dividend_collected']
            roi_total = (total_gain / data['total_cost'] * 100) if data['total_cost'] > 0 else 0
            
            fill_status = "✅已填" if current_price >= avg_cost else "🔻貼息"
            
            results.append({
                "代號": ticker,
                "策略": data['strategy'],
                "庫存": data['shares'],
                "平均成本": round(avg_cost, 2),
                "市價": round(current_price, 2),
                "波動率%": round(volatility, 1), # 新指標
                "市值": round(market_value, 0),
                "帳面損益": round(unrealized_pl, 0),
                "成本殖利率%": round(yield_on_cost, 2), # 新指標
                "含息總報%": round(roi_total, 2),
                "已領股息": round(data['dividend_collected'], 0),
                "填息": fill_status
            })
            
    return pd.DataFrame(results)

def analyze_period(df, start_date, end_date, selected_tickers):
    # 篩選區間
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    if selected_tickers:
        mask = mask & (df['Ticker'].isin(selected_tickers))
    
    period_df = df[mask].copy()
    
    if period_df.empty:
        return None, pd.DataFrame()

    # 計算區間統計
    total_dividend = period_df[period_df['Action'] == 'Dividend']['Total_Amount'].sum()
    total_buy = period_df[period_df['Action'] == 'Buy']['Total_Amount'].sum()
    total_sell = period_df[period_df['Action'] == 'Sell']['Total_Amount'].sum()
    
    # 估算單筆交易損益 (針對賣出)
    # 簡單邏輯：賣出總額 - (賣出股數 * 該筆交易當下的估計平均成本)
    # 注意：這裡無法精確回推當時的平均成本，因此改用比較直觀的 "交易現金流"
    # 若要顯示單筆損益，需在原始資料表標註。這裡我們做「賣出明細表」
    
    sell_records = period_df[period_df['Action'] == 'Sell'].copy()
    # 這裡可以加一個欄位顯示單次賣出金額
    
    net_cashflow = (total_sell + total_dividend) - total_buy
    
    summary = {
        "區間": f"{start_date} ~ {end_date}",
        "總領股息": total_dividend,
        "總買入": total_buy,
        "總賣出": total_sell,
        "淨現金流": net_cashflow
    }
    
    return summary, period_df

# ==========================================
# 3. 前端介面 (UI)
# ==========================================

# --- Sidebar: 只放輸入 ---
with st.sidebar:
    st.header("📝 交易輸入")
    with st.form("entry_form"):
        date_in = st.date_input("日期")
        ticker = st.text_input("代號 (如 2330.TW)", value="").upper()
        
        typ_display = st.selectbox("種類", ["股票 (Stock)", "基金 (Fund)"])
        strategy_display = st.selectbox("策略", ["存股 (Dividend)", "波段 (Swing)"])
        action_display = st.selectbox("動作", ["買入 (Buy)", "賣出 (Sell)", "領息 (Dividend)"])
        
        price = st.number_input("單價 / 淨值", min_value=0.0, format="%.2f")
        shares = st.number_input("股數 / 單位數", min_value=0.0, format="%.2f")
        fee = st.number_input("手續費 (TWD)", min_value=0, value=0)
        total_amt = st.number_input("總金額 (TWD)", min_value=0.0, format="%.2f")
        note = st.text_input("備註")
        
        submitted = st.form_submit_button("送出紀錄")
        
        if submitted:
            typ_map = {"股票 (Stock)": "Stock", "基金 (Fund)": "Fund"}
            strat_map = {"存股 (Dividend)": "Dividend", "波段 (Swing)": "Swing"}
            act_map = {"買入 (Buy)": "Buy", "賣出 (Sell)": "Sell", "領息 (Dividend)": "Dividend"}
            
            db_type = typ_map[typ_display]
            db_strat = strat_map[strategy_display]
            db_action = act_map[action_display]
            
            final_shares = shares
            final_price = price
            final_total = total_amt

            if db_action == "Dividend":
                final_shares = 0
                final_price = 0
                if final_total == 0:
                     st.error("⚠️ 領息模式下，「總金額」不能為 0！")
                     st.stop()
            else:
                if final_total == 0:
                    calculated_total = (price * shares) + fee
                    final_total = calculated_total

            new_row = [str(date_in), ticker, db_type, db_strat, db_action, final_price, final_shares, fee, final_total, note]
            ws_records.append_row(new_row)
            st.success("✅ 交易已儲存！")
            st.cache_data.clear()

    st.divider()
    st.caption("基金淨值更新")
    with st.form("fund_update_form"):
        f_ticker = st.text_input("基金代號").upper()
        f_net_val = st.number_input("最新淨值 (USD)", min_value=0.0, format="%.4f")
        f_submitted = st.form_submit_button("更新")
        if f_submitted:
            try:
                cell = ws_funds.find(f_ticker)
                ws_funds.update_cell(cell.row, 2, f_net_val)
                ws_funds.update_cell(cell.row, 3, str(datetime.now().date()))
            except:
                ws_funds.append_row([f_ticker, f_net_val, str(datetime.now().date())])
            st.success("已更新")
            st.cache_data.clear()

# --- Main Dashboard ---
st.title("📊 投資戰情室 v1.5")

# 1. 戰情分析篩選器 (移至主畫面)
with st.expander("🔍 戰情分析篩選器 (日期/代號)", expanded=True):
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        analysis_start = st.date_input("開始日期", value=date(datetime.now().year, 1, 1))
    with col_f2:
        analysis_end = st.date_input("結束日期", value=datetime.now().date())
    
    # 預載資料以取得代號清單
    _df, _, _ = load_data()
    all_tickers = _df['Ticker'].unique().tolist() if not _df.empty else []
    selected_tickers = st.multiselect("篩選代號 (可多選，留空則全選)", all_tickers)

df, df_funds, usd_rate = load_data()

if df.empty:
    st.info("尚無資料，請先輸入交易。")
else:
    # 2. 庫存總覽 (Snapshot - 不受日期篩選影響，永遠顯示當下)
    portfolio_df = calculate_portfolio(df, df_funds, usd_rate)
    
    if not portfolio_df.empty:
        # 計算總資產指標
        total_market_value = portfolio_df['市值'].sum()
        total_unrealized = portfolio_df['帳面損益'].sum()
        total_div_all_time = portfolio_df['已領股息'].sum()
        
        # 顯示指標卡片
        m1, m2, m3 = st.columns(3)
        m1.metric("目前總市值", f"${total_market_value:,.0f}")
        m2.metric("總帳面損益 (未實現)", f"${total_unrealized:,.0f}", delta_color="normal")
        m3.metric("歷史總領息", f"${total_div_all_time:,.0f}")
        
        st.subheader("📦 現有庫存明細")
        
        # 欄位顯示設定 (包含新指標)
        cols_show = ["代號", "庫存", "平均成本", "市價", "波動率%", "市值", "帳面損益", "成本殖利率%", "含息總報%", "填息"]
        
        tab_div, tab_swing = st.tabs(["💰 存股 / 基金", "🚀 波段交易"])
        
        with tab_div:
            # 顯示存股與基金
            div_assets = portfolio_df[portfolio_df['策略'] == 'Dividend']
            if not div_assets.empty:
                st.dataframe(div_assets[cols_show], use_container_width=True, hide_index=True)
            else:
                st.write("無存股資產")
                
        with tab_swing:
            # 顯示波段
            swing_assets = portfolio_df[portfolio_df['策略'] == 'Swing']
            if not swing_assets.empty:
                st.dataframe(swing_assets[cols_show], use_container_width=True, hide_index=True)
            else:
                st.write("無波段資產")

    # 3. 區間歷史分析 (受上方篩選器控制)
    st.divider()
    st.subheader(f"📅 區間績效回測 ({analysis_start} ~ {analysis_end})")
    
    summary, period_df = analyze_period(df, analysis_start, analysis_end, selected_tickers)
    
    if summary:
        # 區間績效指標
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("區間已領股息", f"${summary['總領股息']:,.0f}")
        k2.metric("區間賣出金額", f"${summary['總賣出']:,.0f}")
        k3.metric("區間買入投入", f"${summary['總買入']:,.0f}")
        k4.metric("區間淨現金流", f"${summary['淨現金流']:,.0f}", help="正值代表資金淨回收，負值代表資金淨投入")
        
        # 顯示區間內的「賣出」與「領息」明細 (單筆檢視)
        with st.expander("查看區間交易明細 (賣出/領息)", expanded=True):
            # 只顯示賣出和領息，因為這些代表獲利/現金流
            view_df = period_df[period_df['Action'].isin(['Sell', 'Dividend'])].copy()
            if not view_df.empty:
                st.dataframe(view_df[['Date', 'Ticker', 'Action', 'Price', 'Shares', 'Total_Amount', 'Note']], use_container_width=True)
            else:
                st.info("此區間內無賣出或領息紀錄。")
    else:
        st.info("此篩選條件下無任何交易。")
