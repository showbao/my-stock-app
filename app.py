# Version: v2.1
# CTOSignature: CAGR Fix - Include Unrealized Market Value in Return Calculation
import streamlit as st
import pandas as pd
import yfinance as yf
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, date, timedelta
import numpy as np

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
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1mo', auto_adjust=True)
        
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            if len(hist) > 1:
                log_ret = np.log(hist['Close'] / hist['Close'].shift(1))
                volatility = log_ret.std() * np.sqrt(252) * 100
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
                pct_sold = qty / p['shares']
                cost_of_sold_shares = p['total_cost'] * pct_sold
                p['realized_pl'] += (amount - cost_of_sold_shares)
                p['total_cost'] -= cost_of_sold_shares
                p['shares'] -= qty
                
                if p['shares'] <= 0.001: 
                    p['shares'] = 0
                    p['total_cost'] = 0
                    
        elif action == 'Dividend':
            p['dividend_collected'] += amount
            
        elif action == 'Split': 
            p['shares'] += qty
            if p['shares'] <= 0.001:
                p['shares'] = 0
                p['total_cost'] = 0
            
    results = []
    for ticker, data in portfolio.items():
        current_price = 0
        market_value = 0
        volatility = 0
        
        if data['shares'] > 0.001:
            if data['type'] == 'Stock':
                current_price, volatility = get_stock_data(ticker)
                market_value = current_price * data['shares']
            elif data['type'] == 'Fund':
                if not df_funds.empty and ticker in df_funds['Ticker'].values:
                    usd_net = df_funds[df_funds['Ticker'] == ticker]['Net_Value_USD'].values[0]
                    current_price = usd_net * current_usd_rate
                    market_value = data['shares'] * usd_net * current_usd_rate
                    volatility = 0
            
            avg_cost = data['total_cost'] / data['shares']
            unrealized_pl = market_value - data['total_cost']
            
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
                "波動率%": round(volatility, 1),
                "市值": round(market_value, 0),
                "帳面損益": round(unrealized_pl, 0),
                "成本殖利率%": round(yield_on_cost, 2),
                "含息總報%": round(roi_total, 2),
                "已領股息": round(data['dividend_collected'], 0),
                "填息": fill_status,
                "總成本": round(data['total_cost'], 0)
            })
            
    return pd.DataFrame(results)

# --- v2.1 重點修正函數 ---
def analyze_period(df, start_date, end_date, selected_tickers, current_portfolio_df):
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    if selected_tickers:
        mask = mask & (df['Ticker'].isin(selected_tickers))
    period_df = df[mask].copy()
    
    if period_df.empty: return None, pd.DataFrame(), pd.DataFrame()

    total_dividend = period_df[period_df['Action'] == 'Dividend']['Total_Amount'].sum()
    total_buy = period_df[period_df['Action'] == 'Buy']['Total_Amount'].sum()
    total_sell = period_df[period_df['Action'] == 'Sell']['Total_Amount'].sum()
    
    # 核心修正：計算「期末庫存價值」
    # 邏輯：如果結束日期是今天 (或未來)，我們就把「目前手上的股票市值」加回去當作回收
    # 注意：如果使用者選的是「過去某一年」，這個算法會不準 (因為不知道當時的市值)，所以只在查看 "至今" 時啟用
    
    ending_inventory_value = 0
    is_current = end_date >= datetime.now().date()
    
    if is_current and not current_portfolio_df.empty:
        # 篩選出符合條件的庫存
        if selected_tickers:
             target_inventory = current_portfolio_df[current_portfolio_df['代號'].isin(selected_tickers)]
        else:
             target_inventory = current_portfolio_df
        
        ending_inventory_value = target_inventory['市值'].sum()

    # 分子：賣掉拿回的錢 + 領到的股息 + (還沒賣但值錢的庫存)
    total_recovered = total_sell + total_dividend + ending_inventory_value
    
    return_rate = (total_recovered / total_buy * 100) if total_buy > 0 else 0
    
    days = (end_date - start_date).days
    # 年化計算 (CAGR)
    # 公式：(期末總值 / 總投入)^(1/年數) - 1
    # 期末總值 = 賣出 + 股息 + 現有庫存市值
    
    if days > 365 and total_buy > 0 and total_recovered > 0:
        years = days / 365
        annualized_return = (pow(total_recovered / total_buy, 1/years) - 1) * 100
    else:
        annualized_return = None

    summary = {
        "總領股息": total_dividend,
        "總買入": total_buy,
        "總賣出": total_sell,
        "淨現金流": (total_sell + total_dividend) - total_buy,
        "總回報率%": return_rate - 100, # 扣掉本金顯示淨利%
        "年化回收率%": annualized_return,
        "期末庫存市值": ending_inventory_value # 用於 debug 或顯示
    }

    # 年度分列
    years_data = []
    start_y = start_date.year
    end_y = end_date.year
    
    for y in range(start_y, end_y + 1):
        y_df = period_df[pd.to_datetime(period_df['Date']).dt.year == y]
        if not y_df.empty:
            y_div = y_df[y_df['Action'] == 'Dividend']['Total_Amount'].sum()
            y_buy = y_df[y_df['Action'] == 'Buy']['Total_Amount'].sum()
            y_sell = y_df[y_df['Action'] == 'Sell']['Total_Amount'].sum()
            y_net = (y_sell + y_div) - y_buy
            years_data.append({
                "年度": str(y),
                "領息金額": f"${y_div:,.0f}",
                "買入投入": f"${y_buy:,.0f}",
                "賣出變現": f"${y_sell:,.0f}",
                "淨現金流": f"${y_net:,.0f}"
            })
            
    years_df = pd.DataFrame(years_data)

    return summary, period_df, years_df

# ==========================================
# 3. 前端介面
# ==========================================
with st.sidebar:
    st.header("📝 交易輸入")
    with st.form("entry_form"):
        date_in = st.date_input("日期")
        ticker = st.text_input("代號", value="").upper()
        typ_display = st.selectbox("種類", ["股票 (Stock)", "基金 (Fund)"])
        strategy_display = st.selectbox("策略", ["存股 (Dividend)", "波段 (Swing)"])
        action_display = st.selectbox("動作", ["買入 (Buy)", "賣出 (Sell)", "領息 (Dividend)", "分割/減資 (Split)"])
        
        if "Split" in action_display:
            st.info("💡 分割：輸入正數股數。\n💡 減資：輸入負數股數。\n💰 金額請填 0。")

        price = st.number_input("單價 / 淨值", min_value=0.0, format="%.2f")
        shares = st.number_input("股數 / 單位數", min_value=-100000.0, max_value=100000.0, format="%.2f")
        fee = st.number_input("手續費", min_value=0, value=0)
        total_amt = st.number_input("總金額", min_value=0.0, format="%.2f")
        note = st.text_input("備註")
        
        submitted = st.form_submit_button("送出紀錄")
        if submitted:
            typ_map = {"股票 (Stock)": "Stock", "基金 (Fund)": "Fund"}
            strat_map = {"存股 (Dividend)": "Dividend", "波段 (Swing)": "Swing"}
            act_map = {"買入 (Buy)": "Buy", "賣出 (Sell)": "Sell", "領息 (Dividend)": "Dividend", "分割/減資 (Split)": "Split"}
            
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
            elif db_action == "Split":
                final_total = 0
                final_price = 0
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
st.title("📊 投資戰情室 v2.1")

# 1. 篩選器
_df, _, _ = load_data()
all_tickers = _df['Ticker'].unique().tolist() if not _df.empty else []

col_s1, col_s2, col_s3 = st.columns([1, 1, 2])
with col_s1:
    analysis_start = st.date_input("開始日期", value=date(datetime.now().year, 1, 1))
with col_s2:
    analysis_end = st.date_input("結束日期", value=datetime.now().date())
with col_s3:
    selected_tickers = st.multiselect("篩選代號", all_tickers)

# 載入正式資料
df, df_funds, usd_rate = load_data()

if not df.empty:
    # 先計算庫存，因為分析需要用到庫存市值
    portfolio_df = calculate_portfolio(df, df_funds, usd_rate)
    
    # 2. 查詢區
    summary, period_df, years_df = analyze_period(df, analysis_start, analysis_end, selected_tickers, portfolio_df)
    
    if summary:
        st.markdown("### 📈 區間績效看板")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("區間已領股息", f"${summary['總領股息']:,.0f}")
        k2.metric("區間淨現金流", f"${summary['淨現金流']:,.0f}", help="正數=回收資金, 負數=持續投入")
        
        # 顯示年化報酬
        if summary['年化回收率%'] is not None:
            # 如果有加回庫存市值，標題改一下讓使用者知道
            if summary['期末庫存市值'] > 0:
                k3.metric("年化報酬率 (CAGR)", f"{summary['年化回收率%']:.2f}%", help="含息 + 含目前庫存市值")
            else:
                k3.metric("年化回收率 (CAGR)", f"{summary['年化回收率%']:.2f}%", help="僅計算現金回收 (已清倉適用)")
        else:
            k3.metric("區間總回報", f"{summary['總回報率%']:.2f}%", help="絕對報酬率")
            
        k4.metric("庫存潛在價值", f"${summary['期末庫存市值']:,.0f}", help="若現在全賣掉可拿回的錢 (已加計入CAGR)")

        if not years_df.empty and len(years_df) > 1:
            with st.expander("📅 年度績效比較表", expanded=True):
                st.dataframe(years_df, use_container_width=True, hide_index=True)
        
        st.divider()

    # 3. 現有庫存區
    st.markdown("### 📦 現有庫存明細")
    if not portfolio_df.empty:
        if selected_tickers:
            portfolio_df = portfolio_df[portfolio_df['代號'].isin(selected_tickers)]

        total_mv = portfolio_df['市值'].sum()
        total_cost = portfolio_df['總成本'].sum()
        total_pl = portfolio_df['帳面損益'].sum()
        
        st.info(f"📊 **庫存總覽**｜ 總市值: **${total_mv:,.0f}** ｜ 總投入成本: **${total_cost:,.0f}** ｜ 總帳面損益: **${total_pl:,.0f}**")

        cols_show = ["代號", "庫存", "平均成本", "市價", "波動率%", "市值", "帳面損益", "成本殖利率%", "含息總報%", "填息"]
        
        tab_div, tab_swing = st.tabs(["💰 存股 / 基金", "🚀 波段交易"])
        with tab_div:
            div_assets = portfolio_df[portfolio_df['策略'] == 'Dividend']
            if not div_assets.empty: st.dataframe(div_assets[cols_show], use_container_width=True, hide_index=True)
            else: st.write("無符合條件的存股資產")
        with tab_swing:
            swing_assets = portfolio_df[portfolio_df['策略'] == 'Swing']
            if not swing_assets.empty: st.dataframe(swing_assets[cols_show], use_container_width=True, hide_index=True)
            else: st.write("無符合條件的波段資產")
            
else:
    st.info("尚無資料，請先輸入交易。")
