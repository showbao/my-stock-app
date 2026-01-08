# Version: v3.2
# CTOSignature: Unified UI, Auto-Fee Logic (0.1425%), Buy/Sell Math Fix
import streamlit as st
import pandas as pd
import yfinance as yf
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, date
import numpy as np

# ==========================================
# 1. 系統設定與連線
# ==========================================
st.set_page_config(page_title="投資追蹤指揮中心", layout="wide")

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
                'dividend_collected': 0, 'type': typ, 
                'strategy': str(row['Strategy'])
            }
            
        p = portfolio[ticker]
        p['strategy'] = str(row['Strategy']) 

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

def analyze_period(df, start_date, end_date, selected_tickers, current_portfolio_df):
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    if selected_tickers:
        mask = mask & (df['Ticker'].isin(selected_tickers))
    period_df = df[mask].copy()
    
    if period_df.empty: return None, pd.DataFrame(), pd.DataFrame()

    total_dividend = period_df[period_df['Action'] == 'Dividend']['Total_Amount'].sum()
    total_buy = period_df[period_df['Action'] == 'Buy']['Total_Amount'].sum()
    total_sell = period_df[period_df['Action'] == 'Sell']['Total_Amount'].sum()
    
    ending_inventory_value = 0
    is_current = end_date >= datetime.now().date()
    
    if is_current and not current_portfolio_df.empty:
        if selected_tickers:
             target_inventory = current_portfolio_df[current_portfolio_df['代號'].isin(selected_tickers)]
        else:
             target_inventory = current_portfolio_df
        ending_inventory_value = target_inventory['市值'].sum()

    total_recovered = total_sell + total_dividend + ending_inventory_value
    return_rate = (total_recovered / total_buy * 100) if total_buy > 0 else 0
    
    days = (end_date - start_date).days
    if days > 365 and total_buy > 0 and total_recovered > 0:
        years = days / 365
        annualized_return = (pow(total_recovered / total_buy, 1/years) - 1) * 100
    else:
        annualized_return = None

    summary = {
        "總領股息": total_dividend,
        "淨現金流": (total_sell + total_dividend) - total_buy,
        "總回報率%": return_rate - 100,
        "年化回收率%": annualized_return,
        "期末庫存市值": ending_inventory_value
    }

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
# 3. 統一的交易輸入處理函數 (Helper)
# ==========================================

def handle_transaction_submit(date_in, ticker, type_display, strategy_list, action_display, price, shares, fee, total_amt, note):
    """統一處理交易計算與寫入，包含手續費邏輯"""
    
    typ_map = {"股票 (Stock)": "Stock", "基金 (Fund)": "Fund"}
    act_map = {"買入 (Buy)": "Buy", "賣出 (Sell)": "Sell", "領息 (Dividend)": "Dividend", "分割/減資 (Split)": "Split"}
    strat_map = {"存股 (Dividend)": "Dividend", "波段 (Swing)": "Swing"}
    
    # 策略處理
    selected_strats = [strat_map[s] for s in strategy_list]
    db_strat = ",".join(selected_strats)
    db_type = typ_map[type_display]
    db_action = act_map[action_display]
    
    final_shares = shares
    final_price = price
    final_fee = fee
    final_total = total_amt

    # --- 1. 手續費自動計算邏輯 ---
    # 如果使用者沒填手續費 (0)，且不是領息或分割，則自動計算
    if final_fee == 0 and db_action in ["Buy", "Sell"]:
        # 費率 0.1425% (0.001425)
        calculated_fee = int(price * shares * 0.001425)
        final_fee = calculated_fee
        # 這裡不設低消 20，依需求單純算 %

    # --- 2. 總金額自動計算邏輯 (含買賣加減修正) ---
    if db_action == "Dividend":
        final_shares = 0
        final_price = 0
        if final_total == 0:
                st.error("領息金額不能為 0")
                return False # 失敗
    elif db_action == "Split":
        final_total = 0
        final_price = 0
    else:
        # 如果使用者沒填總金額，自動計算
        if final_total == 0:
            basic_amt = price * shares
            if db_action == "Buy":
                # 買入 = 股價 + 手續費
                final_total = basic_amt + final_fee
            elif db_action == "Sell":
                # 賣出 = 股價 - 手續費 - 證交稅
                # 證交稅估算: 股票 0.3%, ETF 0.1%。這裡統一先用 0.3% 估算以求保守，或單純減手續費
                # 為了避免複雜，我們這裡先只減手續費 (依您的需求)，但強烈建議扣稅
                tax_rate = 0.003 # 預設千分之三
                tax = int(basic_amt * tax_rate)
                final_total = basic_amt - final_fee - tax
                
                # 顯示提示讓使用者知道我們扣了稅
                if tax > 0:
                    note = f"{note} (系統自動扣除證交稅約 ${tax})".strip()

    new_row = [str(date_in), ticker, db_type, db_strat, db_action, final_price, final_shares, final_fee, final_total, note]
    ws_records.append_row(new_row)
    return True

# ==========================================
# 4. 彈出視窗與介面 (Modals)
# ==========================================

@st.dialog("新增交易紀錄")
def entry_dialog(default_ticker=None):
    with st.form("entry_form_dialog", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            date_in = st.date_input("日期")
            ticker_val = default_ticker if default_ticker else ""
            ticker = st.text_input("代號", value=ticker_val).upper()
            typ_display = st.selectbox("種類", ["股票 (Stock)", "基金 (Fund)"])
        
        with col2:
            strategy_opts = ["存股 (Dividend)", "波段 (Swing)"]
            strategy_display = st.multiselect("策略 (可複選)", strategy_opts, default=["存股 (Dividend)"])
            action_display = st.selectbox("動作", ["買入 (Buy)", "賣出 (Sell)", "領息 (Dividend)", "分割/減資 (Split)"])

        if "Split" in action_display:
            st.info("💡 分割填正數，減資填負數，金額填 0")

        col3, col4 = st.columns(2)
        with col3:
            price = st.number_input("單價 / 淨值", min_value=0.0, format="%.2f")
            shares = st.number_input("股數 / 單位數", min_value=-100000.0, max_value=100000.0, format="%.2f")
        with col4:
            # 加入手續費欄位
            fee = st.number_input("手續費 (0為自動以0.1425%計算)", min_value=0, value=0)
            total_amt = st.number_input("總金額 (0為自動計算)", min_value=0.0, format="%.2f")
        
        note = st.text_input("備註")
        
        submit_col, close_col = st.columns([1, 1])
        with submit_col:
            submitted = st.form_submit_button("送出並新增下一筆", use_container_width=True)
        
        if submitted:
            success = handle_transaction_submit(date_in, ticker, typ_display, strategy_display, action_display, price, shares, fee, total_amt, note)
            if success:
                st.success(f"已儲存 {ticker}！(若未填金額，系統已自動依買賣加減手續費與證交稅)")
                st.cache_data.clear()

    if st.button("關閉視窗回到主畫面"):
        st.rerun()

@st.dialog("更新基金淨值")
def fund_update_dialog():
    with st.form("fund_form", clear_on_submit=True):
        f_ticker = st.text_input("基金代號").upper()
        f_net_val = st.number_input("最新淨值 (USD)", min_value=0.0, format="%.4f")
        f_submitted = st.form_submit_button("更新並輸入下一筆", use_container_width=True)
        
        if f_submitted:
            try:
                cell = ws_funds.find(f_ticker)
                ws_funds.update_cell(cell.row, 2, f_net_val)
                ws_funds.update_cell(cell.row, 3, str(datetime.now().date()))
            except:
                ws_funds.append_row([f_ticker, f_net_val, str(datetime.now().date())])
            st.success(f"{f_ticker} 更新成功！")
            st.cache_data.clear()
    
    if st.button("關閉視窗"):
        st.rerun()

# ==========================================
# 5. 前端介面組合 (Main Layout)
# ==========================================
st.title("📊 投資戰情室 v3.2")

# --- Top Buttons ---
col_btn1, col_btn2, col_dummy = st.columns([1, 1, 4])
with col_btn1:
    if st.button("➕ 新增股票/基金交易", type="primary", use_container_width=True):
        entry_dialog()
with col_btn2:
    if st.button("💵 更新基金淨值", use_container_width=True):
        fund_update_dialog()

# --- 載入資料 ---
df, df_funds, usd_rate = load_data()
_df = df.copy() 
all_tickers = df['Ticker'].unique().tolist() if not df.empty else []

# --- 戰情分析區 ---
with st.expander("🔍 全域戰情分析 & 篩選器", expanded=True):
    c_s1, c_s2, c_s3 = st.columns([1, 1, 2])
    with c_s1:
        analysis_start = st.date_input("開始日期", value=date(datetime.now().year, 1, 1))
    with c_s2:
        analysis_end = st.date_input("結束日期", value=datetime.now().date())
    with c_s3:
        selected_tickers_dashboard = st.multiselect("篩選代號 (僅影響此分析區塊)", all_tickers)

    if not df.empty:
        full_portfolio_df = calculate_portfolio(df, df_funds, usd_rate)
        summary, period_df, years_df = analyze_period(df, analysis_start, analysis_end, selected_tickers_dashboard, full_portfolio_df)
        
        if summary:
            st.markdown("#### 📈 績效指標")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("區間已領股息", f"${summary['總領股息']:,.0f}")
            k2.metric("區間淨現金流", f"${summary['淨現金流']:,.0f}")
            if summary['年化回收率%'] is not None:
                k3.metric("年化報酬率 (CAGR)", f"{summary['年化回收率%']:.2f}%")
            else:
                k3.metric("區間總回報", f"{summary['總回報率%']:.2f}%")
            k4.metric("目前庫存價值", f"${summary['期末庫存市值']:,.0f}")

            if not years_df.empty and len(years_df) > 1:
                st.markdown("#### 📅 年度分列比較")
                st.dataframe(years_df, use_container_width=True, hide_index=True)
            
            # 分面詳情
            if selected_tickers_dashboard:
                st.divider()
                st.markdown("#### 🏷️ 個股交易詳情 (分析區)")
                tabs = st.tabs(selected_tickers_dashboard)
                for i, ticker in enumerate(selected_tickers_dashboard):
                    with tabs[i]:
                        ticker_history = df[df['Ticker'] == ticker].sort_values('Date', ascending=False)
                        display_history = ticker_history[['Date', 'Action', 'Strategy', 'Price', 'Shares', 'Total_Amount', 'Note']].copy()
                        display_history.columns = ['日期', '動作', '策略', '單價', '股數', '總金額', '備註']
                        st.dataframe(display_history, use_container_width=True, hide_index=True)
                        if st.button(f"➕ 新增 {ticker} 交易", key=f"add_btn_dash_{ticker}"):
                            entry_dialog(default_ticker=ticker)

# --- 現有庫存區 ---
st.markdown("### 📦 現有庫存總覽")
if not df.empty and not full_portfolio_df.empty:
    
    total_mv = full_portfolio_df['市值'].sum()
    total_cost = full_portfolio_df['總成本'].sum()
    total_pl = full_portfolio_df['帳面損益'].sum()
    st.info(f"📊 **合計 (全持股)**｜ 市值: **${total_mv:,.0f}** ｜ 成本: **${total_cost:,.0f}** ｜ 損益: **${total_pl:,.0f}**")

    cols_show = ["代號", "庫存", "平均成本", "市價", "市值", "帳面損益", "含息總報%", "策略"]
    st.caption("👇 **點擊表格任一行，下方即會顯示該標的詳細歷史與操作**")
    
    event = st.dataframe(
        full_portfolio_df[cols_show],
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="inventory_table"
    )
    
    # --- 下方展開區 ---
    if len(event.selection.rows) > 0:
        selected_index = event.selection.rows[0]
        selected_row = full_portfolio_df.iloc[selected_index]
        target_ticker = selected_row['代號']
        
        st.divider()
        st.markdown(f"### 📂 {target_ticker} 交易詳情")
        
        t1, t2 = st.tabs(["📜 歷史紀錄", "⚡ 快速新增"])
        
        with t1:
            target_df = df[df['Ticker'] == target_ticker].sort_values('Date', ascending=False)
            if not target_df.empty:
                view_df = target_df[['Date', 'Action', 'Strategy', 'Price', 'Shares', 'Fee', 'Total_Amount', 'Note']].copy()
                view_df.columns = ['日期', '動作', '策略', '單價', '股數', '手續費', '總金額', '備註']
                st.dataframe(view_df, use_container_width=True, hide_index=True)
            else:
                st.info("無交易紀錄")
        
        with t2:
            # 這裡改成跟上方 Dialog 一樣的完整表單邏輯
            with st.form(f"quick_add_inline_{target_ticker}", clear_on_submit=True):
                # 這裡為了版面好看，我們排版稍微緊湊一點
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    q_date = st.date_input("日期")
                    q_action = st.selectbox("動作", ["買入 (Buy)", "賣出 (Sell)", "領息 (Dividend)"]) # 簡化版動作
                with c2:
                    q_shares = st.number_input("股數", step=100.0)
                    q_price = st.number_input("單價", step=0.1)
                with c3:
                    # v3.2 新增：手續費欄位
                    q_fee = st.number_input("手續費 (0自動算)", min_value=0)
                    q_total = st.number_input("總金額 (0自動算)", step=1000.0)
                with c4:
                    q_note = st.text_input("備註")
                    st.write("") # Spacer
                    st.write("") # Spacer
                    q_submit = st.form_submit_button(f"新增 {target_ticker}")
                
                if q_submit:
                    # 快速新增這裡預設策略為 "Dividend" (存股)，種類為 "Stock"
                    # 若要更複雜，也可以加欄位，但通常快速新增就是求快
                    success = handle_transaction_submit(
                        q_date, target_ticker, "股票 (Stock)", ["存股 (Dividend)"], q_action, 
                        q_price, q_shares, q_fee, q_total, q_note
                    )
                    if success:
                        st.success("已新增！請重新整理頁面。")
                        st.cache_data.clear()

else:
    st.info("尚無庫存或交易資料。")
