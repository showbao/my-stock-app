# Version: v6.0 (Final Architecture: Unified Swing, Tiered Metrics, Optimized Layout)
# CTOSignature: Strategy Merge (Swing), Refined KPIs, Bottom Inventory Metrics
import streamlit as st
import pandas as pd
import yfinance as yf
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, date
import numpy as np
from scipy import optimize
import altair as alt

# ==========================================
# 1. 系統設定與連線
# ==========================================
st.set_page_config(page_title="投資戰情室 v6.0", layout="wide")

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
# 2. 核心邏輯函數 (含策略合併與中文化)
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

def normalize_data(df):
    """
    資料標準化：
    1. 動作中文化
    2. 策略合併：將 'Swing Short', 'Swing Long' 等全部統一為 '波段'
    """
    if df.empty: return df
    
    act_map = {
        'Buy': '買入', 'Sell': '賣出', 'Dividend': '領息', 'Split': '分割',
        'Buy (Buy)': '買入', 'Sell (Sell)': '賣出'
    }
    
    # 策略正規化：強制合併
    strat_map = {
        'Dividend': '存股',
        'Swing': '波段',
        'Swing Short': '波段', # 合併
        'Swing Long': '波段',  # 合併
        '波段-短期': '波段',   # 合併
        '波段-長期': '波段',   # 合併
        '波動': '波段',       # 容錯
        '波動-短期': '波段',
        '波動-長期': '波段'
    }
    
    type_map = {'Stock': '股票', 'Fund': '基金'}

    if 'Action' in df.columns:
        df['Action'] = df['Action'].replace(act_map)
    if 'Strategy' in df.columns:
        # 使用 replace 進行取代
        for old, new in strat_map.items():
            df['Strategy'] = df['Strategy'].str.replace(old, new, regex=False)
    if 'Type' in df.columns:
        df['Type'] = df['Type'].replace(type_map)
        
    return df

def load_data():
    records_data = ws_records.get_all_records()
    df = pd.DataFrame(records_data)
    funds_data = ws_funds.get_all_records()
    df_funds = pd.DataFrame(funds_data)
    
    if df.empty:
        return df, df_funds, 32.0
        
    numeric_cols = ['Price', 'Shares', 'Fee', 'Total_Amount']
    for col in numeric_cols:
        if df[col].dtype == object:
             df[col] = df[col].astype(str).str.replace(',','').str.replace('$','')
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    
    # 資料標準化 (v6.0 Key Step)
    df = normalize_data(df)
    
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
    try:
        return optimize.newton(lambda r: xnpv(r, amounts, dates), 0.1)
    except:
        return None

def calculate_portfolio(df, df_funds, current_usd_rate):
    portfolio = {}
    trade_log = [] 
    
    df = df.sort_values('Date')
    
    for _, row in df.iterrows():
        ticker = row['Ticker']
        action = row['Action']
        qty = row['Shares']
        amount = row['Total_Amount']
        date_txn = row['Date']
        typ = row['Type']
        strat = str(row['Strategy'])
        
        if ticker not in portfolio:
            portfolio[ticker] = {
                'shares': 0, 'total_cost': 0, 'realized_pl': 0, 
                'dividend_collected': 0, 'type': typ, 
                'strategy': strat
            }
        
        p = portfolio[ticker]
        p['strategy'] = strat # 更新為最新策略 (通常是同一隻股票策略一致)

        if action == '買入':
            p['shares'] += qty
            p['total_cost'] += amount
            
        elif action == '賣出':
            if p['shares'] > 0:
                pct_sold = qty / p['shares']
                cost_of_sold_shares = p['total_cost'] * pct_sold
                pnl = amount - cost_of_sold_shares
                
                p['realized_pl'] += pnl
                p['total_cost'] -= cost_of_sold_shares
                p['shares'] -= qty
                
                trade_log.append({
                    'Date': date_txn,
                    'Ticker': ticker,
                    'Strategy': p['strategy'],
                    'PnL': pnl,
                    'SellAmount': amount
                })

                if p['shares'] <= 0.001: 
                    p['shares'] = 0
                    p['total_cost'] = 0
                    
        elif action == '領息':
            p['dividend_collected'] += amount
            
        elif action == '分割': 
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
            if data['type'] == '股票':
                current_price, volatility = get_stock_data(ticker)
                market_value = current_price * data['shares']
            elif data['type'] == '基金':
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
                "庫存現值": round(market_value, 0),
                "帳面損益": round(unrealized_pl, 0),
                "成本殖利率%": round(yield_on_cost, 2),
                "含息總報%": round(roi_total, 2),
                "已領股息": round(data['dividend_collected'], 0),
                "填息": fill_status,
                "總成本": round(data['total_cost'], 0)
            })
            
    return pd.DataFrame(results), pd.DataFrame(trade_log)

def analyze_period_advanced(df, start_date, end_date, selected_tickers, current_portfolio_df, trade_log_df, strategy_filter=None):
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    if selected_tickers:
        mask = mask & (df['Ticker'].isin(selected_tickers))
    if strategy_filter:
        mask = mask & (df['Strategy'].str.contains(strategy_filter, na=False))
        
    period_df = df[mask].copy()
    if period_df.empty: return None, pd.DataFrame(), pd.DataFrame()

    total_dividend = period_df[period_df['Action'] == '領息']['Total_Amount'].sum()
    total_buy = period_df[period_df['Action'] == '買入']['Total_Amount'].sum()
    total_sell = period_df[period_df['Action'] == '賣出']['Total_Amount'].sum()
    
    ending_inventory_value = 0
    total_cost_basis = 0
    is_current = end_date >= datetime.now().date()
    
    if is_current and not current_portfolio_df.empty:
        target_inv = current_portfolio_df
        if selected_tickers:
             target_inv = target_inv[target_inv['代號'].isin(selected_tickers)]
        if strategy_filter:
             target_inv = target_inv[target_inv['策略'].str.contains(strategy_filter, na=False)]
        ending_inventory_value = target_inv['庫存現值'].sum()
        total_cost_basis = target_inv['總成本'].sum()

    # 波段相關指標
    win_rate = 0
    profit_factor = 0 # 雖然 v6.0 不顯示，但保留運算邏輯
    realized_pnl_period = 0
    
    if not trade_log_df.empty:
        t_mask = (trade_log_df['Date'] >= start_date) & (trade_log_df['Date'] <= end_date)
        if selected_tickers:
            t_mask = t_mask & (trade_log_df['Ticker'].isin(selected_tickers))
        if strategy_filter:
            t_mask = t_mask & (trade_log_df['Strategy'].str.contains(strategy_filter, na=False))
        period_trades = trade_log_df[t_mask]
        
        if not period_trades.empty:
            realized_pnl_period = period_trades['PnL'].sum()
            wins = period_trades[period_trades['PnL'] > 0]
            if len(period_trades) > 0:
                win_rate = (len(wins) / len(period_trades)) * 100

    # XIRR 現金流
    cash_flows = []
    for _, row in period_df.iterrows():
        d = row['Date']
        amt = row['Total_Amount']
        act = row['Action']
        if act == '買入':
            cash_flows.append((d, -amt))
        elif act in ['賣出', '領息']:
            cash_flows.append((d, amt))
    if ending_inventory_value > 0:
        cash_flows.append((end_date, ending_inventory_value))
    xirr_val = xirr(cash_flows)
    if xirr_val: xirr_val *= 100 

    # 存股相關指標
    yoc_period = 0
    if total_cost_basis > 0:
        yoc_period = (total_dividend / total_cost_basis) * 100
    
    payback_progress = 0 
    if total_buy > 0:
        payback_progress = (total_dividend / total_buy) * 100

    summary = {
        "已領股息": total_dividend,
        "淨現金流": (total_sell + total_dividend) - total_buy,
        "總投入": total_buy,
        "庫存現值": ending_inventory_value,
        "累積總損益": (ending_inventory_value + total_sell + total_dividend) - total_buy,
        "已實現損益": realized_pnl_period,
        "勝率%": win_rate,
        "XIRR%": xirr_val,
        "YoC%": yoc_period,
        "回本率%": payback_progress
    }

    years_data = []
    start_y = start_date.year
    end_y = end_date.year
    for y in range(start_y, end_y + 1):
        y_df = period_df[pd.to_datetime(period_df['Date']).dt.year == y]
        if not y_df.empty:
            y_div = y_df[y_df['Action'] == '領息']['Total_Amount'].sum()
            y_buy = y_df[y_df['Action'] == '買入']['Total_Amount'].sum()
            y_sell = y_df[y_df['Action'] == '賣出']['Total_Amount'].sum()
            y_net = (y_sell + y_div) - y_buy
            years_data.append({
                "年度": str(y),
                "已領股息": f"${y_div:,.0f}",
                "投入": f"${y_buy:,.0f}",
                "變現": f"${y_sell:,.0f}",
                "淨流": f"${y_net:,.0f}"
            })
    years_df = pd.DataFrame(years_data)
    return summary, period_df, years_df

def handle_transaction_submit(date_in, ticker, type_display, strategy_list, action_display, price, shares, fee, total_amt, note):
    db_strat = ",".join(strategy_list)
    final_shares = shares
    final_price = price
    final_fee = fee
    final_total = total_amt

    if final_fee == 0 and action_display in ["買入", "賣出"]:
        calculated_fee = int(price * shares * 0.001425)
        final_fee = calculated_fee

    if action_display == "領息":
        final_shares = 0
        final_price = 0
        if final_total == 0:
                st.error("領息金額不能為 0")
                return False
    elif action_display == "分割":
        final_total = 0
        final_price = 0
    else:
        if final_total == 0:
            basic_amt = price * shares
            if action_display == "買入":
                final_total = basic_amt + final_fee
            elif action_display == "賣出":
                tax_rate = 0.003
                tax = int(basic_amt * tax_rate)
                final_total = basic_amt - final_fee - tax
                if tax > 0:
                    note = f"{note} (系統自動扣除證交稅約 ${tax})".strip()

    new_row = [str(date_in), ticker, type_display, db_strat, action_display, final_price, final_shares, final_fee, final_total, note]
    ws_records.append_row(new_row)
    return True

# ==========================================
# 4. 儀表板渲染組件 (v6.0 新指標配置)
# ==========================================
def render_metrics_cards(summary, mode):
    k1, k2, k3, k4 = st.columns(4)
    
    if mode == "swing": # 波段
        k1.metric("累積總損益", f"${summary['累積總損益']:,.0f}", help="包含未實現、已實現與股息")
        k2.metric("年化報酬率 (XIRR)", f"{summary['XIRR%']:.2f}%" if summary['XIRR%'] else "N/A")
        k3.metric("已實現損益", f"${summary['已實現損益']:,.0f}", help="實際賣出落袋金額")
        k4.metric("交易勝率", f"{summary['勝率%']:.1f}%")
        
    elif mode == "dividend": # 存股
        k1.metric("成本殖利率 (YoC)", f"{summary['YoC%']:.2f}%")
        k2.metric("已領股息", f"${summary['已領股息']:,.0f}")
        k3.metric("回本率", f"{summary['回本率%']:.1f}%")
        k4.metric("庫存現值", f"${summary['庫存現值']:,.0f}")
        
    else: # general (全域總覽)
        k1.metric("累積總損益", f"${summary['累積總損益']:,.0f}")
        k2.metric("年化報酬率 (XIRR)", f"{summary['XIRR%']:.2f}%" if summary['XIRR%'] else "N/A")
        k3.metric("已領股息", f"${summary['已領股息']:,.0f}")

def render_chart_swing(trade_log_df, strategy_filter=None):
    if not trade_log_df.empty:
        if strategy_filter:
            swing_trades = trade_log_df[trade_log_df['Strategy'].str.contains(strategy_filter, na=False)]
        else:
            swing_trades = trade_log_df
            
        if not swing_trades.empty:
            swing_trades = swing_trades.sort_values('Date')
            swing_trades['累積損益'] = swing_trades['PnL'].cumsum()
            
            line = alt.Chart(swing_trades).mark_line(color='purple').encode(
                x='Date:T', y='累積損益:Q', tooltip=['Date', '累積損益']
            )
            points = alt.Chart(swing_trades).mark_circle(size=60).encode(
                x='Date:T', y='PnL:Q', 
                color=alt.condition(alt.datum.PnL > 0, alt.value("green"), alt.value("red")), 
                tooltip=['Date', 'Ticker', 'PnL']
            )
            st.altair_chart((line + points).interactive(), use_container_width=True)
        else:
            st.info("尚無交易損益紀錄")

def render_chart_dividend(years_df):
    if not years_df.empty:
        bar_data = years_df[['年度', '已領股息']].copy()
        bar_data['已領股息'] = bar_data['已領股息'].str.replace('$','').str.replace(',','').astype(float)
        chart = alt.Chart(bar_data).mark_bar(color='#ff7f0e').encode(
            x='年度:O', y='已領股息:Q', tooltip=['年度', '已領股息']
        ).properties(height=300).interactive()
        st.altair_chart(chart, use_container_width=True)

def render_strategy_view(df, start_date, end_date, selected_tickers, strategy_filter, full_portfolio_df, trade_log_df, mode_name):
    summary, period_df, years_df = analyze_period_advanced(
        df, start_date, end_date, selected_tickers, full_portfolio_df, trade_log_df, strategy_filter
    )
    if summary:
        render_metrics_cards(summary, mode_name)
        st.divider()
        if mode_name == "dividend":
            st.markdown("##### 💰 歷年股息成長")
            render_chart_dividend(years_df)
        elif "swing" in mode_name:
            st.markdown("##### 📈 交易損益曲線 & 落點")
            render_chart_swing(trade_log_df, strategy_filter)
        if not years_df.empty:
            st.markdown("##### 📅 年度績效表")
            st.dataframe(years_df, use_container_width=True, hide_index=True)
    else:
        st.info("此區間無相關數據")

# ==========================================
# 5. 主程式佈局
# ==========================================
st.title("📊 投資戰情室 v6.0")

df, df_funds, usd_rate = load_data()
if df.empty:
    st.warning("目前無任何交易紀錄")
    st.stop()

all_tickers = df['Ticker'].unique().tolist()
full_portfolio_df, trade_log_df = calculate_portfolio(df, df_funds, usd_rate)

# --- 1. 最上方：全域總覽區 (All Time) ---
st.markdown("### 🌍 全資產總覽 (All Time)")
# 計算全域總覽 (無策略篩選)
total_summary, _, _ = analyze_period_advanced(df, df['Date'].min(), date.today(), None, full_portfolio_df, trade_log_df, None)
if total_summary:
    # v6.0 老闆視角三指標
    k1, k2, k3 = st.columns(3)
    k1.metric("累積總損益", f"${total_summary['累積總損益']:,.0f}", help="含未實現+已實現+股息")
    k2.metric("年化報酬率 (XIRR)", f"{total_summary['XIRR%']:.2f}%" if total_summary['XIRR%'] else "N/A")
    k3.metric("已領股息", f"${total_summary['已領股息']:,.0f}")
    
    chart_data = []
    cum_cost = 0; cum_profit = 0
    sorted_df = df.sort_values('Date')
    for _, r in sorted_df.iterrows():
        act = r['Action']; amt = r['Total_Amount']
        if act == '買入': cum_cost += amt
        elif act == '賣出': 
            cum_cost -= (amt * 0.8)
            cum_profit += (amt * 0.2)
        elif act == '領息': cum_profit += amt
        chart_data.append({'日期': r['Date'], '淨本金': cum_cost, '累積獲利': cum_profit})
    
    if chart_data:
        c_df = pd.DataFrame(chart_data).melt('日期', var_name='類別', value_name='金額')
        chart = alt.Chart(c_df).mark_area().encode(
            x='日期:T', y='金額:Q', color=alt.Color('類別:N', scale=alt.Scale(range=['#1f77b4', '#2ca02c'])), 
            tooltip=['日期', '類別', '金額']
        ).properties(height=250).interactive()
        st.altair_chart(chart, use_container_width=True)

st.divider()

# --- 2. 左右分欄篩選與報表 ---
col_filter, col_display = st.columns([1, 3])

# [左欄] 篩選區
with col_filter:
    st.subheader("🔍 篩選條件")
    min_date = df['Date'].min()
    max_date = date.today()
    analysis_start = st.date_input("開始日期", value=min_date, min_value=min_date, max_value=max_date)
    analysis_end = st.date_input("結束日期", value=max_date, min_value=min_date, max_value=max_date)
    selected_tickers = st.multiselect("投資標的 (可複選)", all_tickers, default=None)
    st.caption("💡 未選擇標的則顯示所有策略彙整。")

# [右欄] 智慧呈現區 (v6.0 策略合併邏輯)
with col_display:
    if not selected_tickers:
        t1, t2 = st.tabs(["⚡ 波段", "💰 存股"])
        with t1:
            st.caption("所有「波段」策略彙整")
            render_strategy_view(df, analysis_start, analysis_end, None, "波段", full_portfolio_df, trade_log_df, "swing")
        with t2:
            st.caption("所有「存股」策略彙整")
            render_strategy_view(df, analysis_start, analysis_end, None, "存股", full_portfolio_df, trade_log_df, "dividend")
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
                    render_strategy_view(df, analysis_start, analysis_end, [ticker], None, full_portfolio_df, trade_log_df, "general")
                else:
                    if has_swing:
                        with st.expander("⚡ 策略分析：波段", expanded=True):
                            render_strategy_view(df, analysis_start, analysis_end, [ticker], "波段", full_portfolio_df, trade_log_df, "swing")
                    if has_div:
                        with st.expander("💰 策略分析：存股", expanded=True):
                            render_strategy_view(df, analysis_start, analysis_end, [ticker], "存股", full_portfolio_df, trade_log_df, "dividend")

st.divider()

# --- 3. 庫存與新增交易區 ---
st.markdown("### 📦 庫存管理與交易登錄")

if not full_portfolio_df.empty:
    # v6.0 庫存區三指標 (置頂)
    total_mv = full_portfolio_df['庫存現值'].sum()
    total_cost = full_portfolio_df['總成本'].sum()
    total_unrealized = full_portfolio_df['帳面損益'].sum()
    
    i1, i2, i3 = st.columns(3)
    i1.metric("庫存總現值", f"${total_mv:,.0f}")
    i2.metric("庫存總成本", f"${total_cost:,.0f}")
    i3.metric("庫存帳面損益", f"${total_unrealized:,.0f}", delta_color="normal")
    
    st.write("") # Spacer

    cols_show = ["代號", "庫存", "平均成本", "市價", "庫存現值", "帳面損益", "含息總報%", "策略"]
    event = st.dataframe(
        full_portfolio_df[cols_show],
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="inventory_table"
    )
    
    default_ticker = ""
    default_strat = ["存股"]
    
    if len(event.selection.rows) > 0:
        selected_index = event.selection.rows[0]
        selected_row = full_portfolio_df.iloc[selected_index]
        default_ticker = selected_row['代號']
        
        last_strat_str = df[df['Ticker'] == default_ticker].iloc[-1]['Strategy']
        possible_strats = ["存股", "波段"]
        for s in possible_strats:
            if s in last_strat_str:
                default_strat = [s]
                break
        
        st.divider()
        st.markdown(f"#### 📂 {default_ticker} 歷史與操作")
        target_hist = df[df['Ticker'] == default_ticker].sort_values('Date', ascending=False)
        st.dataframe(target_hist[['Date', 'Action', 'Strategy', 'Price', 'Shares', 'Total_Amount']].head(5), use_container_width=True, hide_index=True)

    col_input1, col_input2 = st.columns([2, 1])
    
    with col_input1:
        with st.form("bottom_entry_form", clear_on_submit=True):
            st.markdown(f"**➕ 新增交易** {f'({default_ticker})' if default_ticker else ''}")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                d_date = st.date_input("日期")
                d_ticker = st.text_input("代號", value=default_ticker).upper()
            with c2:
                d_type = st.selectbox("種類", ["股票", "基金"])
                d_action = st.selectbox("動作", ["買入", "賣出", "領息", "分割"])
            with c3:
                d_strat = st.multiselect("策略", ["存股", "波段"], default=default_strat)
                d_price = st.number_input("單價", min_value=0.0, format="%.2f")
            with c4:
                d_shares = st.number_input("股數", step=100.0)
                d_fee = st.number_input("手續費 (0自動算)", min_value=0)
            
            c5, c6 = st.columns([3, 1])
            with c5:
                d_total = st.number_input("總金額 (0自動算)", step=1000.0)
                d_note = st.text_input("備註")
            with c6:
                st.write("")
                st.write("")
                submitted = st.form_submit_button("送出交易", use_container_width=True)
            
            if submitted:
                if not d_ticker:
                    st.error("請輸入代號")
                else:
                    success = handle_transaction_submit(d_date, d_ticker, d_type, d_strat, d_action, d_price, d_shares, d_fee, d_total, d_note)
                    if success:
                        st.success(f"已儲存 {d_ticker}！")
                        st.cache_data.clear()

    with col_input2:
        with st.form("bottom_fund_form", clear_on_submit=True):
            st.markdown("**💵 更新基金淨值**")
            f_ticker = st.text_input("基金代號").upper()
            f_net_val = st.number_input("最新淨值 (USD)", min_value=0.0, format="%.4f")
            st.write("")
            f_btn = st.form_submit_button("更新", use_container_width=True)
            if f_btn:
                try:
                    cell = ws_funds.find(f_ticker)
                    ws_funds.update_cell(cell.row, 2, f_net_val)
                    ws_funds.update_cell(cell.row, 3, str(datetime.now().date()))
                except:
                    ws_funds.append_row([f_ticker, f_net_val, str(datetime.now().date())])
                st.success("更新成功")
                st.cache_data.clear()
else:
    st.info("尚無資料，請先新增第一筆交易。")
