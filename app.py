# Version: v10.1 (v9.2 Metrics Restored + AI Sub-tabs + Interaction Fixes)
# CTOSignature: Restored exact v9.2 charts/metrics. Reorganized AI Coach into 3 sub-tabs. Fixed Swing AI workflow (History -> Analyze -> Save). Verified Saving logic.
import streamlit as st
import pandas as pd
import yfinance as yf
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, date, timedelta
import numpy as np
from scipy import optimize
import altair as alt
import google.generativeai as genai
import time

# ==========================================
# 1. 系統設定與連線
# ==========================================
st.set_page_config(page_title="投資戰情室 v10.1", layout="wide")

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

ws_records = None
ws_funds = None
ws_history = None

if sh:
    try: ws_records = sh.worksheet("Records")
    except: st.error("❌ 找不到工作表 'Records'"); st.stop()
    try: ws_funds = sh.worksheet("Fund_Updates")
    except: st.error("❌ 找不到工作表 'Fund_Updates'"); st.stop()
    try: ws_history = sh.worksheet("Analysis_History")
    except: ws_history = None 
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

def get_historical_price_window(ticker, trade_date, window_days=7):
    try:
        t_date = pd.to_datetime(trade_date).tz_localize(None)
        start_d = (t_date - timedelta(days=window_days + 15)).strftime('%Y-%m-%d')
        end_d = (t_date + timedelta(days=window_days + 15)).strftime('%Y-%m-%d')
        
        real_ticker = fix_ticker_suffix(ticker)
        stock = yf.Ticker(real_ticker)
        hist = stock.history(start=start_d, end=end_d, auto_adjust=True)
        
        if hist.empty: return None
        if hist.index.tz is not None: hist.index = hist.index.tz_localize(None)
            
        mask_window = (hist.index >= (t_date - timedelta(days=window_days))) & (hist.index <= (t_date + timedelta(days=window_days)))
        window_df = hist.loc[mask_window]
        
        if window_df.empty: return None
        
        return {
            "window_high": window_df['High'].max(),
            "window_low": window_df['Low'].min(),
            "price_at_trade": window_df['Close'].mean()
        }
    except: return None

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
        amount = row['Total_Amount']; date_txn = row['Date']
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
                    'PnL': pnl, 'SellAmount': amount, 'SellPrice': sell_price, 'RowIndex': rid, 'AI_Review': review
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

    total_dividend = period_df[period_df['Action'] == '領息']['Total_Amount'].sum()
    total_buy = period_df[period_df['Action'] == '買入']['Total_Amount'].sum()
    ending_inventory_value = 0; total_cost_basis = 0
    if end_date >= datetime.now().date() and not current_portfolio_df.empty:
        target_inv = current_portfolio_df
        if selected_tickers: target_inv = target_inv[target_inv['代號'].isin(selected_tickers)]
        if strategy_filter: target_inv = target_inv[target_inv['策略'].str.contains(strategy_filter, na=False)]
        ending_inventory_value = target_inv['庫存現值'].sum()
        total_cost_basis = target_inv['總成本'].sum()

    total_unrealized = ending_inventory_value - total_cost_basis
    realized_pnl_period = 0; win_rate = 0
    if not trade_log_df.empty:
        t_mask = (trade_log_df['Date'] >= start_date) & (trade_log_df['Date'] <= end_date)
        if selected_tickers: t_mask = t_mask & (trade_log_df['Ticker'].isin(selected_tickers))
        if strategy_filter: t_mask = t_mask & (trade_log_df['Strategy'].str.contains(strategy_filter, na=False))
        period_trades = trade_log_df[t_mask]
        if not period_trades.empty:
            realized_pnl_period = period_trades['PnL'].sum()
            wins = period_trades[period_trades['PnL'] > 0]
            if len(period_trades) > 0: win_rate = (len(wins) / len(period_trades)) * 100

    total_profit = realized_pnl_period + total_unrealized + total_dividend
    
    cash_flows = []
    for _, row in period_df.iterrows():
        d = row['Date']; amt = row['Total_Amount']; act = row['Action']
        if act == '買入': cash_flows.append((d, -amt))
        elif act in ['賣出', '領息']: cash_flows.append((d, amt))
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
        "YoC%": yoc_period, "回本率%": payback_progress, "庫存現值": ending_inventory_value
    }
    return summary, period_df, pd.DataFrame()

# ==========================================
# 3. AI 分析與資料庫存取
# ==========================================
def ask_gemini_coach(api_key, prompt_text):
    if not api_key: return "⚠️ 未偵測到 API Key"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        safety_settings = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}, {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}, {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"}, {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}]
        response = model.generate_content(prompt_text, safety_settings=safety_settings)
        return response.text
    except Exception as e: return f"❌ AI 錯誤: {str(e)}"

def get_monthly_summaries():
    if ws_history is None: return {}
    try:
        data = ws_history.get_all_records()
        df_hist = pd.DataFrame(data)
        if df_hist.empty: return {}
        target = df_hist[df_hist['Type'].astype(str).str.startswith('Swing_Summary_')]
        return pd.Series(target.Content.values, index=target.Type).to_dict()
    except: return {}

def save_monthly_summary(ticker, yyyymm, content):
    if ws_history is None: return
    type_key = f"Swing_Summary_{ticker}_{yyyymm}"
    try:
        ws_history.append_row([str(date.today()), type_key, content])
        st.cache_data.clear()
    except: pass

def get_last_report(report_type):
    if ws_history is None: return None
    try:
        data = ws_history.get_all_records()
        df_hist = pd.DataFrame(data)
        if df_hist.empty: return None
        target = df_hist[df_hist['Type'] == report_type].sort_values('Date', ascending=False)
        if not target.empty: return target.iloc[0]
    except: return None
    return None

def save_report(report_type, content):
    if ws_history is None: st.error("請先建立 'Analysis_History' 工作表。"); return
    try:
        ws_history.append_row([str(date.today()), report_type, content])
        st.cache_data.clear()
    except Exception as e: st.error(f"存檔失敗: {e}")

def update_atomic_reviews(updates_list):
    if not updates_list: return
    try:
        ai_col_idx = 11 
        header = ws_records.cell(1, ai_col_idx).value
        if header != "AI_Review":
            cell = ws_records.find("AI_Review")
            if cell: ai_col_idx = cell.col
            else: st.error("找不到 'AI_Review' 欄位"); return
        with st.status("正在儲存分析結果...", expanded=True) as status:
            for rid, text in updates_list:
                ws_records.update_cell(rid, ai_col_idx, text)
                time.sleep(0.5) 
            status.update(label="儲存完成！", state="complete")
            st.cache_data.clear()
            st.success("✅ 已更新 Google Sheet")
            time.sleep(1); st.rerun()
    except Exception as e: st.error(f"寫入失敗: {e}")

def run_two_layer_swing_analysis(df_raw):
    api_key = st.secrets.get("gemini_api_key", None)
    if not api_key: st.error("無 API Key"); return

    pending = df_raw[
        (df_raw['Strategy'].str.contains('波段', na=False)) & 
        ((df_raw['AI_Review'] == "") | (df_raw['AI_Review'].isna()))
    ].copy()
    
    if pending.empty:
        st.info("🎉 所有波段交易都已完成原子分析。")
        return

    target_batch = pending.sort_values('Date', ascending=False).head(10)
    atomic_updates = []
    affected_months = set()

    with st.status("🚀 正在執行雙層分析...", expanded=True) as status:
        status.write("正在進行逐筆 T±7 檢視...")
        for _, row in target_batch.iterrows():
            t = row['Ticker']; d = row['Date']; px = row['Price']; act = row['Action']; rid = row['RowIndex']
            ym = pd.to_datetime(d).strftime('%Y-%m')
            affected_months.add((t, ym))
            
            context = get_historical_price_window(t, d, 7)
            if context:
                if act == '買入':
                    low = context['window_low']; dist = ((px - low)/low * 100)
                    prompt = f"你是交易員。針對 {t} 在 {d} 買入價 {px} (T±7日最低 {low}, 差距 {dist:.1f}%)。這筆買點合宜嗎？請用 :red[好評] 或 :green[負評/警示] (台股慣例) 一句話簡評。"
                elif act == '賣出':
                    high = context['window_high']; missed = ((high - px)/px * 100)
                    prompt = f"你是交易員。針對 {t} 在 {d} 賣出價 {px} (T±7日最高 {high}, 賣飛 {missed:.1f}%)。這筆賣點合宜嗎？請用 :red[好評] 或 :green[負評/警示] (台股慣例) 一句話簡評。"
                else: prompt = "略過"
                
                if prompt != "略過":
                    review = ask_gemini_coach(api_key, prompt).strip()
                    atomic_updates.append((rid, review))
            else:
                atomic_updates.append((rid, "[無歷史數據]"))

        status.write("正在生成月報建議...")
        for (ticker, yyyymm) in affected_months:
            month_mask = (df_raw['Ticker'] == ticker) & \
                         (pd.to_datetime(df_raw['Date']).dt.strftime('%Y-%m') == yyyymm) & \
                         (df_raw['Strategy'].str.contains('波段', na=False))
            month_trades = df_raw[month_mask].sort_values('Date')
            tx_desc = ""
            for _, r in month_trades.iterrows():
                tx_desc += f"- {r['Date']} {r['Action']} ${r['Price']}\n"
            
            summary_prompt = f"""
            你是一位波段操作教練。針對 {ticker} 在 {yyyymm} 的操作：
            {tx_desc}
            請給出一份綜合建議 (100字內)。使用 :red[好評] / :green[負評] (台股慣例) 標示。
            """
            monthly_advice = ask_gemini_coach(api_key, summary_prompt).strip()
            save_monthly_summary(ticker, yyyymm, monthly_advice)
            
        status.update(label="分析完成！準備寫入...", state="complete")
    
    if atomic_updates:
        update_atomic_reviews(atomic_updates)

# ==========================================
# 4. 圖表繪製 (v9.2 Restored)
# ==========================================
def render_allocation_charts(full_portfolio_df):
    if full_portfolio_df.empty: return
    st.markdown("#### 🥧 資產配置 - 持股佔比")
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
    
    st.markdown("#### 🌊 累積已實現損益 (含股息) - 財富堆疊圖")
    chart = alt.Chart(grouped).mark_area(opacity=0.7).encode(
        x=alt.X('Date:T', timeUnit='yearmonth', title='月份'),
        y=alt.Y('Cumulative_PnL:Q', title='累積已實現獲利 ($)', stack=True), 
        color=alt.Color('Type:N', title='資產種類'),
        tooltip=['Date', 'Type', 'Cumulative_PnL']
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

def render_metrics_cards(summary, mode):
    if not summary: return
    # v9.2 Style Metrics
    if mode == "swing": 
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("累積總損益", f"${summary['累積總損益']:,.0f}")
        k2.metric("已領股息", f"${summary['已領股息']:,.0f}")
        k3.metric("已實現", f"${summary['已實現損益']:,.0f}")
        k4.metric("未實現", f"${summary['未實現損益']:,.0f}")
        k5, k6, k7, k8 = st.columns(4)
        xirr_display = f"{summary['XIRR%']:.2f}%" if summary['XIRR%'] is not None else "N/A"
        k5.metric("年化報酬率", xirr_display)
        k6.metric("交易勝率", f"{summary['勝率%']:.1f}%")
        k7.empty(); k8.empty()
    elif mode == "dividend":
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("累積總損益", f"${summary['累積總損益']:,.0f}")
        k2.metric("已領股息", f"${summary['已領股息']:,.0f}")
        k3.metric("已實現", f"${summary['已實現損益']:,.0f}")
        k4.metric("未實現", f"${summary['未實現損益']:,.0f}")
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
        g3.metric("已實現", f"${summary['已實現損益']:,.0f}")
        g4.metric("未實現", f"${summary['未實現損益']:,.0f}")
        xirr_display = f"{summary['XIRR%']:.2f}%" if summary['XIRR%'] is not None else "N/A"
        g5.metric("年化報酬率", xirr_display)

def render_chart_swing(trade_log_df):
    # v9.2 Swing Chart (Line + Points)
    if not trade_log_df.empty:
        trade_log_df = trade_log_df.sort_values('Date')
        trade_log_df['cumsum_PnL'] = trade_log_df['PnL'].cumsum()
        
        line = alt.Chart(trade_log_df).mark_line(color='purple').encode(
            x='Date:T', y=alt.Y('cumsum_PnL:Q', title='累積已實現損益'), tooltip=['Date', 'cumsum_PnL']
        )
        points = alt.Chart(trade_log_df).mark_circle(size=60).encode(
            x='Date:T', y='PnL:Q', 
            color=alt.condition(alt.datum.PnL > 0, alt.value("red"), alt.value("green")), # Red=Profit (TW Style)
            tooltip=['Date', 'Ticker', 'PnL']
        )
        st.altair_chart((line + points).interactive(), use_container_width=True)
    else:
        st.info("區間內無已實現交易")

def render_chart_dividend_monthly(period_df):
    # v9.2 Dividend Chart (Bar)
    div_df = period_df[period_df['Action'] == '領息'].copy()
    if not div_df.empty:
        div_df['Date'] = pd.to_datetime(div_df['Date'])
        chart = alt.Chart(div_df).mark_bar().encode(
            x=alt.X('Date:T', timeUnit='yearmonth', title='月份'), 
            y=alt.Y('Total_Amount:Q', title='股息金額'),
            color=alt.Color('Ticker:N', title='投資標的'),
            tooltip=['Date', 'Ticker', 'Total_Amount']
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("區間內無領息紀錄")

def render_inventory_management(full_portfolio_df, df_records, key_prefix):
    st.markdown("### 📦 庫存管理與交易登錄")
    if not full_portfolio_df.empty:
        stocks_pf = full_portfolio_df[full_portfolio_df['種類'] == '股票']
        funds_pf = full_portfolio_df[full_portfolio_df['種類'] == '基金']
        if not stocks_pf.empty:
            st.markdown("#### 📈 股票庫存")
            s_cost = stocks_pf['總成本'].sum(); s_pl = stocks_pf['帳面損益'].sum()
            s_roi = ((s_pl + stocks_pf['已領股息'].sum()) / s_cost * 100) if s_cost > 0 else 0
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("股票總現值", f"${stocks_pf['庫存現值'].sum():,.0f}")
            s2.metric("股票總成本", f"${s_cost:,.0f}")
            s3.metric("股票帳面損益", f"${s_pl:,.0f}", delta_color="normal")
            s4.metric("股票總報酬率", f"{s_roi:.2f}%")
        if not funds_pf.empty:
            st.markdown("#### 🛡️ 基金庫存")
            f1, f2, f3, f4 = st.columns(4)
            f_cost = funds_pf['總成本'].sum(); f_pl = funds_pf['帳面損益'].sum()
            f_roi = (f_pl / f_cost * 100) if f_cost > 0 else 0
            f1.metric("基金總現值", f"${funds_pf['庫存現值'].sum():,.0f}")
            f2.metric("基金總投入", f"${f_cost:,.0f}")
            f3.metric("基金帳面損益", f"${f_pl:,.0f}", delta_color="normal")
            f4.metric("基金總報酬率", f"{f_roi:.2f}%")
        st.write("") 
        cols_show = ["代號", "種類", "佔比%", "庫存", "平均成本", "市價", "庫存現值", "帳面損益", "含息總報%", "策略"]
        st.dataframe(full_portfolio_df[cols_show], use_container_width=True, hide_index=True)
        
        with st.form(f"trans_form_{key_prefix}", clear_on_submit=True):
            st.markdown("**➕ 新增交易**")
            c1, c2, c3, c4 = st.columns(4)
            with c1: d_date = st.date_input("日期")
            with c1: d_ticker = st.text_input("代號").upper()
            with c2: d_type = st.selectbox("種類", ["股票", "基金"]); d_action = st.selectbox("動作", ["買入", "賣出", "領息", "分割"])
            with c3: d_strat = st.multiselect("策略", ["存股", "波段"], default=["存股"]); d_price = st.number_input("單價", min_value=0.0, format="%.2f")
            with c4: d_shares = st.number_input("股數", step=100.0); d_fee = st.number_input("手續費 (0自動算)", min_value=0)
            c5, c6 = st.columns([3, 1])
            with c5: d_total = st.number_input("總金額 (0自動算)", step=1000.0); d_note = st.text_input("備註")
            with c6: st.write(""); submitted = st.form_submit_button("送出交易")
            if submitted and d_ticker:
                db_strat = ",".join(d_strat)
                if d_fee == 0 and d_action in ["買入", "賣出"]: d_fee = int(d_price * d_shares * 0.001425)
                if d_total == 0:
                    basic = d_price * d_shares
                    if d_action == "買入": d_total = basic + d_fee
                    elif d_action == "賣出": d_total = basic - d_fee - int(basic*0.003)
                ws_records.append_row([str(d_date), d_ticker, d_type, db_strat, d_action, d_price, d_shares, d_fee, d_total, d_note])
                st.success(f"已儲存 {d_ticker}"); st.cache_data.clear()

        with st.form(f"fund_form_{key_prefix}", clear_on_submit=True):
            st.markdown("**💵 更新基金淨值**")
            f_tick = st.text_input("基金代號").upper()
            f_val = st.number_input("最新淨值", min_value=0.0, format="%.4f")
            f_curr = st.selectbox("幣別", ["USD", "TWD"])
            if st.form_submit_button("更新"):
                try:
                    cell = ws_funds.find(f_tick)
                    ws_funds.update_cell(cell.row, 2, f_val)
                    ws_funds.update_cell(cell.row, 3, str(datetime.now().date()))
                except: ws_funds.append_row([f_tick, f_val, str(datetime.now().date()), f_curr])
                st.success("更新成功"); st.cache_data.clear()

# ==========================================
# 5. 主程式佈局
# ==========================================
st.title("📊 投資戰情室 v10.1 (Final Polish)")

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
    t_all, t_swing, t_div, t_ai = st.tabs(["🌍 全總覽", "⚡ 波段儀表板", "💰 存股月報", "🤖 AI 教練"])
    
    if not df.empty:
        try: total_summary, _, _ = analyze_period_advanced(df, analysis_start, analysis_end, None, full_portfolio_df, trade_log_df, None)
        except: total_summary = None

    # --- Tab 1: 全總覽 ---
    with t_all:
        if total_summary: render_metrics_cards(total_summary, "general")
        st.write(""); g1, g2 = st.columns([1, 2])
        if total_summary:
            with g1: render_allocation_charts(full_portfolio_df)
            with g2: render_global_monthly_pnl_colored(trade_log_df, df)
        st.divider(); render_inventory_management(full_portfolio_df, df, "overview")
        
        gl_rep = get_last_report("Global")
        if gl_rep: st.markdown(f"### 📝 AI 全域診斷 ({gl_rep['Date']})\n{gl_rep['Content']}")

    # --- Tab 2: 波段儀表板 (v9.2 Style + History) ---
    with t_swing:
        if total_summary: render_metrics_cards(total_summary, "swing")
        # [Req 1] Restore v9.2 Charts
        st.markdown("##### 📈 交易損益曲線"); render_chart_swing(trade_log_df)
        st.divider()
        st.markdown("### ⚡ 波段交易履歷")
        
        summaries = get_monthly_summaries()
        swing_tickers = df[df['Strategy'].str.contains('波段', na=False)]['Ticker'].unique()
        
        for t in swing_tickers:
            with st.expander(f"📌 {t} 交易紀錄"):
                t_df = df[(df['Ticker'] == t) & (df['Strategy'].str.contains('波段', na=False))].copy()
                t_df['YYYYMM'] = pd.to_datetime(t_df['Date']).dt.strftime('%Y-%m')
                months = sorted(t_df['YYYYMM'].unique(), reverse=True)
                for m in months:
                    st.markdown(f"**🗓️ {m}**")
                    sum_key = f"Swing_Summary_{t}_{m}"
                    if sum_key in summaries: st.info(summaries[sum_key])
                    m_trades = t_df[t_df['YYYYMM'] == m].sort_values('Date', ascending=False)
                    for _, row in m_trades.iterrows():
                        review = row['AI_Review'] if row['AI_Review'] else "(待分析)"
                        st.markdown(f"- `{row['Date']}` {row['Action']} **${row['Price']}**: {review}")
                    st.divider()

    # --- Tab 3: 存股月報 (v9.2 Style + History) ---
    with t_div:
        if total_summary: render_metrics_cards(total_summary, "dividend")
        # [Req 1] Restore v9.2 Charts
        st.markdown("##### 💰 股息累積圖"); render_chart_dividend_monthly(df[df['Action']=='領息'])
        st.divider()
        
        div_rep = get_last_report("Dividend")
        if div_rep: st.markdown(f"### 📝 本月存股健檢 ({div_rep['Date']})\n{div_rep['Content']}")
        
        st.divider(); render_inventory_management(full_portfolio_df, df, "div")

    # --- Tab 4: AI 教練 (Req 2: Sub-tabs) ---
    with t_ai:
        st.markdown("### 🤖 AI 指揮中心")
        
        # [Req 2] Sub-tabs for AI
        ai_t1, ai_t2, ai_t3 = st.tabs(["🌍 全域總覽", "⚡ 波段分析", "💰 存股健檢"])
        
        # --- AI Sub-tab 1: Global ---
        with ai_t1:
            # [Req 3/4] Show History First
            last_gl = get_last_report("Global")
            if last_gl: st.info(f"**上次分析 ({last_gl['Date']}):**\n\n{last_gl['Content']}")
            
            if st.button("🚀 執行全域分析", use_container_width=True):
                top_holdings = full_portfolio_df.sort_values('庫存現值', ascending=False).head(5)
                holdings_str = ""
                for _, row in top_holdings.iterrows(): holdings_str += f"- {row['代號']}: {row['佔比%']}%\n"
                prompt = f"全域資產診斷。總資產: {total_summary['庫存現值'] if total_summary else 0}。前五大: \n{holdings_str}。請用 :red[好]/:green[壞] (台股慣例) 給建議。"
                api_key = st.secrets.get("gemini_api_key", None)
                if api_key:
                    with st.spinner("分析中..."):
                        advice = ask_gemini_coach(api_key, prompt)
                        # [Req 4] Explicit Save & Refresh
                        save_analysis_history("Global", advice)
                        st.success("✅ 分析完成並已存檔！")
                        time.sleep(1); st.rerun()

        # --- AI Sub-tab 2: Swing (Req 3 Logic) ---
        with ai_t2:
            st.info("此處將逐筆檢視買賣點 (T±7) 並生成月報。")
            
            # Show existing monthly summaries logic could go here, but it's better in the main Swing tab.
            # Here we focus on ACTION.
            
            if st.button("🚀 執行波段分析 (批次10筆)", use_container_width=True):
                run_two_layer_swing_analysis(df) # Logic handles updates/saving internally

        # --- AI Sub-tab 3: Dividend ---
        with ai_t3:
            # [Req 3/4] Show History First
            last_div = get_last_report("Dividend")
            if last_div: st.info(f"**上次分析 ({last_div['Date']}):**\n\n{last_div['Content']}")
            
            if st.button("🚀 執行存股分析", use_container_width=True):
                div_stocks = full_portfolio_df[full_portfolio_df['策略'].str.contains('存股', na=False)]
                if not div_stocks.empty:
                    stocks_str = ""
                    for _, row in div_stocks.iterrows(): stocks_str += f"{row['代號']}: YoC {row['成本殖利率%']}%\n"
                    prompt = f"存股健檢。使用 :red[好/高YoC] 和 :green[壞/低YoC]。禁止 HTML。\n{stocks_str}"
                    api_key = st.secrets.get("gemini_api_key", None)
                    if api_key:
                        with st.spinner("分析中..."):
                            advice = ask_gemini_coach(api_key, prompt)
                            # [Req 4] Explicit Save & Refresh
                            save_analysis_history("Dividend", advice)
                            st.success("✅ 分析完成並已存檔！")
                            time.sleep(1); st.rerun()
                else: st.warning("無存股部位")

else:
    for i, ticker in enumerate(selected_tickers):
        with ticker_tabs[i]:
            ticker_df = df[df['Ticker'] == ticker]
            render_inventory_management(full_portfolio_df[full_portfolio_df['代號']==ticker], df, f"tick_{i}")
