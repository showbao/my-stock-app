# Version: v9.9.1 (Taiwan Colors + Fund Fix + Charts Restored + 2-Layer Swing Logic)
# CTOSignature: Enforced Red=Good/Green=Bad. Fixed Fund price fetch. Restored v9.2 charts. Implemented Monthly Swing Summaries in AI Coach.
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
st.set_page_config(page_title="投資戰情室 v9.9.1", layout="wide")

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
    except: st.error("❌ 找不到 'Records' 工作表"); st.stop()
    try: ws_funds = sh.worksheet("Fund_Updates")
    except: st.error("❌ 找不到 'Fund_Updates' 工作表"); st.stop()
    try: ws_history = sh.worksheet("Analysis_History")
    except: ws_history = None
else:
    st.error("❌ Google Sheet 連線失敗"); st.stop()

# ==========================================
# 2. 核心邏輯函數
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
        # 確保欄位名稱正確，防止大小寫問題
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
            # [v9.9.1 Fix] Fund Logic
            if data['type'] == '基金':
                # 優先查找 Google Sheet 的基金更新表
                if not df_funds.empty and ticker in df_funds['Ticker'].values:
                    fund_row = df_funds[df_funds['Ticker'] == ticker].iloc[0]
                    # 假設 Fund_Updates 有 'Price' 或 'Net_Value' 欄位
                    price_col = 'Price' if 'Price' in df_funds.columns else 'Net_Value' if 'Net_Value' in df_funds.columns else df_funds.columns[1] # Fallback to 2nd col
                    net_val = pd.to_numeric(fund_row[price_col], errors='coerce')
                    
                    currency = 'USD'
                    if 'Currency' in df_funds.columns: currency = fund_row['Currency']
                    current_price = net_val if currency == 'TWD' else net_val * current_usd_rate
            else:
                current_price, _ = get_stock_data(ticker)
            
            # 若仍為 0，嘗試最後手段
            if current_price == 0 and data['type'] == '股票': current_price, _ = get_stock_data(ticker)
                
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
    xirr_val = xirr(cash_flows)
    if xirr_val: xirr_val *= 100
    if xirr_val and (xirr_val > 10000 or xirr_val < -10000): xirr_val = None

    yoc_period = (total_dividend / total_cost_basis * 100) if total_cost_basis > 0 else 0
    payback_progress = (total_dividend / total_buy * 100) if total_buy > 0 else 0

    summary = {
        "累積總損益": total_profit, "已領股息": total_dividend, "已實現損益": realized_pnl_period,
        "未實現損益": total_unrealized, "勝率%": win_rate, "XIRR%": xirr_val, 
        "YoC%": yoc_period, "回本率%": payback_progress, "庫存現值": ending_inventory_value
    }
    return summary, period_df, pd.DataFrame()

# ==========================================
# 3. AI 教練核心邏輯
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

# --- Analysis History Manager ---
def get_analysis_history(report_type_prefix):
    """
    獲取指定類型的歷史報告，回傳 Dictionary: {Key: Content}
    Key for Swing: 'Swing_Summary_{Ticker}_{Year_Month}'
    """
    if ws_history is None: return {}
    try:
        data = ws_history.get_all_records()
        df_hist = pd.DataFrame(data)
        if df_hist.empty: return {}
        # Filter where Type starts with prefix
        target = df_hist[df_hist['Type'].astype(str).str.startswith(report_type_prefix)]
        # Convert to dict for O(1) lookup
        return pd.Series(target.Content.values, index=target.Type).to_dict()
    except: return {}

def save_analysis_history(report_type, content):
    if ws_history is None: st.error("無歷史工作表"); return
    try:
        ws_history.append_row([str(date.today()), report_type, content])
        st.cache_data.clear()
    except: pass

def update_google_sheet_review(updates_list):
    if not updates_list: return
    try:
        ai_col_idx = 11 
        header = ws_records.cell(1, ai_col_idx).value
        if header != "AI_Review":
            cell = ws_records.find("AI_Review")
            if cell: ai_col_idx = cell.col
            else: st.error("找不到 'AI_Review' 欄位"); return
        with st.status("正在儲存單筆交易評語...", expanded=True) as status:
            for row_idx, text in updates_list:
                ws_records.update_cell(row_idx, ai_col_idx, text)
                time.sleep(0.5) 
            status.update(label="儲存完成！", state="complete")
            st.cache_data.clear() 
    except Exception as e: st.error(f"寫入失敗: {e}")

# --- AI Action Logic ---
def run_swing_analysis_action(df_raw, trade_log_df):
    api_key = st.secrets.get("gemini_api_key", None)
    if not api_key: st.error("無 API Key"); return None

    # 1. 找出未分析的交易
    pending = df_raw[
        (df_raw['Strategy'].str.contains('波段', na=False)) & 
        ((df_raw['AI_Review'] == "") | (df_raw['AI_Review'].isna()))
    ].copy()
    
    if pending.empty: return "🎉 所有交易皆已分析完成。"

    # 2. 取最新 10 筆進行處理
    target_batch = pending.sort_values('Date', ascending=False).head(10)
    updates_to_commit = []
    monthly_summaries_to_save = {} # Key: Ticker_YYYY-MM, Value: Text

    with st.status("🚀 AI 正在逐筆檢視並生成月報...", expanded=True) as status:
        # A. 逐筆分析 (Row Level)
        for _, row in target_batch.iterrows():
            t = row['Ticker']; d = row['Date']; px = row['Price']; act = row['Action']; rid = row['RowIndex']
            
            context = get_historical_price_window(t, d, 7) # T+/-7
            if context:
                if act == '買入':
                    low = context['window_low']; dist = ((px - low)/low * 100)
                    prompt = f"評估買點: {t} {d} 買 {px}, 7日低點 {low} (差距 {dist:.1f}%). 用繁中, :red[好]/:green[壞] 標示."
                else:
                    high = context['window_high']; missed = ((high - px)/px * 100)
                    prompt = f"評估賣點: {t} {d} 賣 {px}, 7日高點 {high} (賣飛 {missed:.1f}%). 用繁中, :red[好]/:green[壞] 標示."
                
                review = ask_gemini_coach(api_key, prompt).strip()
                final_text = f"[{date.today()}] {review}"
                updates_to_commit.append((rid, final_text))
            else:
                updates_to_commit.append((rid, "[無股價數據]"))

        # B. 生成月報 (Monthly Summary Level)
        # 找出這批交易涉及哪些 (Ticker, Month)
        target_batch['YYYYMM'] = pd.to_datetime(target_batch['Date']).dt.strftime('%Y-%m')
        groups = target_batch.groupby(['Ticker', 'YYYYMM'])
        
        for (ticker, yyyymm), group in groups:
            tx_str = ""
            for _, r in group.iterrows():
                tx_str += f"{r['Date']} {r['Action']} ${r['Price']}\n"
            
            summary_prompt = f"""
            請總結 {ticker} 在 {yyyymm} 的波段操作表現。
            交易紀錄:
            {tx_str}
            請給出操作建議 (50字內)。
            使用 :red[紅色代表獲利/操作佳], :green[綠色代表虧損/操作差/警示] (台股慣例)。
            """
            summary = ask_gemini_coach(api_key, summary_prompt).strip()
            save_key = f"Swing_Summary_{ticker}_{yyyymm}"
            monthly_summaries_to_save[save_key] = summary
            
        status.update(label="分析完成！", state="complete")
    
    return updates_to_commit, monthly_summaries_to_save

# ==========================================
# 5. 主程式佈局
# ==========================================
st.title("📊 投資戰情室 v9.9.1 (Flagship)")

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
        try:
            total_summary, _, _ = analyze_period_advanced(df, analysis_start, analysis_end, None, full_portfolio_df, trade_log_df, None)
        except: total_summary = None

    # --- Tab 1: 全總覽 ---
    with t_all:
        if total_summary:
            render_metrics_cards(total_summary, "general")
        
        st.write("")
        g_col1, g_col2 = st.columns([1, 2])
        if total_summary:
            with g_col1: render_allocation_charts(full_portfolio_df)
            with g_col2: render_global_monthly_pnl_colored(trade_log_df, df)
                
        st.divider(); render_inventory_management(full_portfolio_df, df, "overview")
        
        # 讀取全域報告
        global_report = get_analysis_history("Global")
        if global_report:
            st.markdown("### 📝 最新全域診斷")
            st.markdown(list(global_report.values())[-1]) # Show latest

    # --- Tab 2: 波段儀表板 ---
    with t_swing:
        if total_summary:
            render_metrics_cards(total_summary, "swing")
            st.markdown("##### 📈 交易損益曲線")
            render_chart_swing(trade_log_df)
            
        st.divider()
        st.markdown("### ⚡ 波段交易履歷 (按月歸檔)")
        
        # 讀取所有波段月報
        swing_summaries = get_analysis_history("Swing_Summary_")
        
        swing_tickers = df[df['Strategy'].str.contains('波段', na=False)]['Ticker'].unique()
        
        for t in swing_tickers:
            with st.expander(f"📌 {t} 交易紀錄"):
                # 1. 找出該標的的所有波段交易，按月分組
                t_df = df[(df['Ticker'] == t) & (df['Strategy'].str.contains('波段', na=False))].copy()
                t_df['YYYYMM'] = pd.to_datetime(t_df['Date']).dt.strftime('%Y-%m')
                months = t_df['YYYYMM'].unique()
                months = sorted(months, reverse=True) # Newest month first
                
                for m in months:
                    st.markdown(f"**🗓️ {m}**")
                    
                    # 顯示該月 AI 總結 (若有)
                    sum_key = f"Swing_Summary_{t}_{m}"
                    if sum_key in swing_summaries:
                        st.info(swing_summaries[sum_key])
                    
                    # 顯示該月逐筆交易
                    m_trades = t_df[t_df['YYYYMM'] == m].sort_values('Date', ascending=False)
                    for _, row in m_trades.iterrows():
                        review = row['AI_Review'] if row['AI_Review'] else "(待分析)"
                        st.markdown(f"- `{row['Date']}` {row['Action']} **${row['Price']}**: {review}")
                    st.divider()

    # --- Tab 3: 存股月報 ---
    with t_div:
        if total_summary:
            render_metrics_cards(total_summary, "dividend")
            st.markdown("##### 💰 股息累積圖")
            render_chart_dividend_monthly(df[df['Action']=='領息'])
            
        st.divider()
        
        div_report = get_analysis_history("Dividend")
        if div_report:
            st.markdown("### 📝 本月存股健檢")
            st.markdown(list(div_report.values())[-1])
        else:
            st.info("尚無本月報告，請至 AI 教練執行分析。")
            
        st.divider(); render_inventory_management(full_portfolio_df, df, "div")

    # --- Tab 4: AI 教練 (Command Center) ---
    with t_ai:
        st.markdown("### 🤖 AI 指揮中心")
        st.caption("所有分析指令皆在此執行。分析結果將自動存檔並顯示於對應儀表板。")
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("#### 🌍 1. 全域分析")
            if st.button("執行全域診斷", use_container_width=True):
                # ... Global Logic ...
                top_holdings = full_portfolio_df.sort_values('庫存現值', ascending=False).head(5)
                holdings_str = ""
                for _, row in top_holdings.iterrows():
                    holdings_str += f"- {row['代號']}: {row['佔比%']}%\n"
                
                prompt = f"""
                全域資產診斷。
                總資產: {total_summary['庫存現值']} (未含現金)。
                前五大持股: \n{holdings_str}
                請使用台股慣例顏色 (:red[好/獲利], :green[壞/虧損]) 給予建議。
                """
                api_key = st.secrets.get("gemini_api_key", None)
                if api_key:
                    with st.spinner("分析中..."):
                        advice = ask_gemini_coach(api_key, prompt)
                        save_analysis_history("Global", advice)
                        st.success("✅ 全域報告已更新！")

        with c2:
            st.markdown("#### ⚡ 2. 波段批次覆盤")
            if st.button("執行波段分析 (10筆)", use_container_width=True):
                updates, summaries = run_swing_analysis_action(df, trade_log_df)
                if isinstance(updates, list) and updates:
                    st.session_state['swing_updates'] = updates
                    st.session_state['swing_summaries'] = summaries
                    st.success(f"分析完成！共 {len(updates)} 筆交易。請按下方存檔。")
                else:
                    st.info(updates) # "全部完成" message
            
            if st.session_state.get('swing_updates'):
                if st.button("💾 確認存檔 (寫入 Sheet)", use_container_width=True):
                    # 1. Save Row Reviews
                    update_google_sheet_review(st.session_state['swing_updates'])
                    # 2. Save Monthly Summaries
                    for k, v in st.session_state['swing_summaries'].items():
                        save_analysis_history(k, v)
                    
                    del st.session_state['swing_updates']
                    del st.session_state['swing_summaries']
                    st.success("✅ 存檔成功！請至波段儀表板查看。")
                    time.sleep(2); st.rerun()

        with c3:
            st.markdown("#### 💰 3. 存股健檢")
            if st.button("執行存股分析", use_container_width=True):
                div_stocks = full_portfolio_df[full_portfolio_df['策略'].str.contains('存股', na=False)]
                if not div_stocks.empty:
                    stocks_str = ""
                    for _, row in div_stocks.iterrows():
                        stocks_str += f"{row['代號']}: YoC {row['成本殖利率%']}%\n"
                    prompt = f"存股健檢。使用 :red[好/高YoC] 和 :green[壞/低YoC]。禁止 HTML。\n{stocks_str}"
                    api_key = st.secrets.get("gemini_api_key", None)
                    if api_key:
                        with st.spinner("分析中..."):
                            advice = ask_gemini_coach(api_key, prompt)
                            save_analysis_history("Dividend", advice)
                            st.success("✅ 存股報告已更新！")
                else:
                    st.warning("無存股部位")

else:
    # Single Ticker View (Simple)
    for i, ticker in enumerate(selected_tickers):
        st.markdown(f"### 🔍 {ticker}")
        render_inventory_management(full_portfolio_df[full_portfolio_df['代號']==ticker], df, f"tick_{i}")
