# Version: v9.10 (Based on v9.2 + Two-Layer Swing Analysis)
# CTOSignature: Rebuilt on v9.2 core. Implemented T+/-7 atomic analysis (stored in Records) and Monthly Aggregated Advice (stored in Analysis_History).
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
st.set_page_config(page_title="投資戰情室 v9.10", layout="wide")

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
    except: ws_history = None # 容錯，若無則無法存月報
else:
    st.error("❌ Google Sheet 連線失敗"); st.stop()

# ==========================================
# 2. 核心邏輯函數 (v9.2 Base)
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
    # 改為 T+/-7
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
        # 增加 RowIndex 供寫回使用 (gspread index starts at 2 for data)
        df['RowIndex'] = range(2, len(df) + 2)
    except: return pd.DataFrame(), pd.DataFrame(), 32.0

    try:
        funds_data = ws_funds.get_all_records()
        df_funds = pd.DataFrame(funds_data)
        if not df_funds.empty and 'Currency' not in df_funds.columns:
            df_funds['Currency'] = 'USD'
    except: df_funds = pd.DataFrame()
    
    if df.empty: return df, df_funds, 32.0
    
    numeric_cols = ['Price', 'Shares', 'Fee', 'Total_Amount']
    for col in numeric_cols:
        if df[col].dtype == object: df[col] = df[col].astype(str).str.replace(',','').str.replace('$','')
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df = normalize_data(df)
    
    # 確保 AI_Review 欄位存在
    if 'AI_Review' not in df.columns: df['AI_Review'] = ""
    
    current_usd_rate = get_usd_twd_rate()
    return df, df_funds, current_usd_rate

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
            if data['type'] == '股票': current_price, _ = get_stock_data(ticker)
            elif data['type'] == '基金': pass 
            if current_price == 0 and data['type'] == '股票': current_price, _ = get_stock_data(ticker)
            market_value = current_price * data['shares']
            unrealized_pl = market_value - data['total_cost']
            roi_total = ((unrealized_pl + data['dividend_collected']) / data['total_cost'] * 100) if data['total_cost'] > 0 else 0
            results.append({
                "代號": ticker, "種類": data['type'], "策略": data['strategy'], "庫存": data['shares'], "平均成本": round(data['total_cost'] / data['shares'], 2),
                "市價": round(current_price, 2), "庫存現值": round(market_value, 0), "帳面損益": round(unrealized_pl, 0),
                "含息總報%": round(roi_total, 2)
            })
    return pd.DataFrame(results), pd.DataFrame(trade_log)

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

# 讀取月報歷史
def get_monthly_summaries():
    if ws_history is None: return {}
    try:
        data = ws_history.get_all_records()
        df_hist = pd.DataFrame(data)
        if df_hist.empty: return {}
        # 篩選 Swing_Summary 類型的資料
        target = df_hist[df_hist['Type'].astype(str).str.startswith('Swing_Summary_')]
        # 轉成 Dictionary {Type: Content} 以便快速查詢
        return pd.Series(target.Content.values, index=target.Type).to_dict()
    except: return {}

# 儲存月報歷史
def save_monthly_summary(ticker, yyyymm, content):
    if ws_history is None: return
    type_key = f"Swing_Summary_{ticker}_{yyyymm}"
    try:
        ws_history.append_row([str(date.today()), type_key, content])
    except: pass

# 寫回單筆交易紀錄 (原子化更新)
def update_atomic_reviews(updates_list):
    """
    updates_list: [(row_index, text), ...]
    """
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
                time.sleep(0.5) # 防止 API 超速
            status.update(label="儲存完成！", state="complete")
            st.cache_data.clear()
            st.success("✅ 已更新 Google Sheet")
            time.sleep(1); st.rerun()
    except Exception as e: st.error(f"寫入失敗: {e}")

def run_two_layer_swing_analysis(df_raw):
    api_key = st.secrets.get("gemini_api_key", None)
    if not api_key: st.error("無 API Key"); return

    # 1. 找出未分析的波段交易 (原子層)
    pending = df_raw[
        (df_raw['Strategy'].str.contains('波段', na=False)) & 
        ((df_raw['AI_Review'] == "") | (df_raw['AI_Review'].isna()))
    ].copy()
    
    if pending.empty:
        st.info("🎉 所有波段交易都已完成原子分析。")
        return

    # 每次處理 10 筆，避免超時
    target_batch = pending.sort_values('Date', ascending=False).head(10)
    
    atomic_updates = []
    
    # 用來記錄哪些月份的資料被更新了，稍後要重跑月報
    affected_months = set() # (Ticker, YYYY-MM)

    with st.status("🚀 正在執行雙層分析...", expanded=True) as status:
        
        # --- 第一層：原子分析 (Atomic Analysis) ---
        status.write("正在進行逐筆 T±7 檢視...")
        for _, row in target_batch.iterrows():
            t = row['Ticker']; d = row['Date']; px = row['Price']; act = row['Action']; rid = row['RowIndex']
            
            # 記錄受影響的月份
            ym = pd.to_datetime(d).strftime('%Y-%m')
            affected_months.add((t, ym))
            
            context = get_historical_price_window(t, d, 7)
            if context:
                if act == '買入':
                    low = context['window_low']; dist = ((px - low)/low * 100)
                    prompt = f"你是交易員。針對 {t} 在 {d} 買入價 {px} (T±7日最低 {low}, 差距 {dist:.1f}%)。這筆買點合宜嗎？請用一句話簡評，買在低點請稱讚。"
                elif act == '賣出':
                    high = context['window_high']; missed = ((high - px)/px * 100)
                    prompt = f"你是交易員。針對 {t} 在 {d} 賣出價 {px} (T±7日最高 {high}, 賣飛 {missed:.1f}%)。這筆賣點合宜嗎？請用一句話簡評，賣在高點請稱讚。"
                else:
                    prompt = "略過"
                
                if prompt != "略過":
                    review = ask_gemini_coach(api_key, prompt).strip()
                    atomic_updates.append((rid, review))
            else:
                atomic_updates.append((rid, "[無歷史數據]"))

        # --- 第二層：月報生成 (Monthly Summary) ---
        # 針對這次有更新到的月份，重新生成一份總結建議
        status.write("正在生成月報建議...")
        for (ticker, yyyymm) in affected_months:
            # 撈出該標的、該月份的所有交易 (包含舊的已經分析過的)
            month_mask = (df_raw['Ticker'] == ticker) & \
                         (pd.to_datetime(df_raw['Date']).dt.strftime('%Y-%m') == yyyymm) & \
                         (df_raw['Strategy'].str.contains('波段', na=False))
            month_trades = df_raw[month_mask].sort_values('Date')
            
            tx_desc = ""
            for _, r in month_trades.iterrows():
                # 如果是剛剛分析的，用新生成的評語；如果是舊的，用原本的
                # 這裡簡單處理：直接把交易數據餵給 AI，讓它重新綜觀
                tx_desc += f"- {r['Date']} {r['Action']} ${r['Price']}\n"
            
            summary_prompt = f"""
            你是一位波段操作教練。
            請針對 {ticker} 在 {yyyymm} 的所有操作紀錄進行「月度總結」：
            
            {tx_desc}
            
            請給出一份綜合建議 (繁體中文, 100字內)，分析進出場時機是否恰當？獲利節奏如何？
            """
            monthly_advice = ask_gemini_coach(api_key, summary_prompt).strip()
            
            # 存入 Analysis_History
            save_monthly_summary(ticker, yyyymm, monthly_advice)
            
        status.update(label="分析完成！準備寫入...", state="complete")
    
    # 寫回 Google Sheet
    if atomic_updates:
        update_atomic_reviews(atomic_updates)

# ==========================================
# 5. 主程式佈局
# ==========================================
st.title("📊 投資戰情室 v9.10 (Dual-Layer)")

df, df_funds, usd_rate = load_data()
if df.empty: st.warning("目前無任何交易紀錄"); st.stop()

all_tickers = df['Ticker'].unique().tolist()
pf_df, trade_log_df = calculate_portfolio(df, df_funds, usd_rate)

st.markdown("#### 🔍 篩選條件")
f1, f2 = st.columns([1, 2])
with f1: selected_ticker = st.selectbox("選擇投資標的", ["全部"] + all_tickers)

st.divider()

if selected_ticker == "全部":
    t_all, t_ai = st.tabs(["🌍 全總覽", "🤖 AI 教練 (執行分析)"])
    
    with t_all:
        st.info("請選擇單一標的以查看詳細波段月報。")
        st.dataframe(pf_df, use_container_width=True)
        
    with t_ai:
        st.markdown("### ⚡ 波段自動分析器")
        st.write("此功能會：")
        st.write("1. **逐筆檢視** T±7 買賣點，寫入 `Records`。")
        st.write("2. **按月總結** 操作建議，寫入 `Analysis_History`。")
        if st.button("🚀 執行波段分析 (批次 10 筆)", use_container_width=True):
            run_two_layer_swing_analysis(df)

else:
    # 單一標的檢視模式
    ticker_df = df[df['Ticker'] == selected_ticker]
    strategies = ticker_df['Strategy'].unique()
    is_swing = any("波段" in str(s) for s in strategies)
    
    st.markdown(f"### 📌 {selected_ticker} 投資歷程")
    
    if is_swing:
        st.subheader("⚡ 波段操作月報 (Monthly Report)")
        
        # 讀取已存的月報
        summaries = get_monthly_summaries()
        
        # 按月份分組顯示
        ticker_df['YYYYMM'] = pd.to_datetime(ticker_df['Date']).dt.strftime('%Y-%m')
        months = sorted(ticker_df['YYYYMM'].unique(), reverse=True)
        
        for m in months:
            with st.expander(f"🗓️ {m} 操作紀錄", expanded=True):
                # 1. 顯示月報建議 (從 Analysis_History 讀取)
                sum_key = f"Swing_Summary_{selected_ticker}_{m}"
                if sum_key in summaries:
                    st.info(f"💡 **AI 綜合建議**：\n\n{summaries[sum_key]}")
                else:
                    st.caption("尚無此月份的綜合建議 (請至 AI 教練執行分析)")
                
                # 2. 顯示逐筆紀錄 (從 Records 讀取)
                m_trades = ticker_df[ticker_df['YYYYMM'] == m].sort_values('Date', ascending=False)
                for _, row in m_trades.iterrows():
                    review = row['AI_Review'] if row['AI_Review'] else "(等待分析中...)"
                    st.markdown(f"""
                    * **{row['Date']}** `{row['Action']}` ${row['Price']}
                        * 💬 {review}
                    """)
    else:
        st.info("此標的無波段策略紀錄。")
        st.dataframe(ticker_df)
