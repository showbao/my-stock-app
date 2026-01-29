# Version: v9.7.4 (Layout Scope Fix: Decoupled st.columns from logic)
# CTOSignature: Moved st.columns initialization outside conditional blocks to guarantee variable existence. Fixed NameError.
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
st.set_page_config(page_title="投資戰情室 v9.7.4", layout="wide")

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
    except: 
        st.error("❌ 找不到工作表 'Records'。")
        st.stop()
        
    try: ws_funds = sh.worksheet("Fund_Updates")
    except: 
        st.error("❌ 找不到工作表 'Fund_Updates'。")
        st.stop()
        
    try: ws_history = sh.worksheet("Analysis_History")
    except: ws_history = None
else:
    st.error("❌ Google Sheet 連線失敗。")
    st.stop()

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

def get_historical_price_window(ticker, trade_date, window_days=10):
    try:
        t_date = pd.to_datetime(trade_date).tz_localize(None)
        today = datetime.now()
        is_mature = (today - t_date).days >= window_days
        
        start_d = (t_date - timedelta(days=window_days + 20)).strftime('%Y-%m-%d')
        end_d = (t_date + timedelta(days=window_days + 15)).strftime('%Y-%m-%d')
        
        real_ticker = fix_ticker_suffix(ticker)
        stock = yf.Ticker(real_ticker)
        hist = stock.history(start=start_d, end=end_d, auto_adjust=True)
        
        if hist.empty: return None, f"無數據 ({real_ticker})", False
        if hist.index.tz is not None: hist.index = hist.index.tz_localize(None)
            
        mask_window = (hist.index >= (t_date - timedelta(days=window_days))) & (hist.index <= (t_date + timedelta(days=window_days)))
        window_df = hist.loc[mask_window]
        
        if window_df.empty: return None, f"區間無數據", False
        
        return {
            "window_high": window_df['High'].max(),
            "window_low": window_df['Low'].min(),
            "price_at_trade": window_df['Close'].mean(),
            "real_ticker": real_ticker 
        }, "Success", is_mature
    except Exception as e: return None, str(e), False

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
            if data['type'] == '股票': current_price, _ = get_stock_data(ticker)
            elif data['type'] == '基金': pass 
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

# --- Soft Lock Manager ---
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
    if ws_history is None:
        st.error("請先建立 'Analysis_History' 工作表。")
        return
    try:
        ws_history.append_row([str(date.today()), report_type, content])
        st.cache_data.clear()
    except Exception as e: st.error(f"存檔失敗: {e}")

# --- Consolidated Swing Logic (Sliding Window) ---
def cluster_trades_by_gap(df_trades, gap_days=7):
    if df_trades.empty: return []
    df = df_trades.sort_values('Date').copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    clusters = []
    current_cluster = []
    last_date = None
    
    for _, row in df.iterrows():
        curr_date = row['Date']
        if last_date is None:
            current_cluster.append(row)
        else:
            diff = (curr_date - last_date).days
            if diff <= gap_days:
                current_cluster.append(row)
            else:
                clusters.append(pd.DataFrame(current_cluster))
                current_cluster = [row]
        last_date = curr_date
        
    if current_cluster:
        clusters.append(pd.DataFrame(current_cluster))
    return clusters

def update_google_sheet_review(updates_list):
    if not updates_list: return
    try:
        ai_col_idx = 11 
        header = ws_records.cell(1, ai_col_idx).value
        if header != "AI_Review":
            cell = ws_records.find("AI_Review")
            if cell: ai_col_idx = cell.col
            else: 
                st.error("找不到 'AI_Review' 欄位。")
                return
        with st.status("正在儲存...", expanded=True) as status:
            for row_idx, text in updates_list:
                ws_records.update_cell(row_idx, ai_col_idx, text)
                time.sleep(0.5) 
            status.update(label="儲存完成！", state="complete")
            st.cache_data.clear() 
    except Exception as e: st.error(f"寫入失敗: {e}")

def run_consolidated_swing_analysis(df_raw, trade_log_df):
    api_key = st.secrets.get("gemini_api_key", None)
    if not api_key: st.error("無 API Key"); return None, None, None

    pending = df_raw[
        (df_raw['Strategy'].str.contains('波段', na=False)) & 
        ((df_raw['AI_Review'] == "") | (df_raw['AI_Review'].isna()))
    ].copy()
    
    if pending.empty: return None, [], []

    ticker_groups = pending.groupby('Ticker')
    
    updates_to_commit = []
    ticker_summaries = {}
    processed_count = 0
    
    with st.status("🚀 正在執行智慧分群分析 (Gap=7天)...", expanded=True) as status:
        for ticker, t_df in ticker_groups:
            clusters = cluster_trades_by_gap(t_df, gap_days=7)
            
            for cluster in clusters:
                if processed_count >= 10: break 
                
                start_date = cluster['Date'].min().strftime('%Y-%m-%d')
                end_date = cluster['Date'].max().strftime('%Y-%m-%d')
                status.write(f"分析 {ticker} ({start_date} ~ {end_date}) - 共 {len(cluster)} 筆...")
                
                tx_details = ""
                row_indices = []
                
                for _, row in cluster.iterrows():
                    d = row['Date'].strftime('%Y-%m-%d'); act = row['Action']; px = row['Price']
                    row_indices.append(row['RowIndex'])
                    
                    context, _, is_mature = get_historical_price_window(ticker, d)
                    if context:
                        if act == '買入':
                            low = context['window_low']
                            dist = ((px - low)/low * 100)
                            tx_details += f"- {d} 買入 {px}元 (區間最低 {low}, 差距 +{dist:.1f}%)\n"
                        elif act == '賣出':
                            high = context['window_high']
                            missed = ((high - px)/px * 100)
                            tx_details += f"- {d} 賣出 {px}元 (區間最高 {high}, 賣飛 {missed:.1f}%)\n"
                    else:
                        tx_details += f"- {d} {act} {px}元 (數據不足)\n"

                prompt = f"""
                你是一位交易教練。請針對以下「同一波段」的操作(差距7天內)進行綜合點評。
                
                【交易數據: {ticker}】
                {tx_details}
                
                【任務】
                1. 給出一段「綜合評語」(繁體中文, 60字內)。
                2. 使用 :green[...] 或 :red[...] 標示。
                3. 重點：這波操作的節奏與進出場時機。
                """
                
                review = ask_gemini_coach(api_key, prompt).strip()
                final_text = f"[{date.today()}] {review}"
                
                for rid in row_indices:
                    updates_to_commit.append((rid, final_text))
                
                group_key = f"{ticker}_{start_date}"
                ticker_summaries[group_key] = review
                
                processed_count += 1
            
            if processed_count >= 10: break
            
        status.update(label="智慧分析完成！", state="complete")
        
    return updates_to_commit, ticker_summaries, pending

@st.dialog("🌍 全域總覽 (月報模式)")
def dialog_global_analysis(full_portfolio_df, summary_metrics):
    last_report = get_last_report("Global")
    cooldown = False
    btn_label = "🚀 啟動本月分析"
    
    if last_report:
        last_date = datetime.strptime(last_report['Date'], "%Y-%m-%d").date()
        days_diff = (date.today() - last_date).days
        if days_diff < 30:
            cooldown = True
            btn_label = f"⚠️ 強制更新 (上次: {days_diff}天前)"
            st.markdown(f"### 📅 上次報告 ({last_report['Date']})")
            st.markdown(last_report['Content'])
            st.divider()

    cash_balance = st.number_input("請輸入現金 (TWD)", min_value=0, value=0, step=10000)
    
    if st.button(btn_label, use_container_width=True):
        api_key = st.secrets.get("gemini_api_key", None)
        if not api_key: st.error("無 API Key"); return
        
        total_assets = summary_metrics['庫存現值'] + cash_balance
        cash_ratio = (cash_balance / total_assets * 100) if total_assets > 0 else 0
        top_holdings = full_portfolio_df.sort_values('庫存現值', ascending=False).head(5)
        holdings_str = ""
        for _, row in top_holdings.iterrows():
            holdings_str += f"- {row['代號']} ({row['種類']}): 佔比 {row['佔比%']}%\n"
            
        prompt = f"""
        (全域分析 Prompt...)
        資產現值: {summary_metrics['庫存現值']}, 現金: {cash_balance}, 現金水位: {cash_ratio:.1f}%.
        前五大: {holdings_str}
        請給予資產配置建議 (Markdown格式, 紅色警示).
        """
        with st.spinner("AI 分析中..."):
            advice = ask_gemini_coach(api_key, prompt)
            save_report("Global", advice)
            st.rerun()

def run_dividend_soft_lock(full_portfolio_df):
    last_report = get_last_report("Dividend")
    
    if last_report:
        last_date = datetime.strptime(last_report['Date'], "%Y-%m-%d").date()
        if (date.today() - last_date).days < 30:
            st.info(f"📅 顯示上月報告 ({last_report['Date']}) - 未滿 30 天")
            st.markdown(last_report['Content'])
            if st.button("⚠️ 強制更新 (消耗 API)"):
                pass 
            else:
                return 

    api_key = st.secrets.get("gemini_api_key", None)
    if not api_key: return
    
    div_stocks = full_portfolio_df[full_portfolio_df['策略'].str.contains('存股', na=False)]
    if div_stocks.empty: st.warning("無存股"); return
    
    stocks_str = ""
    for _, row in div_stocks.iterrows():
        stocks_str += f"{row['代號']}: YoC {row['成本殖利率%']}%\n"

    prompt = f"存股健檢 (Markdown, 紅色警示): \n{stocks_str}"
    
    with st.spinner("分析存股..."):
        advice = ask_gemini_coach(api_key, prompt)
        save_report("Dividend", advice)
        st.markdown(advice)

def render_metrics_cards(summary, mode):
    if not summary: return
    if mode == "swing": 
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("累積總損益", f"${summary['累積總損益']:,.0f}")
        k2.metric("已領股息", f"${summary['已領股息']:,.0f}")
        k3.metric("已實現", f"${summary['已實現損益']:,.0f}")
        k4.metric("未實現", f"${summary['未實現損益']:,.0f}")
        k5, k6, k7, k8 = st.columns(4)
        k5.metric("年化報酬率", f"{summary['XIRR%']:.2f}%" if summary['XIRR%'] else "N/A")
        k6.metric("交易勝率", f"{summary['勝率%']:.1f}%")
        k7.empty(); k8.empty()
    elif mode == "dividend":
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("累積總損益", f"${summary['累積總損益']:,.0f}")
        k2.metric("已領股息", f"${summary['已領股息']:,.0f}")
        k3.metric("已實現", f"${summary['已實現損益']:,.0f}")
        k4.metric("未實現", f"${summary['未實現損益']:,.0f}")
        k5, k6, k7, k8 = st.columns(4)
        k5.metric("年化報酬率", f"{summary['XIRR%']:.2f}%" if summary['XIRR%'] else "N/A")
        k6.metric("成本殖利率 (YoC)", f"{summary['YoC%']:.2f}%")
        k7.metric("回本率", f"{summary['回本率%']:.1f}%")
        k8.empty()
    else: 
        g1, g2, g3, g4, g5 = st.columns(5)
        g1.metric("累積總損益", f"${summary['累積總損益']:,.0f}")
        g2.metric("已領股息", f"${summary['已領股息']:,.0f}")
        g3.metric("已實現", f"${summary['已實現損益']:,.0f}")
        g4.metric("未實現", f"${summary['未實現損益']:,.0f}")
        g5.metric("年化報酬率", f"{summary['XIRR%']:.2f}%" if summary['XIRR%'] else "N/A")

def render_chart_swing(trade_log_df, strategy_filter=None):
    if not trade_log_df.empty:
        if strategy_filter:
            swing_trades = trade_log_df[trade_log_df['Strategy'].str.contains(strategy_filter, na=False)]
        else:
            swing_trades = trade_log_df
        if not swing_trades.empty:
            swing_trades = swing_trades.sort_values('Date')
            swing_trades['累積損益'] = swing_trades['PnL'].cumsum()
            line = alt.Chart(swing_trades).mark_line(color='purple').encode(x='Date:T', y='累積損益:Q', tooltip=['Date', '累積損益'])
            points = alt.Chart(swing_trades).mark_circle(size=60).encode(
                x='Date:T', y='PnL:Q', color=alt.condition(alt.datum.PnL > 0, alt.value("green"), alt.value("red")), tooltip=['Date', 'Ticker', 'PnL']
            )
            st.altair_chart((line + points).interactive(), use_container_width=True)
        else:
            st.info("尚無交易紀錄")

def render_chart_dividend_monthly(period_df):
    div_df = period_df[period_df['Action'] == '領息'].copy()
    if not div_df.empty:
        div_df['Date'] = pd.to_datetime(div_df['Date'])
        chart = alt.Chart(div_df).mark_bar().encode(
            x=alt.X('Date:T', timeUnit='yearmonth', title='月份'), 
            y=alt.Y('Total_Amount:Q', title='股息金額'),
            color=alt.Color('Ticker:N', title='投資標的'),
            tooltip=[alt.Tooltip('Date', timeUnit='yearmonth', title='月份'), 'Ticker', 'Total_Amount']
        ).properties(height=350).interactive()
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("區間內無領息紀錄")

def render_strategy_view(df, start_date, end_date, selected_tickers, strategy_filter, full_portfolio_df, trade_log_df, mode_name):
    summary, period_df, years_df = analyze_period_advanced(
        df, start_date, end_date, selected_tickers, full_portfolio_df, trade_log_df, strategy_filter
    )
    if summary:
        render_metrics_cards(summary, mode_name)
        st.divider()
        if mode_name == "dividend":
            st.markdown("##### 💰 歷年股息成長 (月堆疊)")
            render_chart_dividend_monthly(period_df)
        elif "swing" in mode_name:
            st.markdown("##### 📈 交易損益曲線 & 落點")
            render_chart_swing(trade_log_df, strategy_filter)
        if not years_df.empty:
            st.markdown("##### 📅 年度績效表")
            cols_to_show = []
            if mode_name == "swing": cols_to_show = ["年度", "已實現", "交易勝率", "年度投資報酬率"]
            elif mode_name == "dividend": cols_to_show = ["年度", "已領股息", "成本殖利率 (YoC)", "回本率"]
            final_cols = [c for c in cols_to_show if c in years_df.columns]
            st.dataframe(years_df[final_cols], use_container_width=True, hide_index=True)
    else:
        st.info("此區間無相關數據")

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
        event = st.dataframe(full_portfolio_df[cols_show], use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key=f"inventory_table_{key_prefix}")
        default_ticker = ""; default_strat = ["存股"]
        if len(event.selection.rows) > 0:
            selected_index = event.selection.rows[0]
            selected_row = full_portfolio_df.iloc[selected_index]
            default_ticker = selected_row['代號']
            last_strat_str = df_records[df_records['Ticker'] == default_ticker].iloc[-1]['Strategy']
            possible_strats = ["存股", "波段"]
            for s in possible_strats:
                if s in last_strat_str: default_strat = [s]; break
            st.divider(); st.markdown(f"#### 📂 {default_ticker} 歷史與操作")
            target_hist = df_records[df_records['Ticker'] == default_ticker].sort_values('Date', ascending=False)
            st.dataframe(target_hist[['Date', 'Action', 'Strategy', 'Price', 'Shares', 'Total_Amount']].head(5), use_container_width=True, hide_index=True)
        col_input1, col_input2 = st.columns([2, 1])
        with col_input1:
            with st.form(f"bottom_entry_form_{key_prefix}", clear_on_submit=True):
                st.markdown(f"**➕ 新增交易** {f'({default_ticker})' if default_ticker else ''}")
                c1, c2, c3, c4 = st.columns(4)
                with c1: d_date = st.date_input("日期")
                with c1: d_ticker = st.text_input("代號", value=default_ticker).upper()
                with c2: d_type = st.selectbox("種類", ["股票", "基金"]); d_action = st.selectbox("動作", ["買入", "賣出", "領息", "分割"])
                with c3: d_strat = st.multiselect("策略", ["存股", "波段"], default=default_strat); d_price = st.number_input("單價", min_value=0.0, format="%.2f")
                with c4: d_shares = st.number_input("股數", step=100.0); d_fee = st.number_input("手續費 (0自動算)", min_value=0)
                c5, c6 = st.columns([3, 1])
                with c5: d_total = st.number_input("總金額 (0自動算)", step=1000.0); d_note = st.text_input("備註")
                with c6: st.write(""); st.write(""); submitted = st.form_submit_button("送出交易", use_container_width=True)
                if submitted:
                    if not d_ticker: st.error("請輸入代號")
                    else:
                        success = handle_transaction_submit(d_date, d_ticker, d_type, d_strat, d_action, d_price, d_shares, d_fee, d_total, d_note)
                        if success: st.success(f"已儲存 {d_ticker}！"); st.cache_data.clear()
        with col_input2:
            with st.form(f"bottom_fund_form_{key_prefix}", clear_on_submit=True):
                st.markdown("**💵 更新基金淨值**")
                f_ticker = st.text_input("基金代號").upper()
                f_net_val = st.number_input("最新淨值", min_value=0.0, format="%.4f")
                f_currency = st.selectbox("幣別", ["USD", "TWD"])
                st.write(""); f_btn = st.form_submit_button("更新", use_container_width=True)
                if f_btn:
                    try:
                        cell = ws_funds.find(f_ticker)
                        ws_funds.update_cell(cell.row, 2, f_net_val)
                        ws_funds.update_cell(cell.row, 3, str(datetime.now().date()))
                        ws_funds.update_cell(cell.row, 4, f_currency)
                    except:
                        ws_funds.append_row([f_ticker, f_net_val, str(datetime.now().date()), f_currency])
                    st.success("更新成功"); st.cache_data.clear()
    else: st.info("尚無資料，請先新增第一筆交易。")

# ==========================================
# 5. 主程式佈局
# ==========================================
st.title("📊 投資戰情室 v9.7.4 (Layout Fix)")

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

# [v9.7.4 FIX] Initialize total_summary to avoid NameError if try block fails
total_summary = None

if not selected_tickers:
    t_all, t_swing, t_div, t_ai = st.tabs(["🌍 全總覽", "⚡ 波段儀表板", "💰 存股月報", "🤖 AI 設定"])
    
    # Calculate summary safely
    if not df.empty:
        try:
            total_summary, _, _ = analyze_period_advanced(df, analysis_start, analysis_end, None, full_portfolio_df, trade_log_df, None)
        except Exception as e:
            st.error(f"計算數據時發生錯誤: {e}")
            total_summary = None

    with t_all:
        if total_summary:
            render_metrics_cards(total_summary, "general")
        
        # [v9.7.4 FIX] Unpack columns outside the 'if' block to ensure variables exist
        st.write("")
        g_col1, g_col2 = st.columns([1, 2])
        
        if total_summary: # Only populate if data exists
            with g_col1: render_allocation_charts(full_portfolio_df)
            with g_col2: render_global_monthly_pnl_colored(trade_log_df, df)
                
        st.divider(); render_inventory_management(full_portfolio_df, df, "overview")
        
        st.markdown("### 🤖 全域診斷報告")
        if st.button("開啟全域分析視窗"):
            dialog_global_analysis(full_portfolio_df, total_summary)
    
    with t_swing:
        st.markdown("### ⚡ 波段交易覆盤 (智慧分群版)")
        st.caption("系統會將「7天內」的連續交易視為同一波操作，進行合併分析。")
        
        col_act, col_info = st.columns([1, 3])
        if col_act.button("🚀 執行批次分析 (10組)", use_container_width=True):
            updates, summaries, _ = run_consolidated_swing_analysis(df, trade_log_df)
            st.session_state['swing_updates'] = updates
            st.session_state['swing_summaries'] = summaries
        
        if st.session_state.get('swing_updates'):
            st.success(f"已分析完成！請確認後存檔。")
            if st.button("💾 全部存檔 (寫入 Google Sheet)", use_container_width=True):
                update_google_sheet_review(st.session_state['swing_updates'])
                del st.session_state['swing_updates']
                del st.session_state['swing_summaries']
                st.rerun()

        st.divider()
        
        swing_tickers = df[df['Strategy'].str.contains('波段', na=False)]['Ticker'].unique()
        
        for t in swing_tickers:
            has_new_summary = False
            ticker_summary_text = ""
            if 'swing_summaries' in st.session_state:
                for key, val in st.session_state['swing_summaries'].items():
                    if key.startswith(f"{t}_"):
                        has_new_summary = True
                        ticker_summary_text = val
                        break
            
            with st.expander(f"📌 {t}", expanded=has_new_summary):
                if has_new_summary:
                    st.info(f"💡 **本波操作總評**：{ticker_summary_text}")
                
                t_history = df[
                    (df['Ticker'] == t) & 
                    (df['Strategy'].str.contains('波段', na=False))
                ].sort_values('Date', ascending=False)
                
                for _, row in t_history.iterrows():
                    review = row['AI_Review']
                    is_pending = False
                    if st.session_state.get('swing_updates'):
                        for rid, txt in st.session_state['swing_updates']:
                            if rid == row['RowIndex']:
                                review = f"🆕 {txt}"
                                is_pending = True
                    
                    if review:
                        msg = f"**{row['Date']} {row['Action']} {row['Price']}**: {review}"
                        if is_pending: st.warning(msg)
                        else: st.markdown(msg)
                    else:
                        st.caption(f"{row['Date']} {row['Action']} {row['Price']} (尚未分析)")

    with t_div:
        run_dividend_soft_lock(full_portfolio_df)
        st.divider()
        render_inventory_management(full_portfolio_df, df, "div")
    
    with t_ai:
        st.info("此區保留給未來的 AI 設定功能。")
        
else:
    ticker_tabs = st.tabs([f"🔍 {t}" for t in selected_tickers])
    for i, ticker in enumerate(selected_tickers):
        with ticker_tabs[i]:
            ticker_df = df[df['Ticker'] == ticker]
            strategies_used = ticker_df['Strategy'].unique().tolist()
            combined_strategies = ",".join([str(s) for s in strategies_used])
            has_swing = "波段" in combined_strategies; has_div = "存股" in combined_strategies
            
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
            render_inventory_management(full_portfolio_df, df, f"ticker_{i}")
