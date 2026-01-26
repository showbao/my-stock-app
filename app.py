# Version: v8.6 Patch (Fix NameError & Connection Safety)
# CTOSignature: Initialized variables to prevent NameError, Added connection try-except blocks
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

# ==========================================
# 1. 系統設定與連線
# ==========================================
st.set_page_config(page_title="投資戰情室 v8.6 Patch", layout="wide")

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
        # Return None to handle gracefully later
        return None

sh = connect_google_sheet()
# Safety check
if sh:
    try:
        ws_records = sh.worksheet("Records")
        ws_funds = sh.worksheet("Fund_Updates")
    except:
        st.warning("找不到工作表 'Records' 或 'Fund_Updates'，請檢查 Google Sheet。")
        st.stop()
else:
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
            else: volatility = 0.0
            return current_price, volatility
        return 0.0, 0.0
    except: return 0.0, 0.0

@st.cache_data(ttl=86400) 
def get_historical_price_window(ticker, trade_date, window_days=10):
    try:
        t_date = pd.to_datetime(trade_date)
        start_d = t_date - timedelta(days=window_days + 5)
        end_d = t_date + timedelta(days=window_days + 5)
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_d.strftime('%Y-%m-%d'), end=end_d.strftime('%Y-%m-%d'), auto_adjust=True)
        if hist.empty: return None
        mask_window = (hist.index >= (t_date - timedelta(days=window_days))) & (hist.index <= (t_date + timedelta(days=window_days)))
        window_df = hist.loc[mask_window]
        if window_df.empty: return None
        return {
            "window_high": window_df['High'].max(),
            "window_low": window_df['Low'].min(),
            "price_at_trade": window_df.loc[window_df.index.normalize() == t_date.normalize()]['Close'].mean()
        }
    except: return None

def normalize_data(df):
    if df.empty: return df
    act_map = {'Buy': '買入', 'Sell': '賣出', 'Dividend': '領息', 'Split': '分割', 'Buy (Buy)': '買入', 'Sell (Sell)': '賣出'}
    strat_map = {'Dividend': '存股', 'Swing': '波段', 'Swing Short': '波段', 'Swing Long': '波段', '波段-短期': '波段', '波段-長期': '波段', '波動': '波段', '波動-短期': '波段', '波動-長期': '波段'}
    type_map = {'Stock': '股票', 'Fund': '基金'}
    if 'Action' in df.columns: df['Action'] = df['Action'].replace(act_map)
    if 'Strategy' in df.columns:
        for old, new in strat_map.items(): df['Strategy'] = df['Strategy'].str.replace(old, new, regex=False)
    if 'Type' in df.columns: df['Type'] = df['Type'].replace(type_map)
    return df

def load_data():
    try:
        records_data = ws_records.get_all_records()
        df = pd.DataFrame(records_data)
    except:
        return pd.DataFrame(), pd.DataFrame(), 32.0 # Return empty if sheet fails

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
        if ticker not in portfolio:
            portfolio[ticker] = {'shares': 0, 'total_cost': 0, 'realized_pl': 0, 'dividend_collected': 0, 'type': typ, 'strategy': strat}
        p = portfolio[ticker]; p['strategy'] = strat 
        if action == '買入': p['shares'] += qty; p['total_cost'] += amount
        elif action == '賣出':
            if p['shares'] > 0:
                pct_sold = qty / p['shares']
                cost_of_sold_shares = p['total_cost'] * pct_sold
                pnl = amount - cost_of_sold_shares
                p['realized_pl'] += pnl; p['total_cost'] -= cost_of_sold_shares; p['shares'] -= qty
                trade_log.append({'Date': date_txn, 'Ticker': ticker, 'Strategy': p['strategy'], 'Type': p['type'], 'PnL': pnl, 'SellAmount': amount, 'SellPrice': (amount/qty) if qty>0 else 0})
                if p['shares'] <= 0.001: p['shares'] = 0; p['total_cost'] = 0
        elif action == '領息': p['dividend_collected'] += amount
        elif action == '分割': p['shares'] += qty
            
    results = []
    for ticker, data in portfolio.items():
        current_price = 0; market_value = 0
        if data['shares'] > 0.001:
            if data['type'] == '股票': current_price, _ = get_stock_data(ticker)
            elif data['type'] == '基金':
                if not df_funds.empty and ticker in df_funds['Ticker'].values:
                    fund_row = df_funds[df_funds['Ticker'] == ticker].iloc[0]
                    net_val = fund_row['Net_Value_USD']
                    currency = 'USD'
                    if 'Currency' in df_funds.columns: currency = fund_row['Currency']
                    current_price = net_val if currency == 'TWD' else net_val * current_usd_rate
            market_value = current_price * data['shares']
            avg_cost = data['total_cost'] / data['shares']
            unrealized_pl = market_value - data['total_cost']
            total_gain = unrealized_pl + data['dividend_collected']
            roi_total = (total_gain / data['total_cost'] * 100) if data['total_cost'] > 0 else 0
            results.append({
                "代號": ticker, "種類": data['type'], "策略": data['strategy'], "庫存": data['shares'], "平均成本": round(avg_cost, 2),
                "市價": round(current_price, 2), "庫存現值": round(market_value, 0), "帳面損益": round(unrealized_pl, 0),
                "已領股息": round(data['dividend_collected'], 0), "含息總報%": round(roi_total, 2), "總成本": round(data['total_cost'], 0)
            })
    pf_df = pd.DataFrame(results)
    if not pf_df.empty:
        total_mv = pf_df['庫存現值'].sum()
        if total_mv > 0: pf_df['佔比%'] = (pf_df['庫存現值'] / total_mv * 100).round(1)
        else: pf_df['佔比%'] = 0.0
    return pf_df, pd.DataFrame(trade_log)

def get_historical_cost_basis(df, cutoff_date, selected_tickers=None, strategy_filter=None):
    hist_df = df[df['Date'] < cutoff_date].sort_values('Date')
    if selected_tickers: hist_df = hist_df[hist_df['Ticker'].isin(selected_tickers)]
    if strategy_filter: hist_df = hist_df[hist_df['Strategy'].str.contains(strategy_filter, na=False)]
    portfolio_temp = {}
    for _, row in hist_df.iterrows():
        ticker = row['Ticker']; action = row['Action']; qty = row['Shares']; amount = row['Total_Amount']
        if ticker not in portfolio_temp: portfolio_temp[ticker] = {'shares': 0, 'total_cost': 0}
        p = portfolio_temp[ticker]
        if action == '買入': p['shares'] += qty; p['total_cost'] += amount
        elif action == '賣出':
            if p['shares'] > 0:
                pct_sold = qty / p['shares']
                p['shares'] -= qty; p['total_cost'] -= (p['total_cost'] * pct_sold)
        elif action == '分割': p['shares'] += qty
    return sum([d['total_cost'] for d in portfolio_temp.values() if d['shares'] > 0.001])

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

def ask_gemini_coach(api_key, prompt_text):
    if not api_key: return "⚠️ 未偵測到 API Key，請檢查 Secrets 設定。"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt_text)
        return response.text
    except Exception as e: return f"❌ AI 連線錯誤: {str(e)}"

@st.dialog("🌍 輸入現金餘額以進行全域分析")
def dialog_global_analysis(full_portfolio_df, summary_metrics):
    cash_balance = st.number_input("請輸入目前帳戶閒置現金 (TWD)", min_value=0, value=0, step=10000)
    st.caption("輸入現金能讓 AI 協助判斷資金效率與加碼彈性。")
    if st.button("開始全域診斷", use_container_width=True):
        api_key = st.secrets.get("gemini_api_key", None)
        if not api_key: st.error("無 API Key"); return
        total_assets = summary_metrics['庫存現值'] + cash_balance
        cash_ratio = (cash_balance / total_assets * 100) if total_assets > 0 else 0
        top_holdings = full_portfolio_df.sort_values('庫存現值', ascending=False).head(5)
        holdings_str = ""
        for _, row in top_holdings.iterrows():
            holdings_str += f"- {row['代號']} ({row['種類']}): 佔比 {row['佔比%']}%\n"
        prompt = f"""
        你是一位專業的資產配置顧問。請依據以下【事實數據】進行全域診斷。
        【嚴格指令】1. 絕對依據提供的數據。2. 使用繁體中文。
        【資產數據】
        - 股票庫存現值: ${summary_metrics['庫存現值']:,.0f}
        - 閒置現金餘額: ${cash_balance:,.0f}
        - 總資產: ${total_assets:,.0f}
        - 現金水位: {cash_ratio:.1f}%
        - 總體未實現損益: ${summary_metrics['未實現損益']:,.0f}
        【前五大持股】{holdings_str}
        【分析重點】1. 資金效率。2. 集中度風險。3. 整體建議。
        """
        with st.spinner("AI 正在計算資金效率與風險..."):
            advice = ask_gemini_coach(api_key, prompt)
            st.session_state['ai_result'] = advice; st.rerun()

def run_swing_analysis_advanced(df_raw, trade_log_df):
    api_key = st.secrets.get("gemini_api_key", None)
    if not api_key: st.error("無 API Key"); return
    buys = df_raw[(df_raw['Strategy'].str.contains('波段', na=False)) & (df_raw['Action'] == '買入')].tail(5) 
    sells = trade_log_df[trade_log_df['Strategy'].str.contains('波段', na=False)].tail(5)
    analysis_log = "[買入點位回測 - Entry Analysis]\n"
    with st.status("正在進行前後 10 日股價回測...", expanded=True) as status:
        for _, row in buys.iterrows():
            ticker = row['Ticker']; buy_date = row['Date']; buy_price = row['Price']
            context = get_historical_price_window(ticker, buy_date)
            if context:
                win_low = context['window_low']
                dist_from_low = ((buy_price - win_low) / win_low * 100)
                status.write(f"回測買入 {ticker}: 買價 {buy_price}, 區間最低 {win_low:.2f} (距離 +{dist_from_low:.1f}%)")
                analysis_log += f"- {ticker} 買入 {buy_date}: 買價 {buy_price}, 視窗(±10天)最低價 {win_low:.2f}, 買點距離最低點僅 +{dist_from_low:.1f}%\n"
        analysis_log += "\n[賣出點位回測 - Exit Analysis]\n"
        for _, row in sells.iterrows():
            ticker = row['Ticker']; sell_date = row['Date']; sell_price = row['SellPrice']
            context = get_historical_price_window(ticker, sell_date)
            if context:
                win_high = context['window_high']
                missed_gain = ((win_high - sell_price) / sell_price * 100)
                status.write(f"回測賣出 {ticker}: 賣價 {sell_price:.2f}, 區間最高 {win_high:.2f} (賣飛 -{missed_gain:.1f}%)")
                analysis_log += f"- {ticker} 賣出 {sell_date}: 賣價 {sell_price:.2f}, 視窗(±10天)最高價 {win_high:.2f}, 賣飛幅度 {missed_gain:.1f}%, 該筆損益 ${row['PnL']:.0f}\n"
        status.update(label="回測完成！正在生成 AI 報告...", state="complete", expanded=False)
    prompt = f"""
    你是一位嚴格的波段交易教練。我剛完成了「交易日 前後10天」的雙向股價回測，請分析我的擇時能力。
    【波段交易回測數據 (最近交易)】{analysis_log}
    【分析重點】
    1. **買點精準度 (Entry)**：
       - 若買價距離最低點很近 (<3%)，請稱讚「抄底精準」。
       - 若距離很遠，請提醒「追高風險」。
    2. **賣點精準度 (Exit)**：
       - 若賣價距離最高點很近，請稱讚「賣得漂亮」。
       - 若賣飛幅度大 (>10%)，請分析是否有「太早獲利了結」的心態。
    3. **總結建議**：針對買賣操作給予一個具體的改進方向。
    """
    return ask_gemini_coach(api_key, prompt)

def run_dividend_analysis(full_portfolio_df):
    api_key = st.secrets.get("gemini_api_key", None)
    if not api_key: st.error("無 API Key"); return
    div_stocks = full_portfolio_df[full_portfolio_df['策略'].str.contains('存股', na=False)]
    if div_stocks.empty: return "無存股庫存。"
    stocks_str = ""
    for _, row in div_stocks.iterrows():
        yoc = row['成本殖利率%']
        stocks_str += f"- {row['代號']}: 總成本 ${row['總成本']:,.0f}, 已領股息 ${row['已領股息']:,.0f}, 帳面損益 ${row['帳面損益']:,.0f}, YoC {yoc}%\n"
    prompt = f"""
    你是一位價值投資專家。請檢視以下的存股組合健康度。
    【存股庫存數據】{stocks_str}
    【分析重點】
    1. **高殖利率陷阱偵測**：是否有「賺了股息、賠了價差」的股票？
    2. **持有信心**：針對 YoC 高的標的給予鼓勵。
    3. **複利建議**：簡述再投入的重要性。
    """
    return ask_gemini_coach(api_key, prompt)

def prepare_data_for_ai(full_portfolio_df, summary_metrics, swing_metrics):
    if full_portfolio_df.empty: return "目前無庫存資料。"
    top_holdings = full_portfolio_df.sort_values('庫存現值', ascending=False).head(5)
    holdings_str = ""
    for _, row in top_holdings.iterrows():
        holdings_str += f"- 代號 {row['代號']} ({row['策略']}): 市值 ${row['庫存現值']:,.0f} (佔比 {row['佔比%']}%), 帳面損益 ${row['帳面損益']:,.0f}, 含息報酬率 {row['含息總報%']}%\n"
    swing_win_rate = f"{swing_metrics['勝率%']:.1f}%" if swing_metrics else "無資料"
    swing_pnl = f"${swing_metrics['已實現損益']:,.0f}" if swing_metrics else "0"
    text_report = f"""
    [整體帳戶摘要]
    - 總庫存現值: ${summary_metrics['庫存現值']:,.0f}
    - 累積總損益: ${summary_metrics['累積總損益']:,.0f}
    [波段策略專屬績效]
    - 波段交易勝率: {swing_win_rate}
    - 波段已實現獲利: {swing_pnl}
    [前五大重倉持股]
    {holdings_str}
    """
    return text_report

def handle_transaction_submit(date_in, ticker, type_display, strategy_list, action_display, price, shares, fee, total_amt, note):
    db_strat = ",".join(strategy_list)
    final_shares = shares; final_price = price; final_fee = fee; final_total = total_amt
    if final_fee == 0 and action_display in ["買入", "賣出"]:
        final_fee = int(price * shares * 0.001425)
    if action_display == "領息":
        final_shares = 0; final_price = 0
        if final_total == 0: st.error("領息金額不能為 0"); return False
    elif action_display == "分割": final_total = 0; final_price = 0
    else:
        if final_total == 0:
            basic_amt = price * shares
            if action_display == "買入": final_total = basic_amt + final_fee
            elif action_display == "賣出":
                tax_rate = 0.003; tax = int(basic_amt * tax_rate)
                final_total = basic_amt - final_fee - tax
                if tax > 0: note = f"{note} (稅 ${tax})".strip()
    new_row = [str(date_in), ticker, type_display, db_strat, action_display, final_price, final_shares, final_fee, final_total, note]
    ws_records.append_row(new_row); return True

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
st.title("📊 投資戰情室 v8.6 (Pro AI)")

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

if not selected_tickers:
    t_all, t_swing, t_div, t_ai = st.tabs(["🌍 全總覽", "⚡ 波段", "💰 存股", "🤖 AI 教練"])
    # 1. 預先計算總表
    total_summary = None
    if not df.empty:
        total_summary, _, _ = analyze_period_advanced(df, analysis_start, analysis_end, None, full_portfolio_df, trade_log_df, None)

    with t_all:
        if total_summary:
            render_metrics_cards(total_summary, "general")
            st.write(""); g_col1, g_col2 = st.columns([1, 2])
            with g_col1: render_allocation_charts(full_portfolio_df)
            with g_col2: render_global_monthly_pnl_colored(trade_log_df, df)
        st.divider(); render_inventory_management(full_portfolio_df, df, "overview")
    
    with t_swing:
        render_strategy_view(df, analysis_start, analysis_end, None, "波段", full_portfolio_df, trade_log_df, "swing")
        st.divider(); render_inventory_management(full_portfolio_df, df, "swing")
    
    with t_div:
        render_strategy_view(df, analysis_start, analysis_end, None, "存股", full_portfolio_df, trade_log_df, "dividend")
        st.divider(); render_inventory_management(full_portfolio_df, df, "div")
    
    with t_ai:
        st.markdown("### 🤖 您的專屬 AI 投資顧問")
        st.info("請選擇您想進行的分析面向。AI 將根據您的選擇，載入不同的數據模型進行運算。")
        c_ai_1, c_ai_2, c_ai_3 = st.columns(3)
        
        with c_ai_1:
            # 確保 total_summary 已計算，否則重新計算
            total_sum_ai = total_summary if total_summary else analyze_period_advanced(df, min_date, date.today(), None, full_portfolio_df, trade_log_df, None)[0]
            if st.button("🌍 全域總覽診斷", use_container_width=True):
                dialog_global_analysis(full_portfolio_df, total_sum_ai)
        
        with c_ai_2:
            if st.button("⚡ 波段交易回測 (±10日)", use_container_width=True):
                with st.spinner("正在抓取歷史股價並進行分析..."):
                    advice = run_swing_analysis_advanced(df, trade_log_df)
                    st.session_state['ai_result'] = advice
        
        with c_ai_3:
            if st.button("💰 存股體質健檢", use_container_width=True):
                with st.spinner("正在分析存股績效..."):
                    advice = run_dividend_analysis(full_portfolio_df)
                    st.session_state['ai_result'] = advice
        
        st.divider()
        if 'ai_result' in st.session_state: st.markdown(st.session_state['ai_result'])
        
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
