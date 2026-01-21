# Version: v8.2 (Streamlit Cloud Secrets Only)
# CTOSignature: Removed Sidebar API Input, Auto-load 'gemini_api_key' from Cloud Secrets
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
st.set_page_config(page_title="投資戰情室 v8.2 (Cloud AI)", layout="wide")

@st.cache_resource
def connect_google_sheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    try:
        # 優先嘗試讀取 Streamlit Cloud 的 Secrets
        if "gcp_service_account" in st.secrets:
            creds_dict = st.secrets["gcp_service_account"]
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        else:
            # 本機備援 (若有 secrets.json)
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

def normalize_data(df):
    if df.empty: return df
    act_map = {'Buy': '買入', 'Sell': '賣出', 'Dividend': '領息', 'Split': '分割', 'Buy (Buy)': '買入', 'Sell (Sell)': '賣出'}
    strat_map = {
        'Dividend': '存股', 'Swing': '波段', 'Swing Short': '波段', 'Swing Long': '波段',
        '波段-短期': '波段', '波段-長期': '波段', '波動': '波段', '波動-短期': '波段', '波動-長期': '波段'
    }
    type_map = {'Stock': '股票', 'Fund': '基金'}

    if 'Action' in df.columns:
        df['Action'] = df['Action'].replace(act_map)
    if 'Strategy' in df.columns:
        for old, new in strat_map.items():
            df['Strategy'] = df['Strategy'].str.replace(old, new, regex=False)
    if 'Type' in df.columns:
        df['Type'] = df['Type'].replace(type_map)
    return df

def load_data():
    records_data = ws_records.get_all_records()
    df = pd.DataFrame(records_data)
    
    try:
        funds_data = ws_funds.get_all_records()
        df_funds = pd.DataFrame(funds_data)
        if not df_funds.empty and 'Currency' not in df_funds.columns:
            df_funds['Currency'] = 'USD'
            if ws_funds.cell(1, 4).value != "Currency":
                ws_funds.update_cell(1, 4, "Currency")
    except:
        df_funds = pd.DataFrame()
    
    if df.empty: return df, df_funds, 32.0
        
    numeric_cols = ['Price', 'Shares', 'Fee', 'Total_Amount']
    for col in numeric_cols:
        if df[col].dtype == object:
             df[col] = df[col].astype(str).str.replace(',','').str.replace('$','')
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
            portfolio[ticker] = {'shares': 0, 'total_cost': 0, 'realized_pl': 0, 'dividend_collected': 0, 'type': typ, 'strategy': strat}
        
        p = portfolio[ticker]
        p['strategy'] = strat 

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
                trade_log.append({'Date': date_txn, 'Ticker': ticker, 'Strategy': p['strategy'], 'Type': p['type'], 'PnL': pnl, 'SellAmount': amount})
                if p['shares'] <= 0.001: p['shares'] = 0; p['total_cost'] = 0
                    
        elif action == '領息':
            p['dividend_collected'] += amount
            
        elif action == '分割': 
            p['shares'] += qty
            if p['shares'] <= 0.001: p['shares'] = 0; p['total_cost'] = 0
            
    results = []
    for ticker, data in portfolio.items():
        current_price = 0; market_value = 0; volatility = 0
        if data['shares'] > 0.001:
            if data['type'] == '股票':
                current_price, volatility = get_stock_data(ticker)
                market_value = current_price * data['shares']
            elif data['type'] == '基金':
                if not df_funds.empty and ticker in df_funds['Ticker'].values:
                    fund_row = df_funds[df_funds['Ticker'] == ticker].iloc[0]
                    net_val = fund_row['Net_Value_USD']
                    currency = 'USD'
                    if 'Currency' in df_funds.columns:
                        currency = fund_row['Currency']
                    
                    if currency == 'TWD':
                        current_price = net_val
                        market_value = data['shares'] * net_val
                    else:
                        current_price = net_val * current_usd_rate
                        market_value = data['shares'] * net_val * current_usd_rate
            
            avg_cost = data['total_cost'] / data['shares']
            unrealized_pl = market_value - data['total_cost']
            
            yield_on_cost = (data['dividend_collected'] / data['total_cost'] * 100) if data['total_cost'] > 0 else 0
            roi_price = (unrealized_pl / data['total_cost'] * 100) if data['total_cost'] > 0 else 0
            total_gain = unrealized_pl + data['dividend_collected']
            roi_total = (total_gain / data['total_cost'] * 100) if data['total_cost'] > 0 else 0
            fill_status = "✅已填" if current_price >= avg_cost else "🔻貼息"
            
            results.append({
                "代號": ticker, "種類": data['type'], "策略": data['strategy'], "庫存": data['shares'], "平均成本": round(avg_cost, 2),
                "市價": round(current_price, 2), "庫存現值": round(market_value, 0), "帳面損益": round(unrealized_pl, 0),
                "成本殖利率%": round(yield_on_cost, 2), "含息總報%": round(roi_total, 2), "已領股息": round(data['dividend_collected'], 0),
                "填息": fill_status, "總成本": round(data['total_cost'], 0)
            })
    
    pf_df = pd.DataFrame(results)
    if not pf_df.empty:
        total_mv = pf_df['庫存現值'].sum()
        if total_mv > 0:
            pf_df['佔比%'] = (pf_df['庫存現值'] / total_mv * 100).round(1)
        else:
            pf_df['佔比%'] = 0.0
    
    return pf_df, pd.DataFrame(trade_log)

def get_historical_cost_basis(df, cutoff_date, selected_tickers=None, strategy_filter=None):
    hist_df = df[df['Date'] < cutoff_date].sort_values('Date')
    
    if selected_tickers: hist_df = hist_df[hist_df['Ticker'].isin(selected_tickers)]
    if strategy_filter: hist_df = hist_df[hist_df['Strategy'].str.contains(strategy_filter, na=False)]
    
    portfolio_temp = {}
    for _, row in hist_df.iterrows():
        ticker = row['Ticker']; action = row['Action']
        qty = row['Shares']; amount = row['Total_Amount']
        
        if ticker not in portfolio_temp: portfolio_temp[ticker] = {'shares': 0, 'total_cost': 0}
        p = portfolio_temp[ticker]
        
        if action == '買入':
            p['shares'] += qty; p['total_cost'] += amount
        elif action == '賣出':
            if p['shares'] > 0:
                pct_sold = qty / p['shares']
                cost_of_sold = p['total_cost'] * pct_sold
                p['shares'] -= qty; p['total_cost'] -= cost_of_sold
        elif action == '分割':
            p['shares'] += qty
            if p['shares'] <= 0.001: p['shares'] = 0; p['total_cost'] = 0
            
    total_basis = sum([d['total_cost'] for d in portfolio_temp.values() if d['shares'] > 0.001])
    return total_basis

def analyze_period_advanced(df, start_date, end_date, selected_tickers, current_portfolio_df, trade_log_df, strategy_filter=None):
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    if selected_tickers: mask = mask & (df['Ticker'].isin(selected_tickers))
    if strategy_filter: mask = mask & (df['Strategy'].str.contains(strategy_filter, na=False))
        
    period_df = df[mask].copy()
    if period_df.empty: return None, pd.DataFrame(), pd.DataFrame()

    total_dividend = period_df[period_df['Action'] == '領息']['Total_Amount'].sum()
    total_buy = period_df[period_df['Action'] == '買入']['Total_Amount'].sum()
    total_sell = period_df[period_df['Action'] == '賣出']['Total_Amount'].sum()
    
    ending_inventory_value = 0; total_cost_basis = 0
    is_current = end_date >= datetime.now().date()
    
    if is_current and not current_portfolio_df.empty:
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

    yoc_period = 0
    if total_cost_basis > 0: yoc_period = (total_dividend / total_cost_basis) * 100
    payback_progress = 0 
    if total_buy > 0: payback_progress = (total_dividend / total_buy) * 100

    summary = {
        "累積總損益": total_profit, "已領股息": total_dividend, "已實現損益": realized_pnl_period,
        "未實現損益": total_unrealized, "勝率%": win_rate, "XIRR%": xirr_val,
        "YoC%": yoc_period, "回本率%": payback_progress, "庫存現值": ending_inventory_value
    }

    years_data = []
    start_y = start_date.year; end_y = end_date.year
    for y in range(start_y, end_y + 1):
        y_df = period_df[pd.to_datetime(period_df['Date']).dt.year == y]
        y_trades = pd.DataFrame()
        if not trade_log_df.empty:
            y_trades = trade_log_df[(pd.to_datetime(trade_log_df['Date']).dt.year == y)]
            if selected_tickers: y_trades = y_trades[y_trades['Ticker'].isin(selected_tickers)]
            if strategy_filter: y_trades = y_trades[y_trades['Strategy'].str.contains(strategy_filter, na=False)]

        if not y_df.empty or not y_trades.empty:
            y_div = y_df[y_df['Action'] == '領息']['Total_Amount'].sum()
            y_buy = y_df[y_df['Action'] == '買入']['Total_Amount'].sum()
            y_realized = 0; y_win = 0
            if not y_trades.empty:
                y_realized = y_trades['PnL'].sum()
                wins = y_trades[y_trades['PnL'] > 0]
                if len(y_trades) > 0: y_win = (len(wins) / len(y_trades)) * 100
            
            y_cash_flows = []
            for _, r in y_df.iterrows():
                d = r['Date']; amt = r['Total_Amount']; act = r['Action']
                if act == '買入': y_cash_flows.append((d, -amt))
                elif act in ['賣出', '領息']: y_cash_flows.append((d, amt))
            y_xirr = xirr(y_cash_flows)
            y_xirr_str = f"{y_xirr*100:.2f}%" if y_xirr else "N/A"

            y_start_date = date(y, 1, 1)
            y_start_cost_basis = get_historical_cost_basis(df, y_start_date, selected_tickers, strategy_filter)
            
            y_roi_denominator = y_start_cost_basis + y_buy
            y_roi = "N/A"
            if y_roi_denominator > 0:
                y_roi_val = ((y_realized + y_div) / y_roi_denominator) * 100
                y_roi = f"{y_roi_val:.2f}%"

            y_yoc = "N/A"
            if y_buy > 0: y_yoc = f"{(y_div/y_buy)*100:.2f}%"

            row_data = {
                "年度": str(y),
                "已領股息": f"${y_div:,.0f}",
                "已實現": f"${y_realized:,.0f}",
                "年度投資報酬率": y_roi,
                "年化報酬率": y_xirr_str,
                "交易勝率": f"{y_win:.1f}%",
                "成本殖利率 (YoC)": y_yoc,
                "回本率": f"{(y_div/y_buy)*100:.1f}%" if y_buy > 0 else "N/A"
            }
            years_data.append(row_data)

    years_df = pd.DataFrame(years_data)
    return summary, period_df, years_df

# ==========================================
# 3. AI 教練核心邏輯 (Secrets Only)
# ==========================================
def ask_gemini_coach(api_key, portfolio_summary_text):
    """呼叫 Gemini API 進行投資診斷"""
    if not api_key: return "⚠️ 未偵測到 API Key，請檢查 Secrets 設定。"
        
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        你是一位專業、嚴格但富有洞察力的避險基金經理人與投資教練。
        請根據以下使用者的投資組合數據進行分析，並給出具體的策略建議、風險警示與鼓勵。
        請使用繁體中文回答，語氣專業且條理分明。

        【投資組合數據摘要】
        {portfolio_summary_text}

        【你的分析重點】
        1. **資產配置健康度**：檢查是否有過度集中（單一標的佔比過高）或過度分散的問題。
        2. **策略執行效率**：
           - 針對「波段」策略：檢視勝率與獲利能力。如果勝率低於 50%，請給予停損或選股建議。
           - 針對「存股」策略：檢視是否有「賺了股息賠了價差」的狀況（例如 YoC 高但帳面大賠）。
        3. **風險警示**：指出潛在的最大風險點。
        4. **下一步行動建議**：給出 3 點具體可執行的建議。

        請開始你的分析：
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ AI 連線錯誤: {str(e)}\n請檢查 API Key 額度或網路狀態。"

def prepare_data_for_ai(full_portfolio_df, summary_metrics):
    if full_portfolio_df.empty: return "目前無庫存資料。"
    top_holdings = full_portfolio_df.sort_values('庫存現值', ascending=False).head(5)
    holdings_str = ""
    for _, row in top_holdings.iterrows():
        holdings_str += f"- {row['代號']} ({row['策略']}): 佔比 {row['佔比%']}%, 帳面損益 ${row['帳面損益']:,.0f}, 含息報酬 {row['含息總報%']}%\n"
    
    text_report = f"""
    [整體績效]
    - 總庫存現值: ${summary_metrics['庫存現值']:,.0f}
    - 累積總損益: ${summary_metrics['累積總損益']:,.0f}
    - 未實現損益: ${summary_metrics['未實現損益']:,.0f}
    - 已實現損益: ${summary_metrics['已實現損益']:,.0f}
    - 已領股息: ${summary_metrics['已領股息']:,.0f}
    - 年化報酬率 (XIRR): {summary_metrics['XIRR%']:.2f}%
    - 波段交易勝率: {summary_metrics['勝率%']:.1f}%
    
    [前五大持股風險曝險]
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
                tax_rate = 0.003
                tax = int(basic_amt * tax_rate)
                final_total = basic_amt - final_fee - tax
                if tax > 0: note = f"{note} (稅 ${tax})".strip()

    new_row = [str(date_in), ticker, type_display, db_strat, action_display, final_price, final_shares, final_fee, final_total, note]
    ws_records.append_row(new_row)
    return True

# ==========================================
# 4. 儀表板與圖表
# ==========================================
def render_allocation_charts(full_portfolio_df):
    """[v7.2 Fixed] 資產配置圓餅圖 - 單一甜甜圈圖"""
    if full_portfolio_df.empty: return
    st.markdown("#### 🥧 資產配置 - 持股佔比")
    base = alt.Chart(full_portfolio_df).encode(theta=alt.Theta("庫存現值", stack=True))
    pie = base.mark_arc(outerRadius=120, innerRadius=60).encode(
        color=alt.Color("代號", title="投資標的", sort=alt.EncodingSortField(field="庫存現值", order="descending")),
        order=alt.Order("庫存現值", sort="descending"),
        tooltip=["代號", "庫存現值", "佔比%", "策略", "種類"]
    )
    st.altair_chart(pie, use_container_width=True)

def render_global_monthly_pnl_colored(trade_log_df, df_records):
    """[v7.2 Kept] 累積已實現損益 (含股息) - 堆疊面積圖 + 時間軸修正"""
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
    
    domain_end = datetime.now().date()
    domain_start = domain_end - timedelta(days=365)

    st.markdown("#### 🌊 累積已實現損益 (含股息) - 財富堆疊圖")
    chart = alt.Chart(grouped).mark_area(opacity=0.7).encode(
        x=alt.X('Date:T', timeUnit='yearmonth', title='月份', 
                scale=alt.Scale(domain=[pd.to_datetime(domain_start), pd.to_datetime(domain_end)])),
        y=alt.Y('Cumulative_PnL:Q', title='累積已實現獲利 ($)', stack=True), 
        color=alt.Color('Type:N', title='資產種類', scale=alt.Scale(domain=['股票', '基金'], range=['#1f77b4', '#ff7f0e'])),
        tooltip=[alt.Tooltip('Date', timeUnit='yearmonth', title='月份'), 'Type', 'Cumulative_PnL', 'PnL']
    ).properties(height=350).interactive()
    st.altair_chart(chart, use_container_width=True)

def render_metrics_cards(summary, mode):
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

# ==========================================
# 5. 主程式佈局
# ==========================================
st.title("📊 投資戰情室 v8.2 (Cloud AI)")

df, df_funds, usd_rate = load_data()
if df.empty:
    st.warning("目前無任何交易紀錄")
    st.stop()

all_tickers = df['Ticker'].unique().tolist()
full_portfolio_df, trade_log_df = calculate_portfolio(df, df_funds, usd_rate)

# [v7.1 Layout] 篩選列水平置頂
st.markdown("#### 🔍 篩選條件")
f1, f2, f3 = st.columns([1, 1, 2])
with f1:
    min_date = df['Date'].min()
    max_date = date.today()
    analysis_start = st.date_input("開始日期", value=min_date, min_value=min_date, max_value=max_date)
with f2:
    analysis_end = st.date_input("結束日期", value=max_date, min_value=min_date, max_value=max_date)
with f3:
    selected_tickers = st.multiselect("投資標的", all_tickers, default=None)

st.divider()

# 主呈現區 (全寬度)
if not selected_tickers:
    t_all, t_swing, t_div, t_ai = st.tabs(["🌍 全總覽", "⚡ 波段", "💰 存股", "🤖 AI 教練"])
    
    with t_all:
        total_summary, _, _ = analyze_period_advanced(df, analysis_start, analysis_end, None, full_portfolio_df, trade_log_df, None)
        if total_summary:
            render_metrics_cards(total_summary, "general")
            st.write("")
            g_col1, g_col2 = st.columns([1, 2])
            with g_col1: render_allocation_charts(full_portfolio_df)
            with g_col2: render_global_monthly_pnl_colored(trade_log_df, df)
    
    with t_swing:
        render_strategy_view(df, analysis_start, analysis_end, None, "波段", full_portfolio_df, trade_log_df, "swing")
    
    with t_div:
        render_strategy_view(df, analysis_start, analysis_end, None, "存股", full_portfolio_df, trade_log_df, "dividend")
    
    # [v8.2 Cloud AI] 分頁 4: AI 教練 (自動讀取 Secrets)
    with t_ai:
        st.markdown("### 🤖 您的專屬 AI 投資顧問")
        st.info("AI 教練將分析您的「庫存結構」、「策略績效」與「交易紀錄」，提供客觀的診斷報告。")
        
        if st.button("🚀 開始 AI 診斷 (Call Gemini)", use_container_width=True):
            # 優先檢查 Secrets 是否有 Key
            api_key_to_use = st.secrets.get("gemini_api_key", None)
            
            if not api_key_to_use:
                st.error("❌ 未偵測到 API Key！請確認 Streamlit Cloud 後台 Secrets 是否已設定 `gemini_api_key`。")
            else:
                total_summary, _, _ = analyze_period_advanced(df, min_date, date.today(), None, full_portfolio_df, trade_log_df, None)
                data_prompt = prepare_data_for_ai(full_portfolio_df, total_summary)
                
                with st.spinner("AI 教練正在分析您的投資組合..."):
                    advice = ask_gemini_coach(api_key_to_use, data_prompt)
                    st.markdown(advice)
        
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
    stocks_pf = full_portfolio_df[full_portfolio_df['種類'] == '股票']
    funds_pf = full_portfolio_df[full_portfolio_df['種類'] == '基金']
    
    if not stocks_pf.empty:
        st.markdown("#### 📈 股票庫存")
        s_cost = stocks_pf['總成本'].sum()
        s_pl = stocks_pf['帳面損益'].sum()
        s_div = stocks_pf['已領股息'].sum()
        s_roi = ((s_pl + s_div) / s_cost * 100) if s_cost > 0 else 0
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("股票總現值", f"${stocks_pf['庫存現值'].sum():,.0f}")
        s2.metric("股票總成本", f"${s_cost:,.0f}")
        s3.metric("股票帳面損益", f"${s_pl:,.0f}", delta_color="normal")
        s4.metric("股票總報酬率", f"{s_roi:.2f}%")
    
    if not funds_pf.empty:
        st.markdown("#### 🛡️ 基金庫存")
        f1, f2, f3, f4 = st.columns(4)
        f_cost = funds_pf['總成本'].sum()
        f_pl = funds_pf['帳面損益'].sum()
        f_roi = (f_pl / f_cost * 100) if f_cost > 0 else 0
        f1.metric("基金總現值", f"${funds_pf['庫存現值'].sum():,.0f}")
        f2.metric("基金總投入", f"${f_cost:,.0f}")
        f3.metric("基金帳面損益", f"${f_pl:,.0f}", delta_color="normal")
        f4.metric("基金總報酬率", f"{f_roi:.2f}%")

    st.write("") 

    cols_show = ["代號", "種類", "佔比%", "庫存", "平均成本", "市價", "庫存現值", "帳面損益", "含息總報%", "策略"]
    event = st.dataframe(
        full_portfolio_df[cols_show], use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key="inventory_table"
    )
    
    default_ticker = ""; default_strat = ["存股"]
    if len(event.selection.rows) > 0:
        selected_index = event.selection.rows[0]
        selected_row = full_portfolio_df.iloc[selected_index]
        default_ticker = selected_row['代號']
        last_strat_str = df[df['Ticker'] == default_ticker].iloc[-1]['Strategy']
        possible_strats = ["存股", "波段"]
        for s in possible_strats:
            if s in last_strat_str: default_strat = [s]; break
        
        st.divider()
        st.markdown(f"#### 📂 {default_ticker} 歷史與操作")
        target_hist = df[df['Ticker'] == default_ticker].sort_values('Date', ascending=False)
        st.dataframe(target_hist[['Date', 'Action', 'Strategy', 'Price', 'Shares', 'Total_Amount']].head(5), use_container_width=True, hide_index=True)

    col_input1, col_input2 = st.columns([2, 1])
    with col_input1:
        with st.form("bottom_entry_form", clear_on_submit=True):
            st.markdown(f"**➕ 新增交易** {f'({default_ticker})' if default_ticker else ''}")
            c1, c2, c3, c4 = st.columns(4)
            with c1: d_date = st.date_input("日期"); d_ticker = st.text_input("代號", value=default_ticker).upper()
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
        with st.form("bottom_fund_form", clear_on_submit=True):
            st.markdown("**💵 更新基金淨值**")
            f_ticker = st.text_input("基金代號").upper()
            f_net_val = st.number_input("最新淨值", min_value=0.0, format="%.4f")
            f_currency = st.selectbox("幣別", ["USD", "TWD"])
            st.write("")
            f_btn = st.form_submit_button("更新", use_container_width=True)
            if f_btn:
                try:
                    cell = ws_funds.find(f_ticker)
                    ws_funds.update_cell(cell.row, 2, f_net_val)
                    ws_funds.update_cell(cell.row, 3, str(datetime.now().date()))
                    ws_funds.update_cell(cell.row, 4, f_currency)
                except:
                    ws_funds.append_row([f_ticker, f_net_val, str(datetime.now().date()), f_currency])
                st.success("更新成功"); st.cache_data.clear()
else:
    st.info("尚無資料，請先新增第一筆交易。")
