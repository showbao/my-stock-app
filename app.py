import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from google.oauth2 import service_account
from gspread_pandas import Spread, Client
import plotly.graph_objects as go

# --- 頁面配置 ---
st.set_page_config(page_title="個人投資記錄", layout="wide", initial_sidebar_state="collapsed")

# --- 自定義 CSS (現代化 UI & 隱藏側邊欄 & 懸浮按鈕) ---
st.markdown("""
    <style>
    /* 隱藏側邊欄 */
    [data-testid="stSidebar"] {display: none;}
    
    /* 全域背景與字體 */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* 數據卡片設計 */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #eee;
        text-align: center;
    }
    .metric-label { color: #666; font-size: 0.9rem; margin-bottom: 5px; }
    .metric-value { color: #1f1f1f; font-size: 1.5rem; font-weight: 700; }
    
    /* 右下角大圓加號按鈕 (FAB) */
    .fab-container {
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 1000;
    }
    .fab-button {
        width: 60px;
        height: 60px;
        background-color: #007bff;
        border-radius: 50%;
        display: flex;
        justify-content: center;
        align-items: center;
        color: white;
        font-size: 30px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        cursor: pointer;
        border: none;
        transition: transform 0.2s;
    }
    .fab-button:hover { transform: scale(1.1); }

    /* 適配手機 */
    @media (max-width: 600px) {
        .metric-value { font-size: 1.2rem; }
    }
    </style>
    """, unsafe_allow_stdio=True)

# --- 初始化 Google Sheets 連線 ---
# 在 Streamlit Secrets 中設定連線資訊
def get_gsheet_connection():
    try:
        # 使用 streamlit 的 secrets
        creds_dict = st.secrets["gcp_service_account"]
        scope = ['https://www.googleapis.com/auth/spreadsheets']
        creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=scope)
        # 這裡簡化流程，實際建議用 gspread 或 st.connection
        return creds
    except Exception as e:
        st.error(f"無法連線至 Google Sheets: {e}")
        return None

# --- 核心計算邏輯 (平均成本法) ---
def calculate_portfolio(df_tx, df_prices, filter_strategy=None, filter_symbols=None, start_date=None, end_date=None):
    if df_tx.empty:
        return None, "尚未有交易紀錄"

    # 基礎篩選
    df = df_tx.copy()
    df['date'] = pd.to_datetime(df['date'])
    if start_date: df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date: df = df[df['date'] <= pd.to_datetime(end_date)]
    if filter_strategy and filter_strategy != "全部":
        df = df[df['strategy'] == filter_strategy]
    if filter_symbols:
        df = df[df['symbol'].isin(filter_symbols)]

    # 依 Symbol 排序計算
    df = df.sort_values(['symbol', 'date'])
    
    results = {}
    total_dividend = 0
    
    for symbol, group in df.groupby('symbol'):
        holding_qty = 0
        total_cost_twd = 0
        realized_pnl = 0
        accumulated_dividend = 0
        
        # 取得現價與匯率
        current_price = df_prices.loc[df_prices['symbol'] == symbol, 'price'].values[0] if symbol in df_prices['symbol'].values else 0
        usd_twd = df_prices.loc[df_prices['symbol'] == 'USD_TWD', 'price'].values[0] if 'USD_TWD' in df_prices['symbol'].values else 32.0
        asset_type = df_prices.loc[df_prices['symbol'] == symbol, 'asset_type'].values[0] if symbol in df_prices['symbol'].values else 'stock'

        for _, row in group.iterrows():
            action = row['action']
            qty = row['qty']
            amt_twd = row['amount_twd']
            
            if action == 'initial' or action == 'buy':
                total_cost_twd += amt_twd
                holding_qty += qty
            elif action == 'sell':
                if qty > holding_qty:
                    return None, f"資料異常：{symbol} 持股不足以賣出"
                # 平均成本賣出 (不影響單位成本，但減少總成本)
                avg_cost = total_cost_twd / holding_qty if holding_qty > 0 else 0
                total_cost_twd -= (avg_cost * qty)
                holding_qty -= qty
            elif action == 'dividend':
                accumulated_dividend += amt_twd
        
        if holding_qty < 0:
            return None, f"資料異常：{symbol} 持股為負"
            
        # 計算市值 (美股需換算)
        market_val = holding_qty * current_price
        if symbol.endswith('_USD') or (hasattr(row, 'currency') and row['currency'] == 'USD'):
            market_val *= usd_twd
            
        results[symbol] = {
            'holding_qty': holding_qty,
            'total_cost': total_cost_twd,
            'market_value': market_val,
            'dividend': accumulated_dividend,
            'asset_type': asset_type
        }
        total_dividend += accumulated_dividend

    # 彙整五大指標
    total_invested = sum(v['total_cost'] for v in results.values())
    current_market_value = sum(v['market_value'] for v in results.values())
    total_pnl = current_market_value - total_invested + total_dividend
    roi = (total_pnl / total_invested) if total_invested > 0 else 0
    
    return {
        'invested': total_invested,
        'market_value': current_market_value,
        'dividend': total_dividend,
        'pnl': total_pnl,
        'roi': roi
    }, None

# --- UI 元件：新增交易彈窗 ---
@st.dialog("新增交易紀錄")
def add_transaction_dialog():
    with st.form("tx_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        asset_type = col1.selectbox("資產類型", ["stock", "fund"])
        action = col2.selectbox("行為", ["buy", "sell", "dividend", "initial"])
        
        date = st.date_input("日期", datetime.now())
        symbol = st.text_input("標的代碼 (Symbol)").strip().upper()
        strategy = st.selectbox("策略", ["存股", "波段"])
        
        currency = st.radio("幣別", ["TWD", "USD"], horizontal=True)
        fx_rate = 1.0
        if currency == "USD":
            fx_rate = st.number_input("匯率 (USD/TWD)", value=32.0, format="%.2f")
            
        qty = 0.0
        price = 0.0
        amt_orig = 0.0
        
        if action in ['buy', 'sell', 'initial']:
            qty = st.number_input("股數/單位數", min_value=0.0, step=1.0)
            if action == 'initial':
                amt_twd_input = st.number_input("總投入金額 (台幣)", min_value=0.0)
            else:
                price = st.number_input("單價", min_value=0.0)
        else: # dividend
            amt_orig = st.number_input("配息金額", min_value=0.0)

        submitted = st.form_submit_button("確認送出", use_container_width=True)
        if submitted:
            # 這裡執行寫入邏輯與驗證
            st.success(f"{symbol} 交易已記錄！")
            st.rerun()

# --- 主程式控制 ---
def main():
    # 1. 登入驗證 (簡化模擬)
    if 'authenticated' not in st.session_state:
        st.title("🔐 投資管理系統登入")
        password = st.text_input("請輸入訪問密碼", type="password")
        if st.button("登入"):
            if password == st.secrets["app_password"]: # 密碼存於 Secrets
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("密碼錯誤")
        return

    # 2. 頂部標題與刷新
    col_t, col_r = st.columns([5, 1])
    col_t.title("📈 我的投資總覽")
    if col_r.button("🔄 刷新價格"):
        # 模擬冷卻邏輯
        st.toast("價格已更新！")

    # 3. 篩選區
    with st.container():
        c1, c2, c3 = st.columns([2, 2, 4])
        date_range = c1.date_input("日期區間", [datetime.now() - timedelta(days=365), datetime.now()])
        strategy_filter = c2.selectbox("投資策略", ["全部", "存股", "波段"])
        
        # 模擬標的篩選 Chips
        all_symbols = ["2330", "0050", "AAPL", "NVDA", "VOO"]
        selected_symbols = c3.multiselect("選擇標的 (可多選)", all_symbols)

    # 4. 數據分頁 (Tabs)
    tab_all, tab_stock, tab_fund = st.tabs(["💎 全部資產", "🏦 股票", "📊 基金"])

    # 模擬資料計算 (正式開發請替換為 calculate_portfolio 讀取 Sheets 資料)
    metrics = {
        'invested': 1000000,
        'market_value': 1250000,
        'dividend': 50000,
        'pnl': 300000,
        'roi': 0.3
    }
    
    def render_metrics(m):
        if not m:
            st.warning("資料異常：持股為負")
            return
        
        # 摘要列
        st.markdown(f"### 總市值: NT$ {m['market_value']:,.0f} | 總報酬率: {m['roi']:.2%}")
        
        cols = st.columns(5)
        labels = ["總投入", "目前市值", "已領息", "總報酬", "總報酬率"]
        values = [
            f"{m['invested']:,.0f}",
            f"{m['market_value']:,.0f}",
            f"{m['dividend']:,.0f}",
            f"{m['pnl']:,.0f}",
            f"{m['roi']:.2%}"
        ]
        
        for i, col in enumerate(cols):
            with col:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{labels[i]}</div>
                        <div class="metric-value">{values[i]}</div>
                    </div>
                """, unsafe_allow_stdio=True)

    with tab_all:
        render_metrics(metrics)
        # 畫個簡單的圖
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = metrics['roi']*100,
            title = {'text': "總獲利 %"},
            gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "#007bff"}}
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with tab_stock:
        render_metrics(metrics) # 這裡應傳入股票類別的數據

    with tab_fund:
        render_metrics(metrics) # 這裡應傳入基金類別的數據

    # 5. 懸浮按鈕 (FAB)
    st.markdown("""
        <div class="fab-container">
            <button class="fab-button" onclick="document.querySelector('button[kind=secondary]').click()">+</button>
        </div>
    """, unsafe_allow_stdio=True)
    
    # 隱藏一個真正的 Streamlit 按鈕，由 FAB 觸發
    if st.button("+", key="real_add_btn", help="新增交易"):
        add_transaction_dialog()

if __name__ == "__main__":
    main()
