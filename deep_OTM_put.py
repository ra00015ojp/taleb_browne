import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

@st.cache_data(ttl=60)
def fetch_live_price(ticker):
    """Fetch real-time last price using 1-minute intraday data."""
    try:
        hist = yf.Ticker(ticker).history(period="1d", interval="1m")
        if hist is None or hist.empty:
            return None, None
        # history() always returns single-level columns — safe on all versions
        last_price = float(hist['Close'].iloc[-1])
        last_time  = hist.index[-1]
        return last_price, last_time
    except Exception as e:
        st.warning(f"Live price fetch failed for {ticker}: {e}")
        return None, None


@st.cache_data(ttl=300)
def fetch_market_data(start, end, asset):
    try:
        # Use Ticker.history() instead of yf.download() — avoids ALL MultiIndex issues
        asset_hist = yf.Ticker(asset).history(start=str(start), end=str(end), interval="1d")
        vix_hist   = yf.Ticker('^VIX').history(start=str(start), end=str(end), interval="1d")

        if asset_hist.empty or vix_hist.empty:
            st.error("No data returned. Market may be closed or ticker invalid.")
            return None

        # .history() returns timezone-aware index — normalize to date only for alignment
        asset_hist.index = asset_hist.index.normalize()
        vix_hist.index   = vix_hist.index.normalize()

        asset_series = asset_hist['Close'].squeeze()
        vix_series   = vix_hist['Close'].squeeze()

        data = pd.DataFrame({asset: asset_series, 'VIX': vix_series}).dropna()

        if data.empty:
            st.error("Data aligned but empty — check date range.")
            return None

        return data

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

st.set_page_config(page_title="Browne Portfolio Put Option Advisor", layout="wide")

# Auto-refresh every 3 hours (10800 seconds)
REFRESH_INTERVAL = 3600

# Get current time for display
current_time = datetime.datetime.now()
last_refresh = current_time.strftime("%B %d, %Y at %I:%M %p %Z")

# Display refresh info at the very top
st.markdown(f"""
<div style="background-color: #1f77b4; padding: 12px; border-radius: 8px; margin-bottom: 20px; border: 2px solid #0d47a1;">
    <p style="margin: 0; text-align: center; color: white; font-size: 16px; font-weight: 500;">
        🕐 <b>Last Data Refresh:</b> {last_refresh} | 
        <b>Auto-refresh:</b> Every hour | 
        <b>Market Data:</b> Real-time from Yahoo Finance
    </p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh mechanism
if 'last_refresh_time' not in st.session_state:
    st.session_state.last_refresh_time = time.time()

# Check if 3 hours have passed
time_elapsed = time.time() - st.session_state.last_refresh_time
if time_elapsed >= REFRESH_INTERVAL:
    st.session_state.last_refresh_time = time.time()
    st.rerun()

# Black-Scholes for Put Option Pricing & greeks
def black_scholes_put(S, K, T, r, sigma):
    if T <= 0:
        return max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def black_scholes_greeks(S, K, T, r, sigma):
    if T <= 0:
        return 0, 0, 0, 0
        
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    delta = -norm.cdf(-d1)
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
    theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(-d2))/365
    vega = S*norm.pdf(d1)*np.sqrt(T)/100
    
    return delta, gamma, theta, vega

def get_skewed_implied_vol(S, K, vix, T):
    base_iv = vix / 100
    moneyness = K / S
    
    if vix < 15:
        skew_slope = 2.5
    elif vix < 25:
        skew_slope = 3.0
    elif vix < 40:
        skew_slope = 3.5
    else:
        skew_slope = 4.0
    
    otm_percent = 1 - moneyness
    skew_multiplier = 1 + (skew_slope * otm_percent)
    time_adjustment = 1 + (0.3 * (1 - min(T * 365 / 180, 1)))
    adjusted_iv = base_iv * skew_multiplier * time_adjustment
    
    min_iv = max(0.15, base_iv * 0.8)
    adjusted_iv = max(adjusted_iv, min_iv)
    
    return adjusted_iv

def price_otm_put(S, K, T, r, vix):
    adjusted_iv = get_skewed_implied_vol(S, K, vix, T)
    return black_scholes_put(S, K, T, r, adjusted_iv)

# Strategy parameters
OTM_PERCENT = 0.20
TIME_TO_EXPIRY_DAYS = 180
TIME_TO_EXPIRY = TIME_TO_EXPIRY_DAYS / 365
RISK_FREE_RATE = 0.02
IV_BUY_THRESHOLD_NORMAL = 0.2
IV_BUY_THRESHOLD_RELAXED = 0.4
IV_SELL_THRESHOLD = 0.6
DAYS_AFTER_EXPIRY_RELAXED = 7

# Title and description
st.title("📊 Browne Portfolio Put Option Advisor")
st.markdown("### Tail Risk Hedging Strategy Recommendation System")
st.markdown("---")

# Sidebar for user inputs
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Asset Selection
    st.markdown("### 📊 Select Asset")
    selected_asset = st.radio(
        "Choose asset for put option analysis:",
        options=["SPY (S&P 500)", "GLD (Gold)"],
        index=0,
        help="SPY for equity protection, GLD for inflation/crisis hedge"
    )
    
    # Parse selection
    asset_ticker = selected_asset.split(" ")[0]  # Get "SPY" or "GLD"
    asset_name = "S&P 500" if asset_ticker == "SPY" else "Gold"
    
    # Date range (1 month max)
    end_date = datetime.date.today() + datetime.timedelta(days=1)   # make end exclusive → includes today
    start_date = end_date - datetime.timedelta(days=31)
    
    st.info(f"📅 Analysis Period: {start_date} to {end_date}")
    
    st.markdown("---")
    
    # Do you currently have a put option?
    has_position = st.radio(
        "Do you currently have a put option position?",
        options=["No", "Yes"],
        index=0
    )
    
    # If yes, ask for entry details
    entry_date = None
    entry_price = None
    entry_strike = None
    entry_spy_price = None
    
    if has_position == "Yes":
        st.markdown("#### Position Details")
        entry_date = st.date_input(
            "Entry Date",
            value=end_date - datetime.timedelta(days=14),
            max_value=end_date
        )
        entry_spy_price = st.number_input(
            f"{asset_ticker} Price at Entry ($)",
            min_value=10.0 if asset_ticker == "GLD" else 100.0,
            max_value=1000.0,
            value=220.0 if asset_ticker == "GLD" else 580.0,
            step=1.0
        )
        entry_strike = st.number_input(
            "Strike Price ($)",
            min_value=10.0 if asset_ticker == "GLD" else 100.0,
            max_value=1000.0,
            value=176.0 if asset_ticker == "GLD" else 464.0,
            step=1.0
        )
        entry_price = st.number_input(
            "Entry Put Price ($)",
            min_value=0.01,
            max_value=100.0,
            value=5.0,
            step=0.1
        )
    
    st.markdown("---")
    st.markdown("#### Strategy Parameters")
    st.metric("Buy Threshold (Normal)", f"{IV_BUY_THRESHOLD_NORMAL*100:.0f}%")
    st.metric("Buy Threshold (Relaxed)", f"{IV_BUY_THRESHOLD_RELAXED*100:.0f}%")
    st.metric("Sell Threshold", f"{IV_SELL_THRESHOLD*100:.0f}%")
    st.metric("OTM Percentage", f"{OTM_PERCENT*100:.0f}%")
    st.metric("Days to Expiry", f"{TIME_TO_EXPIRY_DAYS}")
    
    st.markdown("---")
    
    # Manual refresh button
    if st.button("🔄 Force Refresh Data", width='stretch'):
        st.cache_data.clear()
        st.session_state.last_refresh_time = time.time()
        st.rerun()
    
    # Show time until next auto-refresh
    time_until_refresh = REFRESH_INTERVAL - time_elapsed
    hours_left = int(time_until_refresh // 3600)
    minutes_left = int((time_until_refresh % 3600) // 60)
    st.caption(f"⏱️ Next auto-refresh in: {hours_left}h {minutes_left}m")

# Fetch data
@st.cache_data(ttl=REFRESH_INTERVAL)  # Cache for 3 hours
def fetch_market_data(start, end, asset):
    try:
        asset_data = yf.download(asset, start=start, end=end, auto_adjust=False, progress=False)['Adj Close']
        vix = yf.download('^VIX', start=start, end=end, auto_adjust=False, progress=False)['Close']
        
        if isinstance(asset_data, pd.DataFrame): asset_data = asset_data.squeeze()
        if isinstance(vix, pd.DataFrame): vix = vix.squeeze()
        
        data = pd.DataFrame({asset: asset_data, 'VIX': vix}).dropna()
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Main content
with st.spinner(f"Fetching {asset_name} market data..."):
    data = fetch_market_data(start_date, end_date, asset_ticker)

if data is not None and len(data) > 0:
    # Calculate adjusted IV for each day
    adj_ivs = []
    strikes = []
    put_prices = []
    
    for idx, row in data.iterrows():
        S = row[asset_ticker]
        vix = row['VIX']
        strike = S * (1 - OTM_PERCENT)
        adj_iv = get_skewed_implied_vol(S, strike, vix, TIME_TO_EXPIRY)
        put_price = price_otm_put(S, strike, TIME_TO_EXPIRY, RISK_FREE_RATE, vix)
        
        adj_ivs.append(adj_iv * 100)
        strikes.append(strike)
        put_prices.append(put_price)
        adj_iv_val = get_skewed_implied_vol(S, strike, vix, TIME_TO_EXPIRY)
        delta, gamma, theta, vega = black_scholes_greeks(S, strike, TIME_TO_EXPIRY, RISK_FREE_RATE, adj_iv_val)
        antifragile = (gamma * abs(delta)) / (put_price * abs(theta) * vega) if (put_price * theta * vega) != 0 else 0
    
    data['Adj_IV'] = adj_ivs
    data['Strike'] = strikes
    data['Put_Price'] = put_prices
    data['Antifragile'] = antifragile
    
    # Current market conditions
    live_price, live_time = fetch_live_price(asset_ticker)
    live_vix,   _         = fetch_live_price('^VIX')
    latest_date  = data.index[-1]
    latest_price = live_price  if live_price else data[asset_ticker].iloc[-1]
    latest_vix   = live_vix    if live_vix   else data['VIX'].iloc[-1]
    if live_time:
        st.caption(f"⚡ Live price as of {live_time.strftime('%I:%M %p ET')}")
    latest_adj_iv = data['Adj_IV'].iloc[-1]
    latest_strike = data['Strike'].iloc[-1]
    latest_put_price = data['Put_Price'].iloc[-1]
    
    # Display current market conditions
    st.header(f"📈 Current Market Conditions - {asset_name}")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(f"{asset_ticker} Price", f"${latest_price:.2f}")
    with col2:
        st.metric("VIX", f"{latest_vix:.2f}")
    with col3:
        st.metric("Adjusted IV", f"{latest_adj_iv:.1f}%")
    with col4:
        st.metric("Strike (20% OTM)", f"${latest_strike:.2f}")
    with col5:
        st.metric("Put Price", f"${latest_put_price:.2f}")
    
    st.markdown("---")
    
    # Recommendation logic
    if has_position == "No":
        st.header("🎯 BUY RECOMMENDATION")
        
        # Find buy opportunities
        buy_normal = data[data['Adj_IV'] <= IV_BUY_THRESHOLD_NORMAL * 100]
        buy_relaxed = data[data['Adj_IV'] <= IV_BUY_THRESHOLD_RELAXED * 100]
        
        # Check if we should buy now
        should_buy_normal = latest_adj_iv <= IV_BUY_THRESHOLD_NORMAL * 100
        should_buy_relaxed = latest_adj_iv <= IV_BUY_THRESHOLD_RELAXED * 100
        
        if should_buy_normal:
            st.success("✅ **BUY NOW** - Adjusted IV is below 20% threshold!")
            st.markdown(f"""
            ### Recommended Action
            - **Action**: Buy {asset_ticker} Put Options
            - **Strike**: ${latest_strike:.2f} (20% OTM)
            - **Expiry**: {TIME_TO_EXPIRY_DAYS} days
            - **Estimated Cost**: ${latest_put_price:.2f} per contract
            - **Current Adj IV**: {latest_adj_iv:.1f}% (Target: ≤20%)
            - **Reason**: Normal buy threshold met
            """)
        elif should_buy_relaxed:
            st.warning("⚠️ **CONSIDER BUYING** - Adjusted IV is below 40% relaxed threshold")
            st.markdown(f"""
            ### Recommended Action
            - **Action**: Consider buying if 7+ days since last position
            - **Strike**: ${latest_strike:.2f} (20% OTM)
            - **Expiry**: {TIME_TO_EXPIRY_DAYS} days
            - **Estimated Cost**: ${latest_put_price:.2f} per contract
            - **Current Adj IV**: {latest_adj_iv:.1f}% (Target: ≤40% after 7 days)
            - **Reason**: Relaxed buy threshold met
            """)
        else:
            st.info("⏳ **WAIT** - Adjusted IV is too high")
            st.markdown(f"""
            ### Current Status
            - **Current Adj IV**: {latest_adj_iv:.1f}%
            - **Target (Normal)**: ≤{IV_BUY_THRESHOLD_NORMAL*100:.0f}%
            - **Target (Relaxed)**: ≤{IV_BUY_THRESHOLD_RELAXED*100:.0f}% (after 7 days)
            - **Recommendation**: Wait for lower volatility before entering position
            """)
        
        # Show historical buy opportunities
        st.markdown("### 📅 Recent Buy Opportunities (Last 30 Days)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Normal Buy Signals (IV ≤ 20%)")
            if len(buy_normal) > 0:
                for idx, row in buy_normal.tail(5).iterrows():
                    st.write(f"- {idx.date()}: IV={row['Adj_IV']:.1f}%, {asset_ticker}=${row[asset_ticker]:.2f}, Put=${row['Put_Price']:.2f}")
            else:
                st.write("No opportunities in the last 30 days")
        
        with col2:
            st.markdown("#### Relaxed Buy Signals (IV ≤ 40%)")
            if len(buy_relaxed) > 0:
                for idx, row in buy_relaxed.tail(5).iterrows():
                    st.write(f"- {idx.date()}: IV={row['Adj_IV']:.1f}%, {asset_ticker}=${row[asset_ticker]:.2f}, Put=${row['Put_Price']:.2f}")
            else:
                st.write("No opportunities in the last 30 days")
    
    else:  # Has position
        st.header("💰 SELL RECOMMENDATION")
        
        # Calculate current position value
        days_held = (latest_date.date() - entry_date).days
        time_left = max((TIME_TO_EXPIRY_DAYS - days_held) / 365, 0.001)
        
        current_put_price = price_otm_put(latest_price, entry_strike, time_left, RISK_FREE_RATE, latest_vix)
        current_adj_iv = get_skewed_implied_vol(latest_price, entry_strike, latest_vix, time_left)
        
        profit_loss = current_put_price - entry_price
        profit_loss_pct = (profit_loss / entry_price) * 100
        
        # Display position summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Days Held", f"{days_held}")
        with col2:
            st.metric("Entry Price", f"${entry_price:.2f}")
        with col3:
            st.metric("Current Price", f"${current_put_price:.2f}", f"{profit_loss:+.2f}")
        with col4:
            st.metric("P&L %", f"{profit_loss_pct:+.1f}%")
        
        # Check if we should sell
        should_sell = latest_adj_iv >= IV_SELL_THRESHOLD * 100
        
        if should_sell:
            st.success("✅ **SELL NOW** - Adjusted IV is above 60% threshold!")
            st.markdown(f"""
            ### Recommended Action
            - **Action**: SELL your put options
            - **Current Strike**: ${entry_strike:.2f}
            - **Current Put Price**: ${current_put_price:.2f}
            - **Entry Put Price**: ${entry_price:.2f}
            - **Profit/Loss**: ${profit_loss:+.2f} ({profit_loss_pct:+.1f}%)
            - **Current Adj IV**: {latest_adj_iv:.1f}% (Target: ≥60%)
            - **Days Held**: {days_held} days
            - **Reason**: Sell threshold met - volatility spike detected
            """)
        else:
            st.info("⏳ **HOLD** - Adjusted IV has not reached sell threshold yet")
            st.markdown(f"""
            ### Current Position Status
            - **Current Adj IV**: {latest_adj_iv:.1f}%
            - **Sell Target**: ≥{IV_SELL_THRESHOLD*100:.0f}%
            - **Current P&L**: ${profit_loss:+.2f} ({profit_loss_pct:+.1f}%)
            - **Days Held**: {days_held} / {TIME_TO_EXPIRY_DAYS}
            - **Recommendation**: Hold and wait for volatility spike
            """)
        
        # Show position performance
        st.markdown("### 📊 Position Performance")
        
        # Calculate historical values for this position
        position_values = []
        position_dates = []
        
        for idx, row in data[data.index >= pd.Timestamp(entry_date)].iterrows():
            days_from_entry = (idx.date() - entry_date).days
            time_remaining = max((TIME_TO_EXPIRY_DAYS - days_from_entry) / 365, 0.001)
            pos_price = price_otm_put(row[asset_ticker], entry_strike, time_remaining, RISK_FREE_RATE, row['VIX'])
            position_values.append(pos_price)
            position_dates.append(idx)
        
        if len(position_values) > 0:
            fig_pos = go.Figure()
            fig_pos.add_trace(go.Scatter(
                x=position_dates,
                y=position_values,
                mode='lines',
                name='Put Value',
                line=dict(color='blue', width=2)
            ))
            fig_pos.add_hline(y=entry_price, line_dash="dash", line_color="gray", 
                            annotation_text="Entry Price")
            fig_pos.update_layout(
                title="Put Option Value Since Entry",
                xaxis_title="Date",
                yaxis_title="Put Price ($)",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig_pos, use_container_width=True)
    
    # Visualization
    # Option Strategy Matrix Analysis
    st.markdown("---")
    st.header(f"🎲 Option Strategy Matrix - {asset_name} Put Options")
    st.markdown("*Compare different OTM depths and expiration dates to find optimal tail hedge*")
    
    # Define comparison parameters
    otm_levels = [0.15, 0.20, 0.25, 0.30]  # 15%, 20%, 25%, 30% OTM
    expiry_months = [3, 6, 9, 12]  # months
    
    # Calculate matrix
    matrix_data = []
    
    for otm in otm_levels:
        row_data = {'OTM %': f"{otm*100:.0f}%"}
        strike = latest_price * (1 - otm)
        
        for months in expiry_months:
            days = months * 30
            T = days / 365
            put_price = price_otm_put(latest_price, strike, T, RISK_FREE_RATE, latest_vix)
            adj_iv = get_skewed_implied_vol(latest_price, strike, latest_vix, T)
            
            adj_iv_val = get_skewed_implied_vol(latest_price, strike, latest_vix, T)
            d, g, t, v = black_scholes_greeks(latest_price, strike, T, RISK_FREE_RATE, adj_iv_val)
            row_data[f'{months}M'] = (g * abs(d)) / (put_price * abs(t) * v) if (put_price * t * v) != 0 else 0
            row_data[f'{months}M_IV'] = adj_iv
            row_data[f'{months}M_Strike'] = strike
        
        matrix_data.append(row_data)
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["📈 Heatmaps", "💡 Recommendations"])
    
    with tab1:
        st.markdown("### Visual Comparison Heatmaps")
        
        # Prepare data for heatmaps
        price_matrix = []
        iv_matrix = []
        annual_cost_matrix = []
        
        for otm in otm_levels:
            price_row = []
            iv_row = []
            antifragile_row = []
            strike = latest_price * (1 - otm)
            
            for months in expiry_months:
                days = months * 30
                T = days / 365
                put_price = price_otm_put(latest_price, strike, T, RISK_FREE_RATE, latest_vix)
                adj_iv = get_skewed_implied_vol(latest_price, strike, latest_vix, T)
                adj_iv_val = get_skewed_implied_vol(latest_price, strike, latest_vix, T)
                d, g, t, v = black_scholes_greeks(latest_price, strike, T, RISK_FREE_RATE, adj_iv_val)
                antifragile_val = (g * abs(d)) / (put_price * abs(t) * v) if (put_price * t * v) != 0 else 0
                
                price_row.append(put_price)
                iv_row.append(adj_iv * 100)
                antifragile_row.append(antifragile_val)
            
            price_matrix.append(price_row)
            iv_matrix.append(iv_row)
            annual_cost_matrix.append(antifragile_row)
        
        # Create heatmaps
        otm_labels = [f"{int(otm*100)}% OTM" for otm in otm_levels]
        expiry_labels = [f"{m} Months" for m in expiry_months]
        
        fig_heat = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Antifragile Value', 'Adjusted IV (%)', 'Antifragile Heatmap'),
            horizontal_spacing=0.15
        )
        
        # Price heatmap
        fig_heat.add_trace(
            go.Heatmap(
                z=price_matrix,
                x=expiry_labels,
                y=otm_labels,
                colorscale='Greens',
                text=[[f'${val:.4f}' for val in row] for row in price_matrix],
                texttemplate='%{text}',
                textfont={"size": 10},
                showscale=True,
                colorbar=dict(x=0.28)
            ), 
            row=1, 
            col=1
        )
        
        # IV heatmap
        fig_heat.add_trace(
            go.Heatmap(
                z=iv_matrix,
                x=expiry_labels,
                y=otm_labels,
                colorscale='Blues',
                text=[[f'{val:.1f}%' for val in row] for row in iv_matrix],
                texttemplate='%{text}',
                textfont={"size": 10},
                showscale=True,
                colorbar=dict(x=0.63)
            ),
            row=1, col=2
        )
        
        # Annual cost heatmap
        fig_heat.add_trace(
            go.Heatmap(
                z=annual_cost_matrix,
                x=expiry_labels,
                y=otm_labels,
                colorscale='Reds',
                text=[[f'${val:.4f}' for val in row] for row in annual_cost_matrix],
                texttemplate='%{text}',
                textfont={"size": 10},
                showscale=True,
                colorbar=dict(x=0.98)
            ),
            row=1, col=3
        )
        
        fig_heat.update_layout(height=400)
        st.plotly_chart(fig_heat, width='stretch')
        
        # 3D surface plot
        st.markdown("### 3D Price Surface")
        
        fig_3d = go.Figure(data=[go.Surface(
            z=price_matrix,
            x=expiry_months,
            y=[otm*100 for otm in otm_levels],
            colorscale='Viridis',
            showscale=True
        )])
        
        fig_3d.update_layout(
            title='Put Option Prices: OTM % vs Expiry',
            scene=dict(
                xaxis_title='Months to Expiry',
                yaxis_title='OTM %',
                zaxis_title='Antifragile Value',
            ),
            height=600
        )
        
        st.plotly_chart(fig_3d, width='stretch')
    
    with tab2:
        st.markdown("### 💡 Strategy Recommendations")
        
        # Taleb/Universa style recommendation
        st.markdown("#### 🎯 Tail Risk Hedge (Taleb/Universa Style)")
        st.info("""
        **Deep OTM Puts for Convexity:**
        - Universa typically uses 25-30% OTM puts for maximum convexity
        - Cheaper upfront cost allows for more contracts
        - Massive asymmetric payoff during tail events
        - Accept high theta decay for extreme downside protection
        """)
        
        # Find the cheapest deep OTM option
        deep_otm = [0.25, 0.30]
        taleb_options = []
        
        for otm in deep_otm:
            strike = latest_price * (1 - otm)
            for months in [3, 6]:
                T = months * 30 / 365
                put_price = price_otm_put(latest_price, strike, T, RISK_FREE_RATE, latest_vix)
                annual_cost = put_price * (12 / months)
                adj_iv_val = get_skewed_implied_vol(latest_price, strike, latest_vix, T)
                d, g, t, v = black_scholes_greeks(latest_price, strike, T, RISK_FREE_RATE, adj_iv_val)
                af = (g * abs(d)) / (put_price * abs(t) * v) if (put_price * t * v) != 0 else 0
                
                taleb_options.append({
                    'OTM': f"{otm*100:.0f}%",
                    'Expiry': f"{months}M",
                    'Strike': f"${strike:.2f}",
                    'Antifragile': f"{af:.4f}",
                    'Annual Cost (Rolling)': f"${annual_cost:.2f}"
                })
        
        #st.plotly_chart(fig, width='stretch')
        
        # Balanced approach
        st.markdown("#### ⚖️ Balanced Approach")
        st.success("""
        **Moderate OTM with Regular Rolling:**
        - 20% OTM strikes balance cost and protection
        - 6-month expiry reduces roll frequency
        - More likely to profit in moderate corrections
        - Good for typical portfolio hedging
        """)
        
        # Cost comparison
        st.markdown("#### 💵 Example: $800 Hedge Budget")
        
        budget = 800
        
        comparison = []
        
        # Strategy 1: Deep OTM, short dated
        strike_30 = latest_price * 0.70
        price_30_3m = price_otm_put(latest_price, strike_30, 0.25, RISK_FREE_RATE, latest_vix)
        contracts_30 = budget / (price_30_3m * 100)
        comparison.append({
            'Strategy': '30% OTM, 3M (Taleb Style)',
            'Strike': f"${strike_30:.2f}",
            'Price per Contract': f"${price_30_3m:.2f}",
            'Contracts Affordable': f"{contracts_30:.1f}",
            'Total Notional': f"${strike_30 * contracts_30 * 100:,.0f}",
            'Roll Frequency': '4x per year'
        })
        
        # Strategy 2: Moderate OTM, medium dated
        strike_20 = latest_price * 0.80
        price_20_6m = price_otm_put(latest_price, strike_20, 0.5, RISK_FREE_RATE, latest_vix)
        contracts_20 = budget / (price_20_6m * 100)
        comparison.append({
            'Strategy': '20% OTM, 6M (Balanced)',
            'Strike': f"${strike_20:.2f}",
            'Price per Contract': f"${price_20_6m:.2f}",
            'Contracts Affordable': f"{contracts_20:.1f}",
            'Total Notional': f"${strike_20 * contracts_20 * 100:,.0f}",
            'Roll Frequency': '2x per year'
        })
        
        # Strategy 3: Closer OTM, long dated
        strike_15 = latest_price * 0.85
        price_15_12m = price_otm_put(latest_price, strike_15, 1.0, RISK_FREE_RATE, latest_vix)
        contracts_15 = budget / (price_15_12m * 100)
        comparison.append({
            'Strategy': '15% OTM, 12M (Conservative)',
            'Strike': f"${strike_15:.2f}",
            'Price per Contract': f"${price_15_12m:.2f}",
            'Contracts Affordable': f"{contracts_15:.1f}",
            'Total Notional': f"${strike_15 * contracts_15 * 100:,.0f}",
            'Roll Frequency': '1x per year'
        })
        
        st.table(pd.DataFrame(comparison))
        
        st.markdown("""
        **Key Insights:**
        - **Deeper OTM** = More contracts for same budget, but further from current price
        - **Shorter expiry** = Cheaper per contract, but more frequent rolling
        - **Taleb's approach**: Maximize convexity with deep OTM, accept higher roll costs
        - **Traditional approach**: Balance between cost, protection level, and roll frequency
        """)
    
    st.markdown("---")
    st.header(f"📉 Market Analysis - {asset_name} Last 30 Days")
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(f'{asset_ticker} Price', 'VIX Index', 'Adjusted Implied Volatility'),
        vertical_spacing=0.1,
        row_heights=[0.33, 0.33, 0.34]
    )
    
    # Asset Price
    fig.add_trace(
        go.Scatter(x=data.index, y=data[asset_ticker], name=asset_ticker, 
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    # VIX
    fig.add_trace(
        go.Scatter(x=data.index, y=data['VIX'], name='VIX',
                  line=dict(color='orange', width=2)),
        row=2, col=1
    )
    
    # Adjusted IV with thresholds
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Adj_IV'], name='Adj IV',
                  line=dict(color='purple', width=2)),
        row=3, col=1
    )
    
    # Add threshold lines
    fig.add_hline(y=IV_BUY_THRESHOLD_NORMAL*100, line_dash="dash", line_color="green",
                 annotation_text="Buy (20%)", row=3, col=1)
    fig.add_hline(y=IV_BUY_THRESHOLD_RELAXED*100, line_dash="dash", line_color="lightgreen",
                 annotation_text="Buy Relaxed (40%)", row=3, col=1)
    fig.add_hline(y=IV_SELL_THRESHOLD*100, line_dash="dash", line_color="red",
                 annotation_text="Sell (60%)", row=3, col=1)
    
    # Update layout
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="VIX", row=2, col=1)
    fig.update_yaxes(title_text="Adj IV (%)", row=3, col=1)
    
    fig.update_layout(height=900, showlegend=True, hovermode='x unified')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    with st.expander("📋 View Detailed Data"):
        display_data = data[[asset_ticker, 'VIX', 'Adj_IV', 'Strike', 'Put_Price']].copy()
        display_data.columns = [f'{asset_ticker} Price', 'VIX', 'Adj IV (%)', 'Strike Price', 'Put Price']
        st.table(display_data.tail(20).sort_index(ascending=False))

else:
    st.error("Unable to fetch market data. Please try again later.")

# Footer
st.markdown("---")
st.markdown("""
### Strategy Overview
- **Normal Buy**: Adjusted IV ≤ 20% (always)
- **Relaxed Buy**: Adjusted IV ≤ 40% (only if 7+ days since last position expired)
- **Sell**: Adjusted IV ≥ 60% (volatility spike)
- **Strike**: 20% OTM
- **Expiry**: 180 days

*This is for educational purposes only. Not financial advice.*
""")
