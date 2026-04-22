import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

st.set_page_config(page_title="Browne Portfolio Put Option Advisor", layout="wide")

# Auto-refresh every hour
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

# Check if 1 hour has passed
time_elapsed = time.time() - st.session_state.last_refresh_time
if time_elapsed >= REFRESH_INTERVAL:
    st.session_state.last_refresh_time = time.time()
    st.rerun()

# Enhanced Black-Scholes with Greeks
def black_scholes_put_with_greeks(S, K, T, r, sigma):
    """Calculate put price and Greeks (Delta, Gamma, Theta, Vega)"""
    if T <= 0:
        intrinsic = max(K - S, 0)
        return {
            'price': intrinsic,
            'delta': -1.0 if S < K else 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0
        }
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Put price
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    # Greeks
    delta = -norm.cdf(-d1)  # Put delta is negative
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
             - r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365  # Daily theta
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Vega per 1% change in IV
    
    return {
        'price': put_price,
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega
    }

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

def price_otm_put_with_greeks(S, K, T, r, vix):
    adjusted_iv = get_skewed_implied_vol(S, K, vix, T)
    return black_scholes_put_with_greeks(S, K, T, r, adjusted_iv)

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
    asset_ticker = selected_asset.split(" ")[0]
    asset_name = "S&P 500" if asset_ticker == "SPY" else "Gold"
    
    # Date range (1 month max)
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=30)
    
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
    if st.button("🔄 Force Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.session_state.last_refresh_time = time.time()
        st.rerun()
    
    # Show time until next auto-refresh
    time_until_refresh = REFRESH_INTERVAL - time_elapsed
    hours_left = int(time_until_refresh // 3600)
    minutes_left = int((time_until_refresh % 3600) // 60)
    st.caption(f"⏱️ Next auto-refresh in: {hours_left}h {minutes_left}m")

# Fetch data
@st.cache_data(ttl=60)
def fetch_live_price(ticker):
    """Fetch real-time last price using 1-minute intraday data."""
    try:
        hist = yf.Ticker(ticker).history(period="1d", interval="1m")
        if hist is None or hist.empty:
            return None, None
        last_price = float(hist['Close'].iloc[-1])
        last_time = hist.index[-1]
        return last_price, last_time
    except Exception as e:
        st.warning(f"Live price fetch failed for {ticker}: {e}")
        return None, None

@st.cache_data(ttl=REFRESH_INTERVAL)
def fetch_market_data(start, end, asset):
    try:
        asset_hist = yf.Ticker(asset).history(start=(start), end=(end), interval="1d")
        vix_hist = yf.Ticker('^VIX').history(start=(start), end=(end), interval="1d")

        if asset_hist.empty or vix_hist.empty:
            st.error("No data returned. Market may be closed or ticker invalid.")
            return None

        asset_hist.index = asset_hist.index.normalize()
        vix_hist.index = vix_hist.index.normalize()

        asset_series = asset_hist['Close'].squeeze()
        vix_series = vix_hist['Close'].squeeze()

        data = pd.DataFrame({asset: asset_series, 'VIX': vix_series})
        if data.empty:
            st.error("Data aligned but empty — check date range.")
            return None
        data = data.dropna()

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
        greeks = price_otm_put_with_greeks(S, strike, TIME_TO_EXPIRY, RISK_FREE_RATE, vix)
        
        adj_ivs.append(adj_iv * 100)
        strikes.append(strike)
        put_prices.append(greeks['price'])
    
    data['Adj_IV'] = adj_ivs
    data['Strike'] = strikes
    data['Put_Price'] = put_prices
    
    # Current market conditions
    live_price, live_time = fetch_live_price(asset_ticker)
    live_vix, _ = fetch_live_price('^VIX')
    latest_date = data.index[-1]
    latest_price = live_price if live_price else data[asset_ticker].iloc[-1]
    latest_vix = live_vix if live_vix else data['VIX'].iloc[-1]
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
        
        current_greeks = price_otm_put_with_greeks(latest_price, entry_strike, time_left, RISK_FREE_RATE, latest_vix)
        current_put_price = current_greeks['price']
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
            pos_greeks = price_otm_put_with_greeks(row[asset_ticker], entry_strike, time_remaining, RISK_FREE_RATE, row['VIX'])
            position_values.append(pos_greeks['price'])
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
    
    # Option Strategy Matrix Analysis
    st.markdown("---")
    st.header(f"🎲 Option Strategy Matrix - {asset_name} Put Options")
    st.markdown("*Compare different OTM depths and expiration dates to find optimal tail hedge*")
    
    # Define comparison parameters
    otm_levels = [0.15, 0.20, 0.25, 0.30]
    expiry_months = [3, 6, 9, 12]
    
    # Calculate matrix with Greeks
    matrix_data = []
    
    for otm in otm_levels:
        strike = latest_price * (1 - otm)
        
        for months in expiry_months:
            days = months * 30
            T = days / 365
            greeks = price_otm_put_with_greeks(latest_price, strike, T, RISK_FREE_RATE, latest_vix)
            adj_iv = get_skewed_implied_vol(latest_price, strike, latest_vix, T)
            
            # Calculate normalized metrics
            normalized_price = greeks['price'] / strike
            
            # Convexity metric: (price * gamma) / (strike * |theta| * vega)
            # Higher is better - want high gamma (convexity) relative to decay and vega
            if abs(greeks['theta']) > 0.001 and greeks['vega'] > 0.001:
                convexity_metric = (greeks['price'] * greeks['gamma']) / (strike * abs(greeks['theta']) * greeks['vega'])
            else:
                convexity_metric = 0
            
            matrix_data.append({
                'OTM': f"{otm*100:.0f}%",
                'Expiry': f"{months}M",
                'Strike': strike,
                'Price': greeks['price'],
                'Delta': greeks['delta'],
                'Gamma': greeks['gamma'],
                'Theta': greeks['theta'],
                'Vega': greeks['vega'],
                'Adj_IV': adj_iv * 100,
                'Normalized_Price': normalized_price,
                'Convexity_Metric': convexity_metric
            })
    
    matrix_df = pd.DataFrame(matrix_data)
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["📈 Heatmaps & Analysis", "💡 Recommendations"])
    
    with tab1:
        st.markdown("### 📊 Option Metrics Comparison")
        
        # Find cheapest options by different metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏆 Cheapest by Normalized Price (Price/Strike)")
            cheapest_norm = matrix_df.nsmallest(5, 'Normalized_Price')[['OTM', 'Expiry', 'Normalized_Price', 'Price', 'Strike']].copy()
            cheapest_norm['Normalized_Price'] = cheapest_norm['Normalized_Price'].apply(lambda x: f"{x:.4f}")
            cheapest_norm['Price'] = cheapest_norm['Price'].apply(lambda x: f"${x:.2f}")
            cheapest_norm['Strike'] = cheapest_norm['Strike'].apply(lambda x: f"${x:.2f}")
            st.table(cheapest_norm)
            st.caption("Lower is cheaper per dollar of strike protection")
        
        with col2:
            st.markdown("#### 🎯 Best Convexity (Price×Gamma / Strike×|Theta|×Vega)")
            best_convexity = matrix_df.nlargest(5, 'Convexity_Metric')[['OTM', 'Expiry', 'Convexity_Metric', 'Price', 'Gamma']].copy()
            best_convexity['Convexity_Metric'] = best_convexity['Convexity_Metric'].apply(lambda x: f"{x:.6f}")
            best_convexity['Price'] = best_convexity['Price'].apply(lambda x: f"${x:.2f}")
            best_convexity['Gamma'] = best_convexity['Gamma'].apply(lambda x: f"{x:.4f}")
            st.table(best_convexity)
            st.caption("Higher is better - max convexity per unit of time decay & vega risk")
        
        st.markdown("---")
        st.markdown("### 🔥 Complete Greeks Analysis")
        
        # Full table with all metrics
        display_matrix = matrix_df.copy()
        display_matrix['Strike'] = display_matrix['Strike'].apply(lambda x: f"${x:.2f}")
        display_matrix['Price'] = display_matrix['Price'].apply(lambda x: f"${x:.2f}")
        display_matrix['Delta'] = display_matrix['Delta'].apply(lambda x: f"{x:.3f}")
        display_matrix['Gamma'] = display_matrix['Gamma'].apply(lambda x: f"{x:.4f}")
        display_matrix['Theta'] = display_matrix['Theta'].apply(lambda x: f"${x:.3f}")
        display_matrix['Vega'] = display_matrix['Vega'].apply(lambda x: f"${x:.3f}")
        display_matrix['Adj_IV'] = display_matrix['Adj_IV'].apply(lambda x: f"{x:.1f}%")
        display_matrix['Normalized_Price'] = display_matrix['Normalized_Price'].apply(lambda x: f"{x:.4f}")
        display_matrix['Convexity_Metric'] = display_matrix['Convexity_Metric'].apply(lambda x: f"{x:.6f}")
        
        st.table(display_matrix)
        
        st.markdown("---")
        st.markdown("### 📊 Visual Heatmaps")
        
        # Prepare data for heatmaps
        norm_price_matrix = []
        convexity_matrix = []
        gamma_matrix = []
        
        for otm in otm_levels:
            norm_row = []
            conv_row = []
            gamma_row = []
            
            for months in expiry_months:
                subset = matrix_df[(matrix_df['OTM'] == f"{otm*100:.0f}%") & (matrix_df['Expiry'] == f"{months}M")]
                if len(subset) > 0:
                    norm_row.append(subset.iloc[0]['Normalized_Price'])
                    conv_row.append(subset.iloc[0]['Convexity_Metric'])
                    gamma_row.append(subset.iloc[0]['Gamma'])
            
            norm_price_matrix.append(norm_row)
            convexity_matrix.append(conv_row)
            gamma_matrix.append(gamma_row)
        
        otm_labels = [f"{int(otm*100)}% OTM" for otm in otm_levels]
        expiry_labels = [f"{m} Months" for m in expiry_months]
        
        fig_heat = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Normalized Price (Lower=Better)', 'Convexity Metric (Higher=Better)', 'Gamma (Higher=Better)'),
            horizontal_spacing=0.12
        )
        
        # Normalized Price heatmap (lower is better - use reversed colorscale)
        fig_heat.add_trace(
            go.Heatmap(
                z=norm_price_matrix,
                x=expiry_labels,
                y=otm_labels,
                colorscale='RdYlGn_r',
                text=[[f'{val:.4f}' for val in row] for row in norm_price_matrix],
                texttemplate='%{text}',
                textfont={"size": 9},
                showscale=True,
                colorbar=dict(x=0.30)
            ),
            row=1, col=1
        )
        
        # Convexity metric heatmap (higher is better)
        fig_heat.add_trace(
            go.Heatmap(
                z=convexity_matrix,
                x=expiry_labels,
                y=otm_labels,
                colorscale='RdYlGn',
                text=[[f'{val:.5f}' for val in row] for row in convexity_matrix],
                texttemplate='%{text}',
                textfont={"size": 9},
                showscale=True,
                colorbar=dict(x=0.64)
            ),
            row=1, col=2
        )
        
        # Gamma heatmap (higher is better)
        fig_heat.add_trace(
            go.Heatmap(
                z=gamma_matrix,
                x=expiry_labels,
                y=otm_labels,
                colorscale='Blues',
                text=[[f'{val:.4f}' for val in row] for row in gamma_matrix],
                texttemplate='%{text}',
                textfont={"size": 9},
                showscale=True,
                colorbar=dict(x=0.98)
            ),
            row=1, col=3
        )
        
        fig_heat.update_layout(height=400)
        st.plotly_chart(fig_heat, use_container_width=True)
    
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
        - Focus on high Gamma (convexity) relative to Theta (decay)
        """)
        
        # Find the best convexity options
        deep_otm_options = matrix_df[matrix_df['OTM'].isin(['25%', '30%'])].nlargest(4, 'Convexity_Metric')
        
        taleb_display = deep_otm_options[['OTM', 'Expiry', 'Strike', 'Price', 'Gamma', 'Theta', 'Convexity_Metric']].copy()
        taleb_display['Strike'] = taleb_display['Strike'].apply(lambda x: f"${x:.2f}")
        taleb_display['Price'] = taleb_display['Price'].apply(lambda x: f"${x:.2f}")
        taleb_display['Gamma'] = taleb_display['Gamma'].apply(lambda x: f"{x:.4f}")
        taleb_display['Theta'] = taleb_display['Theta'].apply(lambda x: f"${x:.3f}/day")
        taleb_display['Convexity_Metric'] = taleb_display['Convexity_Metric'].apply(lambda x: f"{x:.6f}")
        
        st.table(taleb_display)
        
        # Balanced approach
        st.markdown("#### ⚖️ Balanced Approach")
        st.success("""
        **Moderate OTM with Regular Rolling:**
        - 20% OTM strikes balance cost and protection
        - 6-month expiry reduces roll frequency
        - More likely to profit in moderate corrections
        - Good for typical portfolio hedging
        """)
        
        balanced_options = matrix_df[(matrix_df['OTM'] == '20%') & (matrix_df['Expiry'].isin(['3M', '6M']))]
        balanced_display = balanced_options[['OTM', 'Expiry', 'Strike', 'Price', 'Normalized_Price', 'Convexity_Metric']].copy()
        balanced_display['Strike'] = balanced_display['Strike'].apply(lambda x: f"${x:.2f}")
        balanced_display['Price'] = balanced_display['Price'].apply(lambda x: f"${x:.2f}")
        balanced_display['Normalized_Price'] = balanced_display['Normalized_Price'].apply(lambda x: f"{x:.4f}")
        balanced_display['Convexity_Metric'] = balanced_display['Convexity_Metric'].apply(lambda x: f"{x:.6f}")
        
        st.table(balanced_display)
        
        st.markdown("""
        **Key Insights:**
        - **Normalized Price**: Lower values mean cheaper cost per dollar of strike protection
        - **Convexity Metric**: Higher values indicate better asymmetric payoff (high gamma) relative to time decay (theta) and volatility risk (vega)
        - **Gamma**: Measures convexity - how fast delta changes. Higher gamma = more explosive gains in crashes
        - **Theta**: Daily time decay. Negative values show how much you lose per day
        - **Vega**: Sensitivity to IV changes. Higher vega = more profit from volatility spikes
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

### Greeks Explanation
- **Delta**: Rate of change of option price with underlying price (-1 to 0 for puts)
- **Gamma**: Rate of change of delta (convexity - higher = more explosive gains)
- **Theta**: Daily time decay (cost per day of holding the option)
- **Vega**: Sensitivity to 1% change in implied volatility

*This is for educational purposes only. Not financial advice.*
""")
