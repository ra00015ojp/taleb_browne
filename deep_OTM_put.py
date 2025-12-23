import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Browne Portfolio Put Option Advisor", layout="wide")

# Black-Scholes for Put Option Pricing
def black_scholes_put(S, K, T, r, sigma):
    if T <= 0:
        return max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

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
st.title("📊 Browne Portfolio with an Hedge")
st.markdown("### Tail Risk Hedging Strategy Recommendation System (Educational Only)")
st.markdown("---")

# Sidebar for user inputs
with st.sidebar:
    st.header("⚙️ Configuration")
    
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
            "SPY Price at Entry ($)",
            min_value=100.0,
            max_value=1000.0,
            value=580.0,
            step=1.0
        )
        entry_strike = st.number_input(
            "Strike Price ($)",
            min_value=100.0,
            max_value=1000.0,
            value=464.0,
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

# Fetch data
@st.cache_data(ttl=3600)
def fetch_market_data(start, end):
    try:
        spy = yf.download('SPY', start=start, end=end, auto_adjust=False, progress=False)['Adj Close']
        vix = yf.download('^VIX', start=start, end=end, auto_adjust=False, progress=False)['Close']
        
        if isinstance(spy, pd.DataFrame): spy = spy.squeeze()
        if isinstance(vix, pd.DataFrame): vix = vix.squeeze()
        
        data = pd.DataFrame({'SPY': spy, 'VIX': vix}).dropna()
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Main content
with st.spinner("Fetching market data..."):
    data = fetch_market_data(start_date, end_date)

if data is not None and len(data) > 0:
    # Calculate adjusted IV for each day
    adj_ivs = []
    strikes = []
    put_prices = []
    
    for idx, row in data.iterrows():
        S = row['SPY']
        vix = row['VIX']
        strike = S * (1 - OTM_PERCENT)
        adj_iv = get_skewed_implied_vol(S, strike, vix, TIME_TO_EXPIRY)
        put_price = price_otm_put(S, strike, TIME_TO_EXPIRY, RISK_FREE_RATE, vix)
        
        adj_ivs.append(adj_iv * 100)
        strikes.append(strike)
        put_prices.append(put_price)
    
    data['Adj_IV'] = adj_ivs
    data['Strike'] = strikes
    data['Put_Price'] = put_prices
    
    # Current market conditions
    latest_date = data.index[-1]
    latest_spy = data['SPY'].iloc[-1]
    latest_vix = data['VIX'].iloc[-1]
    latest_adj_iv = data['Adj_IV'].iloc[-1]
    latest_strike = data['Strike'].iloc[-1]
    latest_put_price = data['Put_Price'].iloc[-1]
    
    # Display current market conditions
    st.header("📈 Current Market Conditions")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("SPY Price", f"${latest_spy:.2f}")
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
            - **Action**: Buy SPY Put Options
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
                    st.write(f"- {idx.date()}: IV={row['Adj_IV']:.1f}%, SPY=${row['SPY']:.2f}, Put=${row['Put_Price']:.2f}")
            else:
                st.write("No opportunities in the last 30 days")
        
        with col2:
            st.markdown("#### Relaxed Buy Signals (IV ≤ 40%)")
            if len(buy_relaxed) > 0:
                for idx, row in buy_relaxed.tail(5).iterrows():
                    st.write(f"- {idx.date()}: IV={row['Adj_IV']:.1f}%, SPY=${row['SPY']:.2f}, Put=${row['Put_Price']:.2f}")
            else:
                st.write("No opportunities in the last 30 days")
    
    else:  # Has position
        st.header("💰 SELL RECOMMENDATION")
        
        # Calculate current position value
        days_held = (latest_date.date() - entry_date).days
        time_left = max((TIME_TO_EXPIRY_DAYS - days_held) / 365, 0.001)
        
        current_put_price = price_otm_put(latest_spy, entry_strike, time_left, RISK_FREE_RATE, latest_vix)
        current_adj_iv = get_skewed_implied_vol(latest_spy, entry_strike, latest_vix, time_left)
        
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
            pos_price = price_otm_put(row['SPY'], entry_strike, time_remaining, RISK_FREE_RATE, row['VIX'])
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
    st.markdown("---")
    st.header("📉 Market Analysis - Last 30 Days")
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('SPY Price', 'VIX Index', 'Adjusted Implied Volatility'),
        vertical_spacing=0.1,
        row_heights=[0.33, 0.33, 0.34]
    )
    
    # SPY Price
    fig.add_trace(
        go.Scatter(x=data.index, y=data['SPY'], name='SPY', 
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
        display_data = data[['SPY', 'VIX', 'Adj_IV', 'Strike', 'Put_Price']].copy()
        display_data.columns = ['SPY Price', 'VIX', 'Adj IV (%)', 'Strike Price', 'Put Price']
        st.dataframe(display_data.tail(20).sort_index(ascending=False), use_container_width=True)

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
