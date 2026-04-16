import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
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
        last_price = float(hist['Close'].iloc[-1])
        last_time  = hist.index[-1]
        return last_price, last_time
    except Exception as e:
        st.warning(f"Live price fetch failed for {ticker}: {e}")
        return None, None


@st.cache_data(ttl=300)
def fetch_market_data(start, end, asset):
    try:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        today = pd.Timestamp('today').normalize()

        if start_ts >= end_ts:
            st.error("Invalid date range: start date must be before end date.")
            return None

        if end_ts > today + pd.Timedelta(days=1):
            end_ts = today + pd.Timedelta(days=1)

        asset_ticker = asset.upper()
        asset_hist = yf.Ticker(asset_ticker).history(start=start_ts, end=end_ts, interval="1d")
        vix_hist   = yf.Ticker('^VIX').history(start=start_ts, end=end_ts, interval="1d")

        if asset_hist is None or asset_hist.empty or vix_hist is None or vix_hist.empty:
            # Retry with a shorter recent window if the requested range returns no rows
            asset_hist = yf.Ticker(asset_ticker).history(period="60d", interval="1d")
            vix_hist   = yf.Ticker('^VIX').history(period="60d", interval="1d")

        if asset_hist is None or asset_hist.empty or vix_hist is None or vix_hist.empty:
            st.error("Unable to fetch market data for the selected ticker or date range.")
            return None

        asset_hist.index = asset_hist.index.normalize()
        vix_hist.index   = vix_hist.index.normalize()

        asset_series = asset_hist['Close'].squeeze()
        vix_series   = vix_hist['Close'].squeeze()

        data = pd.DataFrame({asset_ticker: asset_series, 'VIX': vix_series}).dropna()

        if data.empty:
            overlap_start = max(asset_series.index.min(), vix_series.index.min())
            overlap_end = min(asset_series.index.max(), vix_series.index.max())
            st.error(
                f"Data aligned but empty — no overlapping trading days between {overlap_start.date()} and {overlap_end.date()}."
            )
            return None

        return data

    except Exception as e:
        st.error("Unable to fetch market data. Please try again later.")
        st.error(str(e))
        return None


@st.cache_data(ttl=3600)
def fetch_option_chain(ticker, expiration_date=None):
    """Fetch real option chain data from yfinance."""
    try:
        t = yf.Ticker(ticker)
        
        # Get available expirations
        if not hasattr(t, 'options') or not t.options:
            st.warning(f"No option chains available for {ticker}")
            return None, None
        
        # If no expiration specified, use the first available
        if expiration_date is None:
            expiration_date = t.options[0]
        
        # Fetch the chain
        chain = t.option_chain(expiration_date)
        return chain.puts, t.options
    
    except Exception as e:
        st.warning(f"Failed to fetch option chain for {ticker}: {e}")
        return None, None


def filter_otm_puts(puts_df, current_price, otm_percent=0.20):
    """Filter puts that are OTM by the specified percent."""
    if puts_df is None or puts_df.empty:
        return None
    
    strike_threshold = current_price * (1 - otm_percent)
    otm_puts = puts_df[puts_df['strike'] <= strike_threshold].copy()
    
    return otm_puts if not otm_puts.empty else None


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

time_elapsed = time.time() - st.session_state.last_refresh_time
if time_elapsed >= REFRESH_INTERVAL:
    st.session_state.last_refresh_time = time.time()
    st.rerun()

# Strategy parameters
OTM_PERCENT = 0.20
TIME_TO_EXPIRY_DAYS = 180
RISK_FREE_RATE = 0.02
IV_BUY_THRESHOLD_NORMAL = 0.2
IV_BUY_THRESHOLD_RELAXED = 0.4
IV_SELL_THRESHOLD = 0.6

# Title and description
st.title("📊 Browne Portfolio Put Option Advisor")
st.markdown("### Tail Risk Hedging Strategy with Real Option Chains")
st.markdown("---")

# Sidebar for user inputs
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Asset Selection
    st.markdown("### 📊 Select Asset")
    selected_asset = st.radio(
        "Choose asset for put option analysis:",
        options=["SPY (S&P 500)", "GLD (Gold)", "FEZ (European Financials)"],
        index=0,
        help="SPY for equity protection, GLD for inflation/crisis hedge, FEZ for European exposure"
    )
    
    # Parse selection
    asset_ticker = selected_asset.split(" ")[0]
    asset_name = selected_asset.split("(")[1].rstrip(")")
    
    # Date range (1 month max)
    end_date = datetime.date.today() + datetime.timedelta(days=1)
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
    entry_asset_price = None
    
    if has_position == "Yes":
        st.markdown("#### Position Details")
        entry_date = st.date_input(
            "Entry Date",
            value=end_date - datetime.timedelta(days=14),
            max_value=end_date
        )
        entry_asset_price = st.number_input(
            f"{asset_ticker} Price at Entry ($)",
            min_value=5.0,
            max_value=1000.0,
            value=220.0 if asset_ticker == "GLD" else (100.0 if asset_ticker == "FEZ" else 580.0),
            step=1.0
        )
        entry_strike = st.number_input(
            "Strike Price ($)",
            min_value=5.0,
            max_value=1000.0,
            value=176.0 if asset_ticker == "GLD" else (80.0 if asset_ticker == "FEZ" else 464.0),
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
    st.metric("OTM Percentage", f"{OTM_PERCENT*100:.0f}%")
    st.metric("Days to Expiry", f"{TIME_TO_EXPIRY_DAYS}")
    st.metric("Buy Threshold (Normal)", f"{IV_BUY_THRESHOLD_NORMAL*100:.0f}%")
    st.metric("Buy Threshold (Relaxed)", f"{IV_BUY_THRESHOLD_RELAXED*100:.0f}%")
    st.metric("Sell Threshold", f"{IV_SELL_THRESHOLD*100:.0f}%")
    
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

# Main content
with st.spinner(f"Fetching {asset_name} market data..."):
    data = fetch_market_data(start_date, end_date, asset_ticker)

if data is not None and len(data) > 0:
    
    # Fetch option chain data
    with st.spinner(f"Fetching {asset_ticker} option chain..."):
        puts_chain, available_expirations = fetch_option_chain(asset_ticker)
    
    # Current market conditions
    live_price, live_time = fetch_live_price(asset_ticker)
    live_vix, _ = fetch_live_price('^VIX')
    latest_date = data.index[-1]
    latest_price = live_price if live_price else data[asset_ticker].iloc[-1]
    latest_vix = live_vix if live_vix else data['VIX'].iloc[-1]
    
    if live_time:
        st.caption(f"⚡ Live price as of {live_time.strftime('%I:%M %p ET')}")
    
    # Display current market conditions
    st.header(f"📈 Current Market Conditions - {asset_name}")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(f"{asset_ticker} Price", f"${latest_price:.2f}")
    with col2:
        st.metric("VIX", f"{latest_vix:.2f}")
    with col3:
        st.metric("Implied Vol (VIX/100)", f"{(latest_vix/100):.1%}")
    with col4:
        st.metric("Strike (20% OTM)", f"${latest_price * (1-OTM_PERCENT):.2f}")
    with col5:
        if puts_chain is not None and not puts_chain.empty:
            otm_puts = filter_otm_puts(puts_chain, latest_price, OTM_PERCENT)
            if otm_puts is not None:
                closest_put = otm_puts.iloc[-1]  # Closest to 20% OTM
                st.metric("Market Put Price", f"${closest_put['lastPrice']:.2f}")
            else:
                st.metric("Status", "No OTM puts")
        else:
            st.metric("Status", "Chain unavailable")
    
    st.markdown("---")
    
    # Recommendation logic
    if has_position == "No":
        st.header("🎯 POSITION RECOMMENDATION")
        
        if puts_chain is not None and not puts_chain.empty:
            otm_puts = filter_otm_puts(puts_chain, latest_price, OTM_PERCENT)
            
            if otm_puts is not None:
                # Get the closest strike to 20% OTM
                target_strike = latest_price * (1 - OTM_PERCENT)
                closest_idx = (otm_puts['strike'] - target_strike).abs().idxmin()
                closest_put = otm_puts.loc[closest_idx]
                
                current_iv = latest_vix / 100
                
                # Assess recommendation
                if current_iv <= IV_BUY_THRESHOLD_NORMAL:
                    st.success("✅ **BUY NOW** - Volatility is below 20% threshold!")
                    st.markdown(f"""
                    ### Recommended Action
                    - **Action**: Buy {asset_ticker} Put Options
                    - **Strike**: ${closest_put['strike']:.2f} (≈20% OTM)
                    - **Ask Price**: ${closest_put['ask']:.2f}
                    - **Bid Price**: ${closest_put['bid']:.2f}
                    - **Last Price**: ${closest_put['lastPrice']:.2f}
                    - **Expiry Date**: {closest_put['contractSymbol']}
                    - **Current IV (VIX/100)**: {current_iv:.1%}
                    - **Threshold**: ≤{IV_BUY_THRESHOLD_NORMAL*100:.0f}%
                    - **Reason**: Normal buy threshold met
                    """)
                
                elif current_iv <= IV_BUY_THRESHOLD_RELAXED:
                    st.warning("⚠️ **CONSIDER BUYING** - Volatility is elevated but below 40%")
                    st.markdown(f"""
                    ### Recommended Action
                    - **Action**: Consider buying if no recent position
                    - **Strike**: ${closest_put['strike']:.2f} (≈20% OTM)
                    - **Ask Price**: ${closest_put['ask']:.2f}
                    - **Last Price**: ${closest_put['lastPrice']:.2f}
                    - **Current IV (VIX/100)**: {current_iv:.1%}
                    - **Threshold**: ≤{IV_BUY_THRESHOLD_RELAXED*100:.0f}%
                    - **Reason**: Relaxed buy threshold met
                    """)
                
                else:
                    st.info("⏳ **WAIT** - Volatility is too high")
                    st.markdown(f"""
                    ### Current Status
                    - **Current IV (VIX/100)**: {current_iv:.1%}
                    - **Target (Normal)**: ≤{IV_BUY_THRESHOLD_NORMAL*100:.0f}%
                    - **Target (Relaxed)**: ≤{IV_BUY_THRESHOLD_RELAXED*100:.0f}%
                    - **Recommendation**: Wait for lower volatility before entering position
                    """)
                
                # Show available OTM puts
                st.markdown("### 📋 Available OTM Put Options")
                display_puts = otm_puts[['strike', 'bid', 'ask', 'lastPrice', 'impliedVolatility', 'openInterest']].copy()
                display_puts.columns = ['Strike', 'Bid', 'Ask', 'Last Price', 'Implied Vol', 'Open Interest']
                display_puts = display_puts.sort_values('Strike', ascending=False)
                st.table(display_puts.tail(10))
            else:
                st.warning("⚠️ No OTM puts available at 20% threshold")
        else:
            st.error("Unable to fetch option chain data. Please try again.")
    
    else:  # Has position
        st.header("💰 SELL RECOMMENDATION")
        
        current_iv = latest_vix / 100
        
        # Display position summary
        col1, col2, col3, col4 = st.columns(4)
        
        days_held = (latest_date.date() - entry_date).days
        
        with col1:
            st.metric("Days Held", f"{days_held}")
        with col2:
            st.metric("Entry Price", f"${entry_price:.2f}")
        with col3:
            st.metric("Current IV (VIX/100)", f"{current_iv:.1%}")
        with col4:
            st.metric("Current Volatility", f"{latest_vix:.2f} (VIX)")
        
        # Check if we should sell
        should_sell = current_iv >= IV_SELL_THRESHOLD
        
        if should_sell:
            st.success("✅ **SELL NOW** - Volatility spike detected (IV ≥ 60%)")
            st.markdown(f"""
            ### Recommended Action
            - **Action**: SELL your put options
            - **Current Strike**: ${entry_strike:.2f}
            - **Current IV (VIX/100)**: {current_iv:.1%}
            - **Entry IV (VIX/100)**: Estimated from entry date
            - **Days Held**: {days_held} days
            - **Reason**: Sell threshold met - volatility spike detected
            - **Target Volatility**: ≥{IV_SELL_THRESHOLD*100:.0f}%
            """)
        else:
            st.info("⏳ **HOLD** - Volatility has not reached sell threshold")
            st.markdown(f"""
            ### Current Position Status
            - **Current IV (VIX/100)**: {current_iv:.1%}
            - **Sell Target**: ≥{IV_SELL_THRESHOLD*100:.0f}%
            - **Days Held**: {days_held} / {TIME_TO_EXPIRY_DAYS}
            - **Current Volatility (VIX)**: {latest_vix:.2f}
            - **Recommendation**: Hold and wait for volatility spike
            """)
    
    # Visualization
    st.markdown("---")
    st.header(f"📉 Market Analysis - {asset_name} Last 30 Days")
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(f'{asset_ticker} Price', 'VIX Index', 'Historical Volatility'),
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
    
    # Calculate rolling volatility from asset returns
    rolling_vol = data[asset_ticker].pct_change().rolling(window=20).std() * np.sqrt(252) * 100
    fig.add_trace(
        go.Scatter(x=data.index, y=rolling_vol, name='20-Day Vol',
                  line=dict(color='purple', width=2)),
        row=3, col=1
    )
    
    # Add threshold lines on VIX
    fig.add_hline(y=IV_BUY_THRESHOLD_NORMAL*100, line_dash="dash", line_color="green",
                 annotation_text="Buy (20%)", row=2, col=1)
    fig.add_hline(y=IV_BUY_THRESHOLD_RELAXED*100, line_dash="dash", line_color="lightgreen",
                 annotation_text="Buy Relaxed (40%)", row=2, col=1)
    fig.add_hline(y=IV_SELL_THRESHOLD*100, line_dash="dash", line_color="red",
                 annotation_text="Sell (60%)", row=2, col=1)
    
    # Update layout
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="VIX", row=2, col=1)
    fig.update_yaxes(title_text="Volatility (%)", row=3, col=1)
    
    fig.update_layout(height=900, showlegend=True, hovermode='x unified')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Option Chain Visualization
    if puts_chain is not None and not puts_chain.empty:
        st.markdown("---")
        st.header("🎯 Option Chain Analysis")
        
        tab1, tab2, tab3 = st.tabs(["💵 Price by Strike", "📊 Implied Vol Surface", "💡 Strategy Comparison"])
        
        with tab1:
            st.markdown("### Put Prices Across Strikes")
            
            # Filter for puts with reasonable data
            clean_puts = puts_chain[puts_chain['lastPrice'] > 0].copy()
            clean_puts = clean_puts.sort_values('strike')
            
            fig_prices = go.Figure()
            
            fig_prices.add_trace(go.Scatter(
                x=clean_puts['strike'],
                y=clean_puts['lastPrice'],
                mode='lines+markers',
                name='Last Price',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ))
            
            fig_prices.add_vline(x=latest_price, line_dash="dash", line_color="green",
                               annotation_text=f"Current Price: ${latest_price:.2f}")
            fig_prices.add_vline(x=latest_price * (1-OTM_PERCENT), line_dash="dash", line_color="orange",
                               annotation_text=f"20% OTM: ${latest_price * (1-OTM_PERCENT):.2f}")
            
            fig_prices.update_layout(
                title=f"Put Option Prices for {asset_ticker}",
                xaxis_title="Strike Price ($)",
                yaxis_title="Price ($)",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_prices, use_container_width=True)
        
        with tab2:
            st.markdown("### Implied Volatility Skew")
            
            clean_puts = puts_chain[puts_chain['impliedVolatility'] > 0].copy()
            clean_puts = clean_puts.sort_values('strike')
            
            fig_iv = go.Figure()
            
            fig_iv.add_trace(go.Scatter(
                x=clean_puts['strike'],
                y=clean_puts['impliedVolatility'] * 100,
                mode='lines+markers',
                name='Implied Vol (%)',
                line=dict(color='purple', width=2),
                marker=dict(size=6)
            ))
            
            fig_iv.add_vline(x=latest_price, line_dash="dash", line_color="green",
                           annotation_text="Current Price")
            
            fig_iv.update_layout(
                title=f"Implied Volatility Skew for {asset_ticker} Puts",
                xaxis_title="Strike Price ($)",
                yaxis_title="Implied Volatility (%)",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_iv, use_container_width=True)
        
        with tab3:
            st.markdown("### Strategy Comparison by OTM Level")
            
            # Compare different OTM percentages
            otm_levels = [0.15, 0.20, 0.25, 0.30]
            strategy_comparison = []
            
            for otm in otm_levels:
                strike = latest_price * (1 - otm)
                # Find closest strike
                closest_idx = (puts_chain['strike'] - strike).abs().idxmin()
                closest = puts_chain.loc[closest_idx]
                
                strategy_comparison.append({
                    'OTM %': f"{otm*100:.0f}%",
                    'Strike': f"${closest['strike']:.2f}",
                    'Bid': f"${closest['bid']:.2f}",
                    'Ask': f"${closest['ask']:.2f}",
                    'Last Price': f"${closest['lastPrice']:.2f}",
                    'IV': f"{closest['impliedVolatility']*100:.1f}%",
                    'Open Interest': int(closest['openInterest'])
                })
            
            st.dataframe(
                pd.DataFrame(strategy_comparison),
                hide_index=True,
                use_container_width=True
            )
            
            st.markdown("""
            **Key Insights:**
            - **Deeper OTM** = Cheaper options, but further from current price (less likely to be ITM in moderate moves)
            - **Higher IV** at out-of-money strikes reflects tail risk pricing (volatility skew)
            - **Open Interest** shows market activity - higher OI = tighter bid-ask spreads
            - For tail hedging: balance between cost (deeper OTM) and protection level (not too deep)
            """)
    
    # Data table
    with st.expander("📋 View Detailed Market Data (Last 20 Days)"):
        display_data = data[[asset_ticker, 'VIX']].copy()
        display_data.columns = [f'{asset_ticker} Price', 'VIX']
        st.table(display_data.tail(20).sort_index(ascending=False))

else:
    st.error("Unable to fetch market data. Please try again later.")

# Footer
st.markdown("---")
st.markdown("""
### Strategy Overview
- **Normal Buy**: Volatility (VIX/100) ≤ 20%
- **Relaxed Buy**: Volatility (VIX/100) ≤ 40% (if 7+ days since last position)
- **Sell**: Volatility (VIX/100) ≥ 60% (volatility spike)
- **Strike**: 20% OTM (depth varies by market conditions)
- **Tool**: Uses real market option chain data from Yahoo Finance

*This is for educational purposes only. Not financial advice.*
""")
