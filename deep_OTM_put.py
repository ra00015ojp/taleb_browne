import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import os
import hashlib
import re

st.set_page_config(page_title="Browne Portfolio Put Option Advisor", layout="wide")

# File-based user database (for simple deployment)
USER_DB_FILE = "users.json"
POSITION_DB_FILE = "positions.json"

# Email configuration (set these as environment variables or Streamlit secrets)
EMAIL_CONFIG = {
    'smtp_server': st.secrets.get("SMTP_SERVER", "smtp.gmail.com"),
    'smtp_port': st.secrets.get("SMTP_PORT", 587),
    'email_address': st.secrets.get("EMAIL_ADDRESS", "your_email@gmail.com"),
    'email_password': st.secrets.get("EMAIL_PASSWORD", "your_app_password")
}

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_email' not in st.session_state:
    st.session_state.user_email = None
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {}

# Helper functions for user management
def load_users():
    """Load users from JSON file"""
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to JSON file"""
    with open(USER_DB_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def load_positions():
    """Load user positions from JSON file"""
    if os.path.exists(POSITION_DB_FILE):
        with open(POSITION_DB_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_positions(positions):
    """Save user positions to JSON file"""
    with open(POSITION_DB_FILE, 'w') as f:
        json.dump(positions, f, indent=2)

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def send_email_notification(to_email, subject, body):
    """Send email notification"""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG['email_address']
        msg['To'] = to_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'html'))
        
        server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.starttls()
        server.login(EMAIL_CONFIG['email_address'], EMAIL_CONFIG['email_password'])
        
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

def create_recommendation_email(user_email, recommendation_type, data):
    """Create HTML email with trading recommendation"""
    
    if recommendation_type == "BUY":
        subject = "🟢 BUY Signal - Browne Portfolio Put Option"
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2 style="color: #28a745;">🟢 BUY RECOMMENDATION</h2>
            <p>Hello,</p>
            <p>Based on current market conditions, we recommend <strong>BUYING</strong> put options:</p>
            
            <table style="border-collapse: collapse; width: 100%; margin: 20px 0;">
                <tr style="background-color: #f2f2f2;">
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Current SPY Price</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">${data['spy']:.2f}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>VIX</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{data['vix']:.2f}</td>
                </tr>
                <tr style="background-color: #f2f2f2;">
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Adjusted IV</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{data['adj_iv']:.1f}%</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Recommended Strike (20% OTM)</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">${data['strike']:.2f}</td>
                </tr>
                <tr style="background-color: #f2f2f2;">
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Estimated Put Price</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">${data['put_price']:.2f}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Reason</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{data['reason']}</td>
                </tr>
            </table>
            
            <p><a href="http://your-app-url.com" style="background-color: #28a745; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Open App</a></p>
            
            <p style="color: #666; font-size: 12px; margin-top: 30px;">
                This is an automated notification. Not financial advice.
            </p>
        </body>
        </html>
        """
    else:  # SELL
        subject = "🔴 SELL Signal - Browne Portfolio Put Option"
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2 style="color: #dc3545;">🔴 SELL RECOMMENDATION</h2>
            <p>Hello,</p>
            <p>Based on current market conditions, we recommend <strong>SELLING</strong> your put options:</p>
            
            <table style="border-collapse: collapse; width: 100%; margin: 20px 0;">
                <tr style="background-color: #f2f2f2;">
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Current SPY Price</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">${data['spy']:.2f}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>VIX</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{data['vix']:.2f}</td>
                </tr>
                <tr style="background-color: #f2f2f2;">
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Adjusted IV</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{data['adj_iv']:.1f}%</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Current Put Price</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">${data['current_price']:.2f}</td>
                </tr>
                <tr style="background-color: #f2f2f2;">
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Entry Price</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">${data['entry_price']:.2f}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Profit/Loss</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd; color: {'green' if data['pl'] > 0 else 'red'};">${data['pl']:+.2f} ({data['pl_pct']:+.1f}%)</td>
                </tr>
                <tr style="background-color: #f2f2f2;">
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Reason</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">Volatility spike - Adj IV ≥ 60%</td>
                </tr>
            </table>
            
            <p><a href="http://your-app-url.com" style="background-color: #dc3545; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Open App</a></p>
            
            <p style="color: #666; font-size: 12px; margin-top: 30px;">
                This is an automated notification. Not financial advice.
            </p>
        </body>
        </html>
        """
    
    return subject, body

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

# Login/Registration UI
def show_auth_page():
    st.title("🔐 Browne Portfolio Put Option Advisor")
    st.markdown("### Please login or register to continue")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.markdown("#### Login to Your Account")
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if not email or not password:
                    st.error("Please enter both email and password")
                else:
                    users = load_users()
                    if email in users and users[email]['password'] == hash_password(password):
                        st.session_state.logged_in = True
                        st.session_state.user_email = email
                        st.session_state.user_preferences = users[email].get('preferences', {})
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid email or password")
    
    with tab2:
        st.markdown("#### Create New Account")
        with st.form("register_form"):
            new_email = st.text_input("Email")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            email_notifications = st.checkbox("Enable email notifications", value=True)
            submit_register = st.form_submit_button("Register")
            
            if submit_register:
                if not new_email or not new_password:
                    st.error("Please fill in all fields")
                elif not validate_email(new_email):
                    st.error("Please enter a valid email address")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    users = load_users()
                    if new_email in users:
                        st.error("Email already registered")
                    else:
                        users[new_email] = {
                            'password': hash_password(new_password),
                            'preferences': {
                                'email_notifications': email_notifications,
                                'last_notification': None
                            },
                            'created_at': datetime.datetime.now().isoformat()
                        }
                        save_users(users)
                        
                        # Send welcome email
                        if email_notifications:
                            subject = "Welcome to Browne Portfolio Advisor"
                            body = f"""
                            <html>
                            <body style="font-family: Arial, sans-serif;">
                                <h2>Welcome!</h2>
                                <p>Thank you for registering with Browne Portfolio Put Option Advisor.</p>
                                <p>You will receive email notifications when there are buy or sell signals based on our strategy.</p>
                                <p><a href="http://your-app-url.com">Access the app</a></p>
                            </body>
                            </html>
                            """
                            send_email_notification(new_email, subject, body)
                        
                        st.success("Registration successful! Please login.")

# Check notification and send if needed
def check_and_send_notification(user_email, recommendation_type, data):
    """Check if notification should be sent and send it"""
    users = load_users()
    
    if user_email not in users:
        return
    
    user_prefs = users[user_email].get('preferences', {})
    
    # Check if notifications are enabled
    if not user_prefs.get('email_notifications', True):
        return
    
    # Check if we already sent notification recently (within 24 hours)
    last_notification = user_prefs.get('last_notification')
    if last_notification:
        last_time = datetime.datetime.fromisoformat(last_notification)
        if (datetime.datetime.now() - last_time).total_seconds() < 86400:  # 24 hours
            return
    
    # Send notification
    subject, body = create_recommendation_email(user_email, recommendation_type, data)
    if send_email_notification(user_email, subject, body):
        # Update last notification time
        users[user_email]['preferences']['last_notification'] = datetime.datetime.now().isoformat()
        save_users(users)
        st.success(f"📧 Email notification sent to {user_email}")

# Main app (only shown if logged in)
if not st.session_state.logged_in:
    show_auth_page()
else:
    # Show logout button in sidebar
    with st.sidebar:
        st.markdown(f"👤 Logged in as: **{st.session_state.user_email}**")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user_email = None
            st.session_state.user_preferences = {}
            st.rerun()
        
        st.markdown("---")
        
        # User preferences
        st.header("⚙️ Notification Settings")
        users = load_users()
        current_prefs = users[st.session_state.user_email].get('preferences', {})
        
        email_notif = st.checkbox(
            "Enable email notifications",
            value=current_prefs.get('email_notifications', True)
        )
        
        if st.button("Save Preferences"):
            users[st.session_state.user_email]['preferences']['email_notifications'] = email_notif
            save_users(users)
            st.success("Preferences saved!")
        
        st.markdown("---")
        
        # Configuration
        st.header("⚙️ Configuration")
        
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=30)
        
        st.info(f"📅 Analysis Period: {start_date} to {end_date}")
        
        st.markdown("---")
        
        # Position management
        has_position = st.radio(
            "Do you currently have a put option position?",
            options=["No", "Yes"],
            index=0
        )
        
        entry_date = None
        entry_price = None
        entry_strike = None
        entry_spy_price = None
        
        if has_position == "Yes":
            # Load saved position if exists
            positions = load_positions()
            saved_position = positions.get(st.session_state.user_email, {})
            
            st.markdown("#### Position Details")
            entry_date = st.date_input(
                "Entry Date",
                value=pd.to_datetime(saved_position.get('entry_date', end_date - datetime.timedelta(days=14))).date() if saved_position.get('entry_date') else end_date - datetime.timedelta(days=14),
                max_value=end_date
            )
            entry_spy_price = st.number_input(
                "SPY Price at Entry ($)",
                min_value=100.0,
                max_value=1000.0,
                value=float(saved_position.get('entry_spy_price', 580.0)),
                step=1.0
            )
            entry_strike = st.number_input(
                "Strike Price ($)",
                min_value=100.0,
                max_value=1000.0,
                value=float(saved_position.get('entry_strike', 464.0)),
                step=1.0
            )
            entry_price = st.number_input(
                "Entry Put Price ($)",
                min_value=0.01,
                max_value=100.0,
                value=float(saved_position.get('entry_price', 5.0)),
                step=0.1
            )
            
            if st.button("Save Position"):
                positions[st.session_state.user_email] = {
                    'entry_date': entry_date.isoformat(),
                    'entry_spy_price': entry_spy_price,
                    'entry_strike': entry_strike,
                    'entry_price': entry_price
                }
                save_positions(positions)
                st.success("Position saved!")
        
        st.markdown("---")
        st.markdown("#### Strategy Parameters")
        st.metric("Buy Threshold (Normal)", f"{IV_BUY_THRESHOLD_NORMAL*100:.0f}%")
        st.metric("Buy Threshold (Relaxed)", f"{IV_BUY_THRESHOLD_RELAXED*100:.0f}%")
        st.metric("Sell Threshold", f"{IV_SELL_THRESHOLD*100:.0f}%")
    
    # Title
    st.title("📊 Browne Portfolio Put Option Advisor")
    st.markdown("### Tail Risk Hedging Strategy Recommendation System")
    st.markdown("---")
    
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
        
        # Recommendation logic with email notifications
        if has_position == "No":
            st.header("🎯 BUY RECOMMENDATION")
            
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
                
                # Send email notification
                email_data = {
                    'spy': latest_spy,
                    'vix': latest_vix,
                    'adj_iv': latest_adj_iv,
                    'strike': latest_strike,
                    'put_price': latest_put_price,
                    'reason': 'Normal buy threshold met (IV ≤ 20%)'
                }
                check_and_send_notification(st.session_state.user_email, "BUY", email_data)
                
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
        
        else:  # Has position
            st.header("💰 SELL RECOMMENDATION")
            
            days_held = (latest_date.date() - entry_date).days
            time_left = max((TIME_TO_EXPIRY_DAYS - days_held) / 365, 0.001)
            
            current_put_price = price_otm_put(latest_spy, entry_strike, time_left, RISK_FREE_RATE, latest_vix)
            
            profit_loss = current_put_price - entry_price
            profit_loss_pct = (profit_loss / entry_price) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Days Held", f"{days_held}")
            with col2:
                st.metric("Entry Price", f"${entry_price:.2f}")
            with col3:
                st.metric("Current Price", f"${current_put_price:.2f}", f"{profit_loss:+.2f}")
            with col4:
                st.metric("P&L %", f"{profit_loss_pct:+.1f}%")
            
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
                
                # Send email notification
                email_data = {
                    'spy': latest_spy,
                    'vix': latest_vix,
                    'adj_iv': latest_adj_iv,
                    'current_price': current_put_price,
                    'entry_price': entry_price,
                    'pl': profit_loss,
                    'pl_pct': profit_loss_pct
                }
                check_and_send_notification(st.session_state.user_email, "SELL", email_data)
                
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
        
        # Rest of the app continues with Option Strategy Matrix, etc.
        # [Include all the matrix analysis code from previous version here]
        
    else:
        st.error("Unable to fetch market data. Please try again later.")

# Footer
st.markdown("---")
st.markdown("""
*This is for educational purposes only. Not financial advice.*
""")
