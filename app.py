# Set page config must be the very first Streamlit command
import streamlit as st

# Configure page
st.set_page_config(
    page_title="SPARS - Smart Portfolio Analysis and Risk System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import authentication module
from auth import get_authenticator, login

# Initialize session state for authentication
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.authenticator = get_authenticator()

# Show login form if not logged in
if not st.session_state.logged_in:
    # Call login function with the authenticator
    auth_status, username = login(st.session_state.authenticator)
    
    # Update session state if login was successful
    if auth_status:
        st.session_state.logged_in = True
        st.session_state.username = username
        st.experimental_rerun()
    else:
        # Display the login form
        st.stop()

# ===== Only the code below runs if user is authenticated =====

# Now import other dependencies after authentication is confirmed
import yaml
from yaml.loader import SafeLoader
from datetime import datetime, timedelta, date
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from smart_portfolio_analyzer import Portfolio, StockAsset, CryptoAsset, ForexAsset, OptionAsset, DataManager
import os
from pathlib import Path
import json
import time
from typing import Dict, List, Optional, Union, Tuple, Any

# Add logout button to sidebar
with st.sidebar:
    st.write(f"Welcome, *{st.session_state.username}*")
    if st.button('Logout'):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.logged_in = False
        st.experimental_rerun()

# Fix for Material Icons and BaseWeb components
st.markdown("""
    <style>
        /* Completely disable Material Icons */
        [class*="material"],
        [class*="Material"],
        [class*="Mui"],
        [class*="keyboard_arrow"],
        [class*="css"][data-icon*="keyboard"],
        [data-testid="stIconMaterial"],
        .material-icons,
        .material-icons-outlined,
        .material-icons-round,
        .material-icons-sharp,
        .material-symbols-outlined,
        .material-symbols-rounded,
        .material-symbols-sharp {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
            width: 0 !important;
            height: 0 !important;
            font-size: 0 !important;
            line-height: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            position: absolute !important;
            overflow: hidden !important;
            clip: rect(0, 0, 0, 0) !important;
            white-space: nowrap !important;
            border: 0 !important;
        }
        
        /* Fix for Streamlit expanders */
        .streamlit-expanderHeader {
            position: relative;
            padding-left: 1.5rem !important;
        }
        
        .streamlit-expanderHeader::before {
            content: '▶';
            position: absolute;
            left: 0.5rem;
            top: 50%;
            transform: translateY(-50%);
            font-size: 1rem;
            transition: transform 0.2s ease;
        }
        
        .streamlit-expanderHeader[aria-expanded="true"]::before {
            content: '▼';
        }
        
        .streamlit-expanderIcon {
            display: none !important;
        }
        
        /* Fix for select boxes */
        [data-baseweb="select"] {
            position: relative;
        }
        
        [data-baseweb="select"]::after {
            content: '▼' !important;
            position: absolute;
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
            pointer-events: none;
            color: #4b5563;
            font-size: 0.75em;
        }
        
        /* Hide the default select icon */
        [data-baseweb="select"] svg,
        [data-baseweb="select"] [role="button"] svg,
        [data-baseweb="select"] [role="button"]::after {
            display: none !important;
        }
        
        /* Fix for any remaining icon containers */
        [class*="material"],
        [class*="Material"],
        [class*="Mui"],
        [class*="icon"],
        [class*="Icon"] {
            font-family: inherit !important;
        }
    </style>
""", unsafe_allow_html=True)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.spatial import ConvexHull
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from smart_portfolio_analyzer import (
    Portfolio, 
    StockAsset, 
    CryptoAsset,
    ForexAsset,
    OptionAsset,
    RiskSimulator,
    DataManager,
    PortfolioAnalyzer
)

# Initialize DataManager with environment variables
try:
    # Try to get API key from Streamlit secrets first (for production)
    if 'POLYGON_API_KEY' in st.secrets:
        DATA_MANAGER = DataManager(api_key=st.secrets['POLYGON_API_KEY'])
    # Fallback to .env file for local development
    else:
        from dotenv import load_dotenv
        import os
        
        # Load environment variables from .env file
        load_dotenv()
        
        # Get API key from environment variables
        polygon_api_key = os.getenv('POLYGON_API_KEY')
        if not polygon_api_key:
            raise ValueError("POLYGON_API_KEY not found in environment variables")
            
        DATA_MANAGER = DataManager(api_key=polygon_api_key)
        
except Exception as e:
    st.error(f"Error initializing DataManager: {str(e)}")
    st.error("Please ensure you have a valid Polygon.io API key in Streamlit secrets or .env file")
    st.stop()

# Add logo to sidebar with better styling
import base64

def get_base64_encoded_image(image_path):
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

try:
    logo_base64 = get_base64_encoded_image('logo.png')
    st.sidebar.markdown(f"""
        <div style="display: flex; justify-content: center; margin: 0 0 20px 0; padding: 15px; background: #f8f9fa; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <img src="data:image/png;base64,{logo_base64}" width="100%" style="max-width: 180px; height: auto; object-fit: contain;"/>
        </div>
        <div style='margin: 10px 0 20px 0;'><hr style='border: 0.5px solid #e0e0e0;'></div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.sidebar.error("Could not load logo image")
    st.sidebar.text(f"Error: {str(e)}")

# Enhanced Custom CSS with consistent styling
st.markdown("""
    <style>
    /* Import and apply Helvetica Neue font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.cdnfonts.com/css/helvetica-neue-9');
    
    /* Base styles */
    :root {
        --primary-color: #4a90e2;
        --text-color: #2c3e50;
        --text-muted: #6c757d;
        --border-color: #e0e0e0;
        --card-bg: #ffffff;
        --sidebar-bg: #f8f9fa;
        --success-color: #28a745;
        --warning-color: #ffc107;
        --danger-color: #dc3545;
        --border-radius: 10px;
        --box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        --transition: all 0.2s ease;
    }
    
    /* Base styles */
    * {
        font-family: 'Helvetica Neue', 'Inter', Arial, sans-serif !important;
        box-sizing: border-box;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    body {
        color: var(--text-color);
        line-height: 1.6;
        background-color: #f5f7fa;
    }
    
    /* Main content area */
    .main .block-container {
        padding: 2rem 3rem 3rem 3rem;
        max-width: 1800px;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--text-color);
        margin: 1.5em 0 0.75em 0;
        font-weight: 600;
        line-height: 1.2;
        letter-spacing: -0.01em;
    }
    
    h1, .stMarkdown h1 { font-size: 2.25rem; }
    h2, .stMarkdown h2 { font-size: 1.75rem; }
    h3, .stMarkdown h3 { font-size: 1.5rem; }
    h4 { font-size: 1.25rem; }
    h5 { font-size: 1.1rem; }
    h6 { font-size: 1rem; }
    
    /* Cards and containers */
    .stDataFrame, 
    .stAlert, 
    .stExpander, 
    .stTabs [data-baseweb="tab-panel"],
    .stTabs [role="tabpanel"],
    .stTabs [data-baseweb="tab-list"] {
        border-radius: var(--border-radius);
        box-shadow: var(--box-shadow);
        margin: 1.25rem 0;
        padding: 1.75rem;
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        transition: var(--transition);
    }
    
    /* Consistent spacing for expanders */
    .stExpander {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        margin: 1.25rem 0;
    }
    
    .streamlit-expanderHeader {
        padding: 1rem 1.5rem;
        font-weight: 600;
    }
    
    .streamlit-expanderContent {
        padding: 1rem 1.5rem;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        width: 750px !important;
        min-width: 750px !important;
        max-width: 800px !important;
        padding: 3rem 2.5rem 3.5rem 3rem !important;
        transition: all 0.3s ease;
        background-color: #f8f9fa;
        box-shadow: 2px 0 25px rgba(0,0,0,0.1);
    }
    
    /* Hide sidebar by default when toggled */
    [data-testid="stSidebar"][aria-expanded="false"] {
        margin-left: -750px;
        opacity: 0;
        visibility: hidden;
    }
    
    /* Main content adjustment when sidebar is hidden */
    [data-testid="stSidebar"][aria-expanded="false"] + div > div > div > div > div > section > div > div {
        width: calc(100% - 4rem) !important;
        max-width: 100% !important;
        margin-left: 4rem !important;
    }
    
    /* Sidebar content spacing */
    .stSidebar .stSelectbox, 
    .stSidebar .stTextInput, 
    .stSidebar .stNumberInput,
    .stSidebar .stDateInput,
    .stSidebar .stButton,
    .stSidebar .stCheckbox,
    .stSidebar .stRadio {
        margin-bottom: 1.25rem !important;
    }
    
    /* Add more space between form groups */
    .stSidebar > div:not(:first-child) {
        margin-top: 0.5rem !important;
    }
    
    /* Form elements in sidebar */
    .stSidebar .stTextInput > div,
    .stSidebar .stNumberInput > div,
    .stSidebar .stDateInput > div {
        width: 100% !important;
    }
    
    /* Section headers in sidebar */
    .stSidebar h3 {
        margin: 3rem 0 2rem 0 !important;
        padding: 1rem 0;
        border-bottom: 2px solid #e0e0e0;
        font-size: 2rem !important;
        font-weight: 600 !important;
        color: #2c3e50 !important;
        letter-spacing: -0.02em;
    }
    
    /* Add more space between sections */
    .stSidebar > div[data-testid="stVerticalBlock"] {
        border-bottom: 3px solid #3498db;
        background-color: #f8f9fa;
    }
    
    /* Tables */
    table.dataframe {
        width: 100%;
        border-collapse: collapse;
    }
    
    table.dataframe th {
        background-color: #f8f9fa;
        font-weight: 600;
        text-align: left;
        padding: 0.75rem 1rem;
    }
    
    table.dataframe td {
        padding: 0.5rem 1rem;
        border-bottom: 1px solid #e0e0e0;
    }
    
    /* Alerts and messages */
    .stAlert {
        border-left: 4px solid #3498db;
    }
    
    .stAlert.warning {
        border-left-color: #f39c12;
    }
    
    .stAlert.error {
        border-left-color: #e74c3c;
    }
    
    .stAlert.success {
        border-left-color: #2ecc71;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .stTabs [role="tab"] {
            padding: 0.5rem 1rem;
            font-size: 0.9rem;
        }
        
        h1 { font-size: 1.75rem; }
        h2 { font-size: 1.5rem; }
        h3 { font-size: 1.25rem; }
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding: 0 20px;
        border-radius: 8px;
        background-color: #f8f9fa;
        transition: all 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e9ecef;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498db;
        color: white !important;
    }
    
    /* Input fields */
    .stTextInput>div>div>input, 
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>div>div {
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        padding: 1rem;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header { font-size: 24px; }
        .section-header { font-size: 20px; }
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state with sample portfolio
if 'portfolio' not in st.session_state:
    # Load the sample portfolio data
    sample_portfolio = {
        'name': 'Diversified ETF Portfolio',
        'assets': [
            {'ticker': 'VTI', 'quantity': 100, 'current_price': 238.75},
            {'ticker': 'VXUS', 'quantity': 150, 'current_price': 62.30},
            {'ticker': 'IEF', 'quantity': 120, 'current_price': 94.80},
            {'ticker': 'XLK', 'quantity': 75, 'current_price': 198.40},
            {'ticker': 'VIG', 'quantity': 80, 'current_price': 172.80}
        ]
    }
    
    # Create a new portfolio and add the sample assets
    portfolio = Portfolio("Sample Portfolio")
    for asset_data in sample_portfolio['assets']:
        asset = StockAsset(
            asset_id=asset_data['ticker'],  # Using ticker as asset_id for simplicity
            name=asset_data['ticker'],      # Using ticker as name for simplicity
            ticker=asset_data['ticker'],
            purchase_price=asset_data['current_price'],  # Using current as purchase price for simplicity
            quantity=asset_data['quantity'],
            purchase_date=date.today()
        )
        asset.current_price = asset_data['current_price']
        portfolio.add_asset(asset)
    
    st.session_state.portfolio = portfolio

# Initialize data manager if not already set
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = DATA_MANAGER

# Automatic cache cleanup (optional - uncomment to enable)
# Clean cache files older than 90 days on startup
try:
    cache_info = st.session_state.data_manager.get_cache_info()
    if cache_info.get('total_files', 0) > 200:  # Only clean if there are many files
        old_files = cache_info.get('files_by_age', {}).get('older_than_30_days', 0)
        if old_files > 50:  # Only clean if there are many old files
            files_deleted = st.session_state.data_manager.clean_old_cache(days_to_keep=30)
            if files_deleted > 0:
                print(f"Auto-cleaned {files_deleted} old cache files on startup")
except Exception as e:
    print(f"Error during automatic cache cleanup: {str(e)}")
st.sidebar.markdown("""
    <div style='margin: 0 0 3rem 0; padding: 0 0 2.5rem 0; border-bottom: 2px solid #e9ecef;'>
        <h2 style='color: #2c3e50; margin: 0 0 1.25rem 0; font-size: 3rem; font-weight: 700; line-height: 1.1; letter-spacing: -0.03em;'>Portfolio Manager</h2>
        <div style='font-size: 1.3rem; color: #6c757d; line-height: 1.7; max-width: 100%; padding-right: 2rem;'>
            Manage your investment portfolio with powerful analytics and insights
        </div>
    </div>
""", unsafe_allow_html=True)

# Add toggle button and JavaScript
st.markdown(
    """
    <button class="sidebar-toggle" id="sidebarToggle">☰</button>
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        const toggleButton = document.getElementById('sidebarToggle');
        let isExpanded = true;
        
        // Initialize sidebar state
        if (sidebar) {
            sidebar.setAttribute('aria-expanded', 'true');
        }
        
        // Toggle sidebar visibility
        function toggleSidebar() {
            isExpanded = !isExpanded;
            if (sidebar) {
                sidebar.setAttribute('aria-expanded', isExpanded);
                toggleButton.textContent = isExpanded ? '☰' : '☰';
                
                // Store state in session storage
                sessionStorage.setItem('sidebarExpanded', isExpanded);
            }
        }
        
        // Load saved state
        const savedState = sessionStorage.getItem('sidebarExpanded');
        if (savedState !== null) {
            isExpanded = savedState === 'true';
            if (sidebar) {
                sidebar.setAttribute('aria-expanded', isExpanded);
                toggleButton.textContent = isExpanded ? '☰' : '☰';
            }
        }
        
        // Add click event
        if (toggleButton) {
            toggleButton.addEventListener('click', toggleSidebar);
        }
    });
    </script>
    """,
    unsafe_allow_html=True
)

# Add Assets Section in Sidebar
st.markdown("""
    <style>
        /* Custom expander styling with Unicode arrows */
        .stExpander > div:first-child > div:first-child {
            padding-left: 1.5rem !important;
            position: relative;
        }
        
        .stExpander > div:first-child > div:first-child::before {
            content: '▶';
            position: absolute;
            left: 0.5rem;
            top: 50%;
            transform: translateY(-50%);
            font-size: 1rem;
            transition: transform 0.2s ease;
        }
        
        .stExpander > div[aria-expanded="true"] > div:first-child::before {
            content: '▼';
        }
        
        /* Hide the default expander icon */
        .stExpander > div:first-child > div:first-child > div:first-child > div:first-child {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

def get_historical_price(ticker, date):
    """Fetch historical price for a ticker on a specific date"""
    try:
        # Get data for the date range (1 day before to handle timezone issues)
        start_date = (date - timedelta(days=1)).strftime('%Y-%m-%d')
        end_date = (date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Get historical data using the data manager
        hist_data = st.session_state.data_manager.get_historical_prices(
            symbols=[ticker],
            start_date=start_date,
            end_date=end_date,
            interval='day'
        )
        
        if ticker in hist_data and not hist_data[ticker].empty:
            # Find the closest date to the purchase date
            price_data = hist_data[ticker]
            price_data = price_data[price_data.index.date <= date]
            
            if not price_data.empty:
                # Get the last available price before or on the purchase date
                last_row = price_data.iloc[-1]
                # Return the average of high and low for the day
                return (last_row['high'] + last_row['low']) / 2
    except Exception as e:
        st.warning(f"Could not fetch historical price for {ticker} on {date}: {str(e)}")
    return None

with st.sidebar.expander("\u2795 Add Assets", expanded=True):
    st.markdown("""
    <style>
        .asset-type-label {
            font-weight: 600;
            margin-bottom: 8px;
            color: #2c3e50;
        }
        .stRadio > div {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .stRadio > div > label {
            flex: 1 0 calc(50% - 4px);
            margin: 0;
            border-radius: 8px;
            padding: 8px 12px;
            background: #f8f9fa;
            border: 1px solid #e0e0e0;
            text-align: center;
            transition: all 0.2s ease;
        }
        .stRadio > div > label:hover {
            background: #f0f2f6;
            border-color: #3498db;
        }
        .stRadio > div > div[data-baseweb="radio"] > div {
            padding: 0;
            margin: 0;
        }
        .stRadio > div > label > div:first-child {
            display: none;
        }
        .stRadio > div > label > div:last-child {
            width: 100%;
            text-align: center;
        }
        .stRadio > div > div[data-baseweb="radio"] > div > div:first-child {
            margin-right: 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("### Add Assets to Portfolio")
    st.markdown('<p class="asset-type-label">Select Asset Type</p>', unsafe_allow_html=True)
    
    # Define asset types with icons
    asset_types = {
        "Stock": "📈",
        "Crypto": "🪙",
        "Forex": "💱",
        "Options": "📊"
    }
    
    # Create radio buttons with custom styling
    asset_type = st.radio(
        "Asset Type",
        options=list(asset_types.keys()),
        format_func=lambda x: f"{asset_types[x]} {x}",
        label_visibility="collapsed",
        horizontal=True
    )
    
    st.markdown("<div style='margin: 12px 0;'></div>", unsafe_allow_html=True)
    
    if asset_type == "Stock":
        col1, col2 = st.columns(2)
        with col1:
            ticker = st.text_input("Stock Ticker (e.g., AAPL, MSFT)", key="stock_ticker").strip().upper()
        with col2:
            purchase_date = st.date_input("Purchase Date", key="stock_purchase_date")
        
        col1, col2 = st.columns(2)
        with col1:
            shares = st.number_input("Number of Shares", min_value=0.0, step=0.01, 
                                   format="%.2f", key="stock_shares")
        
        with col2:
            # Initialize session state for stock price if not exists
            if 'stock_purchase_price' not in st.session_state:
                st.session_state.stock_purchase_price = 0.0
            
            # Fetch price when ticker or date changes
            if ticker and purchase_date:
                price_key = f"stock_price_{ticker}_{purchase_date}"
                if price_key not in st.session_state:
                    with st.spinner(f"Fetching price for {ticker}..."):
                        avg_price = get_historical_price(ticker, purchase_date)
                        if avg_price is not None:
                            st.session_state.stock_purchase_price = round(avg_price, 2)
                            st.session_state[price_key] = True  # Mark as fetched
                            st.rerun()
                
                # Show the fetched price and allow editing
                purchase_price = st.number_input(
                    "Purchase Price per Share ($)", 
                    min_value=0.0, 
                    step=0.01, 
                    format="%.2f", 
                    value=st.session_state.get('stock_purchase_price', 0.0),
                    key="stock_purchase_price_input"
                )
                
                # Show the source of the price as a caption
                st.caption(f"Average price on {purchase_date}: ${st.session_state.stock_purchase_price:.2f}")
            else:
                # If no ticker or date is selected, show a disabled input
                purchase_price = st.number_input(
                    "Purchase Price per Share ($)", 
                    min_value=0.0, 
                    step=0.01, 
                    format="%.2f", 
                    value=0.0,
                    key="stock_purchase_price_input",
                    disabled=not (ticker and purchase_date)
                )
                if not ticker:
                    st.caption("Enter a stock ticker and date to fetch the price")
                elif not purchase_date:
                    st.caption("Select a purchase date to fetch the price")
        
        if st.button("Add Stock to Portfolio", key="add_stock_btn", type="primary"):
            if ticker and shares > 0 and purchase_price > 0:
                try:
                    from smart_portfolio_analyzer.assets.stock_asset import StockAsset
                    stock = StockAsset(
                        asset_id=f"{ticker}_{purchase_date}",
                        name=ticker,
                        purchase_date=purchase_date,
                        purchase_price=purchase_price,
                        quantity=shares,
                        ticker=ticker
                    )
                    st.session_state.portfolio.add_asset(stock)
                    st.success(f"Successfully added {shares} shares of {ticker} to your portfolio!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding stock: {str(e)}")
    
    elif asset_type == "Bond":
        col1, col2 = st.columns(2)
        with col1:
            ticker = st.text_input("Bond Ticker (e.g., US10Y)", key="bond_ticker").strip().upper()
        with col2:
            purchase_date = st.date_input("Purchase Date", key="bond_purchase_date")
        
        col1, col2 = st.columns(2)
        with col1:
            face_value = st.number_input("Face Value ($)", min_value=0.0, step=100.0, 
                                       format="%.2f", key="bond_face_value")
        with col2:
            coupon_rate = st.number_input("Coupon Rate (%)", min_value=0.0, max_value=100.0, 
                                        step=0.1, format="%.2f", key="bond_coupon_rate")
        
        # Initialize session state for bond price if not exists
        if 'bond_purchase_price' not in st.session_state:
            st.session_state.bond_purchase_price = 100.0  # Default to par value
        
        # Fetch price when ticker or date changes for bonds
        if ticker and purchase_date:
            price_key = f"bond_price_{ticker}_{purchase_date}"
            if price_key not in st.session_state:
                with st.spinner(f"Fetching bond price for {ticker}..."):
                    bond_price = get_historical_price(ticker, purchase_date)
                    if bond_price is not None:
                        st.session_state.bond_purchase_price = round(bond_price, 2)
                        st.session_state[price_key] = True  # Mark as fetched
                        st.rerun()
        
        # Always show the input with the current value, allowing user to edit
        purchase_price = st.number_input(
            "Purchase Price (% of face value)", 
            min_value=0.0, 
            max_value=200.0,
            step=0.1, 
            format="%.2f", 
            value=st.session_state.get('bond_purchase_price', 100.0),
            key="bond_purchase_price_input"
        )
                    
    elif asset_type == "Options":
        col1, col2 = st.columns(2)
        with col1:
            underlying = st.text_input("Underlying Ticker (e.g., AAPL)", value="MSFT", key="option_underlying").strip().upper()
        with col2:
            option_type = st.selectbox("Option Type", ["Call", "Put"], index=0, key="option_type")
            
        col1, col2 = st.columns(2)
        with col1:
            strike_price = st.number_input("Strike Price ($)", min_value=0.01, step=0.5, 
                                         value=480.0, format="%.2f", key="option_strike")
        with col2:
            # Default to third Friday of March 2025 (March 21, 2025)
            expiration_date = st.date_input("Expiration Date", 
                                         value=date(2025, 3, 21), 
                                         key="option_expiration")
            
        col1, col2 = st.columns(2)
        with col1:
            contracts = st.number_input("Number of Contracts", min_value=1, step=1, 
                                      value=200, key="option_contracts")
        with col2:
            premium = st.number_input("Premium per Share ($)", min_value=0.0, step=0.01, 
                                    value=32.00, format="%.2f", key="option_premium")
            
        if st.button("Add Options to Portfolio", key="add_option_btn", type="primary"):
            if underlying and strike_price > 0 and premium > 0:
                try:
                    from smart_portfolio_analyzer.assets.option_asset import OptionAsset
                    
                    option = OptionAsset(
                        asset_id=f"{underlying}_{strike_price}{'C' if option_type == 'Call' else 'P'}_{expiration_date}",
                        name=f"{underlying} {strike_price}{'C' if option_type == 'Call' else 'P'} {expiration_date}",
                        purchase_date=date.today(),
                        purchase_price=premium,  # Premium per share (will be multiplied by 100 in cost_basis)
                        quantity=contracts,
                        underlying_ticker=underlying,
                        option_type=option_type.lower(),
                        strike_price=strike_price,
                        expiration_date=expiration_date
                    )
                    st.session_state.portfolio.add_asset(option)
                    st.success(f"Successfully added {contracts} {option_type} option(s) on {underlying} to your portfolio!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding option: {str(e)}")
                    
    elif asset_type == "Crypto":
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("Symbol (e.g., BTC, ETH)", key="crypto_symbol").strip().upper()
        with col2:
            exchange = st.text_input("Exchange (e.g., Coinbase, Binance)", key="crypto_exchange")
        
        col1, col2 = st.columns(2)
        with col1:
            purchase_date = st.date_input("Purchase Date", key="crypto_purchase_date")
        with col2:
            quantity = st.number_input("Quantity", min_value=0.00000001, step=0.00000001, format="%.8f", key="crypto_quantity")
        
        col1, col2 = st.columns(2)
        with col1:
            purchase_price = st.number_input("Purchase Price (per unit in USD)", min_value=0.0, step=0.01, key="crypto_purchase_price")
        with col2:
            blockchain = st.text_input("Blockchain (optional)", key="crypto_blockchain")
            
        col1, col2 = st.columns(2)
        with col1:
            is_token = st.checkbox("Is this a token?", key="crypto_is_token")
        with col2:
            is_stablecoin = st.checkbox("Is this a stablecoin?", key="crypto_is_stablecoin")
            
        contract_address = ""
        if is_token:
            contract_address = st.text_input("Contract Address (required for tokens)", key="crypto_contract_address")
            
        if st.button("Add Crypto to Portfolio", key="add_crypto_btn", type="primary"):
            if symbol and quantity > 0 and purchase_price > 0:
                if is_token and not contract_address:
                    st.error("Please provide a contract address for the token")
                    st.stop()
                    
                try:
                    from smart_portfolio_analyzer.assets.crypto_asset import CryptoAsset
                    
                    asset = CryptoAsset(
                        asset_id=f"{symbol}_{purchase_date}",
                        name=f"{symbol} Crypto",
                        purchase_date=purchase_date,
                        purchase_price=purchase_price,
                        quantity=quantity,
                        symbol=symbol,
                        exchange=exchange,
                        blockchain=blockchain if blockchain else None,
                        contract_address=contract_address if is_token else None,
                        is_token=is_token,
                        is_stablecoin=is_stablecoin
                    )
                    
                    st.session_state.portfolio.add_asset(asset)
                    st.success(f"Successfully added {symbol} to your portfolio!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding crypto asset: {str(e)}")
    
    elif asset_type == "Forex":
        col1, col2 = st.columns(2)
        with col1:
            base_currency = st.text_input("Base Currency (e.g., EUR, JPY)", key="forex_base_currency").strip().upper()
        with col2:
            quote_currency = st.text_input("Quote Currency (e.g., USD, EUR)", value="USD", key="forex_quote_currency").strip().upper()
        
        col1, col2 = st.columns(2)
        with col1:
            purchase_date = st.date_input("Purchase Date", key="forex_purchase_date")
        with col2:
            quantity = st.number_input("Quantity", min_value=0.00000001, step=0.00000001, format="%.8f", key="forex_quantity")
        
        purchase_price = st.number_input(
            f"Purchase Price (per {base_currency} in {quote_currency})", 
            min_value=0.0, 
            step=0.0001, 
            format="%.6f", 
            key="forex_purchase_price"
        )
        
        exchange = st.text_input("Exchange (optional, e.g., Forex, OANDA)", key="forex_exchange")
            
        if st.button("Add Forex to Portfolio", key="add_forex_btn", type="primary"):
            if base_currency and quote_currency and quantity > 0 and purchase_price > 0:
                try:
                    from smart_portfolio_analyzer.assets.forex_asset import ForexAsset
                    
                    symbol = f"{base_currency}/{quote_currency}"
                    asset = ForexAsset(
                        asset_id=f"{symbol}_{purchase_date}",
                        name=f"{symbol} Forex",
                        purchase_date=purchase_date,
                        purchase_price=purchase_price,
                        quantity=quantity,
                        symbol=symbol,
                        base_currency=base_currency,
                        quote_currency=quote_currency,
                        exchange=exchange if exchange else "Forex"
                    )
                    
                    st.session_state.portfolio.add_asset(asset)
                    st.success(f"Successfully added {symbol} to your portfolio!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding forex asset: {str(e)}")

# Add a separator after the Add Assets section
st.sidebar.markdown("<div style='margin: 20px 0;'><hr style='border: 0.5px solid #e0e0e0;'></div>", unsafe_allow_html=True)

# Cache Management Section
with st.sidebar.expander("Cache Management", expanded=False):
    st.markdown("**Manage cached market data**")
    
    # Show cache info
    cache_info = st.session_state.data_manager.get_cache_info()
    
    if 'error' in cache_info:
        st.error(f"Error reading cache info: {cache_info['error']}")
    else:
        st.write(f"**Total files:** {cache_info['total_files']}")
        st.write(f"**Total size:** {cache_info['total_size_mb']} MB")
        
        if cache_info['oldest_file']:
            st.write(f"**Oldest file:** {cache_info['oldest_file']}")
        if cache_info['newest_file']:
            st.write(f"**Newest file:** {cache_info['newest_file']}")
        
        # Show age distribution
        age_dist = cache_info['files_by_age']
        if age_dist.get('older_than_30_days', 0) > 0:
            st.warning(f"Warning: {age_dist['older_than_30_days']} files are older than 30 days")
        
        # Cache cleanup options
        st.markdown("**Clean up old cache files:**")
        
        col1, col2 = st.columns(2)
        with col1:
            days_to_keep = st.number_input("Keep files (days)", min_value=1, max_value=365, value=30, step=1)
        with col2:
            st.write("")  # Spacer
            if st.button("Clean Cache", type="secondary"):
                with st.spinner("Cleaning cache..."):
                    files_deleted = st.session_state.data_manager.clean_old_cache(days_to_keep)
                    if files_deleted > 0:
                        st.success(f"Deleted {files_deleted} old cache files!")
                        st.rerun()
                    else:
                        st.info("No old cache files to delete.")
        
        # Show detailed breakdown
        if st.checkbox("Show detailed breakdown"):
            st.markdown("**Files by age:**")
            st.write(f"• Less than 1 day: {age_dist.get('less_than_1_day', 0)}")
            st.write(f"• 1-7 days: {age_dist.get('1_to_7_days', 0)}")
            st.write(f"• 7-30 days: {age_dist.get('7_to_30_days', 0)}")
            st.write(f"• Older than 30 days: {age_dist.get('older_than_30_days', 0)}")

# Main app with enhanced header
st.markdown("""
    <div style='background: linear-gradient(135deg, #3498db, #2c3e50); padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 2rem; display: flex; align-items: center;'>
        <div style='margin-right: 25px;'>
            <img src='https://img.icons8.com/color/96/000000/line-chart--v1.png' width='60' style='filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.2));'/>
        </div>
        <div>
            <h1 style='color: white; margin: 0; display: flex; align-items: center; font-family: "Futura PT", "Futura", -apple-system, sans-serif;'>
                <span style='font-weight: 800; letter-spacing: 1px;'>SPARS</span>
                <span style='background: rgba(255,255,255,0.15); padding: 4px 8px; border-radius: 4px; font-size: 14px; margin-left: 12px; font-weight: 500;'>
                    v1.0
                </span>
            </h1>
            <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 15px;'>
                Smart Portfolio Analysis and Risk System
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Tab layout with simple text labels
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", 
    "Performance", 
    "Risk",
    "Optimize",
    "Forecast"
])

# Tab 5: Price Forecast
with tab5:
    st.header("🔮 Price Forecast")
    
    if not st.session_state.portfolio.assets:
        st.warning("Please add assets to your portfolio to use the forecasting tool.")
    else:
        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            history_years = st.slider("Years of historical data:", 1, 10, 5)
        with col2:
            forecast_days = st.slider("Days to forecast:", 30, 365, 90)
        
        if st.button("Generate Forecast", type="primary"):
            st.session_state.show_forecast = True
            st.session_state.forecast_data = None  # Clear previous forecast data

        if st.session_state.get('show_forecast', False):
            try:
                # Get historical data for all assets
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365 * history_years)
                
                # Get historical prices for all assets
                symbols = [asset.ticker for asset in st.session_state.portfolio.assets]
                historical_data = st.session_state.data_manager.get_historical_prices(
                    symbols=symbols,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                
                if not historical_data:
                    st.error("No historical data available for forecasting.")
                else:
                    # Calculate portfolio value over time
                    portfolio_values = pd.DataFrame()
                    asset_values = {}
                    
                    # First, find the common date range where we have data for all assets
                    common_dates = None
                    valid_symbols = []
                    
                    # First pass: find common dates and valid symbols
                    for symbol, data in historical_data.items():
                        if not data.empty and 'close' in data.columns and not data['close'].isna().all():
                            valid_symbols.append(symbol)
                            if common_dates is None:
                                common_dates = set(data.index[~data['close'].isna()])
                            else:
                                common_dates = common_dates.intersection(set(data.index[~data['close'].isna()]))
                    
                    if not valid_symbols:
                        st.error("No valid price data found for any assets. Please check your data sources.")
                    elif not common_dates:
                        st.warning(f"No common date range found for all assets. Please check the data availability for: {', '.join(valid_symbols)}")
                    else:
                        # Convert to sorted list of dates
                        common_dates = sorted(list(common_dates))
                        
                        # Create a new DataFrame with common dates as index
                        portfolio_values = pd.DataFrame(index=common_dates)
                        
                        # Second pass: populate the portfolio values with valid data
                        for symbol in valid_symbols:
                            data = historical_data[symbol]
                            # Get data for common dates and forward fill missing values
                            asset_data = data['close'].reindex(common_dates).ffill()
                            portfolio_values[symbol] = asset_data
                            asset_values[symbol] = asset_data.iloc[-1] if not asset_data.empty else 0
                        
                        # Calculate total portfolio value
                        if not portfolio_values.empty:
                            portfolio_values['total'] = portfolio_values.sum(axis=1)
                            
                            # Prepare data for Prophet
                            df = portfolio_values[['total']].reset_index()
                            df = df.rename(columns={'index': 'ds', 'total': 'y'})
                            
                            # Ensure we have at least 2 data points for forecasting
                            if len(df) < 2:
                                st.error("Not enough data points for forecasting. Try increasing the historical data period.")
                                st.stop()
                        else:
                            st.warning("No valid price data available for forecasting after processing.")
                            st.stop()
                        
                        # Fit Prophet model
                        with st.spinner("Training forecasting model..."):
                            model = Prophet(
                                daily_seasonality=True,
                                weekly_seasonality=True,
                                yearly_seasonality=True,
                                changepoint_prior_scale=0.05
                            )
                            model.fit(df)
                            
                            # Create future dates for forecasting
                            future = model.make_future_dataframe(periods=forecast_days)
                            
                            # Generate forecast
                            forecast = model.predict(future)
                        
                        # Calculate metrics
                        current_value = df['y'].iloc[-1]
                        forecasted_value = forecast['yhat'].iloc[-1]
                        change_pct = ((forecasted_value - current_value) / current_value) * 100
                        volatility = df['y'].pct_change().std() * np.sqrt(252)  # Annualized volatility
                        
                        # Store forecast data in session state
                        st.session_state.forecast_data = {
                            'model': model,
                            'forecast': forecast,
                            'df': df,
                            'end_date': end_date,
                            'forecast_days': forecast_days,
                            'current_value': current_value,
                            'forecasted_value': forecasted_value,
                            'change_pct': change_pct,
                            'volatility': volatility,
                            'asset_values': asset_values,
                            'portfolio_values': portfolio_values
                        }
                    
                    # Display forecast results
                    if 'forecast_data' in st.session_state and st.session_state.forecast_data is not None:
                        fd = st.session_state.forecast_data
                        
                        # Create tabs for different visualizations
                        tab_overview, tab_trend, tab_contributions, tab_scenarios = st.tabs([
                            "📊 Forecast Overview", 
                            "📈 Trend Analysis",
                            "🧩 Asset Contributions",
                            "🔄 Scenario Analysis"
                        ])
                        
                        # Tab 1: Forecast Overview
                        with tab_overview:
                            st.subheader("Portfolio Value Forecast")
                            
                            try:
                                # Safely access forecast data with defaults
                                current_value = fd.get('current_value', 0)
                                forecast_days = fd.get('forecast_days', 90)
                                forecasted_value = fd.get('forecasted_value', current_value)
                                change_pct = fd.get('change_pct', 0)
                                volatility = fd.get('volatility', 0) * 100  # Convert to percentage
                                
                                # Display metrics in columns
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Current Value", f"${current_value:,.2f}")
                                with col2:
                                    st.metric(
                                        f"Forecasted Value in {forecast_days} days",
                                        f"${forecasted_value:,.2f}",
                                        delta=f"{change_pct:,.2f}%" if change_pct is not None else None
                                    )
                                with col3:
                                    st.metric("Volatility (Annualized)", f"{volatility:.2f}%")
                                
                            except Exception as e:
                                st.error(f"Error displaying forecast metrics: {str(e)}")
                                st.warning("Please try regenerating the forecast data.")
                            
                            try:
                                # Check if we have the required data for plotting
                                if all(key in fd for key in ['model', 'forecast', 'df']):
                                    # Create forecast plot
                                    fig_forecast = plot_plotly(fd['model'], fd['forecast'])
                                    
                                    # Add actual data points if available
                                    if not fd['df'].empty and 'ds' in fd['df'] and 'y' in fd['df']:
                                        fig_forecast.add_scatter(
                                            x=fd['df']['ds'],
                                            y=fd['df']['y'],
                                            mode='markers',
                                            name='Actual',
                                            marker=dict(color='blue')
                                        )
                                    
                                    # Add confidence interval if available
                                    if 'yhat_upper' in fd['forecast'] and 'yhat_lower' in fd['forecast']:
                                        fig_forecast.add_trace(go.Scatter(
                                            x=pd.concat([fd['forecast']['ds'], fd['forecast']['ds']][::-1]),
                                            y=pd.concat([fd['forecast']['yhat_upper'], fd['forecast']['yhat_lower']][::-1]),
                                            fill='toself',
                                            fillcolor='rgba(31, 119, 180, 0.2)',
                                            line=dict(color='rgba(255,255,255,0)'),
                                            showlegend=True,
                                            name='Confidence Interval'
                                        ))
                                    
                                    # Update layout
                                    fig_forecast.update_layout(
                                        yaxis_title="Portfolio Value ($)",
                                        xaxis_title="Date",
                                        hovermode='x unified',
                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                    )
                                    
                                    st.plotly_chart(fig_forecast, use_container_width=True, key='forecast_chart')
                                else:
                                    st.warning("Incomplete forecast data for visualization. Please regenerate the forecast.")
                                    
                            except Exception as e:
                                st.error(f"Error generating forecast visualization: {str(e)}")
                                st.warning("Please try regenerating the forecast data.")
                            
                            # Add key insights
                            st.markdown("""
                            ### Key Insights
                            - The forecast suggests a **{}** trend in your portfolio value.
                            - The forecasted range in {} days is **${:,.2f} - ${:,.2f}** (80% confidence).
                            - Consider reviewing your portfolio if the lower bound falls below your risk tolerance.
                            """.format(
                                "positive" if fd['change_pct'] > 0 else "negative",
                                fd['forecast_days'],
                                fd['forecast']['yhat_lower'].iloc[-1],
                                fd['forecast']['yhat_upper'].iloc[-1]
                            ))
                        
                        # Tab 2: Trend Analysis
                        with tab_trend:
                            st.subheader("Trend Analysis")
                            
                            # Show components
                            st.markdown("""
                            ### Decomposed Forecast Components
                            The forecast is broken down into the following components:
                            - **Trend**: Long-term direction of the portfolio value
                            - **Weekly Seasonality**: Weekly patterns in the data
                            - **Yearly Seasonality**: Yearly patterns in the data
                            """)
                            
                            # Show components plot
                            fig_components = plot_components_plotly(fd['model'], fd['forecast'])
                            st.plotly_chart(fig_components, use_container_width=True, key='components_chart')
                            
                            # Tab 3: Asset Contributions
                            with tab_contributions:
                                st.subheader("Asset Contributions to Forecast")
                                
                                if 'asset_values' in fd and fd['asset_values']:
                                    # Calculate total portfolio value
                                    total_value = sum(fd['asset_values'].values())
                                    
                                    if total_value > 0:
                                        # Calculate percentage contributions
                                        contributions = {k: (v / total_value) * 100 for k, v in fd['asset_values'].items()}
                                        
                                        # Create a DataFrame for display
                                        contribution_data = [
                                            {
                                                'Asset': asset,
                                                'Value': fd['asset_values'][asset],
                                                'Contribution (%)': contributions[asset]
                                            }
                                            for asset in contributions
                                        ]
                                        contribution_df = pd.DataFrame(contribution_data)
                                        
                                        # Sort by contribution
                                        contribution_df = contribution_df.sort_values('Contribution (%)', ascending=False)
                                        
                                        # Display the table with remove buttons
                                        if fd['asset_values']:
                                            # Convert to DataFrame for display
                                            df = pd.DataFrame(fd['asset_values'].items(), columns=['Symbol', 'Value'])
                                            
                                            # Display the table with a "Remove" button for each row
                                            for i, row in df.iterrows():
                                                col1, col2 = st.columns([8, 1])
                                                with col1:
                                                    st.write(f"**{row['Symbol']}** - Value: {row['Value']}")
                                                with col2:
                                                    if st.button("❌", key=f"remove_asset_{i}_{row['Symbol']}"):
                                                        try:
                                                            # Remove the asset from the portfolio
                                                            st.session_state.portfolio.remove_asset(row['Symbol'])
                                                            st.success(f"Removed {row['Symbol']} from portfolio")
                                                            st.rerun()
                                                        except Exception as e:
                                                            st.error(f"Error removing asset: {str(e)}")
                                                st.divider()
                                        else:
                                            st.info("No assets in portfolio. Add assets using the sidebar.")
                                        
                                        # Create a pie chart
                                        fig = px.pie(
                                            contribution_df,
                                            values='Contribution (%)',
                                            names='Asset',
                                            title='Asset Allocation by Value',
                                            hover_data=['Value']
                                        )
                                        fig.update_traces(
                                            textposition='inside',
                                            textinfo='percent+label',
                                            hovertemplate='<b>%{label}</b><br>' +
                                                        'Contribution: %{percent}<br>' +
                                                        'Value: $%{customdata[0]:,.2f}<br>' +
                                                        '<extra></extra>'
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning("Total portfolio value is zero. Cannot calculate contributions.")
                                else:
                                    st.warning("No asset value data available for contribution analysis.")
                            
                            # Show asset performance if we have historical data
                            st.markdown("### Asset Performance")
                            fig_assets = go.Figure()
                            
                            try:
                                # Get current values from the portfolio
                                current_values = {asset.ticker: asset.quantity * asset.current_price 
                                               for asset in st.session_state.portfolio.assets}
                                
                                # Check if we have historical data for plotting
                                if 'portfolio_values' in fd and not fd['portfolio_values'].empty:
                                    for symbol in current_values.keys():
                                        if symbol in fd['portfolio_values']:
                                            fig_assets.add_trace(go.Scatter(
                                                x=fd['portfolio_values'].index,
                                                y=fd['portfolio_values'][symbol],
                                                mode='lines',
                                                name=symbol
                                            ))
                                    
                                    fig_assets.update_layout(
                                        title='Asset Performance Over Time',
                                        xaxis_title='Date',
                                        yaxis_title='Price ($)',
                                        hovermode='x unified',
                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                    )
                                    st.plotly_chart(fig_assets, use_container_width=True, key='assets_chart')
                                else:
                                    st.warning("No historical price data available for assets.")
                                    
                            except Exception as e:
                                st.error(f"Error generating asset performance chart: {str(e)}")
                                import traceback
                                st.text(traceback.format_exc())
                                st.info("Historical performance data is not available.")
                
                # Tab 4: Scenario Analysis
                with tab_scenarios:
                    st.subheader("Scenario Analysis")
                    
                    # Calculate historical drawdowns
                    returns = fd['portfolio_values']['total'].pct_change().dropna()
                    cumulative_returns = (1 + returns).cumprod() - 1
                    rolling_max = cumulative_returns.cummax()
                    drawdowns = (cumulative_returns - rolling_max) / (1 + rolling_max)
                    
                    # Calculate VaR and CVaR
                    var_95 = np.percentile(returns, 5)
                    cvar_95 = returns[returns <= var_95].mean()
                    
                    # Display risk metrics
                    st.markdown("### Portfolio Risk Metrics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Max Drawdown", f"{drawdowns.min()*100:.2f}%")
                    with col2:
                        st.metric("95% Daily VaR", f"{var_95*100:.2f}%")
                    with col3:
                        st.metric("95% CVaR", f"{cvar_95*100:.2f}%")
                    
                    # Drawdown chart
                    st.markdown("### Historical Drawdowns")
                    fig_drawdown = go.Figure()
                    fig_drawdown.add_trace(go.Scatter(
                        x=drawdowns.index,
                        y=drawdowns * 100,
                        fill='tozeroy',
                        line=dict(color='crimson'),
                        name='Drawdown %',
                        hovertemplate='%{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%<extra></extra>'
                    ))
                    
                    fig_drawdown.update_layout(
                        yaxis_title="Drawdown (%)",
                        xaxis_title="Date",
                        hovermode='x',
                        showlegend=False,
                        margin=dict(l=20, r=20, t=30, b=20),
                        height=400
                    )
                    
                    st.plotly_chart(fig_drawdown, use_container_width=True, key='scenario_drawdown_chart')
                    
                    # Stress test scenarios
                    st.markdown("### Stress Test Scenarios")
                    
                    scenarios = {
                        'Market Correction (-10%)': 0.9,
                        'Market Crash (-20%)': 0.8,
                        'Mild Growth (+5%)': 1.05,
                        'Strong Growth (+15%)': 1.15
                    }
                    
                    scenario_results = []
                    for scenario, factor in scenarios.items():
                        scenario_value = fd['current_value'] * factor
                        scenario_change = (scenario_value - fd['current_value']) / fd['current_value'] * 100
                        scenario_results.append({
                            'Scenario': scenario,
                            'Value': scenario_value,
                            'Change %': scenario_change
                        })
                    
                    # Display scenario results
                    scenario_df = pd.DataFrame(scenario_results)
                    st.dataframe(
                        scenario_df,
                        column_config={
                            'Scenario': 'Scenario',
                            'Value': st.column_config.NumberColumn(
                                'Value ($)',
                                format='$%.2f'
                            ),
                            'Change %': st.column_config.NumberColumn(
                                'Change %',
                                format='%+.2f%%'
                            )
                        },
                        hide_index=True,
                        use_container_width=True,
                        height=200
                    )
                            
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")
                import traceback
                st.text(traceback.format_exc())

# Tab 1: Portfolio Overview
with tab1:
    st.header("📋 Portfolio Overview")
    
    # Add button to load sample data if portfolio is empty
    if not st.session_state.portfolio.assets:
        if st.button("📊 Load Sample Portfolio"):
            try:
                try:
                    import os
                    import json
                    from pathlib import Path
                    from smart_portfolio_analyzer.portfolio import Portfolio
                    from smart_portfolio_analyzer.assets import StockAsset, CryptoAsset, ForexAsset, OptionAsset
                    
                    # Get the absolute path to the sample data file
                    base_dir = Path(__file__).parent.absolute()
                    sample_file = base_dir / "sample_data" / "sample_portfolio.json"
                    
                    # Debug: Print the file path being used
                    print(f"Looking for sample file at: {sample_file}")
                    
                    if not sample_file.exists():
                        st.error(f"Sample data file not found at: {sample_file}")
                        # Continue with the rest of the code to show additional error details
                    
                    # Load the sample data
                    with open(sample_file, 'r') as f:
                        sample_data = json.load(f)
                    
                    # Debug: Print the loaded data
                    print(f"Loaded sample data: {json.dumps(sample_data, indent=2)[:500]}...")
                    
                    # Create a new portfolio with the sample data
                    portfolio = Portfolio(
                        name=sample_data.get('name', 'Sample Portfolio'),
                        description=sample_data.get('description', ''),
                        risk_free_rate=float(sample_data.get('risk_free_rate', 0.02))
                    )
                    
                    # Add each asset to the portfolio
                    assets_added = 0
                    for i, asset_data in enumerate(sample_data.get('assets', [])):
                        try:
                            asset_type = asset_data.get('asset_type', '').lower()
                            ticker = asset_data.get('ticker', 'unknown')
                            
                            # Add required ID field if missing
                            if 'id' not in asset_data:
                                asset_data['id'] = f"{ticker}_{i}"
                            
                            try:
                                if asset_type in ['stock', 'etf']:
                                    # For both stocks and ETFs, we can use StockAsset
                                    asset_data.setdefault('exchange', 'NYSE')
                                    asset = StockAsset.from_dict(asset_data)
                                    print(f"Created {asset_type.upper()}: {ticker}")
                                elif asset_type == 'bond':
                                    # Ensure required fields are present with defaults
                                    print(f"Skipping bond asset type as it's no longer supported: {ticker}")
                                else:
                                    print(f"Skipping unknown asset type: {asset_type}")
                                    continue
                            except Exception as e:
                                print(f"Error creating asset {ticker}: {str(e)}")
                                continue
                                
                            # Add the asset with its weight if available
                            weight = sample_data.get('weights', {}).get(ticker)
                            portfolio.add_asset(asset, weight)
                            assets_added += 1
                            print(f"Added asset: {ticker} (type: {asset_type}, weight: {weight})")
                            
                        except Exception as e:
                            print(f"Error adding asset {ticker}: {str(e)}")
                            st.error(f"Error adding asset {ticker}: {str(e)}")
                    
                    if assets_added > 0:
                        st.session_state.portfolio = portfolio
                        st.success(f"Successfully loaded {assets_added} assets into the portfolio!")
                        st.rerun()
                    else:
                        st.error("No assets were added to the portfolio. Please check the sample data file.")
                        
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    print(f"Error loading sample portfolio: {error_details}")
                    st.error(f"Error loading sample portfolio: {str(e)}")
                    st.text_area("Error details", error_details, height=200)
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")
    
    # Portfolio metrics
    # Update prices before displaying metrics
    try:
        st.session_state.portfolio.update_prices()
    except Exception as e:
        st.error(f"Error updating prices: {str(e)}")
    
    # Calculate total portfolio value using the latest market data
    try:
        # Use the session's DataManager to get latest prices
        data_manager = st.session_state.data_manager
        
        # Get all tickers from the portfolio
        tickers = [asset.ticker for asset in st.session_state.portfolio.assets 
                  if hasattr(asset, 'ticker') and hasattr(asset, 'quantity')]
        
        if tickers:
            # Get current date for the end date
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')  # Get last 5 days of data
            
            # Fetch the latest prices using DataManager
            hist_data = data_manager.get_historical_prices(
                tickers,
                start_date=start_date,
                end_date=end_date,
                period='5d',
                interval='1d',
                source='polygon'
            )
            
            # Update each asset with the latest price and calculate values
            asset_values = {}
            total_value = 0.0
            
            for asset in st.session_state.portfolio.assets:
                if hasattr(asset, 'ticker') and hasattr(asset, 'quantity'):
                    ticker = asset.ticker
                    quantity = getattr(asset, 'quantity', 0)
                    
                    # Get the latest price from the historical data
                    if ticker in hist_data and not hist_data[ticker].empty:
                        latest_price = hist_data[ticker]['close'].iloc[-1]
                        # Update the asset's current price using the proper method
                        old_price = getattr(asset, 'current_price', 0)
                        asset.update_price(latest_price, date.today())
                        asset_value = latest_price * quantity
                        
                        # Debug info
                        if old_price != latest_price:
                            st.sidebar.write(f"📈 Updated {ticker}: ${old_price:.2f} → ${latest_price:.2f}")
                    else:
                        # Fallback to current_price if available, then purchase price
                        latest_price = getattr(asset, 'current_price', 0)
                        if latest_price is None or latest_price == 0:
                            latest_price = getattr(asset, 'purchase_price', 0)
                        asset_value = latest_price * quantity
                        
                        # Debug info
                        st.sidebar.write(f"⚠️ Using fallback price for {ticker}: ${latest_price:.2f}")
                    
                    asset_values[ticker] = asset_value
                    total_value += asset_value
        else:
            total_value = 0.0
            asset_values = {}
            
    except Exception as e:
        st.error(f"Error fetching latest market data: {str(e)}")
        # Fallback calculation using existing prices
        total_value = 0.0
        asset_values = {}
        for asset in st.session_state.portfolio.assets:
            if hasattr(asset, 'ticker') and hasattr(asset, 'quantity'):
                ticker = asset.ticker
                quantity = getattr(asset, 'quantity', 0)
                latest_price = getattr(asset, 'current_price', 0) or getattr(asset, 'purchase_price', 0)
                asset_value = latest_price * quantity
                asset_values[ticker] = asset_value
                total_value += asset_value
    
    # Calculate total purchase value
    total_purchase_value = sum(
        getattr(asset, 'purchase_price', 0) * getattr(asset, 'quantity', 0)
        for asset in st.session_state.portfolio.assets
    )
    
    # Calculate P&L
    total_pl = total_value - total_purchase_value
    pl_percent = (total_pl / total_purchase_value * 100) if total_purchase_value > 0 else 0
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        # Show the latest portfolio value with a tooltip
        st.metric(
            "Total Portfolio Value", 
            f"${total_value:,.2f}",
            help="Latest value from today's market data"
        )
    with col2:
        st.metric("Number of Assets", len(st.session_state.portfolio.assets))
    with col3:
        st.metric(
            "Total P&L", 
            f"${total_pl:,.2f}",
            delta=f"{pl_percent:.2f}%" if pl_percent != 0 else None
        )
    with col4:
        st.metric(
            "Total Invested", 
            f"${total_purchase_value:,.2f}",
            help="Original investment amount"
        )
    with col5:
        if st.button("🔄 Refresh Prices", help="Fetch latest market prices"):
            st.rerun()
        
    # The portfolio value over time chart is now handled in its own section
    # and uses the same data source as the total value shown above
    
    # Asset allocation with enhanced donut chart
    st.subheader("Asset Allocation")
    if st.session_state.portfolio.assets:
        allocation = st.session_state.portfolio.get_asset_allocation()
        
        # Create a donut chart with better styling
        fig = go.Figure(data=[go.Pie(
            labels=list(allocation.keys()),
            values=list(allocation.values()),
            hole=0.5,
            marker_colors=px.colors.qualitative.Plotly,
            textinfo='percent+label',
            hoverinfo='label+value+percent',
            textposition='inside'
        )])
        
        # Update layout for better appearance
        fig.update_layout(
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
            height=400
        )
        
        st.plotly_chart(fig, config={'displayModeBar': True}, use_container_width=True, key='portfolio_allocation_pie')
        
        # Calculate total portfolio value using current prices
        current_total_value = sum(asset_values.values()) if asset_values else 0
        
        # Use the calculated current value for allocation percentages
        total_value = current_total_value
        
        # Display allocation as a table with percentages
        if allocation and total_value > 0:
            # Convert to list of dicts for the table
            allocation_data = []
            for asset in st.session_state.portfolio.assets:
                if hasattr(asset, 'ticker') and hasattr(asset, 'current_price') and hasattr(asset, 'quantity'):
                    # Get the latest price using Polygon API with yfinance fallback
                    ticker = asset.ticker
                    display_price = 0.0
                    
                    try:
                        from smart_portfolio_analyzer.data_manager import DataManager
                        # Get the API key from environment variables
                        import os
                        from dotenv import load_dotenv
                        load_dotenv()
                        POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
                        
                        if not POLYGON_API_KEY:
                            raise ValueError("POLYGON_API_KEY not found in environment variables")
                            
                        # Initialize DataManager with the Polygon API key
                        data_manager = DataManager(api_key=POLYGON_API_KEY)
                        
                        # Try to get data from Polygon first
                        try:
                            # Get data for the last 5 days to ensure we have the most recent data
                            hist_data = data_manager.get_historical_prices(
                                [ticker], 
                                period="5d", 
                                source='polygon'
                            )
                            
                            if ticker in hist_data and not hist_data[ticker].empty:
                                display_price = hist_data[ticker]['close'].iloc[-1]
                            else:
                                raise ValueError("No data from Polygon")
                                
                        except Exception as poly_error:
                            st.warning(f"Error with Polygon for {ticker}, falling back to yfinance: {str(poly_error)}")
                            # Fall back to yfinance if Polygon fails
                            hist_data = data_manager.get_historical_prices(
                                [ticker], 
                                period="1d", 
                                source='yfinance'
                            )
                            if ticker in hist_data and not hist_data[ticker].empty:
                                display_price = hist_data[ticker]['close'].iloc[-1]
                            else:
                                raise ValueError("No data from yfinance")
                        
                        # Update the asset's current price for consistency
                        asset.current_price = display_price
                        # Update the asset_values dictionary
                        asset_values[ticker] = display_price * asset.quantity
                        
                    except Exception as e:
                        st.warning(f"Error fetching data for {ticker}: {str(e)}")
                        # Fallback to existing value if both Polygon and yfinance fail
                        display_price = getattr(asset, 'current_price', getattr(asset, 'price', 0))
                    
                    # Ensure we're using the latest price from the cache
                    if ticker in asset_values:
                        display_price = asset_values[ticker] / asset.quantity
                    
                    # Calculate asset value using the display price from the data cache
                    asset_value = display_price * asset.quantity
                    
                    # Calculate allocation percentage based on the total portfolio value
                    # Using a small epsilon to prevent division by zero
                    epsilon = 1e-10
                    allocation_pct = (asset_value / (total_value + epsilon)) * 100
                    
                    # Ensure allocation is between 0% and 100%
                    allocation_pct = max(0.0, min(100.0, allocation_pct))
                    
                    # Round to 1 decimal place
                    allocation_pct_rounded = round(allocation_pct, 1)
                    
                    allocation_data.append({
                        'Ticker': ticker,
                        'Current Price': f"${display_price:,.2f}",
                        'Quantity': asset.quantity,
                        'Value': f"${asset_value:,.2f}",
                        'Allocation': allocation_pct_rounded  # Pass the raw value (0-100) for ProgressColumn
                    })
            
            if allocation_data:
                # Create a DataFrame and display it
                import pandas as pd
                df = pd.DataFrame(allocation_data)
                # Add custom CSS for the table
                st.markdown("""
                <style>
                    .table-container {
                        display: flex;
                        justify-content: center;
                        width: 90%;
                        margin: 5px 0;
                    }
                    .compact-table {
                        font-size: 0.8em !important;
                        margin: 0 auto !important;
                        max-width: 90%;
                        min-width: 300px;
                    }
                    .compact-table th, .compact-table td {
                        padding: 1px 6px !important;
                        line-height: 1.2 !important;
                    }
                    .compact-table .stProgress > div > div > div > div {
                        height: 14px !important;
                        min-height: 14px !important;
                    }
                    .compact-table .stProgress > div > div > div > div > div {
                        height: 14px !important;
                        min-height: 14px !important;
                    }
                    .stDataFrame {
                        margin: 0 auto !important;
                    }
                </style>
                """, unsafe_allow_html=True)
                
                # Display centered table
                st.markdown('<div class="table-container">', unsafe_allow_html=True)
                st.dataframe(
                    df,
                    column_config={
                        'Ticker': st.column_config.TextColumn('Ticker', width='small'),
                        'Current Price': st.column_config.NumberColumn('$', format='%.2f', width='small'),
                        'Quantity': st.column_config.NumberColumn('Qty', format='%.2f', width='small'),
                        'Value': st.column_config.NumberColumn('$Value', format='%.2f', width='small'),
                        'Allocation': st.column_config.ProgressColumn(
                            '%',
                            format='%.1f%%',
                            min_value=0,
                            max_value=100,
                            width='small'
                        )
                    },
                    hide_index=True,
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Asset table with remove buttons
        st.subheader("Asset Details")
        asset_data = []
        for i, asset in enumerate(st.session_state.portfolio.assets):
            # Get the latest price from the asset
            ticker = asset.ticker
            display_price = getattr(asset, 'current_price', getattr(asset, 'price', 0))
            if display_price is None:
                display_price = getattr(asset, 'purchase_price', 0)
            
            # Calculate values using the display price
            current_value = display_price * asset._quantity if hasattr(asset, '_quantity') else 0
            purchase_value = asset._purchase_price * asset._quantity if hasattr(asset, '_purchase_price') else 0
            pl_value = current_value - purchase_value
            pl_percent = (pl_value / purchase_value * 100) if purchase_value > 0 else 0
            
            asset_data.append({
                "Symbol": asset.ticker,
                "Name": asset.name,
                "Type": asset.asset_type,
                "Quantity": asset._quantity,
                "Purchase Price": f"${asset._purchase_price:,.2f}",
                "Current Price": f"${display_price:,.2f}",
                "Market Value": f"${current_value:,.2f}",
                "P&L": f"${pl_value:,.2f}",
                "P&L %": f"{pl_percent:.2f}%",
                "Remove": i  # Store index for removal
            })
        
        # Display each asset with a remove button
        for i, asset in enumerate(st.session_state.portfolio.assets):
            col1, col2 = st.columns([9, 1])
            
            # Get asset details
            ticker = getattr(asset, 'ticker', 'N/A')
            name = getattr(asset, 'name', 'N/A')
            asset_type = getattr(asset, 'asset_type', 'N/A')
            quantity = getattr(asset, 'quantity', 0)
            current_price = getattr(asset, 'current_price', 0)
            purchase_price = getattr(asset, 'purchase_price', 0)
            
            # Calculate values
            current_value = current_price * quantity if current_price else 0
            pl_value = (current_price - purchase_price) * quantity if current_price and purchase_price else 0
            pl_percent = (pl_value / (purchase_price * quantity)) * 100 if purchase_price and quantity else 0
            
            # Format the display values
            price = f"${current_price:,.2f}" if current_price is not None else 'N/A'
            current_value_str = f"${current_value:,.2f}" if current_value is not None else 'N/A'
            color = 'red' if pl_value < 0 else 'green' if pl_value > 0 else 'black'
            pl_str = f"{pl_percent:+.2f}% (${pl_value:+,.2f})" if pl_value is not None else 'N/A'
            
            with col1:
                # Create the HTML
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h4 style="margin: 0;">{ticker} - {name}</h4>
                            <p style="margin: 5px 0 0 0; color: #666;">
                                {asset_type} • Qty: {quantity} • Price: {price}
                            </p>
                        </div>
                        <div style="text-align: right;">
                            <p style="margin: 0; font-weight: bold;">{current_value_str}</p>
                            <p style="margin: 5px 0 0 0; color: {color};">
                                {pl_str}
                            </p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                if st.button("×", key=f"remove_asset_compact_{i}_{ticker}"):
                    try:
                        st.session_state.portfolio.remove_asset(asset)
                        st.success(f"Removed {ticker} from portfolio")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error removing asset: {str(e)}")
    else:
        st.info("No assets in portfolio. Add assets using the 'Add Assets' tab.")

# Tab 2: Performance
with tab2:
    st.header("📈 Performance Analysis")
    
    if not st.session_state.portfolio.assets:
        st.warning("Add assets to your portfolio to see performance metrics.")
    else:
        # Create tabs for different performance sections
        perf_tab1, perf_tab2, perf_tab3 = st.tabs([
            "📊 Overview", 
            "📈 Attribution", 
            "🕒 Time Periods"
        ])
        
        # Get symbols and date range
        symbols = [getattr(asset, 'symbol', getattr(asset, 'ticker', '')) 
                 for asset in st.session_state.portfolio.assets]
        symbols = [s for s in symbols if s]  # Remove empty symbols
        
        with perf_tab1:
            # Date range selector
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", 
                                         value=datetime.now() - timedelta(days=365),
                                         max_value=datetime.now() - timedelta(days=1),
                                         key="perf_start_date")
            with col2:
                end_date = st.date_input("End Date", 
                                       value=datetime.now(),
                                       min_value=start_date + timedelta(days=1),
                                       max_value=datetime.now(),
                                       key="perf_end_date")
            
            if not symbols:
                st.warning("No valid assets with symbols found in the portfolio.")
            else:
                with st.spinner("Loading performance data..."):
                    historical_data = st.session_state.data_manager.get_historical_prices(
                        symbols=symbols,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d')
                    )
                    
                    if historical_data:
                        # Calculate portfolio values over time
                        portfolio_values = pd.DataFrame()
                        for symbol, data in historical_data.items():
                            if not data.empty and 'close' in data.columns:
                                # Find the asset by symbol or ticker
                                asset_quantity = 0
                                for asset in st.session_state.portfolio.assets:
                                    asset_symbol = getattr(asset, 'symbol', getattr(asset, 'ticker', ''))
                                    if asset_symbol == symbol:
                                        asset_quantity = getattr(asset, 'quantity', 0)
                                        break
                                
                                if asset_quantity > 0:
                                    portfolio_values[symbol] = data['close'] * asset_quantity
                        
                        if not portfolio_values.empty:
                            portfolio_values['Total'] = portfolio_values.sum(axis=1)
                            
                            # Portfolio value chart
                            st.subheader("Portfolio Value Over Time")
                            fig = px.line(portfolio_values, y='Total', 
                                        title="Portfolio Value Over Time",
                                        labels={'value': 'Value ($)', 'index': 'Date'})
                            st.plotly_chart(fig, config={'displayModeBar': True}, use_container_width=True, key='portfolio_value_overview_chart')
                            
                            # Add Cumulative Returns section
                            st.subheader("Cumulative Returns")
                            
                            # Calculate daily returns
                            if len(portfolio_values) > 1 and 'Total' in portfolio_values.columns:
                                # Calculate portfolio returns
                                returns = portfolio_values['Total'].pct_change().dropna()
                                cum_returns = (1 + returns).cumprod() - 1
                                
                                # Create a benchmark (e.g., S&P 500) - this is a placeholder
                                # In a real app, you would fetch actual benchmark data
                                benchmark_returns = pd.Series(
                                    np.random.normal(0.0005, 0.01, len(returns)),
                                    index=returns.index
                                )
                                benchmark_cum_returns = (1 + benchmark_returns).cumprod() - 1
                                
                                # Create the cumulative returns chart
                                fig = go.Figure()
                                
                                # Add portfolio line
                                fig.add_trace(go.Scatter(
                                    x=cum_returns.index,
                                    y=cum_returns * 100,  # Convert to percentage
                                    mode='lines',
                                    name='Your Portfolio',
                                    line=dict(color='#1f77b4', width=2.5)
                                ))
                                
                                # Add benchmark line
                                fig.add_trace(go.Scatter(
                                    x=benchmark_cum_returns.index,
                                    y=benchmark_cum_returns * 100,  # Convert to percentage
                                    mode='lines',
                                    name='Benchmark (S&P 500)',
                                    line=dict(color='#ff7f0e', width=2, dash='dash')
                                ))
                                
                                # Update layout
                                fig.update_layout(
                                    title='Cumulative Returns Over Time',
                                    xaxis_title='Date',
                                    yaxis_title='Cumulative Return (%)',
                                    legend=dict(
                                        orientation='h',
                                        yanchor='bottom',
                                        y=1.02,
                                        xanchor='right',
                                        x=1
                                    ),
                                    hovermode='x unified',
                                    template='plotly_white',
                                    height=500,
                                    margin=dict(t=40, b=80, l=60, r=40)
                                )
                                
                                # Format hover data
                                fig.update_traces(
                                    hovertemplate='%{y:.2f}%<extra></extra>'
                                )
                                
                                # Add a horizontal line at y=0
                                fig.add_hline(
                                    y=0,
                                    line_width=1,
                                    line_dash="dash",
                                    line_color="gray"
                                )
                                
                                # Display the chart
                                st.plotly_chart(fig, config={'displayModeBar': True}, use_container_width=True, key='portfolio_benchmark_comparison')
                                
                                # Add a note about the benchmark
                                st.caption("ℹ️ Benchmark data is simulated. In a production environment, you would fetch actual benchmark data.")
                            
                            # Performance metrics section
                            st.subheader("Performance Metrics")
                            
                            if len(portfolio_values) > 1 and 'Total' in portfolio_values.columns:
                                try:
                                    # Get risk-free rate with fallback
                                    try:
                                        risk_free_rate = st.session_state.data_manager.get_risk_free_rate()
                                    except Exception as e:
                                        st.warning(f"Could not fetch risk-free rate, using 0%: {str(e)}")
                                        risk_free_rate = 0.0
                                    
                                    # Calculate returns and metrics
                                    returns = portfolio_values['Total'].pct_change().dropna()
                                    if not returns.empty:
                                        metrics = PortfolioAnalyzer.calculate_performance_metrics(
                                            returns.values,
                                            risk_free_rate=risk_free_rate
                                        )
                                        
                                        # Calculate total return safely
                                        total_return = 0.0
                                        if portfolio_values['Total'].iloc[0] != 0:
                                            total_return = (portfolio_values['Total'].iloc[-1] / portfolio_values['Total'].iloc[0] - 1) * 100
                                        
                                        # Display metrics in a grid
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("Total Return", f"{total_return:.2f}%")
                                            st.metric("Annualized Return", f"{metrics.mean_return:.2%}")
                                        with col2:
                                            st.metric("Volatility", f"{metrics.volatility:.2%}")
                                            st.metric("Max Drawdown", f"{-metrics.max_drawdown:.2%}")
                                        with col3:
                                            st.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
                                            st.metric("Sortino Ratio", f"{metrics.sortino_ratio:.2f}")
                                        with col4:
                                            var_95 = -metrics.var_95 * portfolio_values['Total'].iloc[-1] if hasattr(metrics, 'var_95') and metrics.var_95 is not None else 0
                                            cvar_95 = -metrics.cvar_95 * portfolio_values['Total'].iloc[-1] if hasattr(metrics, 'cvar_95') and metrics.cvar_95 is not None else 0
                                            st.metric("95% VaR (1-day)", f"${max(0, var_95):.2f}")
                                            st.metric("95% CVaR (1-day)", f"${max(0, cvar_95):.2f}")
                                except Exception as e:
                                    st.error(f"Error calculating performance metrics: {str(e)}")
        
        with perf_tab2:
            if not symbols:
                st.warning("No valid assets with symbols found in the portfolio.")
            else:
                with st.spinner("Loading performance attribution data..."):
                    # Reuse historical data from the first tab or fetch fresh data
                    if 'historical_data' not in locals():
                        historical_data = st.session_state.data_manager.get_historical_prices(
                            symbols=symbols,
                            start_date=start_date.strftime('%Y-%m-%d'),
                            end_date=end_date.strftime('%Y-%m-%d')
                        )
                    
                    if historical_data:
                        # Calculate portfolio values for attribution
                        portfolio_values = pd.DataFrame()
                        for symbol, data in historical_data.items():
                            if not data.empty and 'close' in data.columns:
                                asset_quantity = 0
                                for asset in st.session_state.portfolio.assets:
                                    asset_symbol = getattr(asset, 'symbol', getattr(asset, 'ticker', ''))
                                    if asset_symbol == symbol:
                                        asset_quantity = getattr(asset, 'quantity', 0)
                                        break
                                
                                if asset_quantity > 0:
                                    portfolio_values[symbol] = data['close'] * asset_quantity
                        
                        if not portfolio_values.empty:
                            portfolio_values['Total'] = portfolio_values.sum(axis=1)
                            
                            # Performance Attribution Analysis
                    
                    try:
                        # Calculate individual asset contributions
                        attribution_data = []
                        total_start_value = portfolio_values['Total'].iloc[0]
                        total_end_value = portfolio_values['Total'].iloc[-1]
                        total_return_pct = (total_end_value / total_start_value - 1) * 100
                        
                        for symbol in portfolio_values.columns:
                            if symbol != 'Total':
                                start_value = portfolio_values[symbol].iloc[0]
                                end_value = portfolio_values[symbol].iloc[-1]
                                
                                # Calculate contribution in dollars and percentage
                                dollar_contribution = end_value - start_value
                                pct_contribution = (dollar_contribution / total_start_value) * 100
                                asset_return = (end_value / start_value - 1) * 100 if start_value > 0 else 0
                                weight_start = (start_value / total_start_value) * 100
                                weight_end = (end_value / total_end_value) * 100 if total_end_value > 0 else 0
                                
                                attribution_data.append({
                                    'Asset': symbol,
                                    'Start Value': start_value,
                                    'End Value': end_value,
                                    'Dollar Contribution': dollar_contribution,
                                    '% Contribution': pct_contribution,
                                    'Asset Return %': asset_return,
                                    'Weight Start %': weight_start,
                                    'Weight End %': weight_end
                                })
                        
                        if attribution_data:
                            # Convert to DataFrame
                            attribution_df = pd.DataFrame(attribution_data)
                            
                            # Sort by absolute contribution
                            attribution_df = attribution_df.sort_values('Dollar Contribution', key=abs, ascending=False)
                            
                            # Display top contributors
                            st.markdown("### Top Contributors to Performance")
                            
                            # Create a bar chart of contributions
                            fig = px.bar(
                                attribution_df,
                                x='Asset',
                                y='Dollar Contribution',
                                title='Dollar Contribution by Asset',
                                color='Dollar Contribution',
                                color_continuous_scale=px.colors.sequential.Viridis,
                                text=attribution_df['Dollar Contribution'].apply(lambda x: f'${x:,.0f}')
                            )
                            
                            fig.update_traces(
                                textposition='outside',
                                texttemplate='%{text}',
                                hovertemplate='<b>%{x}</b><br>' +
                                              'Contribution: $%{y:,.2f}<br>' +
                                              'Return: %{customdata[0]:.1f}%<br>' +
                                              'Weight: %{customdata[1]:.1f}% → %{customdata[2]:.1f}%' +
                                              '<extra></extra>',
                                customdata=attribution_df[['Asset Return %', 'Weight Start %', 'Weight End %']].values
                            )
                            
                            fig.update_layout(
                                yaxis_title='Contribution ($)',
                                xaxis_title='Asset',
                                showlegend=False,
                                height=500,
                                margin=dict(t=40, b=100, l=50, r=50)
                            )
                            
                            st.plotly_chart(fig, config={'displayModeBar': True}, use_container_width=True, key='performance_attribution_chart_v2')
                            
                            # Display detailed attribution table
                            st.markdown("### Detailed Performance Attribution")
                            
                            # Format the table
                            display_df = attribution_df.copy()
                            display_df = display_df[[
                                'Asset', 'Dollar Contribution', '% Contribution',
                                'Asset Return %', 'Weight Start %', 'Weight End %'
                            ]]
                            
                            # Format numbers
                            display_df['Dollar Contribution'] = display_df['Dollar Contribution'].apply(
                                lambda x: f'${x:,.2f}')
                            display_df['% Contribution'] = display_df['% Contribution'].apply(
                                lambda x: f'{x:.2f}%')
                            display_df['Asset Return %'] = display_df['Asset Return %'].apply(
                                lambda x: f'{x:.2f}%')
                            display_df['Weight Start %'] = display_df['Weight Start %'].apply(
                                lambda x: f'{x:.1f}%')
                            display_df['Weight End %'] = display_df['Weight End %'].apply(
                                lambda x: f'{x:.1f}%')
                            
                            # Rename columns for display
                            display_df = display_df.rename(columns={
                                'Dollar Contribution': 'Contribution ($)',
                                '% Contribution': 'Contribution (%)',
                                'Asset Return %': 'Return (%)',
                                'Weight Start %': 'Start Weight',
                                'Weight End %': 'End Weight'
                            })
                            
                            # Display the table
                            st.dataframe(
                                display_df,
                                column_config={
                                    'Asset': st.column_config.TextColumn(
                                        'Asset',
                                        help='Asset ticker symbol'
                                    ),
                                    'Contribution ($)': st.column_config.NumberColumn(
                                        'Contribution ($)',
                                        help='Dollar contribution to portfolio return',
                                        format='$%.2f'
                                    ),
                                    'Contribution (%)': st.column_config.NumberColumn(
                                        'Contribution (%)',
                                        help='Percentage contribution to portfolio return',
                                        format='%.2f%%'
                                    ),
                                    'Return (%)': st.column_config.NumberColumn(
                                        'Return (%)',
                                        help='Asset return over the period',
                                        format='%.2f%%'
                                    ),
                                    'Start Weight': st.column_config.TextColumn(
                                        'Start Weight',
                                        help='Weight at the beginning of the period'
                                    ),
                                    'End Weight': st.column_config.TextColumn(
                                        'End Weight',
                                        help='Weight at the end of the period'
                                    )
                                },
                                hide_index=True,
                                use_container_width=True
                            )
                            
                            # Add download button for the data
                            csv = display_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Attribution Data",
                                data=csv,
                                file_name=f"performance_attribution_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                            
                    except Exception as e:
                        st.error(f"Error generating performance attribution: {str(e)}")
                        import traceback
                        st.text(traceback.format_exc())
        
        # Time Periods Tab
        with perf_tab3:
            st.subheader("Performance by Time Period")
            
            if not symbols:
                st.warning("No valid assets with symbols found in the portfolio.")
            else:
                with st.spinner("Loading performance by time period..."):
                    try:
                        # Define time periods
                        periods = {
                            '1M': 30,
                            '3M': 90,
                            'YTD': (datetime.now() - datetime(datetime.now().year, 1, 1)).days,
                            '1Y': 365,
                            '3Y': 365 * 3,
                            '5Y': 365 * 5,
                            'Max': 365 * 20  # Max 20 years
                        }
                        
                        # Get historical data for max period
                        max_days = max(periods.values())
                        historical_data = st.session_state.data_manager.get_historical_prices(
                            symbols=symbols,
                            start_date=(datetime.now() - timedelta(days=max_days)).strftime('%Y-%m-%d'),
                            end_date=datetime.now().strftime('%Y-%m-%d')
                        )
                        
                        if historical_data:
                            # Calculate portfolio values for each symbol
                            portfolio_values = pd.DataFrame()
                            for symbol, data in historical_data.items():
                                if not data.empty and 'close' in data.columns:
                                    asset_quantity = 0
                                    for asset in st.session_state.portfolio.assets:
                                        asset_symbol = getattr(asset, 'symbol', getattr(asset, 'ticker', ''))
                                        if asset_symbol == symbol:
                                            asset_quantity = getattr(asset, 'quantity', 0)
                                            break
                                    if asset_quantity > 0:
                                        portfolio_values[symbol] = data['close'] * asset_quantity
                            
                            if not portfolio_values.empty:
                                portfolio_values['Total'] = portfolio_values.sum(axis=1)
                                
                                # Calculate returns for each period
                                returns_data = []
                                today = datetime.now().date()
                                
                                for period_name, days in periods.items():
                                    if period_name == 'YTD':
                                        start_date = datetime(today.year, 1, 1).date()
                                    else:
                                        start_date = today - timedelta(days=days)
                                    
                                    period_data = portfolio_values[portfolio_values.index.date >= start_date]
                                    
                                    if len(period_data) > 1:
                                        start_value = period_data['Total'].iloc[0]
                                        end_value = period_data['Total'].iloc[-1]
                                        total_return = (end_value / start_value - 1) * 100 if start_value > 0 else 0
                                        
                                        # Calculate annualized return if period is at least 1 year
                                        annualized_return = 0
                                        if period_name in ['1Y', '3Y', '5Y', 'Max'] and days >= 365:
                                            years = days / 365
                                            annualized_return = ((1 + total_return/100) ** (1/years) - 1) * 100
                                        
                                        returns_data.append({
                                            'Period': period_name,
                                            'Start Date': period_data.index[0].strftime('%Y-%m-%d'),
                                            'End Date': period_data.index[-1].strftime('%Y-%m-%d'),
                                            'Total Return (%)': total_return,
                                            'Annualized Return (%)': annualized_return if annualized_return else None
                                        })
                                
                                if returns_data:
                                    returns_df = pd.DataFrame(returns_data)
                                    
                                    # Display returns table
                                    st.markdown("### Returns by Time Period")
                                    
                                    # Format the table
                                    display_returns = returns_df.copy()
                                    display_returns['Total Return (%)'] = display_returns['Total Return (%)'].apply(lambda x: f"{x:.2f}%")
                                    display_returns['Annualized Return (%)'] = display_returns['Annualized Return (%)'].apply(
                                        lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A"
                                    )
                                    
                                    st.dataframe(
                                        display_returns[['Period', 'Start Date', 'End Date', 'Total Return (%)', 'Annualized Return (%)']],
                                        column_config={
                                            'Period': st.column_config.TextColumn('Period'),
                                            'Start Date': st.column_config.TextColumn('Start Date'),
                                            'End Date': st.column_config.TextColumn('End Date'),
                                            'Total Return (%)': st.column_config.TextColumn('Total Return'),
                                            'Annualized Return (%)': st.column_config.TextColumn('Annualized Return')
                                        },
                                        hide_index=True,
                                        use_container_width=True
                                    )
                                    
                                    # Add download button for the data
                                    csv = returns_df.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        label="Download Returns Data",
                                        data=csv,
                                        file_name=f"portfolio_returns_by_period_{today.strftime('%Y%m%d')}.csv",
                                        mime="text/csv"
                                    )
                                else:
                                    st.warning("Insufficient data to calculate returns for the selected periods.")
                        else:
                            st.warning("No historical data available for the selected time periods.")
                            
                    except Exception as e:
                        st.error(f"Error generating performance by time period: {str(e)}")
                        import traceback
                        st.text(traceback.format_exc())

# Tab 3: Risk Analysis
with tab3:
    st.header("📊 Risk Analysis")
    
    if not st.session_state.portfolio.assets:
        st.warning("Add assets to your portfolio to perform risk analysis.")
    else:
        # Create tabs for different risk analysis sections
        risk_tab1, risk_tab2, risk_tab3, risk_tab4 = st.tabs([
            "📊 Risk Dashboard", 
            "🎲 Monte Carlo", 
            "📉 Value at Risk",
            "📊 Correlation"
        ])
        
        with risk_tab1:
            st.subheader("Risk Metrics Overview")
            
            # Calculate key risk metrics
            with st.spinner("Calculating risk metrics..."):
                simulator = RiskSimulator(st.session_state.portfolio)
                metrics = simulator.estimate_risk_metrics(time_horizon=252)  # 1 year
                
                # Store metrics in session state for other tabs
                if 'risk_metrics' not in st.session_state:
                    st.session_state.risk_metrics = metrics
                
                # Risk metrics cards in a grid
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Portfolio Beta", f"{metrics.get('beta', 0):.2f}", 
                             help="Measures portfolio's sensitivity to market movements")
                with col2:
                    st.metric("Volatility (1Y)", f"{metrics['volatility']:.2%}",
                             help="Annualized standard deviation of returns")
                with col3:
                    st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}",
                             help="Risk-adjusted return (higher is better)")
                with col4:
                    st.metric("Max Drawdown", f"{-metrics['max_drawdown']:.2%}",
                             help="Maximum observed loss from peak to trough")
                
                # Second row of metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("95% VaR (1D)", f"{-metrics['var'].get(0.95, 0)*100:.2f}%",
                             help="1-day 95% Value at Risk")
                with col2:
                    st.metric("95% CVaR (1D)", f"{-metrics['cvar'].get(0.95, 0)*100:.2f}%",
                             help="1-day 95% Conditional Value at Risk")
                with col3:
                    st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}",
                             help="Downside risk-adjusted return")
                with col4:
                    st.metric("Value at Risk (1M, 95%)", 
                             f"${-metrics.get('var_abs', {}).get(0.95, 0) * st.session_state.portfolio.get_total_value():,.0f}",
                             help="1-month 95% Value at Risk in $")
            
            # Drawdown analysis
            st.subheader("Historical Drawdown Analysis")
            try:
                # Get historical prices for drawdown calculation
                symbols = [asset.ticker for asset in st.session_state.portfolio.assets]
                historical_data = st.session_state.data_manager.get_historical_prices(
                    symbols=symbols,
                    start_date=(datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d'),
                    end_date=datetime.now().strftime('%Y-%m-%d')
                )
                
                if not historical_data:
                    st.warning("No historical data available for drawdown analysis.")
                else:
                    # Combine all asset data into a single DataFrame
                    combined_data = None
                    
                    for symbol, df in historical_data.items():
                        if df is None or df.empty:
                            continue
                            
                        # Ensure we have a 'close' column
                        if 'close' not in df.columns and len(df.columns) > 0:
                            # Use the first numeric column if 'close' is not available
                            for col in df.select_dtypes(include=['float64', 'int64']).columns:
                                df = df.rename(columns={col: 'close'})
                                break
                        
                        if 'close' not in df.columns:
                            continue
                            
                        # Create a temporary DataFrame with just the close prices
                        temp_df = pd.DataFrame({
                            symbol: df['close']
                        })
                        
                        if combined_data is None:
                            combined_data = temp_df
                        else:
                            combined_data = combined_data.join(temp_df, how='outer')
                    
                    if combined_data is None or combined_data.empty:
                        st.warning("No valid price data available for drawdown analysis.")
                    else:
                        # Forward fill and backfill any missing values
                        combined_data = combined_data.ffill().bfill()
                        
                        # Calculate portfolio value over time
                        portfolio_values = pd.DataFrame(index=combined_data.index)
                        
                        # Calculate value of each asset over time
                        for asset in st.session_state.portfolio.assets:
                            if asset.ticker in combined_data.columns:
                                shares = asset.quantity
                                portfolio_values[asset.ticker] = combined_data[asset.ticker] * shares
                        
                        if portfolio_values.empty:
                            st.warning("Could not calculate portfolio values. No valid asset data found.")
                        else:
                            # Calculate total portfolio value
                            portfolio_values['Total'] = portfolio_values.sum(axis=1)
                            
                            # Calculate drawdown
                            rolling_max = portfolio_values['Total'].cummax()
                            drawdown = (portfolio_values['Total'] - rolling_max) / rolling_max
                            
                            # Only proceed with plotting if we have valid drawdown data
                            if drawdown.isnull().all() or len(drawdown) == 0:
                                st.warning("Could not calculate drawdown: insufficient data.")
                            else:
                                fig_drawdown = go.Figure()
                                fig_drawdown.add_trace(go.Scatter(
                                    x=drawdown.index,
                                    y=drawdown * 100,
                                fill='tozeroy',
                                line=dict(color='#ff4b4b'),
                                name='Drawdown %'
                            ))
                            
                            fig_drawdown.update_layout(
                                title="Portfolio Drawdown Over Time",
                                xaxis_title="Date",
                                yaxis_title="Drawdown (%)",
                                hovermode="x unified",
                                height=400,
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig_drawdown, config={'displayModeBar': True}, use_container_width=True, key='risk_drawdown_chart')
                            
                            # Add some context about drawdown
                            max_drawdown = drawdown.min() * 100
                            max_dd_date = drawdown.idxmin()
                            current_dd = drawdown.iloc[-1] * 100 if len(drawdown) > 0 else 0
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Maximum Drawdown", f"{max_drawdown:.2f}%", 
                                         delta=f"on {max_dd_date.strftime('%Y-%m-%d')}" if pd.notnull(max_dd_date) else None)
                            with col2:
                                st.metric("Current Drawdown", f"{current_dd:.2f}%")
                            
                            # Drawdown analysis completed successfully
                            drawdown_successful = True
                    
                    # If we get here, we couldn't generate the drawdown analysis
                    st.info("Could not generate drawdown analysis with the available data.")
                    
            except Exception as e:
                st.error(f"Error generating drawdown analysis: {str(e)}")
                import traceback
                st.text(traceback.format_exc())
        
        with risk_tab2:
            st.subheader("Monte Carlo Simulation")
            
            # Simulation controls
            with st.expander("⚙️ Simulation Settings", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    num_simulations = st.slider("Number of Simulations", 100, 10000, 1000, 100)
                with col2:
                    time_horizon = st.slider("Time Horizon (days)", 30, 252*5, 252, 30)
                with col3:
                    initial_investment = st.number_input(
                        "Initial Investment ($)", 
                        min_value=1000, 
                        max_value=10000000, 
                        value=10000, 
                        step=1000
                    )
            
            if st.button("▶️ Run Simulation", type="primary"):
                with st.spinner("Running Monte Carlo simulation..."):
                    simulator = RiskSimulator(st.session_state.portfolio)
                    # Run simulation with only supported parameters
                    result = simulator.run_monte_carlo_simulation(
                        num_simulations=int(num_simulations),
                        time_horizon=int(time_horizon)
                    )
                    
                    # Store results in session state
                    st.session_state.simulation_results = result
                    
                    # Display results in two columns
                    col1, col2 = st.columns(2)
                    
                    # Get the simulation summary
                    summary = result.get_summary()
                    
                    with col1:
                        st.metric("Expected Return", f"{summary['mean_return']*100:.2f}%")
                        # Get the 95% confidence interval
                        ci_95 = result.confidence_intervals.get(0.95, (0, 0))
                        st.metric("95% Confidence Interval", 
                                f"{ci_95[0]*100:.2f}% to {ci_95[1]*100:.2f}%")
                        st.metric("Value at Risk (95%)", 
                                f"{result.var.get(0.95, 0)*100:.2f}%")
                    
                    with col2:
                        st.metric("Volatility", f"{summary['volatility']*100:.2f}%")
                        st.metric("Sharpe Ratio", f"{summary['sharpe_ratio']:.2f}")
                        st.metric("Conditional VaR (95%)", 
                                f"{result.cvar.get(0.95, 0)*100:.2f}%")
                    
                    # Plot simulation results
                    fig = go.Figure()
                    
                    # Add histogram with a more professional look
                    fig.add_trace(go.Histogram(
                        x=result.simulated_returns * 100,
                        nbinsx=50,
                        name='Returns Distribution',
                        marker_color='#1f77b4',
                        opacity=0.8,
                        hovertemplate='Return: %{x:.2f}%<br>Count: %{y}'+
                                    '<extra></extra>'
                    ))
                    
                    # Add VaR lines with better styling
                    var_levels = [(0.95, '#FF6B6B'), (0.99, '#FF2B2B')]
                    for cl, color in var_levels:
                        var_pct = result.var.get(cl, 0) * 100
                        if var_pct < 0:  # Only add if VaR is negative (loss)
                            fig.add_vline(
                                x=var_pct,
                                line=dict(dash="dash", color=color, width=2),
                                annotation=dict(
                                    text=f"{int(cl*100)}% VaR: {var_pct:.2f}%",
                                    showarrow=True,
                                    arrowhead=1,
                                    y=0.9,
                                    yref="paper",
                                    yanchor="bottom",
                                    xanchor="right",
                                    font=dict(color=color)
                                ),
                                name=f"{int(cl*100)}% VaR"
                            )
                    
                    fig.update_layout(
                        title=dict(
                            text="Portfolio Returns Distribution",
                            x=0.5,
                            xanchor='center',
                            font=dict(size=18)
                        ),
                        xaxis_title="Portfolio Return (%)",
                        yaxis_title="Frequency",
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(t=50, b=50, l=50, r=50),
                        height=500
                    )
                    
                    st.plotly_chart(fig, config={'displayModeBar': True}, use_container_width=True, key='monte_carlo_histogram')
                    
                    # Add some space before the next section
                    st.markdown("---")
                    
                    # Show simulation paths if available
                    if hasattr(result, 'simulation_paths') and result.simulation_paths is not None:
                        st.subheader("Monte Carlo Simulation Paths")
                        
                        # Select number of paths to display
                        num_paths = st.slider("Number of paths to show", 10, 100, 20, 10)
                        
                        # Plot sample paths
                        fig_paths = go.Figure()
                        
                        # Add simulation paths
                        for i in range(min(num_paths, len(result.simulation_paths))):
                            fig_paths.add_trace(go.Scatter(
                                x=list(range(len(result.simulation_paths[i]))),
                                y=result.simulation_paths[i],
                                mode='lines',
                                line=dict(width=1, color='rgba(31, 119, 180, 0.1)'),
                                showlegend=False,
                                hoverinfo='none'
                            ))
                        
                        # Add mean path
                        mean_path = np.mean(result.simulation_paths[:num_paths], axis=0)
                        fig_paths.add_trace(go.Scatter(
                            x=list(range(len(mean_path))),
                            y=mean_path,
                            mode='lines',
                            line=dict(width=3, color='#FF2B2B'),
                            name='Average Path',
                            hovertemplate='Day %{x}<br>Value: %{y:,.2f}<extra></extra>'
                        ))
                        
                        # Add percentiles
                        for p in [5, 25, 50, 75, 95]:
                            pct = np.percentile(result.simulation_paths[:num_paths], p, axis=0)
                            fig_paths.add_trace(go.Scatter(
                                x=list(range(len(pct))),
                                y=pct,
                                mode='lines',
                                line=dict(width=1, dash='dash', color='#2B9C2B'),
                                name=f'{p}th Percentile',
                                showlegend=p in [5, 50, 95],  # Only show a few in legend
                                hovertemplate=f'P{p} - Day %{{x}}<br>Value: %{{y:,.2f}}<extra></extra>',
                                visible='legendonly' if p not in [5, 50, 95] else True
                            ))
                        
                        fig_paths.update_layout(
                            title=dict(
                                text="Portfolio Value Simulation Paths",
                                x=0.5,
                                xanchor='center',
                                font=dict(size=18)
                            ),
                            xaxis_title="Trading Days",
                            yaxis_title="Portfolio Value ($)",
                            showlegend=True,
                            plot_bgcolor='rgba(0,0,0,0.02)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            margin=dict(t=50, b=50, l=50, r=50),
                            height=500,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_paths, config={'displayModeBar': True}, use_container_width=True, key='monte_carlo_paths')
                        
                        # Add some space before the next section
                        st.markdown("---")
                        
                        # Add probability of reaching target
                        st.subheader("Probability of Reaching Investment Goals")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            target_return = st.number_input(
                                "Target Return (%)", 
                                min_value=0.0, 
                                max_value=1000.0, 
                                value=10.0, 
                                step=1.0
                            ) / 100
                            
                            # Calculate probability
                            prob = result.probability_of_return(target_return) * 100
                            
                            # Display with a nice gauge
                            fig_gauge = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = prob,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': f"Probability of {target_return*100:.1f}% Return"},
                                gauge = {
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': "#1f77b4"},
                                    'steps' : [
                                        {'range': [0, 33], 'color': "lightgray"},
                                        {'range': [33, 66], 'color': "darkgray"},
                                        {'range': [66, 100], 'color': "gray"}],
                                    'threshold' : {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': prob}
                                },
                                number = {'suffix': "%"}
                            ))
                            
                            fig_gauge.update_layout(
                                height=300,
                                margin=dict(t=50, b=10, l=50, r=50)
                            )
                            
                            st.plotly_chart(fig_gauge, config={'displayModeBar': True}, use_container_width=True, key='probability_gauge')
                        
                        with col2:
                            st.markdown("### Simulation Statistics")
                            
                            # Get the simulation summary
                            summary = result.get_summary()
                            ci_95 = result.confidence_intervals.get(0.95, (0, 0))
                            
                            # Calculate some statistics
                            stats = {
                                "Simulations Run": f"{num_simulations:,}",
                                "Time Horizon": f"{time_horizon} days",
                                "Expected Return": f"{summary['mean_return']*100:.2f}%",
                                "95% Confidence Interval": f"{ci_95[0]*100:.2f}% to {ci_95[1]*100:.2f}%",
                                "Volatility": f"{summary['volatility']*100:.2f}%",
                                "Sharpe Ratio": f"{summary['sharpe_ratio']:.2f}",
                                "Value at Risk (95%)": f"{result.var.get(0.95, 0)*100:.2f}%",
                                "Conditional VaR (95%)": f"{result.cvar.get(0.95, 0)*100:.2f}%"
                            }
                            
                            # Display stats in a nice table
                            for key, value in stats.items():
                                st.markdown(f"**{key}:** {value}")
                            
                            # Add some space
                            st.markdown("")
                            
                            # Add a download button for simulation data
                            if st.button("📥 Download Simulation Data"):
                                # Create a DataFrame with simulation results
                                import io
                                buffer = io.BytesIO()
                                
                                # Create a DataFrame with simulation paths
                                paths_df = pd.DataFrame(result.simulation_paths.T)
                                paths_df.columns = [f"Path_{i+1}" for i in range(paths_df.shape[1])]
                                paths_df.index.name = "Day"
                                
                                # Save to buffer
                                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                    # Save paths
                                    paths_df.to_excel(writer, sheet_name='Simulation Paths')
                                    
                                    # Save summary statistics
                                    summary = pd.DataFrame({
                                        'Metric': [
                                            'Initial Investment',
                                            'Expected Return',
                                            'Volatility',
                                            'Sharpe Ratio',
                                            'Probability of Loss',
                                            '5th Percentile',
                                            '95th Percentile'
                                        ],
                                        'Value': [
                                            initial_investment,
                                            result.expected_return,
                                            result.volatility,
                                            result.sharpe_ratio,
                                            result.probability_of_loss(),
                                            f"${(10000 * (1 + result.percentile(95)/100)):,.2f}" if hasattr(result, 'percentile') else "N/A",
                                            result.percentile(95)
                                        ]
                                    })
                                    summary.to_excel(writer, sheet_name='Summary', index=False)
                                
                                # Create download button
                                st.download_button(
                                    label="📥 Download Simulation Data (Excel)",
                                    data=buffer,
                                    file_name=f"monte_carlo_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
            
            elif st.session_state.get('simulation_results'):
                st.info("Previous simulation results loaded. Click 'Run Simulation' to generate new results.")
                
                # You could add code here to display previous results if needed
            else:
                st.info("Configure the simulation settings and click 'Run Simulation' to begin.")
        
        with risk_tab3:
            st.subheader("Value at Risk (VaR) Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                confidence_level = st.select_slider(
                    "Confidence Level",
                    options=[0.90, 0.95, 0.975, 0.99, 0.995],
                    value=0.95,
                    format_func=lambda x: f"{int(x*100)}%"
                )
                
                time_horizon_var = st.select_slider(
                    "Time Horizon",
                    options=[1, 5, 10, 21, 63, 252],
                    value=1,
                    format_func=lambda x: f"{x} day{'s' if x > 1 else ''}"
                )
                
                if st.button("🔄 Calculate VaR"):
                    with st.spinner("Calculating Value at Risk..."):
                        simulator = RiskSimulator(st.session_state.portfolio)
                        
                        # Calculate VaR using different methods
                        var_results = {}
                        
                        # Get historical returns for VaR calculation
                        try:
                            # Get historical prices for returns calculation
                            symbols = [asset.ticker for asset in st.session_state.portfolio.assets]
                            historical_data = st.session_state.data_manager.get_historical_prices(
                                symbols=symbols,
                                start_date=(datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d'),
                                end_date=datetime.now().strftime('%Y-%m-%d')
                            )
                            
                            if historical_data:
                                # Calculate portfolio returns from historical prices
                                portfolio_returns = []
                                dates = None
                                
                                # Get common date index
                                for symbol, df in historical_data.items():
                                    if df is not None and not df.empty and 'close' in df.columns:
                                        if dates is None:
                                            dates = df.index
                                        else:
                                            dates = dates.intersection(df.index)
                                
                                if dates is not None and len(dates) > 0:
                                    # Calculate weighted returns for each asset
                                    total_value = st.session_state.portfolio.get_total_value()
                                    weights = {asset.ticker: asset.current_value() / total_value 
                                              for asset in st.session_state.portfolio.assets}
                                    
                                    # Initialize portfolio returns with zeros
                                    portfolio_returns = pd.Series(0.0, index=dates[1:])
                                    
                                    for symbol, df in historical_data.items():
                                        if symbol in weights and df is not None and not df.empty and 'close' in df.columns:
                                            # Calculate daily returns for this asset
                                            asset_returns = df['close'].pct_change().dropna()
                                            # Add weighted returns to portfolio
                                            if not asset_returns.empty:
                                                portfolio_returns = portfolio_returns.add(
                                                    asset_returns.reindex(dates).dropna() * weights[symbol], 
                                                    fill_value=0
                                                )
                                    
                                    # Calculate VaR using historical simulation
                                    historical_var = simulator.calculate_historical_var(
                                        returns=portfolio_returns.values,
                                        confidence_level=confidence_level
                                    )
                                    var_results["Historical"] = historical_var
                                    
                                    # Calculate parametric VaR using portfolio statistics
                                    mean_return = portfolio_returns.mean()
                                    volatility = portfolio_returns.std()
                                    normal_var = simulator.calculate_parametric_var(
                                        mean_return=mean_return,
                                        volatility=volatility,
                                        confidence_level=confidence_level
                                    )
                                    var_results["Parametric (Normal)"] = normal_var
                                    
                        except Exception as e:
                            st.error(f"Error calculating VaR: {str(e)}")
                            import traceback
                            st.text(traceback.format_exc())
                        
                        # Monte Carlo VaR (if simulation results exist)
                        if 'simulation_results' in st.session_state and hasattr(st.session_state.simulation_results, 'var'):
                            mc_var = st.session_state.simulation_results.var.get(confidence_level, 0)
                            var_results["Monte Carlo"] = mc_var
                        
                        # Store results
                        st.session_state.var_results = var_results
            
            with col2:
                st.markdown("### About Value at Risk (VaR)")
                st.markdown("""
                **Value at Risk (VaR)** is a statistical measure that quantifies the level of financial risk 
                within a portfolio over a specific time frame. It estimates how much a set of investments 
                might lose, given normal market conditions, in a set time period with a given confidence level.
                
                - **Historical VaR**: Based on historical price movements
                - **Parametric VaR**: Assumes normal distribution of returns
                - **Monte Carlo VaR**: Based on simulated future price paths
                """)
            
            # Display VaR results if available
            if 'var_results' in st.session_state and st.session_state.var_results:
                st.markdown("### VaR Results")
                
                # Create a DataFrame for display
                var_df = pd.DataFrame(
                    [(method, f"{abs(var)*100:.2f}%") 
                     for method, var in st.session_state.var_results.items()],
                    columns=["Method", f"{int(confidence_level*100)}% VaR"]
                )
                
                # Display as a nice table
                st.dataframe(
                    var_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Method": "Calculation Method",
                        f"{int(confidence_level*100)}% VaR": f"{int(confidence_level*100)}% VaR (1-day)"
                    }
                )
                
                # Add some interpretation
                st.markdown("#### Interpretation")
                st.info(
                    f"There is a {int(confidence_level*100)}% confidence that your portfolio will not lose more than "
                    f"**{abs(st.session_state.var_results.get('Historical', 0))*100:.2f}%** in the next {time_horizon_var} day{'s' if time_horizon_var > 1 else ''} "
                    "based on historical data."
                )
                
                # Add a visualization comparing VaR methods
                if len(st.session_state.var_results) > 1:
                    fig_var = go.Figure(go.Bar(
                        x=list(st.session_state.var_results.keys()),
                        y=[abs(v)*100 for v in st.session_state.var_results.values()],
                        text=[f"{abs(v)*100:.2f}%" for v in st.session_state.var_results.values()],
                        textposition='auto',
                        marker_color='#1f77b4',
                        opacity=0.8
                    ))
                    
                    fig_var.update_layout(
                        title=f"{int(confidence_level*100)}% Value at Risk Comparison",
                        xaxis_title="Method",
                        yaxis_title=f"VaR ({time_horizon_var}-day) in %",
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_var, config={'displayModeBar': True}, use_container_width=True, key='var_chart')
        
        with risk_tab4:
            st.subheader("Asset Correlation Matrix")
            
            with st.spinner("Calculating correlations..."):
                try:
                    # Get historical returns for correlation
                    symbols = [asset.ticker for asset in st.session_state.portfolio.assets]
                    historical_data = st.session_state.data_manager.get_historical_prices(
                        symbols=symbols,
                        start_date=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                        end_date=datetime.now().strftime('%Y-%m-%d')
                    )
                    
                    if historical_data and len(historical_data) > 0:
                        # Combine all asset data into a single DataFrame for correlation
                        combined_data = None
                        for symbol, df in historical_data.items():
                            if df is not None and not df.empty:
                                try:
                                    df = df.rename(columns={'close': symbol})
                                    if symbol in df.columns:
                                        temp_df = pd.DataFrame({
                                            symbol: df[symbol]
                                        })
                                        if combined_data is None:
                                            combined_data = temp_df
                                        else:
                                            combined_data = combined_data.join(temp_df, how='outer')
                                except Exception as e:
                                    st.warning(f"Error processing data for {symbol}: {str(e)}")
                    
                    if combined_data is None or combined_data.empty:
                        st.warning("No valid price data available for correlation analysis.")
                    else:
                        # Forward fill and backfill any missing values
                        combined_data = combined_data.ffill().bfill()
                        
                        # Calculate daily returns
                        returns = combined_data.pct_change().dropna()
                        
                        # Calculate correlation matrix
                        corr_matrix = returns.corr()
                        
                        # Create heatmap
                        fig_corr = go.Figure(data=go.Heatmap(
                            z=corr_matrix.values,
                            x=corr_matrix.columns,
                            y=corr_matrix.index,
                            colorscale='RdBu_r',
                            zmin=-1,
                            zmax=1,
                            colorbar=dict(title='Correlation')
                        ))
                        
                        # Add annotations
                        annotations = []
                        for i, row in enumerate(corr_matrix.values):
                            for j, value in enumerate(row):
                                annotations.append(
                                    go.layout.Annotation(
                                        text=f"{value:.2f}",
                                        x=corr_matrix.columns[j],
                                        y=corr_matrix.index[i],
                                        xref='x1',
                                        yref='y1',
                                        showarrow=False,
                                        font=dict(color='white' if abs(value) > 0.5 else 'black')
                                    )
                                )
                        
                        fig_corr.update_layout(
                            title="Asset Correlation Matrix",
                            xaxis_title="",
                            yaxis_title="",
                            height=600,
                            annotations=annotations,
                            margin=dict(l=100, r=100, t=50, b=100)
                        )
                        
                        st.plotly_chart(fig_corr, config={'displayModeBar': True}, use_container_width=True, key='correlation_matrix')
                        
                        # Add some interpretation
                        st.markdown("### Correlation Interpretation")
                        st.markdown("""
                        - **+1**: Perfect positive correlation (assets move in the same direction)
                        - **0**: No correlation (assets move independently)
                        - **-1**: Perfect negative correlation (assets move in opposite directions)
                        
                        A well-diversified portfolio typically contains assets with low or negative correlations.
                        """)
                
                except Exception as e:
                    st.error(f"Error calculating correlations: {str(e)}")
                    st.exception(e)
            
            # Stress Testing
            st.subheader("Stress Testing")
            stress_scenarios = simulator.calculate_stress_scenarios()
            
            for scenario, results in stress_scenarios.items():
                st.metric(
                    f"Scenario: {scenario.replace('_', ' ').title()}",
                    f"-${-results['dollar_impact']:,.2f}",
                    f"{results['percent_impact']:.2f}%"
                )

# Tab 4: Efficient Frontier
with tab4:
    st.header("📊 Portfolio Optimization")
    
    if not st.session_state.portfolio.assets:
        st.warning("Please add assets to your portfolio to view the efficient frontier.")
    else:
        # Get current portfolio value
        total_value = st.session_state.portfolio.get_total_value()
        
        # Create columns for controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Time period selection
            time_period = st.selectbox(
                "Time Period",
                ["1M", "3M", "6M", "1Y", "3Y", "5Y", "Max"],
                index=3
            )
        
        with col2:
            # Number of simulations - reduced from 10,000 to 2,500 for better performance
            num_portfolios = st.slider(
                "Number of Portfolios",
                min_value=1000,
                max_value=10000,  # Reduced max from 20,000 to 10,000
                value=2500,       # Reduced default from 10,000 to 2,500
                step=500,
                help="More portfolios provide a smoother frontier but take longer to calculate"
            )
        
        with col3:
            # Get risk-free rate from DataManager
            try:
                risk_free_rate = st.session_state.data_manager.get_risk_free_rate()
                risk_free_rate_display = risk_free_rate * 100  # Convert to percentage for display
                
                # Allow manual override
                risk_free_rate_override = st.number_input(
                    "Risk-Free Rate (%)",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(f"{risk_free_rate_display:.2f}"),
                    step=0.1,
                    format="%.2f",
                    help=f"Current 10-year Treasury yield: {risk_free_rate_display:.2f}%"
                ) / 100  # Convert to decimal
                
                if abs(risk_free_rate_override - risk_free_rate) > 0.001:  # If user changed the value
                    risk_free_rate = risk_free_rate_override
                    st.session_state.portfolio.risk_free_rate = risk_free_rate
                
            except Exception as e:
                st.error(f"Could not fetch risk-free rate: {str(e)}")
                risk_free_rate = 0.04  # Fallback to 4%
                st.session_state.portfolio.risk_free_rate = risk_free_rate
        
        # Calculate time delta based on selection
        time_deltas = {
            "1M": 30,
            "3M": 90,
            "6M": 180,
            "1Y": 365,
            "3Y": 365 * 3,
            "5Y": 365 * 5,
            "Max": 365 * 20  # Cap at 20 years
        }
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_deltas[time_period])
        
        try:
            with st.spinner(f"Analyzing {len(st.session_state.portfolio.assets)} assets over {time_period} period..."):
                # Get historical prices for all assets in the portfolio
                symbols = [asset.ticker for asset in st.session_state.portfolio.assets]
                
                # Fetch historical data
                historical_data = st.session_state.data_manager.get_historical_prices(
                    symbols=symbols,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                
                if not historical_data:
                    st.error("No historical data returned. Please check your data source.")
                    st.stop()
                
                # Process historical data
                prices_df = pd.DataFrame()
                for ticker, df in historical_data.items():
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        close_col = next((col for col in df.columns if str(col).lower() == 'close'), None)
                        if close_col is not None:
                            df = df.rename(columns={close_col: 'close'})
                            prices_df[ticker] = df['close']
                
                if prices_df.empty:
                    st.error("Could not extract price data from the returned data.")
                    st.stop()
                
                # Ensure we have numeric data and drop missing values
                prices_df = prices_df.apply(pd.to_numeric, errors='coerce').dropna(how='all')
                
                if len(prices_df) < 10:  # Require at least 10 data points
                    st.error("Insufficient data points for analysis. Try a longer time period.")
                    st.stop()
                
                # Calculate daily returns
                returns = prices_df.pct_change().dropna()
                
                if len(returns) < 5:  # Require at least 5 return periods
                    st.error("Insufficient data to calculate returns. Try a longer time period.")
                    st.stop()
                
                # Calculate expected returns (annualized)
                expected_returns = returns.mean() * 252
                
                # Calculate covariance matrix (annualized)
                cov_matrix = returns.cov() * 252
                
                # Store results
                results = np.zeros((4, num_portfolios))  # volatility, return, sharpe, sortino
                weights_record = []
                
                # Calculate Sortino ratio helper function
                def calculate_sortino_ratio(returns, target_return=0, risk_free=0, periods_per_year=252):
                    excess_returns = returns - risk_free / periods_per_year
                    downside_returns = np.minimum(0, excess_returns - target_return / periods_per_year)
                    downside_volatility = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(periods_per_year)
                    
                    if downside_volatility == 0:
                        return 0
                    return (np.mean(excess_returns) * periods_per_year) / downside_volatility
                
                # Generate random portfolios
                for i in range(num_portfolios):
                    # Random weights that sum to 1
                    weights = np.random.random(len(symbols))
                    weights /= np.sum(weights)
                    weights_record.append(weights)
                    
                    # Portfolio return and volatility
                    portfolio_return = np.sum(weights * expected_returns)
                    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    
                    # Calculate portfolio returns for the period
                    portfolio_returns = returns.dot(weights)
                    
                    # Calculate Sharpe and Sortino ratios
                    sharpe_ratio = (portfolio_return - risk_free_rate) / (portfolio_volatility + 1e-10)
                    sortino_ratio = calculate_sortino_ratio(
                        portfolio_returns, 
                        target_return=risk_free_rate,
                        risk_free=risk_free_rate
                    )
                    
                    # Store results
                    results[0, i] = portfolio_volatility
                    results[1, i] = portfolio_return
                    results[2, i] = sharpe_ratio
                    results[3, i] = sortino_ratio
                
                # Calculate current portfolio metrics
                current_weights = np.array([
                    asset.current_value() / total_value 
                    for asset in st.session_state.portfolio.assets
                ])
                
                current_return = np.sum(current_weights * expected_returns)
                current_volatility = np.sqrt(np.dot(current_weights.T, np.dot(cov_matrix, current_weights)))
                current_portfolio_returns = returns.dot(current_weights)
                current_sharpe = (current_return - risk_free_rate) / (current_volatility + 1e-10)
                current_sortino = calculate_sortino_ratio(
                    current_portfolio_returns,
                    target_return=risk_free_rate,
                    risk_free=risk_free_rate
                )
                
                # Find optimal portfolios
                max_sharpe_idx = np.argmax(results[2])
                max_sortino_idx = np.argmax(results[3])
                min_vol_idx = np.argmin(results[0])
                
                # Get optimal portfolios
                optimal_portfolios = {
                    'Max Sharpe': {
                        'weights': weights_record[max_sharpe_idx],
                        'return': results[1, max_sharpe_idx],
                        'volatility': results[0, max_sharpe_idx],
                        'sharpe': results[2, max_sharpe_idx],
                        'sortino': results[3, max_sharpe_idx]
                    },
                    'Max Sortino': {
                        'weights': weights_record[max_sortino_idx],
                        'return': results[1, max_sortino_idx],
                        'volatility': results[0, max_sortino_idx],
                        'sharpe': results[2, max_sortino_idx],
                        'sortino': results[3, max_sortino_idx]
                    },
                    'Min Volatility': {
                        'weights': weights_record[min_vol_idx],
                        'return': results[1, min_vol_idx],
                        'volatility': results[0, min_vol_idx],
                        'sharpe': results[2, min_vol_idx],
                        'sortino': results[3, min_vol_idx]
                    }
                }
                
                # Create tabs for different views with emoji icons
                tab1, tab2 = st.tabs(["📈 Efficient Frontier", "📊 Portfolio Comparison"])
                
                with tab1:
                    # Create efficient frontier plot with secondary y-axis for Sharpe ratio
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Color scale for points based on Sharpe ratio
                    colors = results[2]  # Sharpe ratio for coloring
                    
                    # Add scatter plot of all portfolios with enhanced styling
                    scatter = go.Scatter(
                        x=results[0],
                        y=results[1] * 100,  # Convert to percentage
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=colors,
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(
                                title='Sharpe Ratio',
                                thickness=20,
                                y=0.5,
                                ypad=0,
                                len=0.6
                            ),
                            line=dict(width=0.5, color='DarkSlateGrey'),
                            opacity=0.8
                        ),
                        name='Simulated Portfolios',
                        hovertemplate=
                            '<b>Volatility:</b> %{x:.2f}%<br>' +
                            '<b>Return:</b> %{y:.2f}%<br>' +
                            '<b>Sharpe:</b> %{marker.color:.2f}' +
                            '<extra></extra>',
                        showlegend=False
                    )
                    fig.add_trace(scatter, secondary_y=False)
                    
                    # Add efficient frontier line (using convex hull)
                    points = np.column_stack((results[0], results[1] * 100))
                    hull = ConvexHull(points)
                    efficient_frontier = points[hull.vertices]
                    efficient_frontier = efficient_frontier[np.argsort(efficient_frontier[:, 0])]
                    
                    # Filter to only keep the efficient part (top of the convex hull)
                    max_so_far = -np.inf
                    efficient_mask = []
                    for i in range(len(efficient_frontier)):
                        if efficient_frontier[i, 1] > max_so_far:
                            max_so_far = efficient_frontier[i, 1]
                            efficient_mask.append(True)
                        else:
                            efficient_mask.append(False)
                    efficient_frontier = efficient_frontier[efficient_mask]
                    
                    # Add efficient frontier line
                    fig.add_trace(go.Scatter(
                        x=efficient_frontier[:, 0],
                        y=efficient_frontier[:, 1],
                        mode='lines',
                        line=dict(
                            color='rgba(255, 0, 0, 0.8)',
                            width=2.5,
                            dash='dash'
                        ),
                        name='Efficient Frontier',
                        hovertemplate='<b>Efficient Frontier</b><br>' +
                                    'Volatility: %{x:.2f}%<br>' +
                                    'Return: %{y:.2f}%<br>' +
                                    '<extra></extra>'
                    ), secondary_y=False)
                    
                    # Add Capital Market Line (CML) if we have a risk-free rate
                    if risk_free_rate > 0 and len(efficient_frontier) > 1:
                        # Find the market portfolio (tangency portfolio)
                        market_idx = np.argmax(results[2])  # Max Sharpe ratio
                        market_vol = results[0][market_idx]
                        market_ret = results[1][market_idx] * 100
                        
                        # Calculate CML points (from risk-free to 1.5x max volatility)
                        max_vol = max(market_vol * 1.5, max(results[0]) * 1.1)
                        cml_x = np.linspace(0, max_vol, 2)
                        cml_y = (risk_free_rate * 100) + (cml_x * (market_ret - (risk_free_rate * 100)) / market_vol)
                        
                        # Add CML line
                        fig.add_trace(go.Scatter(
                            x=cml_x,
                            y=cml_y,
                            mode='lines',
                            line=dict(
                                color='rgba(0, 150, 0, 0.7)',
                                width=2,
                                dash='dot'
                            ),
                            name='Capital Market Line',
                            hovertemplate='<b>CML</b><br>' +
                                        'Volatility: %{x:.2f}%<br>' +
                                        'Return: %{y:.2f}%<br>' +
                                        '<extra></extra>'
                        ), secondary_y=False)
                        
                        # Add market portfolio point
                        fig.add_trace(go.Scatter(
                            x=[market_vol],
                            y=[market_ret],
                            mode='markers+text',
                            marker=dict(
                                size=12,
                                color='gold',
                                line=dict(width=1, color='black')
                            ),
                            text=['Market Portfolio'],
                            textposition='top center',
                            name='Market Portfolio',
                            hovertemplate='<b>Market Portfolio</b><br>' +
                                        'Volatility: ' + f'{market_vol:.2f}%<br>' +
                                        'Return: ' + f'{market_ret:.2f}%<br>' +
                                        'Sharpe: ' + f'{results[2][market_idx]:.2f}' +
                                        '<extra></extra>',
                            showlegend=False
                        ), secondary_y=False)
                    
                    # Add optimal portfolios
                    for label, portfolio in optimal_portfolios.items():
                        fig.add_trace(go.Scatter(
                            x=[portfolio['volatility']],
                            y=[portfolio['return'] * 100],
                            mode='markers+text',
                            marker=dict(
                                size=14,
                                line=dict(width=2, color='DarkSlateGrey')
                            ),
                            text=[label],
                            textposition='bottom center',
                            textfont=dict(size=10, color='black'),
                            hovertemplate=f'<b>{label} Portfolio</b><br>' +
                                        f'Return: {portfolio["return"]*100:.2f}%<br>' +
                                        f'Volatility: {portfolio["volatility"]:.2f}%<br>' +
                                        f'Sharpe: {portfolio["sharpe"]:.2f}<br>' +
                                        f'Sortino: {portfolio["sortino"]:.2f}<extra></extra>'
                        ), secondary_y=False)
                    
                    # Add current portfolio
                    fig.add_trace(go.Scatter(
                        x=[current_volatility],
                        y=[current_return * 100],
                        mode='markers+text',
                        marker=dict(
                            color='red',
                            size=16,
                            symbol='diamond',
                            line=dict(width=2, color='black')
                        ),
                        text=['Current'],
                        textposition='bottom center',
                        textfont=dict(size=10, color='black'),
                        hovertemplate='<b>Current Portfolio</b><br>' +
                                    f'Return: {current_return*100:.2f}%<br>' +
                                    f'Volatility: {current_volatility:.2f}%<br>' +
                                    f'Sharpe: {current_sharpe:.2f}<extra></extra>'
                    ), secondary_y=False)
                    
                    # Create custom hover behavior
                    fig.update_traces(
                        hoverlabel=dict(
                            bgcolor='white',
                            font_size=12,
                            font_family='Arial'
                        )
                    )
                    
                    # Update layout with simplified configuration
                    layout = go.Layout(
                        title=dict(
                            text='Efficient Frontier Analysis',
                            x=0.5,
                            xanchor='center',
                            y=0.97,
                            font=dict(size=22, family='Arial', color='#2c3e50')
                        ),
                        xaxis=dict(
                            title=dict(
                                text='Annualized Volatility (%)',
                                font=dict(size=14, family='Arial')
                            ),
                            showgrid=True,
                            gridcolor='rgba(200, 200, 200, 0.3)',
                            tickformat=".0%"
                        ),
                        yaxis=dict(
                            title=dict(
                                text='Annualized Return (%)',
                                font=dict(size=14, family='Arial')
                            ),
                            showgrid=True,
                            gridcolor='rgba(200, 200, 200, 0.3)',
                            tickformat=".0%"
                        )
                    )
                    fig.layout = layout
                    
                    # Add annotations for key metrics
                    annotations = []
                    
                    # Add risk-free rate line
                    max_vol = max(results[0])
                    max_ret = max(results[1])
                    
                    fig.add_shape(
                        type="line",
                        x0=0, y0=risk_free_rate, x1=max_vol, y1=risk_free_rate,
                        line=dict(color="RoyalBlue", width=2, dash="dash"),
                        name=f"Risk-Free Rate ({risk_free_rate:.1%})"
                    )
                    
                    # Add capital market line (tangency line)
                    if max_sharpe_idx >= 0 and results[0, max_sharpe_idx] > 0:
                        # Calculate CML points
                        x_cml = [0, results[0, max_sharpe_idx] * 1.1]
                        y_cml = [risk_free_rate, risk_free_rate + results[2, max_sharpe_idx] * x_cml[1]]
                        
                        fig.add_trace(go.Scatter(
                            x=x_cml,
                            y=y_cml,
                            mode='lines',
                            name='Capital Market Line',
                            line=dict(color='#FF7F0E', width=2, dash='dot'),
                            hovertemplate='<b>CML</b><extra></extra>'
                        ))
                    
                    # Update layout
                    fig.update_layout(
                        height=600,
                        width=800,
                        margin=dict(t=50, b=50, l=50, r=50)
                    )
                    
                    st.plotly_chart(fig, config={'displayModeBar': True}, use_container_width=True, use_container_height=True, key='efficient_frontier_plot')
                    
                    # Create DataFrame and display
                    allocation_df = pd.DataFrame(allocation_data)
                    
                    # Format the DataFrame for display
                    display_df = allocation_df.copy()
                    
                    # Check if columns exist before trying to access them
                    if 'Return' in display_df.columns:
                        display_df['Return'] = (display_df['Return'] * 100).round(2).astype(str) + '%'
                    if 'Volatility' in display_df.columns:
                        display_df['Volatility'] = (display_df['Volatility'] * 100).round(2).astype(str) + '%'
                    if 'Sharpe' in display_df.columns:
                        display_df['Sharpe'] = display_df['Sharpe'].round(2)
                    if 'Sortino' in display_df.columns:
                        display_df['Sortino'] = display_df['Sortino'].round(2)
                    
                    # Create current portfolio row with proper error handling
                    current_row_data = {'Portfolio': 'Current'}
                    
                    # Only add columns that exist in the display_df
                    if 'Return' in display_df.columns:
                        current_row_data['Return'] = f"{current_return*100:.2f}%"
                    if 'Volatility' in display_df.columns:
                        current_row_data['Volatility'] = f"{current_volatility*100:.2f}%"
                    if 'Sharpe' in display_df.columns:
                        current_row_data['Sharpe'] = round(current_sharpe, 2)
                    if 'Sortino' in display_df.columns:
                        current_row_data['Sortino'] = round(current_sortino, 2)
                        
                    current_row = pd.DataFrame([current_row_data])
                    
                    display_df = pd.concat([current_row, display_df]).reset_index(drop=True)
                    
                    # Display the table with better formatting
                    st.dataframe(
                        display_df,
                        column_config={
                            'Portfolio': st.column_config.TextColumn('Portfolio', width='medium'),
                            'Return': st.column_config.TextColumn('Return', help='Annualized expected return'),
                            'Volatility': st.column_config.TextColumn('Volatility', help='Annualized volatility (risk)'),
                            'Sharpe': st.column_config.NumberColumn('Sharpe Ratio', format='%.2f', 
                                                                  help='Risk-adjusted return (higher is better)'),
                            'Sortino': st.column_config.NumberColumn('Sortino Ratio', format='%.2f',
                                                                   help='Downside risk-adjusted return (higher is better)')
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Add some space
                    st.markdown("---")
                    
                    # Display detailed allocations in an expander
                    with st.expander("View Detailed Allocations", expanded=False):
                        # Create a figure for asset allocations
                        fig_allocation = make_subplots(
                            rows=1, 
                            cols=len(optimal_portfolios) + 1, 
                            subplot_titles=['Current'] + list(optimal_portfolios.keys()),
                            specs=[[{"type": "pie"} for _ in range(len(optimal_portfolios) + 1)]]
                        )
                        
                        # Add current portfolio allocation
                        fig_allocation.add_trace(
                            go.Pie(
                                labels=symbols,
                                values=current_weights * 100,
                                name='Current',
                                textinfo='percent+label',
                                textposition='inside',
                                hole=0.4,
                                hovertemplate='<b>%{label}</b><br>%{percent:.1%}<extra></extra>',
                                showlegend=False
                            ),
                            row=1, col=1
                        )
                        
                        # Add optimal portfolios
                        for i, (label, portfolio) in enumerate(optimal_portfolios.items(), 2):
                            fig_allocation.add_trace(
                                go.Pie(
                                    labels=symbols,
                                    values=portfolio['weights'] * 100,
                                    name=label,
                                    textinfo='percent+label',
                                    textposition='inside',
                                    hole=0.4,
                                    hovertemplate='<b>%{label}</b><br>%{percent:.1%}<extra></extra>',
                                    showlegend=False
                                ),
                                row=1, col=i
                            )
                        
                        # Update layout
                        fig_allocation.update_layout(
                            title_text="Asset Allocation by Portfolio Strategy",
                            height=500,
                            showlegend=False,
                            margin=dict(t=80, b=20, l=20, r=20)
                        )
                        
                        st.plotly_chart(fig_allocation, config={'displayModeBar': True}, use_container_width=True, key='allocation_chart')
                        
                        # Add a data table with exact percentages
                        st.subheader("Detailed Weightings")
                        
                        # Create a DataFrame with all allocations
                        weight_data = {'Asset': symbols}
                        
                        # Add current weights
                        weight_data['Current'] = (current_weights * 100).round(2)
                        
                        # Add optimal portfolio weights
                        for label, portfolio in optimal_portfolios.items():
                            weight_data[label] = (portfolio['weights'] * 100).round(2)
                        
                        weight_df = pd.DataFrame(weight_data)
                        
                        # Display the table
                        st.dataframe(
                            weight_df,
                            column_config={
                                'Asset': st.column_config.TextColumn('Asset'),
                                **{
                                    col: st.column_config.NumberColumn(
                                        col,
                                        format='%.2f%%',
                                        min_value=0,
                                        max_value=100
                                    ) for col in weight_df.columns if col != 'Asset'
                                }
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # Add download button
                        csv = weight_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Allocation Data",
                            data=csv,
                            file_name=f"portfolio_allocations_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                
                with tab2:
                    st.subheader("Portfolio Comparison")
                    
                    # Create a DataFrame for comparison
                    compare_data = []
                    
                    # Add current portfolio
                    compare_data.append({
                        'Portfolio': 'Current',
                        'Type': 'Current',
                        'Return': current_return,
                        'Volatility': current_volatility,
                        'Sharpe': current_sharpe,
                        'Sortino': current_sortino
                    })
                    
                    # Add optimal portfolios
                    for label, portfolio in optimal_portfolios.items():
                        compare_data.append({
                            'Portfolio': label,
                            'Type': 'Optimized',
                            'Return': portfolio['return'],
                            'Volatility': portfolio['volatility'],
                            'Sharpe': portfolio['sharpe'],
                            'Sortino': portfolio['sortino']
                        })
                    
                    compare_df = pd.DataFrame(compare_data)
                    
                    # Create tabs for different comparison views
                    overview_tab, radar_tab, metrics_tab, rec_tab = st.tabs([
                        "📊 Overview", 
                        "📈 Radar Chart", 
                        "📋 Metrics Table", 
                        "💡 Recommendations"
                    ])
                    
                    with overview_tab:
                        st.markdown("### Portfolio Performance Overview")
                        
                        # Create a summary figure with subplots
                        fig_overview = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=(
                                'Annualized Return (%)', 
                                'Annualized Volatility (%)',
                                'Sharpe Ratio', 
                                'Sortino Ratio'
                            ),
                            vertical_spacing=0.15,
                            horizontal_spacing=0.1
                        )
                        
                        # Add traces for each metric
                        metrics = ['Return', 'Volatility', 'Sharpe', 'Sortino']
                        titles = ['Return', 'Volatility', 'Sharpe', 'Sortino']
                        
                        for i, (metric, title) in enumerate(zip(metrics, titles)):
                            row = (i // 2) + 1
                            col = (i % 2) + 1
                            
                            # Sort by the current metric (descending, except for Volatility)
                            if metric == 'Volatility':
                                sorted_df = compare_df.sort_values(by=metric, ascending=True)
                            else:
                                sorted_df = compare_df.sort_values(by=metric, ascending=False)
                            
                            # Create the bar chart
                            fig_overview.add_trace(
                                go.Bar(
                                    x=sorted_df['Portfolio'],
                                    y=sorted_df[metric] * (100 if metric in ['Return', 'Volatility'] else 1),
                                    name=title,
                                    text=[f"{x:.1f}%" if metric in ['Return', 'Volatility'] else f"{x:.2f}" 
                                          for x in sorted_df[metric] * (100 if metric in ['Return', 'Volatility'] else 1)],
                                    textposition='auto',
                                    marker_color=['#1f77b4' if p == 'Current' else '#ff7f0e' for p in sorted_df['Portfolio']]
                                ),
                                row=row, col=col
                            )
                            
                            # Update y-axis titles
                            yaxis_title = '%' if metric in ['Return', 'Volatility'] else ''
                            fig_overview.update_yaxes(title_text=yaxis_title, row=row, col=col)
                        
                        # Update layout
                        fig_overview.update_layout(
                            height=700,
                            showlegend=False,
                            margin=dict(t=50, b=50, l=50, r=50),
                            title_text="Portfolio Performance Comparison",
                            title_x=0.5
                        )
                        
                        st.plotly_chart(fig_overview, config={'displayModeBar': True}, use_container_width=True, key='portfolio_overview')
                    
                    with radar_tab:
                        st.markdown("### Radar Chart Comparison")
                        st.markdown("Compare portfolios across multiple metrics (normalized 0-1 scale)")
                        
                        # Create radar chart for comparison
                        categories = ['Return', 'Volatility', 'Sharpe', 'Sortino']
                        
                        # Normalize values for radar chart (0-1)
                        radar_df = compare_df.copy()
                        for col in categories:
                            if col != 'Volatility':  # Higher is better
                                radar_df[col] = (radar_df[col] - radar_df[col].min()) / (radar_df[col].max() - radar_df[col].min() + 1e-10)
                            else:  # Lower is better for volatility
                                radar_df[col] = 1 - (radar_df[col] - radar_df[col].min()) / (radar_df[col].max() - radar_df[col].min() + 1e-10)
                        
                        # Create radar chart
                        fig_radar = go.Figure()
                        
                        # Add traces for each portfolio
                        for _, row in radar_df.iterrows():
                            fig_radar.add_trace(go.Scatterpolar(
                                r=[row[col] for col in categories],
                                theta=categories,
                                name=row['Portfolio'],
                                fill='toself',
                                opacity=0.7
                            ))
                        
                        # Update layout
                        fig_radar.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )
                            ),
                            showlegend=True,
                            title='Portfolio Comparison (Normalized)',
                            height=600,
                            legend=dict(
                                orientation='h',
                                yanchor='bottom',
                                y=1.1,
                                xanchor='center',
                                x=0.5
                            )
                        )
                        
                        st.plotly_chart(fig_radar, config={'displayModeBar': True}, use_container_width=True, key='radar_chart')
                    
                    with metrics_tab:
                        st.markdown("### Detailed Performance Metrics")
                        
                        # Format the comparison table
                        display_compare = compare_df.copy()
                        display_compare['Return'] = (display_compare['Return'] * 100).round(2)
                        display_compare['Volatility'] = (display_compare['Volatility'] * 100).round(2)
                        
                        # Create a styled DataFrame
                        def color_metrics(val, col):
                            if col in ['Return', 'Sharpe', 'Sortino']:
                                return 'color: green' if val == display_compare[col].max() else 'color: black'
                            elif col == 'Volatility':
                                return 'color: red' if val == display_compare[col].max() else 'color: black'
                            return ''
                        
                        # Apply styling
                        styled_df = display_compare.style.apply(
                            lambda x: [color_metrics(val, x.name) for val in x],
                            subset=['Return', 'Volatility', 'Sharpe', 'Sortino']
                        ).format({
                            'Return': '{:.2f}%',
                            'Volatility': '{:.2f}%',
                            'Sharpe': '{:.2f}',
                            'Sortino': '{:.2f}'
                        })
                        
                        # Display the styled table
                        st.dataframe(
                            styled_df,
                            column_config={
                                'Portfolio': st.column_config.TextColumn('Portfolio', width='medium'),
                                'Type': st.column_config.TextColumn('Type', width='small'),
                                'Return': st.column_config.NumberColumn('Return (%)', format='%.2f', 
                                                                      help='Annualized expected return'),
                                'Volatility': st.column_config.NumberColumn('Volatility (%)', format='%.2f',
                                                                          help='Annualized volatility (risk)'),
                                'Sharpe': st.column_config.NumberColumn('Sharpe Ratio', format='%.2f', 
                                                                      help='Risk-adjusted return (higher is better)'),
                                'Sortino': st.column_config.NumberColumn('Sortino Ratio', format='%.2f',
                                                                       help='Downside risk-adjusted return (higher is better)')
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # Add explanation
                        with st.expander("ℹ️ How to interpret these metrics"):
                            st.markdown("""
                            - **Return (%):** Annualized expected return of the portfolio
                            - **Volatility (%):** Annualized standard deviation of returns (risk)
                            - **Sharpe Ratio:** Measures risk-adjusted return (higher is better)
                            - **Sortino Ratio:** Similar to Sharpe but only considers downside risk
                            - **Best values** are highlighted in green (or red for Volatility)
                            """)
                    
                    with rec_tab:
                        st.markdown("### Portfolio Recommendations")
                        
                        try:
                            # Check if we have the necessary data
                            if not hasattr(st.session_state, 'portfolio') or not st.session_state.portfolio.assets:
                                st.warning("No portfolio data available. Please add assets to your portfolio first.")
                            elif 'compare_df' not in locals() or compare_df.empty:
                                st.warning("Optimization data not available. Please run the portfolio optimization first.")
                            else:
                                # Get the best portfolio based on Sharpe ratio
                                best_portfolio = compare_df.loc[compare_df['Sharpe'].idxmax()]
                                
                                if best_portfolio['Portfolio'] == 'Current':
                                    st.success("🎉 Your current portfolio is already well-optimized with the highest Sharpe ratio!")
                                    
                                    # Show current allocation
                                    st.markdown("#### Your Current Allocation")
                                    current_weights = [asset.quantity * asset.current_price / st.session_state.portfolio.get_total_value() 
                                                     for asset in st.session_state.portfolio.assets]
                                    current_allocation = pd.DataFrame({
                                        'Asset': [asset.ticker for asset in st.session_state.portfolio.assets],
                                        'Current Weight': [f"{w*100:.1f}%" for w in current_weights]
                                    })
                                    st.table(current_allocation)
                                    
                                    # Calculate and display current metrics
                                    try:
                                        if 'portfolio_values' in locals() and not portfolio_values.empty and 'Total' in portfolio_values.columns:
                                            returns = portfolio_values['Total'].pct_change().dropna()
                                            if not returns.empty:
                                                current_return = (1 + returns).prod() ** (252/len(returns)) - 1  # Annualized return
                                                current_volatility = returns.std() * np.sqrt(252)  # Annualized volatility
                                                risk_free_rate = 0.02  # Default risk-free rate, can be adjusted
                                                current_sharpe = (current_return - risk_free_rate) / current_volatility if current_volatility > 0 else 0
                                                
                                                st.metric("Current Return", f"{current_return*100:.2f}%")
                                                st.metric("Current Volatility", f"{current_volatility*100:.2f}%")
                                                st.metric("Current Sharpe Ratio", f"{current_sharpe:.2f}")
                                    except Exception as metric_error:
                                        st.warning("Could not calculate all performance metrics. Some data may be missing.")
                                        st.text(f"Error details: {str(metric_error)}")
                                
                                # Show optimal weights if available
                                if 'optimal_weights' in locals() and optimal_weights is not None:
                                    with col2:
                                        st.subheader("Optimal Portfolio Weights")
                                        try:
                                            optimal_weights_df = pd.DataFrame({
                                                'Asset': symbols,
                                                'Optimal Weight': [f"{w*100:.1f}%" for w in optimal_weights]
                                            })
                                            st.table(optimal_weights_df)
                                            
                                            # Display optimal portfolio metrics if available
                                            if 'optimal_return' in locals():
                                                st.metric("Expected Return", f"{optimal_return*100:.2f}%")
                                            if 'optimal_volatility' in locals():
                                                st.metric("Expected Volatility", f"{optimal_volatility*100:.2f}%")
                                            if 'optimal_sharpe' in locals():
                                                st.metric("Sharpe Ratio", f"{optimal_sharpe:.2f}")
                                        except Exception as optimal_error:
                                            st.warning("Could not display optimal portfolio details.")
                                            st.text(f"Error details: {str(optimal_error)}")
                                else:
                                    st.info("Run the portfolio optimization to see recommended allocations.")
                        except Exception as e:
                            st.error("Could not generate portfolio recommendations due to an error.")
                            st.text(f"Error details: {str(e)}")
                            import traceback
                            st.text(traceback.format_exc())
                            st.warning("Could not display portfolio details due to incomplete data.")
                        
                        # Add some space
                        st.markdown("---")
                        
                        # Add a section for risk metrics
                        st.subheader("Risk Metrics")
                        
                        # Calculate additional risk metrics
                        def calculate_risk_metrics(returns, risk_free_rate=0.0):
                            metrics = {}
                            
                            # Annualized metrics
                            metrics['Annualized Return'] = np.mean(returns) * 252
                            metrics['Annualized Volatility'] = np.std(returns) * np.sqrt(252)
                            metrics['Max Drawdown'] = (1 - (1 + returns).cumprod() / (1 + returns).cumprod().cummax()).max()
                            
                            # Risk-adjusted returns
                            metrics['Sharpe Ratio'] = (metrics['Annualized Return'] - risk_free_rate) / (metrics['Annualized Volatility'] + 1e-10)
                            
                            # Sortino ratio (using risk-free rate as target)
                            downside_returns = np.minimum(0, returns - risk_free_rate/252)
                            downside_volatility = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)
                            metrics['Sortino Ratio'] = (metrics['Annualized Return'] - risk_free_rate) / (downside_volatility + 1e-10)
                            
                            # Value at Risk (95% confidence)
                            metrics['Value at Risk (95%)'] = np.percentile(returns, 5) * np.sqrt(252)
                            
                            # Conditional Value at Risk (95% confidence)
                            metrics['CVaR (95%)'] = returns[returns <= np.percentile(returns, 5)].mean() * np.sqrt(252)
                            
                            return metrics
                        
                        # Calculate metrics for current and optimal portfolios
                        risk_metrics = {}
                        
                        # Current portfolio
                        current_port_returns = returns.dot(current_weights)
                        risk_metrics['Current'] = calculate_risk_metrics(current_port_returns, risk_free_rate)
                        
                        # Optimal portfolios
                        for label, portfolio in optimal_portfolios.items():
                            port_returns = returns.dot(portfolio['weights'])
                            risk_metrics[label] = calculate_risk_metrics(port_returns, risk_free_rate)
                        
                        # Create a DataFrame for display
                        risk_df = pd.DataFrame(risk_metrics).T
                        
                        # Format the DataFrame
                        display_risk = risk_df.copy()
                        display_risk['Annualized Return'] = (display_risk['Annualized Return'] * 100).round(2).astype(str) + '%'
                        display_risk['Annualized Volatility'] = (display_risk['Annualized Volatility'] * 100).round(2).astype(str) + '%'
                        display_risk['Max Drawdown'] = (display_risk['Max Drawdown'] * 100).round(2).astype(str) + '%'
                        display_risk['Sharpe Ratio'] = display_risk['Sharpe Ratio'].round(2)
                        display_risk['Sortino Ratio'] = display_risk['Sortino Ratio'].round(2)
                        display_risk['Value at Risk (95%)'] = (display_risk['Value at Risk (95%)'] * 100).round(2).astype(str) + '%'
                        display_risk['CVaR (95%)'] = (display_risk['CVaR (95%)'] * 100).round(2).astype(str) + '%'
                        
                        # Reset index to show portfolio names as a column
                        display_risk = display_risk.reset_index().rename(columns={'index': 'Portfolio'})
                        
                        # Display the table with better formatting
                        st.dataframe(
                            display_risk,
                            column_config={
                                'Portfolio': st.column_config.TextColumn('Portfolio'),
                                'Annualized Return': st.column_config.TextColumn('Return'),
                                'Annualized Volatility': st.column_config.TextColumn('Volatility'),
                                'Max Drawdown': st.column_config.TextColumn('Max Drawdown'),
                                'Sharpe Ratio': st.column_config.NumberColumn('Sharpe', format='%.2f'),
                                'Sortino Ratio': st.column_config.NumberColumn('Sortino', format='%.2f'),
                                'Value at Risk (95%)': st.column_config.TextColumn('VaR (95%)'),
                                'CVaR (95%)': st.column_config.TextColumn('CVaR (95%)')
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # Add some space
                        st.markdown("---")
                        
                        # Add a section for optimization constraints
                        st.subheader("Optimization Settings")
                        
                        # Add optimization constraints
                        st.markdown("""
                        The efficient frontier is calculated using the following settings:
                        
                        - **Time Period:** {} ({} days)
                        - **Risk-Free Rate:** {:.2f}%
                        - **Number of Simulated Portfolios:** {:,}
                        - **Optimization Objectives:**
                          - Maximum Sharpe Ratio
                          - Maximum Sortino Ratio
                          - Minimum Volatility
                          
                        The analysis assumes:
                        - No transaction costs
                        - No taxes
                        - No short selling
                        - No leverage
                        - No position limits
                        
                        For more accurate results, consider:
                        - Using a longer time period
                        - Adjusting the risk-free rate
                        - Adding more assets to the portfolio
                        - Including transaction costs and taxes
                        """.format(
                            time_period,
                            time_deltas[time_period],
                            risk_free_rate * 100,
                            num_portfolios
                        ))
                        
                        # Add a prominent disclaimer
                        st.markdown("---")
                        with st.container():
                            st.markdown("""
                            <div style="background-color: rgba(255, 229, 143, 0.2); 
                                        padding: 15px; 
                                        border-radius: 10px;
                                        border-left: 5px solid #ffc107;
                                        margin: 10px 0;">
                                <h4 style="color: #ffc107; margin-top: 0;">
                                    <i class="fas fa-exclamation-triangle"></i> Important Notice
                                </h4>
                                <p style="margin-bottom: 0;">
                                    Portfolio optimization is based on historical returns and volatility, 
                                    which may not be indicative of future performance. Past performance 
                                    is not a guarantee of future results.
                                </p>
                                <p style="margin: 10px 0 0 0; font-weight: 500;">
                                    <i class="fas fa-check-circle"></i> Always consider your risk tolerance
                                </p>
                                <p style="margin: 5px 0 0 0; font-weight: 500;">
                                    <i class="fas fa-check-circle"></i> Review your investment objectives
                                </p>
                                <p style="margin: 5px 0 0 0; font-weight: 500;">
                                    <i class="fas fa-check-circle"></i> Consult with a financial advisor if needed
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown("---")
        
        except Exception as e:
            st.error(f"An error occurred while calculating the efficient frontier: {str(e)}")
            st.exception(e)  # This will show the full traceback in the app
            
            # Try to display the plot if it was created
            try:
                if 'fig' in locals():
                    st.plotly_chart(fig, config={'displayModeBar': True}, use_container_width=True, key="efficient_frontier_error")
            except Exception as plot_error:
                st.error(f"Error displaying plot: {str(plot_error)}")
            
            # Check if we have the required variables before trying to display them
            if 'symbols' in locals() and 'current_weights' in locals():
                # Display current portfolio weights if available
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Current Portfolio Weights")
                    try:
                        current_weights_df = pd.DataFrame({
                            'Asset': symbols,
                            'Current Weight': [f"{w*100:.2f}%" for w in current_weights]
                        })
                        st.table(current_weights_df)
                    except Exception as weight_error:
                        st.error(f"Error displaying current weights: {str(weight_error)}")
                    
                    # Display current metrics if available
                    try:
                        if 'current_return' in locals():
                            st.metric("Current Return", f"{current_return*100:.2f}%")
                        if 'current_volatility' in locals():
                            st.metric("Current Volatility", f"{current_volatility*100:.2f}%")
                        if 'current_sharpe' in locals():
                            st.metric("Current Sharpe Ratio", f"{current_sharpe:.2f}")
                    except Exception as metric_error:
                        st.error(f"Error displaying current metrics: {str(metric_error)}")
                
                # Only show optimal weights if they're available
                if 'optimal_weights' in locals():
                    with col2:
                        st.subheader("Optimal Portfolio Weights")
                        try:
                            optimal_weights_df = pd.DataFrame({
                                'Asset': symbols,
                                'Optimal Weight': [f"{w*100:.1f}%" for w in optimal_weights]
                            })
                            st.table(optimal_weights_df)
                            
                            if 'optimal_return' in locals():
                                st.metric("Expected Return", f"{optimal_return*100:.2f}%")
                            if 'optimal_volatility' in locals():
                                st.metric("Expected Volatility", f"{optimal_volatility*100:.2f}%")
                            if 'optimal_sharpe' in locals():
                                st.metric("Sharpe Ratio", f"{optimal_sharpe:.2f}")
                        except Exception as optimal_error:
                            st.error(f"Error displaying optimal weights: {str(optimal_error)}")
            else:
                st.warning("Could not display portfolio details due to incomplete data.")
                
        except Exception as e:
            st.error(f"Error calculating efficient frontier: {str(e)}")
            import traceback
            st.text(traceback.format_exc())


# Run the app
if __name__ == "__main__":
    st.write("SPARS is running...")
