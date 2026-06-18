"""
Market Data Module for SPARS (Smart Portfolio Analysis and Risk System)
Handles fetching real-time and historical market data from Polygon.io
"""
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

load_dotenv()

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Polygon.io API Configuration
POLYGON_BASE_URL = "https://api.polygon.io"
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")

# Configure retry strategy
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)

class MarketDataFetcher:
    """Handles fetching market data from Polygon.io API"""
    
    def __init__(self, api_key: str = None):
        """Initialize with API key"""
        self.api_key = api_key or POLYGON_API_KEY
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create a session with retry strategy"""
        session = requests.Session()
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        return session
    
    def _make_request(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make a request to the Polygon API"""
        url = f"{POLYGON_BASE_URL}{endpoint}"
        params = params or {}
        params["apiKey"] = self.api_key
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data from {url}: {str(e)}")
            raise
    
    def get_stock_details(self, ticker: str) -> dict:
        """Get details for a specific stock"""
        endpoint = f"/v3/reference/tickers/{ticker}"
        return self._make_request(endpoint)
    
    def get_stock_aggregates(
        self, 
        ticker: str, 
        from_date: str, 
        to_date: str, 
        timespan: str = "day"
    ) -> pd.DataFrame:
        """
        Get aggregate bars for a stock over a given date range
        
        Args:
            ticker: Stock ticker symbol
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            timespan: Timespan for aggregation (minute, hour, day, week, month, quarter, year)
            
        Returns:
            DataFrame with OHLCV data
        """
        endpoint = f"/v2/aggs/ticker/{ticker}/range/1/{timespan}/{from_date}/{to_date}"
        data = self._make_request(endpoint)
        
        if data.get("results"):
            df = pd.DataFrame(data["results"])
            df["t"] = pd.to_datetime(df["t"], unit="ms")
            df = df.rename(columns={
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "vw": "vwap"
            })
            return df
        return pd.DataFrame()
    
    def get_dividends(self, ticker: str) -> pd.DataFrame:
        """Get dividend history for a stock"""
        endpoint = f"/v3/reference/dividends"
        params = {"ticker": ticker, "limit": 1000}
        data = self._make_request(endpoint, params)
        
        if data.get("results"):
            return pd.DataFrame(data["results"])
        return pd.DataFrame()
    
    def get_previous_close(self, ticker: str) -> dict:
        """Get the previous day's close price"""
        endpoint = f"/v2/aggs/ticker/{ticker}/prev"
        data = self._make_request(endpoint)
        return data.get("results", [{}])[0] if data.get("results") else {}
    
    def get_intraday_prices(
        self, 
        ticker: str, 
        date: str, 
        interval: str = "1"
    ) -> pd.DataFrame:
        """
        Get intraday prices for a stock on a specific date
        
        Args:
            ticker: Stock ticker symbol
            date: Date in YYYY-MM-DD format
            interval: Time interval in minutes (1, 5, 15, 30, 60)
            
        Returns:
            DataFrame with intraday price data
        """
        endpoint = f"/v1/open-close/crypto/{ticker}/{date}"
        params = {"adjusted": "true", "unadjusted": "false"}
        data = self._make_request(endpoint, params)
        
        if data.get("status") == "OK":
            return pd.DataFrame([{
                "symbol": data["symbol"],
                "date": data["from"],
                "open": data["open"],
                "high": data["high"],
                "low": data["low"],
                "close": data["close"],
                "volume": data["volume"]
            }])
        return pd.DataFrame()
    
    def get_market_status(self) -> dict:
        """Get the current market status (open/closed)"""
        endpoint = "/v1/marketstatus/now"
        return self._make_request(endpoint)
    
    def get_ticker_news(
        self, 
        ticker: str, 
        limit: int = 10
    ) -> List[dict]:
        """Get news articles for a specific ticker"""
        endpoint = f"/v2/reference/news"
        params = {"ticker": ticker, "limit": limit}
        data = self._make_request(endpoint, params)
        return data.get("results", [])

# Singleton instance for easy importing
market_data = MarketDataFetcher()

def update_portfolio_prices(portfolio_data: dict) -> dict:
    """
    Update the prices of all assets in a portfolio with real-time data
    
    Args:
        portfolio_data: Portfolio data in dictionary format
        
    Returns:
        Updated portfolio data with current prices
    """
    updated_assets = []
    
    for asset in portfolio_data.get("assets", []):
        ticker = asset.get("ticker")
        if not ticker:
            continue
            
        try:
            if asset.get("asset_type") == "stock":
                # Get the latest price
                prev_close = market_data.get_previous_close(ticker)
                if prev_close:
                    asset["current_price"] = prev_close.get("c", asset.get("current_price"))
                    asset["last_updated"] = datetime.utcnow().isoformat()
            
            # For bonds, we might want to get the latest yield
            elif asset.get("asset_type") == "bond":
                # This would need to be implemented based on bond data source
                pass
                
        except Exception as e:
            logger.error(f"Error updating price for {ticker}: {str(e)}")
        
        updated_assets.append(asset)
    
    portfolio_data["assets"] = updated_assets
    portfolio_data["last_updated"] = datetime.utcnow().isoformat()
    return portfolio_data

def load_sample_portfolio() -> dict:
    """Load the sample portfolio from the JSON file without making API calls"""
    import os
    import json
    from datetime import datetime
    
    file_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "sample_data",
        "sample_portfolio.json"
    )
    
    with open(file_path, 'r') as f:
        portfolio_data = json.load(f)
    
    # Add last_updated timestamp
    portfolio_data["last_updated"] = datetime.utcnow().isoformat()
    
    return portfolio_data

# Example usage
if __name__ == "__main__":
    # Initialize the market data fetcher
    fetcher = MarketDataFetcher()
    
    # Example: Get stock details
    print("Fetching AAPL details...")
    aapl = fetcher.get_stock_details("AAPL")
    print(f"AAPL Details: {json.dumps(aapl, indent=2)}")
    
    # Example: Get previous close
    print("\nFetching previous close for AAPL...")
    prev_close = fetcher.get_previous_close("AAPL")
    print(f"Previous Close: {prev_close}")
    
    # Example: Get historical data
    print("\nFetching historical data for AAPL...")
    hist_data = fetcher.get_stock_aggregates(
        "AAPL", 
        (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
        datetime.now().strftime("%Y-%m-%d")
    )
    print(f"Historical Data (first 5 rows):\n{hist_data.head()}")
