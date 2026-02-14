import os
import json
import time
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Union, Any, Tuple
import requests
from urllib.parse import urljoin
from polygon import RESTClient
import yfinance as yf  # Keep as fallback

from .portfolio import Portfolio
from .assets import StockAsset, CryptoAsset, ForexAsset, OptionAsset


class DataManager:
    """
    Handles data operations including reading from files, APIs, and other data sources.
    """
    
    def __init__(self, api_key: str = None, cache_dir: str = 'data_cache'):
        """
        Initialize the DataManager.
        
        Args:
            api_key: Polygon.io API key (required for Polygon data)
            cache_dir: Directory to store cached data
        """
        if not api_key:
            raise ValueError("Polygon.io API key is required. Get one at https://polygon.io/")
            
        self.api_key = api_key
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.polygon_client = RESTClient(api_key=api_key)
    
    def get_historical_prices(
        self, 
        symbols: List[str], 
        start_date: str = None, 
        end_date: str = None,
        period: str = '1y',
        interval: str = '1d',
        source: str = 'polygon'
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical price data for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format (default: 1 year ago)
            end_date: End date in 'YYYY-MM-DD' format (default: today)
            period: Time period if start_date not provided (e.g., '1y', '2y', '5y')
            interval: Data interval ('1d', '1w', '1m' for daily, weekly, monthly)
            source: Data source ('polygon', 'yfinance' as fallback)
            
        Returns:
            Dictionary mapping symbols to DataFrames with historical data
        """
        if not symbols:
            return {}
            
        # Set default dates if not provided
        end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            if period == '1y':
                # Default to January 1, 2025
                start_date = '2025-01-01'
            elif period.endswith('y'):
                years = int(period[:-1])
                start_date = (datetime.now() - timedelta(days=365*years)).strftime('%Y-%m-%d')
            else:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        # Convert interval to Polygon format
        polygon_interval = 'day'  # default
        if interval == '1w':
            polygon_interval = 'week'
        elif interval == '1m':
            polygon_interval = 'month'
            
        try:
            if source == 'polygon':
                return self._get_polygon_data(symbols, start_date, end_date, polygon_interval)
            else:
                # Fallback to yfinance if specified or if Polygon fails
                return self._get_yfinance_data(symbols, start_date, end_date, period, interval)
        except Exception as e:
            print(f"Error with {source} data source, falling back to yfinance: {str(e)}")
            return self._get_yfinance_data(symbols, start_date, end_date, period, interval)
    
    def _get_yfinance_data(
        self, 
        symbols: List[str], 
        start_date: str = None, 
        end_date: str = None,
        period: str = '1y',
        interval: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
        """Get historical data from Yahoo Finance."""
        data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                
                if start_date and end_date:
                    df = ticker.history(start=start_date, end=end_date, interval=interval)
                else:
                    df = ticker.history(period=period, interval=interval)
                
                if not df.empty:
                    # Clean up the DataFrame
                    df = df.rename(columns={
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume',
                        'Dividends': 'dividends',
                        'Stock Splits': 'splits'
                    })
                    data[symbol] = df
                
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
        
        return data
    
    def _get_polygon_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = 'day'
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical data from Polygon.io.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval ('day', 'week', 'month', 'quarter', 'year')
            
        Returns:
            Dictionary mapping symbols to DataFrames with historical data
        """
        data = {}
        
        for symbol in symbols:
            try:
                # Check cache first
                cache_file = os.path.join(self.cache_dir, f"{symbol}_{start_date}_{end_date}_{interval}.pkl")
                if os.path.exists(cache_file):
                    # Load from cache if not expired (1 day)
                    cache_time = os.path.getmtime(cache_file)
                    if (time.time() - cache_time) < 86400:  # 24 hours in seconds
                        data[symbol] = pd.read_pickle(cache_file)
                        continue
                
                # Ensure dates are in the past
                today = datetime.now().date()
                end_date_obj = min(datetime.strptime(end_date, '%Y-%m-%d').date(), today)
                end_date = end_date_obj.strftime('%Y-%m-%d')
                
                # Fetch data from Polygon
                bars = []
                
                # Try to get aggregate bars
                try:
                    for bar in self.polygon_client.list_aggs(
                        ticker=symbol,
                        multiplier=1,
                        timespan=interval,
                        from_=start_date,
                        to=end_date,
                        limit=50000  # Maximum allowed by Polygon
                    ):
                        bars.append({
                            'date': datetime.utcfromtimestamp(bar.timestamp / 1000).strftime('%Y-%m-%d'),
                            'open': bar.open,
                            'high': bar.high,
                            'low': bar.low,
                            'close': bar.close,
                            'volume': bar.volume,
                            'vwap': bar.vwap if hasattr(bar, 'vwap') else None,
                            'transactions': bar.transactions if hasattr(bar, 'transactions') else None
                        })
                except Exception as e:
                    print(f"Error fetching aggregate bars for {symbol}: {str(e)}")
                
                if bars:
                    df = pd.DataFrame(bars)
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    
                    # Cache the data
                    df.to_pickle(cache_file)
                    data[symbol] = df
                
            except Exception as e:
                print(f"Error fetching Polygon data for {symbol}: {str(e)}")
                # Try yfinance as fallback
                try:
                    fallback_data = self._get_yfinance_data(
                        [symbol], 
                        start_date, 
                        end_date,
                        interval='1d'
                    )
                    if symbol in fallback_data:
                        data[symbol] = fallback_data[symbol]
                except Exception as e2:
                    print(f"Fallback to yfinance also failed for {symbol}: {str(e2)}")
        
        return data
    
    def _get_alpha_vantage_data(
        self, 
        symbols: List[str], 
        start_date: str = None, 
        end_date: str = None,
        interval: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
        """Get historical data from Alpha Vantage."""
        # Implementation for Alpha Vantage API
        # Note: This is a placeholder - you would need to implement the actual API calls
        raise NotImplementedError("Alpha Vantage integration not implemented")
    
    def _get_iex_data(
        self, 
        symbols: List[str], 
        start_date: str = None, 
        end_date: str = None
    ) -> Dict[str, pd.DataFrame]:
        """Get historical data from IEX Cloud."""
        # Implementation for IEX Cloud API
        # Note: This is a placeholder - you would need to implement the actual API calls
        raise NotImplementedError("IEX Cloud integration not implemented")
    
    def get_dividend_history(
        self, 
        symbol: str, 
        start_date: str = None, 
        end_date: str = None
    ) -> pd.DataFrame:
        """Get dividend history for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            divs = ticker.dividends
            
            if start_date:
                divs = divs[divs.index >= start_date]
            if end_date:
                divs = divs[divs.index <= end_date]
                
            return pd.DataFrame(divs)
            
        except Exception as e:
            print(f"Error fetching dividend data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """Get company information for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract relevant information
            company_info = {
                'symbol': symbol,
                'name': info.get('shortName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'website': info.get('website', ''),
                'description': info.get('longBusinessSummary', '')
            }
            
            return company_info
            
        except Exception as e:
            print(f"Error fetching company info for {symbol}: {str(e)}")
            return {}
    
    def get_bond_yield_curve(self, date: str = None) -> Dict[str, float]:
        """Get current or historical yield curve data."""
        # This is a simplified example - in practice, you would fetch this from a data provider
        # like Treasury.gov or FRED (Federal Reserve Economic Data)
        
        # Example yield curve (10-year Treasury yields as of a specific date)
        yield_curve = {
            '1m': 0.05,
            '3m': 0.08,
            '6m': 0.12,
            '1y': 0.18,
            '2y': 0.35,
            '5y': 0.85,
            '10y': 1.45,
            '30y': 2.10
        }
        
        return yield_curve
    
    def save_portfolio(self, portfolio: Portfolio, filename: str) -> None:
        """Save a portfolio to a JSON file."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(portfolio.to_dict(), f, indent=2)
    
    def load_portfolio(self, filename: str) -> Portfolio:
        """Load a portfolio from a JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        return Portfolio.from_dict(data)
    
    def export_to_csv(self, data: Union[Dict, pd.DataFrame], filename: str) -> None:
        """
        Export data to a CSV file.
        
        Args:
            data: Data to export (DataFrame or dictionary of DataFrames)
            filename: Output filename
        """
        if isinstance(data, dict):
            # If it's a dictionary of DataFrames, create a directory and save each one
            os.makedirs(os.path.splitext(filename)[0], exist_ok=True)
            
            for key, df in data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    safe_key = "".join(c if c.isalnum() else "_" for c in str(key))
                    df.to_csv(os.path.join(os.path.splitext(filename)[0], f"{safe_key}.csv"))
        elif isinstance(data, pd.DataFrame):
            # If it's a single DataFrame, save it directly
            data.to_csv(filename)
    
    def get_benchmark_returns(
        self, 
        benchmark: str = '^GSPC',  # S&P 500 by default
        start_date: str = None, 
        end_date: str = None,
        period: str = '1y'
    ) -> pd.Series:
        """
        Get returns for a benchmark index.
        
        Args:
            benchmark: Benchmark symbol (e.g., '^GSPC' for S&P 500, '^IXIC' for NASDAQ)
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (default: today)
            period: Time period if dates not specified
            
        Returns:
            Series of returns
        """
        try:
            if start_date and end_date:
                df = yf.download(benchmark, start=start_date, end=end_date, progress=False)
            else:
                df = yf.download(benchmark, period=period, progress=False)
            
            if not df.empty:
                return df['Adj Close'].pct_change().dropna()
            
        except Exception as e:
            print(f"Error fetching benchmark data for {benchmark}: {str(e)}")
        
        return pd.Series()
    
    def get_risk_free_rate(self, target_date: date = None) -> float:
        """
        Get the risk-free rate using the 10-year US Treasury yield.
        
        Args:
            target_date: Date to get the rate for (defaults to most recent available)
            
        Returns:
            Annualized risk-free rate as a decimal (e.g., 0.05 for 5%)
        """
        if target_date is None:
            target_date = date.today()
            
        # Try to get from Treasury website first
        try:
            year = target_date.year
            cache_file = os.path.join(self.cache_dir, f'treasury_rates_{year}.json')
            
            # Try to get from cache first
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        rates = json.load(f)
                    if str(target_date) in rates:
                        rate = float(rates[str(target_date)]) / 100  # Convert percentage to decimal
                        print(f"Using cached 10-year Treasury yield for {target_date}: {rate*100:.2f}%")
                        return rate
                except (json.JSONDecodeError, KeyError):
                    pass  # Cache is invalid, will fetch fresh data
                    
            # If not in cache or cache is invalid, fetch from Treasury website
            url = f'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value={year}'
            
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse XML response
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            # Namespace handling for the XML response
            ns = {'d': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic'}
            
            rates_dict = {}
            
            # Find all series with 10-year constant maturity
            for series in root.findall(".//d:Series[d:SeriesKey/d:Value[@value='10']]", namespaces=ns):
                date_str = series.find("d:Attributes/d:Value[@concept='NEW_DATE']", namespaces=ns)
                rate = series.find("d:Obs/d:ObsValue", namespaces=ns)
                
                if date_str is not None and rate is not None and 'value' in date_str.attrib and 'value' in rate.attrib:
                    try:
                        rate_date = datetime.strptime(date_str.attrib['value'], '%Y-%m-%d').date()
                        rate_value = float(rate.attrib['value'])
                        rates_dict[str(rate_date)] = rate_value
                    except (ValueError, AttributeError):
                        continue
            
            # Save to cache
            with open(cache_file, 'w') as f:
                json.dump(rates_dict, f)
                
            # Return the rate for the target date, or the most recent available rate
            if str(target_date) in rates_dict:
                return rates_dict[str(target_date)] / 100
            elif rates_dict:  # If we have rates but not for the exact date, return the most recent
                most_recent_date = max(rates_dict.keys())
                rate = rates_dict[most_recent_date] / 100
                print(f"Using most recent 10-year Treasury yield ({most_recent_date}): {rate*100:.2f}%")
                return rate
                
        except Exception as e:
            print(f"Warning: Could not fetch 10-year Treasury rate from Treasury website: {str(e)}")
            # Fall through to Yahoo Finance fallback
        
        # Fallback to Yahoo Finance if Treasury website fails
        try:
            # Try to get the 10-year Treasury yield from Yahoo Finance
            ticker = yf.Ticker('^TNX')  # ^TNX is the ticker for 10-year Treasury yield
            hist = ticker.history(period='1d')
            
            if not hist.empty and 'Close' in hist.columns and not hist['Close'].empty:
                rate = hist['Close'].iloc[-1] / 100.0  # Convert from percentage to decimal
                print(f"Fetched current 10-year Treasury yield from Yahoo Finance: {rate*100:.2f}%")
                return rate
                
            print("Warning: No data returned for 10-year Treasury yield, using fallback")
            return 0.04  # Fallback to 4%
            
        except Exception as e:
            print(f"Error fetching 10-year Treasury yield: {str(e)}")
            return 0.04  # Fallback to 4%
    
    def clean_old_cache(self, days_to_keep: int = 30) -> int:
        """
        Delete cached data files older than the specified number of days.
        
        Args:
            days_to_keep: Number of days to keep cache files (default: 30)
            
        Returns:
            Number of files deleted
        """
        if not os.path.exists(self.cache_dir):
            return 0
            
        files_deleted = 0
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)  # Convert days to seconds
        
        try:
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                
                # Only process files (not directories)
                if os.path.isfile(file_path):
                    file_time = os.path.getmtime(file_path)
                    
                    # Delete if file is older than cutoff time
                    if file_time < cutoff_time:
                        try:
                            os.remove(file_path)
                            files_deleted += 1
                            print(f"Deleted old cache file: {filename}")
                        except Exception as e:
                            print(f"Error deleting cache file {filename}: {str(e)}")
                            
        except Exception as e:
            print(f"Error cleaning cache directory: {str(e)}")
            
        if files_deleted > 0:
            print(f"Cache cleanup completed. Deleted {files_deleted} old files.")
        else:
            print("No old cache files to delete.")
            
        return files_deleted
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached data.
        
        Returns:
            Dictionary with cache statistics
        """
        if not os.path.exists(self.cache_dir):
            return {
                'total_files': 0,
                'total_size_mb': 0,
                'oldest_file': None,
                'newest_file': None,
                'files_by_age': {}
            }
            
        files = []
        total_size = 0
        
        try:
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.isfile(file_path):
                    file_time = os.path.getmtime(file_path)
                    file_size = os.path.getsize(file_path)
                    files.append({
                        'name': filename,
                        'mtime': file_time,
                        'size': file_size
                    })
                    total_size += file_size
        except Exception as e:
            print(f"Error scanning cache directory: {str(e)}")
            return {'error': str(e)}
        
        if not files:
            return {
                'total_files': 0,
                'total_size_mb': 0,
                'oldest_file': None,
                'newest_file': None,
                'files_by_age': {}
            }
        
        # Sort files by modification time
        files.sort(key=lambda x: x['mtime'])
        
        # Calculate age distribution
        now = time.time()
        age_groups = {
            'less_than_1_day': 0,
            '1_to_7_days': 0,
            '7_to_30_days': 0,
            'older_than_30_days': 0
        }
        
        for file in files:
            age_days = (now - file['mtime']) / (24 * 60 * 60)
            if age_days < 1:
                age_groups['less_than_1_day'] += 1
            elif age_days < 7:
                age_groups['1_to_7_days'] += 1
            elif age_days < 30:
                age_groups['7_to_30_days'] += 1
            else:
                age_groups['older_than_30_days'] += 1
        
        return {
            'total_files': len(files),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'oldest_file': datetime.fromtimestamp(files[0]['mtime']).strftime('%Y-%m-%d %H:%M:%S'),
            'newest_file': datetime.fromtimestamp(files[-1]['mtime']).strftime('%Y-%m-%d %H:%M:%S'),
            'files_by_age': age_groups
        }
