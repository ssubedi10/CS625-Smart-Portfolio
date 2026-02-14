from typing import Dict, List, Optional, TypeVar, Tuple, Any, Type
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .assets import Asset, StockAsset, CryptoAsset, ForexAsset, OptionAsset

# Type aliases
FloatArray = npt.NDArray[np.float64]
ReturnArray = npt.NDArray[np.float64]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T', bound='Portfolio')

@dataclass
class Portfolio:
    """
    A portfolio containing multiple financial assets with methods for portfolio analysis.
    
    Attributes:
        name (str): Name of the portfolio
        description (str): Optional description
        assets (List[Asset]): List of assets in the portfolio
        weights (Dict[str, float]): Weights of each asset by ticker
        risk_free_rate (float): Annual risk-free rate for calculations (default: 0.02)
        created_at (datetime): When the portfolio was created
    """
    name: str
    description: str = ""
    assets: List[Asset] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)
    risk_free_rate: float = 0.02
    created_at: datetime = field(default_factory=datetime.now)
    _covariance_matrix: Optional[np.ndarray] = field(init=False, default=None)
    _expected_returns: Optional[np.ndarray] = field(init=False, default=None)
    
    def __post_init__(self):
        """Initialize the portfolio and validate weights."""
        self._validate_weights()
        
        # Initialize price cache
        self._price_cache = {}
        self._price_cache_date = None
    
    def _validate_weights(self) -> None:
        """Ensure weights sum to 1 (within floating point tolerance)."""
        if not self.weights:  # Skip validation if no weights are set yet
            return
            
        total = sum(self.weights.values())
        if not np.isclose(total, 1.0, rtol=1e-5):
            raise ValueError(f"Weights must sum to 1.0, got {total}")
    
    def add_asset(self, asset: Asset, weight: Optional[float] = None, allow_duplicates: bool = False) -> None:
        """
        Add an asset to the portfolio with an optional weight.
        
        Args:
            asset: The asset to add
            weight: Optional weight of the asset in the portfolio (0-1). 
                   If None, weight will be distributed equally among all assets.
            allow_duplicates: If True and asset exists, adds to existing quantity.
                             If False (default), raises an error for duplicate tickers.
            
        Time Complexity: O(n) where n is the number of assets
        """
        # Check if asset with same ticker already exists
        existing_asset = next((a for a in self.assets if a.ticker == asset.ticker), None)
        
        if existing_asset is not None:
            if not allow_duplicates:
                raise ValueError(f"Asset with ticker {asset.ticker} already exists in the portfolio")
            
            # Add to existing quantity
            existing_asset.quantity += asset.quantity
            return  # Skip the rest as we've updated the existing asset
        
        # If we get here, it's a new asset
        self.assets.append(asset)
        
        if weight is not None:
            if not 0 <= weight <= 1:
                raise ValueError(f"Weight must be between 0 and 1, got {weight}")
            self.weights[asset.ticker] = weight
        else:
            n = len(self.assets)
            self.weights[asset.ticker] = 1.0 / n
            
        self._normalize_weights()
        self._invalidate_cache()
    
    def _normalize_weights(self) -> None:
        """Normalize weights to ensure they sum to 1."""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}
    
    def _invalidate_cache(self) -> None:
        """Invalidate cached calculations when portfolio changes."""
        self._covariance_matrix = None
        self._expected_returns = None
        self._cached_total = None
        self._last_updated = None
    
    def calculate_expected_return(self) -> float:
        """
        Calculate the expected return of the portfolio.
        
        Returns:
            float: The expected return as a decimal
            
        Time Complexity: O(n) where n is the number of assets
        """
        if not self.assets:
            return 0.0
            
        if self._expected_returns is None:
            self._expected_returns = np.array([asset.expected_return for asset in self.assets])
        
        weights = np.array([self.weights[asset.ticker] for asset in self.assets])
        return float(np.dot(weights, self._expected_returns))
    
    def calculate_volatility(self) -> float:
        """
        Calculate the annualized volatility of the portfolio.
        
        Returns:
            float: The annualized volatility as a decimal
            
        Time Complexity: O(n²) due to covariance matrix calculation
        """
        if not self.assets:
            return 0.0
            
        if self._covariance_matrix is None:
            self._calculate_covariance_matrix()
            
        weights = np.array([self.weights[asset.ticker] for asset in self.assets])
        portfolio_variance = weights.T @ self._covariance_matrix @ weights
        return float(np.sqrt(portfolio_variance))
    
    def _calculate_covariance_matrix(self) -> None:
        """Calculate and cache the covariance matrix of asset returns."""
        returns = np.column_stack([asset.historical_returns for asset in self.assets])
        self._covariance_matrix = np.cov(returns, rowvar=False) * 252  # Annualize
    
    def calculate_sharpe_ratio(self, risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate the Sharpe ratio of the portfolio.
        
        Args:
            risk_free_rate: Optional override for risk-free rate
            
        Returns:
            float: The annualized Sharpe ratio
            
        Time Complexity: O(n²) due to volatility calculation
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
            
        excess_return = self.calculate_expected_return() - risk_free_rate
        volatility = self.calculate_volatility()
        
        if volatility == 0:
            return 0.0
            
        return excess_return / volatility
    
    def rebalance(self, new_weights: Dict[str, float]) -> None:
        """
        Rebalance the portfolio to the specified weights.
        
        Args:
            new_weights: Dictionary mapping tickers to new weights
            
        Raises:
            ValueError: If weights don't match assets or don't sum to 1
        """
        if set(new_weights.keys()) != {asset.ticker for asset in self.assets}:
            raise ValueError("Weights must be provided for all assets")
            
        self.weights = new_weights
        self._validate_weights()
        self._invalidate_cache()
    
    def get_asset_weights(self) -> Dict[str, float]:
        """Return a copy of the current asset weights."""
        return self.weights.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the portfolio to a dictionary for serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'assets': [asset.to_dict() for asset in self.assets],
            'weights': self.weights,
            'risk_free_rate': self.risk_free_rate,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Portfolio':
        """
        Create a Portfolio from a dictionary representation.
        
        Args:
            data: Dictionary containing portfolio data with keys:
                - name: str
                - description: str (optional)
                - assets: List[Dict] - List of asset dictionaries
                - weights: Dict[str, float] - Asset weights by ticker
                - risk_free_rate: float (optional, default=0.02)
                - created_at: str - ISO format datetime string
                
        Returns:
            Portfolio: A new Portfolio instance
            
        Time Complexity: O(n) where n is the number of assets
        """
        from .assets import StockAsset, CryptoAsset, ForexAsset, OptionAsset  # Import here to avoid circular imports
        
        # Create a new portfolio
        portfolio = cls(
            name=data['name'],
            description=data.get('description', ''),
            risk_free_rate=float(data.get('risk_free_rate', 0.02)),
            created_at=datetime.fromisoformat(data['created_at']) if 'created_at' in data else datetime.now()
        )
        
        # Add assets and weights
        asset_map = {}
        for asset_data in data.get('assets', []):
            asset_type = asset_data.get('asset_type')
            try:
                if asset_type == 'stock':
                    asset = StockAsset.from_dict(asset_data)
                elif asset_type == 'crypto':
                    asset = CryptoAsset.from_dict(asset_data)
                elif asset_type == 'forex':
                    asset = ForexAsset.from_dict(asset_data)
                elif asset_type == 'option':
                    asset = OptionAsset.from_dict(asset_data)
                else:
                    raise ValueError(f"Unknown asset type: {asset_type}")
                
                weight = data.get('weights', {}).get(asset.ticker, 0.0)
                portfolio.add_asset(asset, weight)
                
            except Exception as e:
                logger.error(f"Error loading asset {asset_data.get('ticker', 'unknown')}: {str(e)}")
                continue
                
        return portfolio
    
    def remove_asset(self, ticker: str) -> None:
        """
        Remove an asset from the portfolio by ticker.
        
        Args:
            ticker: The ticker symbol of the asset to remove
            
        Time Complexity: O(n) where n is the number of assets
        """
        # Find and remove the asset
        for i, asset in enumerate(self.assets):
            if asset.ticker == ticker:
                self.assets.pop(i)
                if ticker in self.weights:
                    del self.weights[ticker]
                self._normalize_weights()
                self._invalidate_cache()
                return
                
        raise ValueError(f"Asset with ticker {ticker} not found in portfolio")
    
    def update_asset(self, ticker: str, **updates) -> None:
        """
        Update an existing asset's properties.
        
        Args:
            ticker: The ticker symbol of the asset to update
            **updates: Key-value pairs of properties to update
            
        Time Complexity: O(n) where n is the number of assets
        """
        for asset in self.assets:
            if asset.ticker == ticker:
                for key, value in updates.items():
                    if hasattr(asset, key):
                        setattr(asset, key, value)
                    else:
                        raise AttributeError(f"{key} is not a valid attribute of {asset.__class__.__name__}")
                self._invalidate_cache()
                return
                
        raise ValueError(f"Asset with ticker {ticker} not found in portfolio")
    
    def get_asset(self, ticker: str) -> Asset:
        """
        Get an asset by its ticker symbol.
        
        Args:
            ticker: The ticker symbol of the asset to retrieve
            
        Returns:
            Asset: The requested asset
            
        Raises:
            ValueError: If no asset with the given ticker exists
            
        Time Complexity: O(n) where n is the number of assets
        """
        for asset in self.assets:
            if asset.ticker == ticker:
                return asset
        raise ValueError(f"Asset with ticker {ticker} not found in portfolio")
    
    def get_total_value(self) -> float:
        """
        Calculate the total market value of the portfolio using the latest cached prices.
        
        Returns:
            float: The total market value calculated as sum(latest_price * quantity)
        """
        total = 0.0
        
        # First, ensure we have the latest prices in cache
        self.update_prices()
        
        # Then calculate using the most recent cached prices
        for asset in self.assets:
            if hasattr(asset, 'ticker') and hasattr(asset, 'quantity'):
                ticker = asset.ticker
                quantity = getattr(asset, 'quantity', 0)
                
                # Try to get current value from the asset first
                try:
                    if hasattr(asset, 'current_value') and callable(getattr(asset, 'current_value')):
                        asset_value = asset.current_value()
                        total += asset_value
                        continue
                except (ValueError, AttributeError):
                    # Fall back to manual calculation if current_value() fails
                    pass
                
                # Get the latest price from cache
                latest_price = None
                if ticker in self._price_cache and self._price_cache[ticker]:
                    latest_price = self._price_cache[ticker][-1]
                
                # If no cached price, use current_price as fallback
                if latest_price is None and hasattr(asset, 'current_price'):
                    latest_price = getattr(asset, 'current_price', 0)
                    if latest_price is None:
                        latest_price = 0
                
                # Final fallback to purchase price if no other price is available
                if latest_price is None or latest_price == 0:
                    latest_price = getattr(asset, 'purchase_price', 0)
                
                total += latest_price * quantity
        
        return round(total, 2)
        
    def get_purchase_value(self) -> float:
        """
        Calculate the total amount invested in the portfolio.
        
        Returns:
            float: The total invested amount (sum of purchase_price * quantity for all assets)
        """
        total_invested = 0.0
        for asset in self.assets:
            if hasattr(asset, 'purchase_price') and hasattr(asset, 'quantity'):
                purchase_price = getattr(asset, 'purchase_price', 0)
                quantity = getattr(asset, 'quantity', 0)
                total_invested += purchase_price * quantity
        return round(total_invested, 2)
        
    def get_total_pnl(self) -> Dict[str, float]:
        """
        Calculate the total profit and loss of the portfolio.
        
        Returns:
            Dict[str, float]: Dictionary containing:
                - 'amount': Total P&L in dollars (current_value - total_invested)
                - 'percentage': Total P&L as a percentage of total invested
        """
        total_invested = self.get_purchase_value()
        current_value = self.get_total_value()
        
        # Calculate P&L
        pnl_amount = current_value - total_invested
        pnl_percent = (pnl_amount / total_invested * 100) if total_invested > 0 else 0.0
        
        return {
            'amount': round(pnl_amount, 2),
            'percentage': round(pnl_percent, 2)
        }
        
    def get_asset_details(self) -> List[Dict]:
        """
        Get detailed information about each asset including purchase date, current value, and P&L.
        Uses real-time prices from Polygon API.
        
        Returns:
            List[Dict]: List of dictionaries containing asset details
        """
        # Ensure we have the latest prices
        self.update_prices()
        
        asset_details = []
        for asset in self.assets:
            if hasattr(asset, 'ticker'):
                # Get current price (should be updated by update_prices)
                current_price = getattr(asset, 'current_price', 0)
                purchase_price = getattr(asset, 'purchase_price', 0)
                quantity = getattr(asset, 'quantity', 0)
                purchase_date = getattr(asset, 'purchase_date', 'N/A')
                last_updated = getattr(asset, 'last_updated', datetime.now())
                
                if isinstance(purchase_date, date):
                    purchase_date = purchase_date.strftime('%Y-%m-%d')
                
                # Format last updated time
                last_updated_str = last_updated.strftime('%Y-%m-%d %H:%M:%S') if hasattr(last_updated, 'strftime') else 'N/A'
                
                current_value = current_price * quantity
                purchase_value = purchase_price * quantity
                pnl_amount = current_value - purchase_value
                pnl_percent = (pnl_amount / purchase_value * 100) if purchase_value > 0 else 0
                
                asset_details.append({
                    'ticker': asset.ticker,
                    'name': getattr(asset, 'name', asset.ticker),
                    'purchase_date': purchase_date,
                    'purchase_price': purchase_price,
                    'purchase_value': purchase_value,
                    'current_price': current_price,
                    'quantity': quantity,
                    'current_value': current_value,
                    'pnl_amount': round(pnl_amount, 2),
                    'pnl_percent': round(pnl_percent, 2),
                    'exchange': getattr(asset, 'exchange', 'N/A'),
                    'last_updated': last_updated_str
                })
        return asset_details
    
    def get_asset_allocation(self) -> Dict[str, float]:
        """
        Calculate the allocation percentage of each asset in the portfolio.
        
        Returns:
            Dict[str, float]: Dictionary mapping asset tickers to their allocation percentages
        """
        total_value = self.get_total_value()
        if total_value == 0:
            return {}
            
        asset_allocation = {}
        for asset in self.assets:
            # Only include assets with positive value
            try:
                # Try to get current value from the asset first
                if hasattr(asset, 'current_value') and callable(getattr(asset, 'current_value')):
                    asset_value = asset.current_value()
                else:
                    # Fall back to manual calculation
                    ticker = asset.ticker
                    quantity = getattr(asset, 'quantity', 0)
                    latest_price = getattr(asset, 'current_price', 0) or getattr(asset, 'purchase_price', 0)
                    asset_value = latest_price * quantity
                
                if asset_value > 0:
                    asset_allocation[asset.ticker] = (asset_value / total_value) * 100
            except (ValueError, AttributeError):
                # Skip assets that can't be valued
                continue
                
        return asset_allocation
    
    def _get_polygon_price(self, ticker: str) -> Optional[float]:
        """
        Fetch the latest price for a ticker using Polygon API.
        
        Args:
            ticker: The ticker symbol to fetch price for
            
        Returns:
            float: The latest price, or None if not available
        """
        try:
            from polygon import RESTClient
            import os
            
            api_key = os.getenv('POLYGON_API_KEY')
            if not api_key:
                logger.error("POLYGON_API_KEY environment variable not set")
                return None
                
            client = RESTClient(api_key)
            quote = client.get_last_trade(ticker)
            return float(quote.price) if hasattr(quote, 'price') else None
            
        except Exception as e:
            logger.error(f"Error fetching price for {ticker} from Polygon: {str(e)}")
            return None
    
    def update_prices(self) -> None:
        """
        Update the current prices of all assets in the portfolio using Polygon API.
        Uses a daily cache to avoid unnecessary API calls.
        """
        if not self.assets:
            return
            
        today = date.today()
        cache_is_fresh = (self._price_cache_date == today)
        
        # First, ensure we have the latest prices in the cache
        if not cache_is_fresh:
            for asset in self.assets:
                if not hasattr(asset, 'ticker') or not hasattr(asset, 'purchase_price'):
                    continue
                    
                ticker = asset.ticker
                
                try:
                    from polygon import RESTClient
                    import os
                    
                    api_key = os.getenv('POLYGON_API_KEY')
                    if not api_key:
                        raise ValueError("POLYGON_API_KEY environment variable not set")
                        
                    client = RESTClient(api_key)
                    quote = client.get_last_trade(ticker)
                    current_price = float(quote.price) if hasattr(quote, 'price') else None
                    
                    if current_price is not None:
                        # Update the cache
                        if ticker not in self._price_cache:
                            self._price_cache[ticker] = []
                        self._price_cache[ticker].append(current_price)
                except Exception as e:
                    logger.warning(f"Error fetching price for {ticker} from Polygon: {str(e)}")
            
            # Update cache date after fetching all prices
            self._price_cache_date = today
        
        # Now update all assets with the latest cached prices
        for asset in self.assets:
            try:
                if not hasattr(asset, 'ticker') or not hasattr(asset, 'purchase_price'):
                    continue
                    
                ticker = asset.ticker
                
                # Always use the latest cached price if available
                if ticker in self._price_cache and self._price_cache[ticker]:
                    latest_price = self._price_cache[ticker][-1]
                    asset.current_price = latest_price
                    asset.last_updated = datetime.now()
                # Fall back to existing current_price if no cache
                elif hasattr(asset, 'current_price') and asset.current_price is not None:
                    # Keep the existing price
                    pass
                # Final fallback to purchase price
                else:
                    asset.current_price = getattr(asset, 'purchase_price', 0)
                    asset.last_updated = datetime.now()
                        
            except Exception as e:
                logger.error(f"Error updating price for asset {getattr(asset, 'ticker', 'unknown')}: {str(e)}")
                # If there's an error, try to keep the existing price or fall back to purchase price
                if not hasattr(asset, 'current_price') or asset.current_price is None:
                    asset.current_price = getattr(asset, 'purchase_price', 0)
                    asset.last_updated = datetime.now()
    
    def get_historical_returns(self, days: int = 30) -> Dict[str, List[float]]:
        """Simulate historical returns for the portfolio.
        
        In a real implementation, this would fetch actual historical data.
        
        Args:
            days: Number of days of historical data to generate
            
        Returns:
            Dict with 'dates' and 'returns' keys
        """
        
        # This is a simplified simulation
        np.random.seed(42)  # For reproducibility
        daily_returns = np.random.normal(0.0005, 0.02, days)
        cumulative_returns = 100 * (1 + daily_returns).cumprod()
        
        return {
            'dates': [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') 
                     for i in range(days, 0, -1)],
            'returns': cumulative_returns.tolist()
        }
    
    def to_dict(self) -> Dict:
        """Convert the portfolio to a dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'assets': {asset_id: asset.to_dict() for asset_id, asset in self.assets.items()},
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }
    
    def to_json(self, filepath: str = None) -> Optional[str]:
        """Serialize the portfolio to JSON."""
        data = self.to_dict()
        json_str = json.dumps(data, indent=2)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
            return None
        return json_str
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict) -> T:
        """Create a Portfolio instance from a dictionary."""
        portfolio = cls(
            name=data['name'],
            description=data.get('description', ''),
            created_at=datetime.fromisoformat(data['created_at'])
        )
        
        # Rebuild assets
        for asset_data in data.get('assets', {}).values():
            asset_type = asset_data.get('asset_type', '').upper()
            try:
                if asset_type == 'STOCK':
                    asset = StockAsset.from_dict(asset_data)
                elif asset_type == 'CRYPTO':
                    from .assets.crypto_asset import CryptoAsset
                    asset = CryptoAsset.from_dict(asset_data)
                elif asset_type == 'FOREX':
                    from .assets.forex_asset import ForexAsset
                    asset = ForexAsset.from_dict(asset_data)
                elif asset_type == 'OPTION':
                    from .assets.option_asset import OptionAsset
                    asset = OptionAsset.from_dict(asset_data)
                else:
                    logger.warning(f"Skipping unknown asset type: {asset_type}")
                    continue
                
                portfolio.assets[asset.asset_id] = asset
            except Exception as e:
                logger.error(f"Error creating {asset_type} asset from data: {asset_data}")
                logger.error(f"Error details: {str(e)}")
                continue
        
        portfolio.last_updated = datetime.fromisoformat(data.get('last_updated', data['created_at']))
        return portfolio
    
    @classmethod
    def from_json(cls: Type[T], json_str: str = None, filepath: str = None) -> T:
        """Create a Portfolio instance from a JSON string or file.
        
        Args:
            json_str: JSON string containing portfolio data
            filepath: Path to a JSON file containing portfolio data
            
        Returns:
            Portfolio: A new Portfolio instance
            
        Raises:
            ValueError: If neither json_str nor filepath is provided
            json.JSONDecodeError: If the JSON string is invalid
            FileNotFoundError: If the specified file doesn't exist
            
        Time Complexity: O(n) where n is the number of assets
        """
        if json_str is None and filepath is not None:
            with open(filepath, 'r') as f:
                json_str = f.read()
        
        if json_str is None:
            raise ValueError("Either json_str or filepath must be provided")
            
        data = json.loads(json_str)
        return cls.from_dict(data)
