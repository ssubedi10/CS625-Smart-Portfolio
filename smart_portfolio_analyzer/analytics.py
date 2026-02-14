import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

class ReturnType(Enum):
    """Types of returns calculation methods."""
    SIMPLE = 'simple'
    LOG = 'log'
    ARITHMETIC = 'arithmetic'
    GEOMETRIC = 'geometric'

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    returns: np.ndarray
    mean_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    beta: Optional[float] = None
    alpha: Optional[float] = None
    tracking_error: Optional[float] = None
    information_ratio: Optional[float] = None

class Analytics:
    """
    A utility class with static methods for financial calculations and analytics.
    """
    
    @staticmethod
    def calculate_returns(
        prices: Union[pd.Series, np.ndarray], 
        return_type: ReturnType = ReturnType.SIMPLE,
        periods: int = 1
    ) -> np.ndarray:
        """
        Calculate returns from price data.
        
        Args:
            prices: Array or Series of prices
            return_type: Type of returns to calculate
            periods: Number of periods for the return calculation
            
        Returns:
            Array of returns
        """
        if isinstance(prices, pd.Series):
            prices = prices.values
            
        if return_type == ReturnType.SIMPLE:
            return (prices[periods:] / prices[:-periods]) - 1
        elif return_type == ReturnType.LOG:
            return np.log(prices[periods:] / prices[:-periods])
        else:
            raise ValueError(f"Unsupported return type: {return_type}")
    
    @staticmethod
    def calculate_sharpe_ratio(
        returns: np.ndarray, 
        risk_free_rate: float = 0.0, 
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate the Sharpe ratio.
        
        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate (default: 0.0)
            periods_per_year: Number of periods per year (default: 252 trading days)
            
        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
            
        excess_returns = returns - (risk_free_rate / periods_per_year)
        
        if np.std(returns, ddof=1) == 0:
            return 0.0
            
        return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(returns, ddof=1)
    
    @staticmethod
    def calculate_sortino_ratio(
        returns: np.ndarray, 
        risk_free_rate: float = 0.0, 
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate the Sortino ratio.
        
        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate (default: 0.0)
            periods_per_year: Number of periods per year (default: 252 trading days)
            
        Returns:
            Annualized Sortino ratio
        """
        if len(returns) < 2:
            return 0.0
            
        excess_returns = returns - (risk_free_rate / periods_per_year)
        downside_returns = np.minimum(0, returns)
        
        if np.std(downside_returns, ddof=1) == 0:
            return 0.0
            
        return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(downside_returns, ddof=1)
    
    @staticmethod
    def calculate_max_drawdown(prices: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate the maximum drawdown from a series of prices.
        
        Args:
            prices: Array or Series of prices
            
        Returns:
            Maximum drawdown as a decimal (e.g., 0.15 for 15%)
        """
        if isinstance(prices, pd.Series):
            prices = prices.values
            
        if len(prices) < 2:
            return 0.0
            
        peak = prices[0]
        max_drawdown = 0.0
        
        for price in prices[1:]:
            if price > peak:
                peak = price
            else:
                drawdown = (peak - price) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        
        return max_drawdown
    
    @staticmethod
    def calculate_value_at_risk(
        returns: np.ndarray, 
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (VaR) using historical simulation.
        
        Args:
            returns: Array of returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Value at Risk as a decimal (e.g., -0.1 for -10%)
        """
        if len(returns) == 0:
            return 0.0
            
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    @staticmethod
    def calculate_conditional_var(
        returns: np.ndarray, 
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) or Expected Shortfall.
        
        Args:
            returns: Array of returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Conditional Value at Risk as a decimal
        """
        if len(returns) == 0:
            return 0.0
            
        var = Analytics.calculate_value_at_risk(returns, confidence_level)
        return np.mean(returns[returns <= var])
    
    @staticmethod
    def calculate_beta(
        asset_returns: np.ndarray, 
        market_returns: np.ndarray
    ) -> float:
        """
        Calculate beta of an asset relative to the market.
        
        Args:
            asset_returns: Array of asset returns
            market_returns: Array of market returns (benchmark)
            
        Returns:
            Beta coefficient
        """
        if len(asset_returns) != len(market_returns) or len(asset_returns) < 2:
            return 1.0  # Default to market beta if not enough data
            
        covariance = np.cov(asset_returns, market_returns, ddof=1)[0, 1]
        market_variance = np.var(market_returns, ddof=1)
        
        if market_variance == 0:
            return 1.0
            
        return covariance / market_variance
    
    @staticmethod
    def calculate_alpha(
        asset_returns: np.ndarray,
        market_returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Jensen's alpha.
        
        Args:
            asset_returns: Array of asset returns
            market_returns: Array of market returns (benchmark)
            risk_free_rate: Annual risk-free rate (default: 0.0)
            periods_per_year: Number of periods per year (default: 252 trading days)
            
        Returns:
            Annualized alpha
        """
        if len(asset_returns) != len(market_returns) or len(asset_returns) < 2:
            return 0.0
            
        beta = Analytics.calculate_beta(asset_returns, market_returns)
        
        # Calculate average returns
        avg_asset_return = np.mean(asset_returns) * periods_per_year
        avg_market_return = np.mean(market_returns) * periods_per_year
        
        # CAPM: Expected return = risk_free_rate + beta * (market_return - risk_free_rate)
        expected_return = risk_free_rate + beta * (avg_market_return - risk_free_rate)
        
        # Alpha = Actual return - Expected return
        return avg_asset_return - expected_return
    
    @staticmethod
    def calculate_tracking_error(
        asset_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate tracking error (annualized standard deviation of active returns).
        
        Args:
            asset_returns: Array of asset returns
            benchmark_returns: Array of benchmark returns
            periods_per_year: Number of periods per year
            
        Returns:
            Annualized tracking error
        """
        if len(asset_returns) != len(benchmark_returns) or len(asset_returns) < 2:
            return 0.0
            
        active_returns = asset_returns - benchmark_returns
        return np.std(active_returns, ddof=1) * np.sqrt(periods_per_year)
    
    @staticmethod
    def calculate_information_ratio(
        asset_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate the information ratio.
        
        Args:
            asset_returns: Array of asset returns
            benchmark_returns: Array of benchmark returns
            periods_per_year: Number of periods per year
            
        Returns:
            Information ratio
        """
        if len(asset_returns) != len(benchmark_returns) or len(asset_returns) < 2:
            return 0.0
            
        active_returns = asset_returns - benchmark_returns
        tracking_error = Analytics.calculate_tracking_error(asset_returns, benchmark_returns, periods_per_year)
        
        if tracking_error == 0:
            return 0.0
            
        return np.mean(active_returns) * np.sqrt(periods_per_year) / tracking_error
    
    @staticmethod
    def get_portfolio_overview(self) -> dict:
        """
        Generate a comprehensive overview of the portfolio's performance and risk metrics.
        
        Returns:
            dict: Dictionary containing portfolio overview metrics
        """
        # Calculate basic metrics
        total_value = self.calculate_total_value()
        expected_return = self.calculate_expected_return()
        volatility = self.calculate_volatility()
        sharpe_ratio = self.calculate_sharpe_ratio()
        
        # Calculate asset allocation
        asset_allocation = {}
        for asset in self.portfolio.assets:
            weight = self.portfolio.weights.get(asset.ticker, 0)
            try:
                asset_value = asset.current_value()
            except (ValueError, AttributeError):
                # Fallback to manual calculation if current_value() fails
                quantity = asset.quantity if hasattr(asset, 'quantity') else 1
                latest_price = getattr(asset, 'current_price', 0) or getattr(asset, 'purchase_price', 0)
                asset_value = latest_price * quantity
            asset_allocation[asset.ticker] = {
                'weight': weight,
                'value': asset_value,
                'type': asset.asset_type,
                'sector': getattr(asset, 'sector', 'N/A'),
                'pct_change': ((asset.current_price / asset.purchase_price) - 1) * 100
            }
        
        # Calculate sector allocation (for stocks)
        sector_allocation = {}
        for asset in self.portfolio.assets:
            if hasattr(asset, 'sector'):
                sector = asset.sector or 'Other'
            try:
                value = asset.current_value()
            except (ValueError, AttributeError):
                # Fallback to manual calculation if current_value() fails
                quantity = asset.quantity if hasattr(asset, 'quantity') else 1
                latest_price = getattr(asset, 'current_price', 0) or getattr(asset, 'purchase_price', 0)
                value = latest_price * quantity
                if sector in sector_allocation:
                    sector_allocation[sector] += value
                else:
                    sector_allocation[sector] = value
        
        # Convert to percentages
        if total_value > 0:
            sector_allocation = {k: (v / total_value) * 100 for k, v in sector_allocation.items()}
        
        # Calculate risk metrics
        returns = self.calculate_returns()
        sortino_ratio = self.calculate_sortino_ratio()
        max_drawdown = self.calculate_max_drawdown()
        var_95 = self.calculate_var(confidence_level=0.95)
        cvar_95 = self.calculate_cvar(confidence_level=0.95)
        
        # Prepare the overview dictionary
        overview = {
            'portfolio_summary': {
                'total_value': total_value,
                'total_invested': self.calculate_total_invested(),
                'total_pnl': total_value - self.calculate_total_invested(),
                'pct_return': (total_value / self.calculate_total_invested() - 1) * 100 \
                    if self.calculate_total_invested() > 0 else 0,
                'num_assets': len(self.portfolio.assets),
                'num_stocks': len([a for a in self.portfolio.assets if a.asset_type == 'stock']),
                'num_bonds': len([a for a in self.portfolio.assets if a.asset_type == 'bond']),
            },
            'performance_metrics': {
                'expected_return': expected_return * 100,  # as percentage
                'volatility': volatility * 100,  # as percentage
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown * 100,  # as percentage
                'var_95': var_95 * 100,  # as percentage
                'cvar_95': cvar_95 * 100,  # as percentage
            },
            'asset_allocation': asset_allocation,
            'sector_allocation': sector_allocation,
            'top_performers': self._get_top_performers(),
            'bottom_performers': self._get_bottom_performers(),
            'last_updated': datetime.utcnow().isoformat()
        }
        
        return overview
    
    def _get_top_performers(self, n: int = 3) -> list:
        """Get top n performing assets by return percentage"""
        performers = []
        for asset in self.portfolio.assets:
            if hasattr(asset, 'purchase_price') and asset.purchase_price > 0:
                pct_return = ((asset.current_price / asset.purchase_price) - 1) * 100
                try:
                    value = asset.current_value()
                except (ValueError, AttributeError):
                    # Fallback to manual calculation if current_value() fails
                    quantity = asset.quantity if hasattr(asset, 'quantity') else 1
                    latest_price = getattr(asset, 'current_price', 0) or getattr(asset, 'purchase_price', 0)
                    value = latest_price * quantity
                
                performers.append({
                    'ticker': asset.ticker,
                    'name': getattr(asset, 'name', ''),
                    'pct_return': pct_return,
                    'value': value
                })
        
        return sorted(performers, key=lambda x: x['pct_return'], reverse=True)[:n]
    
    def _get_bottom_performers(self, n: int = 3) -> list:
        """Get bottom n performing assets by return percentage"""
        performers = []
        for asset in self.portfolio.assets:
            if hasattr(asset, 'purchase_price') and asset.purchase_price > 0:
                pct_return = ((asset.current_price / asset.purchase_price) - 1) * 100
                try:
                    value = asset.current_value()
                except (ValueError, AttributeError):
                    # Fallback to manual calculation if current_value() fails
                    quantity = asset.quantity if hasattr(asset, 'quantity') else 1
                    latest_price = getattr(asset, 'current_price', 0) or getattr(asset, 'purchase_price', 0)
                    value = latest_price * quantity
                
                performers.append({
                    'ticker': asset.ticker,
                    'name': getattr(asset, 'name', ''),
                    'pct_return': pct_return,
                    'value': value
                })
        
        return sorted(performers, key=lambda x: x['pct_return'])[:n]
    
    def calculate_performance_metrics(
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> PerformanceMetrics:
        """
        Calculate a comprehensive set of performance metrics.
        
        Args:
            returns: Array of asset returns
            benchmark_returns: Optional array of benchmark returns for relative metrics
            risk_free_rate: Annual risk-free rate (default: 0.0)
            periods_per_year: Number of periods per year (default: 252 trading days)
            
        Returns:
            PerformanceMetrics object containing all calculated metrics
        """
        if len(returns) < 2:
            return PerformanceMetrics(
                returns=returns,
                mean_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                var_95=0.0,
                cvar_95=0.0
            )
        
        # Calculate basic metrics
        mean_return = np.mean(returns) * periods_per_year
        volatility = np.std(returns, ddof=1) * np.sqrt(periods_per_year)
        sharpe_ratio = Analytics.calculate_sharpe_ratio(
            returns, risk_free_rate, periods_per_year
        )
        sortino_ratio = Analytics.calculate_sortino_ratio(
            returns, risk_free_rate, periods_per_year
        )
        
        # Calculate drawdown from cumulative returns
        cum_returns = np.cumprod(1 + returns) - 1
        max_drawdown = Analytics.calculate_max_drawdown(cum_returns + 1)  # Convert to price-like series
        
        # Calculate risk metrics
        var_95 = Analytics.calculate_value_at_risk(returns, 0.95)
        cvar_95 = Analytics.calculate_conditional_var(returns, 0.95)
        
        # Initialize optional metrics
        beta = None
        alpha = None
        tracking_error = None
        information_ratio = None
        
        # Calculate relative metrics if benchmark is provided
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            beta = Analytics.calculate_beta(returns, benchmark_returns)
            alpha = Analytics.calculate_alpha(
                returns, benchmark_returns, risk_free_rate, periods_per_year
            )
            tracking_error = Analytics.calculate_tracking_error(
                returns, benchmark_returns, periods_per_year
            )
            information_ratio = Analytics.calculate_information_ratio(
                returns, benchmark_returns, periods_per_year
            )
        
        return PerformanceMetrics(
            returns=returns,
            mean_return=mean_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            beta=beta,
            alpha=alpha,
            tracking_error=tracking_error,
            information_ratio=information_ratio
        )
    
    @staticmethod
    def calculate_correlation_matrix(returns: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Calculate the correlation matrix for multiple return series.
        
        Args:
            returns: Dictionary of return series with asset names as keys
            
        Returns:
            Correlation matrix as a pandas DataFrame
        """
        if not returns:
            return pd.DataFrame()
            
        # Convert to DataFrame for easy correlation calculation
        df = pd.DataFrame(returns)
        return df.corr()
    
    @staticmethod
    def calculate_efficient_frontier(
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free_rate: float = 0.0,
        n_points: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Calculate the efficient frontier using the Markowitz model.
        
        Args:
            expected_returns: Array of expected returns for each asset
            cov_matrix: Covariance matrix of asset returns
            risk_free_rate: Risk-free rate (for calculating the capital market line)
            n_points: Number of points to generate on the efficient frontier
            
        Returns:
            Dictionary containing:
                - 'weights': Array of optimal weights for each portfolio on the frontier
                - 'returns': Expected returns of the efficient portfolios
                - 'volatility': Volatility (risk) of the efficient portfolios
                - 'sharpe_ratio': Sharpe ratio of each portfolio
        """
        n_assets = len(expected_returns)
        
        # Calculate minimum variance portfolio
        inv_cov = np.linalg.inv(cov_matrix)
        ones = np.ones(n_assets)
        
        # Minimum variance portfolio
        w_min_var = inv_cov @ ones / (ones.T @ inv_cov @ ones)
        min_var_return = w_min_var @ expected_returns
        min_var_vol = np.sqrt(w_min_var.T @ cov_matrix @ w_min_var)
        
        # Maximum return portfolio (100% in the highest returning asset)
        max_return = np.max(expected_returns)
        max_return_idx = np.argmax(expected_returns)
        w_max_return = np.zeros(n_assets)
        w_max_return[max_return_idx] = 1.0
        max_return_vol = np.sqrt(cov_matrix[max_return_idx, max_return_idx])
        
        # Generate target returns between min_var_return and max_return
        target_returns = np.linspace(min_var_return, max_return, n_points)
        
        # Calculate efficient portfolios
        efficient_weights = []
        efficient_returns = []
        efficient_volatilities = []
        
        for target_return in target_returns:
            # Skip if target return is below the minimum variance portfolio return
            if target_return < min_var_return:
                continue
                
            # Solve for optimal weights
            A = np.array([
                [2 * cov_matrix, expected_returns, ones],
                [expected_returns.T, 0, 0],
                [ones.T, 0, 0]
            ])
            
            b = np.zeros(n_assets + 2)
            b[-2] = target_return
            b[-1] = 1
            
            try:
                # Solve the system of equations
                solution = np.linalg.solve(A, b)
                weights = solution[:n_assets]
                
                # Calculate portfolio metrics
                port_return = weights @ expected_returns
                port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
                
                efficient_weights.append(weights)
                efficient_returns.append(port_return)
                efficient_volatilities.append(port_vol)
                
            except np.linalg.LinAlgError:
                # Skip if the system is singular
                continue
        
        if not efficient_returns:
            return {
                'weights': np.array([]),
                'returns': np.array([]),
                'volatility': np.array([]),
                'sharpe_ratio': np.array([])
            }
        
        # Convert to numpy arrays
        efficient_weights = np.array(efficient_weights)
        efficient_returns = np.array(efficient_returns)
        efficient_volatilities = np.array(efficient_volatilities)
        
        # Calculate Sharpe ratios
        sharpe_ratios = (efficient_returns - risk_free_rate) / efficient_volatilities
        
        return {
            'weights': efficient_weights,
            'returns': efficient_returns,
            'volatility': efficient_volatilities,
            'sharpe_ratio': sharpe_ratios
        }
