"""
Portfolio Overview Module

Provides comprehensive analysis and visualization of portfolio performance and risk metrics,
including historical analysis, benchmarking, and rebalancing suggestions.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from dataclasses import dataclass
import yfinance as yf
from scipy.optimize import minimize
from enum import Enum

class TimePeriod(str, Enum):
    ONE_MONTH = "1m"
    THREE_MONTHS = "3m"
    SIX_MONTHS = "6m"
    YEAR_TO_DATE = "ytd"
    ONE_YEAR = "1y"
    THREE_YEARS = "3y"
    FIVE_YEARS = "5y"
    MAX = "max"

class Benchmark(str, Enum):
    SP500 = "^GSPC"  # S&P 500
    DOW_JONES = "^DJI"  # Dow Jones Industrial Average
    NASDAQ = "^IXIC"  # NASDAQ Composite
    RUSSELL_2000 = "^RUT"  # Russell 2000
    AGG = "AGG"  # iShares Core U.S. Aggregate Bond ETF
    CASH = "CASH"  # Cash equivalent (risk-free rate)

@dataclass
class PortfolioOverview:
    """
    A class to generate comprehensive portfolio overviews including performance
    metrics, risk analysis, and visualizations.
    """
    
    def __init__(self, portfolio, benchmark: str = Benchmark.SP500):
        """
        Initialize with a Portfolio instance.
        
        Args:
            portfolio: Instance of the Portfolio class
            benchmark: Benchmark to compare against (default: S&P 500)
        """
        self.portfolio = portfolio
        self.risk_free_rate = getattr(portfolio, 'risk_free_rate', 0.02)
        self.benchmark = benchmark
        self._historical_data = {}
        self._benchmark_data = {}
        self._last_analysis_date = None
        
    def get_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the portfolio.
        
        Returns:
            Dictionary containing portfolio summary metrics
        """
        return {
            'portfolio_summary': self._get_portfolio_summary(),
            'performance_metrics': self._get_performance_metrics(),
            'risk_metrics': self._get_risk_metrics(),
            'asset_allocation': self._get_asset_allocation(),
            'sector_allocation': self._get_sector_allocation(),
            'performance_breakdown': self._get_performance_breakdown(),
            'last_updated': datetime.utcnow().isoformat()
        }
    
    def _get_portfolio_summary(self) -> Dict[str, Any]:
        """Calculate basic portfolio summary metrics."""
        total_value = self.portfolio.calculate_total_value()
        total_invested = self._calculate_total_invested()
        total_pnl = total_value - total_invested
        
        return {
            'total_value': total_value,
            'total_invested': total_invested,
            'total_pnl': total_pnl,
            'pct_return': (total_pnl / total_invested * 100) if total_invested > 0 else 0,
            'num_assets': len(self.portfolio.assets),
            'num_stocks': len([a for a in self.portfolio.assets if a.asset_type == 'stock']),
            'num_bonds': len([a for a in self.portfolio.assets if a.asset_type == 'bond']),
            'start_date': min([asset.purchase_date for asset in self.portfolio.assets]) if self.portfolio.assets else None,
            'last_updated': datetime.utcnow().isoformat()
        }
    
    def _get_performance_metrics(self) -> Dict[str, float]:
        """Calculate key performance metrics."""
        returns = self._get_historical_returns()
        if len(returns) < 2:
            return {}
            
        return {
            'annualized_return': self._calculate_annualized_return(returns) * 100,  # as percentage
            'ytd_return': self._calculate_ytd_return() * 100,  # as percentage
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'alpha': self._calculate_alpha(returns),
            'beta': self._calculate_beta(returns),
            'r_squared': self._calculate_r_squared(returns)
        }
    
    def _get_risk_metrics(self) -> Dict[str, float]:
        """Calculate key risk metrics."""
        returns = self._get_historical_returns()
        if len(returns) < 2:
            return {}
            
        return {
            'volatility': np.std(returns, ddof=1) * np.sqrt(252) * 100,  # annualized, as percentage
            'max_drawdown': self._calculate_max_drawdown() * 100,  # as percentage
            'var_95': self._calculate_var(returns) * 100,  # as percentage
            'cvar_95': self._calculate_cvar(returns) * 100,  # as percentage
            'tracking_error': self._calculate_tracking_error(returns) * 100,  # as percentage
            'information_ratio': self._calculate_information_ratio(returns)
        }
    
    def _get_asset_allocation(self) -> Dict[str, Dict[str, float]]:
        """Calculate asset allocation by asset."""
        total_value = self.portfolio.calculate_total_value()
        if total_value <= 0:
            return {}
            
        allocation = {}
        for asset in self.portfolio.assets:
            weight = self.portfolio.weights.get(asset.ticker, 0)
            try:
                asset_value = asset.current_value()
            except (ValueError, AttributeError):
                # Fallback to manual calculation if current_value() fails
                quantity = getattr(asset, 'quantity', 1) or 1
                latest_price = getattr(asset, 'current_price', 0) or getattr(asset, 'purchase_price', 0)
                asset_value = latest_price * quantity
            allocation[asset.ticker] = {
                'weight': weight * 100,  # as percentage
                'value': asset_value,
                'type': asset.asset_type,
                'sector': getattr(asset, 'sector', 'N/A'),
                'pct_change': ((asset.current_price / asset.purchase_price) - 1) * 100 \
                    if hasattr(asset, 'purchase_price') and asset.purchase_price > 0 else 0
            }
        return allocation
    
    def _get_sector_allocation(self) -> Dict[str, float]:
        """Calculate allocation by sector."""
        sector_allocation = {}
        for asset in self.portfolio.assets:
            if hasattr(asset, 'sector') and asset.sector:
                sector = asset.sector
                try:
                    value = asset.current_value()
                except (ValueError, AttributeError):
                    # Fallback to manual calculation if current_value() fails
                    quantity = getattr(asset, 'quantity', 1) or 1
                    latest_price = getattr(asset, 'current_price', 0) or getattr(asset, 'purchase_price', 0)
                    value = latest_price * quantity
                sector_allocation[sector] = sector_allocation.get(sector, 0) + value
        
        total_value = self.portfolio.calculate_total_value()
        if total_value > 0:
            return {k: (v / total_value) * 100 for k, v in sector_allocation.items()}  # as percentage
        return {}
    
    def _get_performance_breakdown(self) -> Dict[str, Any]:
        """Calculate performance breakdown by asset and sector."""
        assets = []
        for asset in self.portfolio.assets:
            if hasattr(asset, 'purchase_price') and asset.purchase_price > 0:
                pct_return = ((asset.current_price / asset.purchase_price) - 1) * 100
                try:
                    value = asset.current_value()
                except (ValueError, AttributeError):
                    # Fallback to manual calculation if current_value() fails
                    quantity = getattr(asset, 'quantity', 1) or 1
                    latest_price = getattr(asset, 'current_price', 0) or getattr(asset, 'purchase_price', 0)
                    value = latest_price * quantity
                
                assets.append({
                    'ticker': asset.ticker,
                    'name': getattr(asset, 'name', ''),
                    'sector': getattr(asset, 'sector', 'N/A'),
                    'asset_type': asset.asset_type,
                    'pct_return': pct_return,
                    'value': value
                })
        
        # Sort by return (descending)
        assets_sorted = sorted(assets, key=lambda x: x['pct_return'], reverse=True)
        
        # Group by sector
        sectors = {}
        for asset in assets:
            sector = asset['sector']
            if sector not in sectors:
                sectors[sector] = {
                    'pct_return': 0,
                    'value': 0,
                    'assets': []
                }
            sectors[sector]['pct_return'] += asset['pct_return'] * asset['value']
            sectors[sector]['value'] += asset['value']
            sectors[sector]['assets'].append(asset)
        
        # Calculate weighted average return for sectors
        for sector in sectors.values():
            if sector['value'] > 0:
                sector['pct_return'] /= sector['value']
        
        return {
            'top_performers': assets_sorted[:3],
            'bottom_performers': assets_sorted[-3:],
            'by_sector': sectors
        }
    
    def _calculate_total_invested(self) -> float:
        """Calculate total amount invested in the portfolio."""
        total = 0
        for asset in self.portfolio.assets:
            if hasattr(asset, 'purchase_price') and hasattr(asset, 'quantity'):
                total += asset.purchase_price * asset.quantity
        return total
    
    def _get_historical_returns(self, window: Union[str, TimePeriod] = TimePeriod.ONE_YEAR) -> pd.Series:
        """
        Get historical returns for the portfolio.
        
        Args:
            window: Time window for returns (TimePeriod enum or string like '1y', '3m')
            
        Returns:
            Pandas Series of historical returns
        """
        if isinstance(window, TimePeriod):
            window = window.value
            
        # Check if we already have this data
        if window in self._historical_data:
            return self._historical_data[window]
            
        # Get historical prices for all assets
        tickers = [asset.ticker for asset in self.portfolio.assets]
        if not tickers:
            return pd.Series(dtype=float)
            
        # Use yfinance to get historical data
        data = yf.download(tickers, period=window, group_by='ticker')
        
        # Calculate daily returns for each asset
        returns = pd.DataFrame()
        for ticker in tickers:
            if ticker in data.columns.get_level_values(0):
                asset_data = data[ticker]
                if 'Adj Close' in asset_data.columns:
                    returns[ticker] = asset_data['Adj Close'].pct_change().dropna()
        
        # Calculate weighted portfolio returns
        weights = [self.portfolio.weights.get(ticker, 0) for ticker in tickers 
                  if ticker in returns.columns]
        if len(weights) != len(returns.columns):
            # Handle case where not all tickers were found
            return pd.Series(dtype=float)
            
        portfolio_returns = returns.dot(weights)
        
        # Cache the results
        self._historical_data[window] = portfolio_returns
        self._last_analysis_date = datetime.now()
        
        return portfolio_returns
    
    def _get_benchmark_returns(self, window: Union[str, TimePeriod] = TimePeriod.ONE_YEAR) -> pd.Series:
        """
        Get historical returns for the benchmark.
        
        Args:
            window: Time window for returns (TimePeriod enum or string like '1y', '3m')
            
        Returns:
            Pandas Series of historical returns for the benchmark
        """
        if isinstance(window, TimePeriod):
            window = window.value
            
        # Check if we already have this data
        if window in self._benchmark_data:
            return self._benchmark_data[window]
            
        # Skip if benchmark is CASH (risk-free rate)
        if self.benchmark == Benchmark.CASH:
            # Return daily risk-free rate (annual rate / 252 trading days)
            daily_rf = (1 + self.risk_free_rate) ** (1/252) - 1
            if window == '1m':
                days = 21
            elif window == '3m':
                days = 63
            elif window == '6m':
                days = 126
            elif window == '1y':
                days = 252
            elif window == '3y':
                days = 252 * 3
            elif window == '5y':
                days = 252 * 5
            else:  # max or other
                days = 252 * 10  # 10 years as default for max
                
            dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
            benchmark_returns = pd.Series([daily_rf] * len(dates), index=dates)
        else:
            # Get benchmark data using yfinance
            benchmark_data = yf.download(
                self.benchmark,
                period=window,
                progress=False
            )
            
            if benchmark_data.empty:
                return pd.Series(dtype=float)
                
            # Calculate daily returns
            benchmark_returns = benchmark_data['Adj Close'].pct_change().dropna()
        
        # Cache the results
        self._benchmark_data[window] = benchmark_returns
        return benchmark_returns
        
    def _calculate_annualized_return(self, returns: Union[np.ndarray, pd.Series]) -> float:
        """Calculate annualized return from daily returns."""
        if len(returns) == 0 or isinstance(returns, (float, int)):
            return 0.0
        return (1 + np.mean(returns)) ** 252 - 1
    
    def _calculate_ytd_return(self) -> float:
        """Calculate year-to-date return."""
        # This is a simplified implementation
        start_value = self._calculate_total_invested()
        current_value = self.portfolio.calculate_total_value()
        return (current_value / start_value - 1) if start_value > 0 else 0.0
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2 or np.std(returns, ddof=1) == 0:
            return 0.0
        excess_returns = returns - (self.risk_free_rate / 252)
        return np.sqrt(252) * np.mean(excess_returns) / np.std(returns, ddof=1)
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate annualized Sortino ratio."""
        if len(returns) < 2:
            return 0.0
        downside_returns = np.minimum(0, returns - (self.risk_free_rate / 252))
        if np.std(downside_returns, ddof=1) == 0:
            return 0.0
        return np.sqrt(252) * np.mean(returns - (self.risk_free_rate / 252)) / np.std(downside_returns, ddof=1)
    
    def _calculate_alpha(self, returns: np.ndarray) -> float:
        """Calculate portfolio alpha relative to a benchmark."""
        # In a real implementation, you would compare to a benchmark
        # This is a simplified version
        benchmark_returns = np.random.normal(0.0003, 0.015, len(returns))
        beta = self._calculate_beta(returns, benchmark_returns)
        return np.mean(returns) * 252 - (self.risk_free_rate + beta * (np.mean(benchmark_returns) * 252 - self.risk_free_rate))
    
    def _calculate_beta(self, returns: np.ndarray, benchmark_returns: Optional[np.ndarray] = None) -> float:
        """Calculate portfolio beta relative to a benchmark."""
        if benchmark_returns is None:
            benchmark_returns = np.random.normal(0.0003, 0.015, len(returns))
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 1.0
        covariance = np.cov(returns, benchmark_returns, ddof=1)[0, 1]
        variance = np.var(benchmark_returns, ddof=1)
        return covariance / variance if variance != 0 else 1.0
    
    def _calculate_r_squared(self, returns: np.ndarray) -> float:
        """Calculate R-squared relative to a benchmark."""
        # This is a placeholder - in a real implementation, you would use actual benchmark data
        return 0.85  # Dummy value
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        # This is a simplified implementation
        returns = self._get_historical_returns()
        cum_returns = np.cumprod(1 + returns) - 1
        peak = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - peak) / (1 + peak)
        return np.min(drawdowns) if len(drawdowns) > 0 else 0.0
    
    def _calculate_var(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk (VaR) at given confidence level."""
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def _calculate_cvar(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (CVaR) at given confidence level."""
        if len(returns) == 0:
            return 0.0
        var = self._calculate_var(returns, confidence_level)
        return np.mean(returns[returns <= var])
    
    def _calculate_tracking_error(self, returns: np.ndarray) -> float:
        """Calculate tracking error relative to a benchmark."""
        # This is a placeholder - in a real implementation, you would use actual benchmark data
        benchmark_returns = np.random.normal(0.0003, 0.015, len(returns))
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 0.0
        active_returns = returns - benchmark_returns
        return np.std(active_returns, ddof=1) * np.sqrt(252)
    
    def _calculate_information_ratio(self, returns: np.ndarray) -> float:
        """Calculate information ratio relative to a benchmark."""
        # This is a placeholder - in a real implementation, you would use actual benchmark data
        benchmark_returns = np.random.normal(0.0003, 0.015, len(returns))
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 0.0
        active_returns = returns - benchmark_returns
        tracking_error = np.std(active_returns, ddof=1)
        if tracking_error == 0:
            return 0.0
        return np.mean(active_returns) * np.sqrt(252) / tracking_error
    
    # Visualization methods would be added here
    def plot_asset_allocation(self, save_path: Optional[str] = None) -> None:
        """Generate a pie chart of asset allocation."""
        allocation = self._get_asset_allocation()
        if not allocation:
            return
            
        labels = list(allocation.keys())
        sizes = [a['weight'] for a in allocation.values()]
        
        plt.figure(figsize=(10, 8))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')
        plt.title('Asset Allocation')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_sector_allocation(self, save_path: Optional[str] = None) -> None:
        """Generate a pie chart of sector allocation."""
        sector_allocation = self._get_sector_allocation()
        if not sector_allocation:
            return
            
        labels = list(sector_allocation.keys())
        sizes = list(sector_allocation.values())
        
        plt.figure(figsize=(10, 8))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')
        plt.title('Sector Allocation')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_performance(self, save_path: Optional[str] = None) -> None:
        """Generate a line chart of portfolio performance."""
        returns = self._get_historical_returns()
        if len(returns) < 2:
            return
            
        cum_returns = np.cumprod(1 + returns) - 1
        
        plt.figure(figsize=(12, 6))
        plt.plot(cum_returns * 100)  # as percentage
        plt.title('Cumulative Returns')
        plt.xlabel('Days')
        plt.ylabel('Return (%)')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_historical_returns(self, period: Union[str, TimePeriod] = TimePeriod.ONE_YEAR, 
                              benchmark: Optional[str] = None) -> plt.Figure:
        """
        Plot the historical returns of the portfolio against a benchmark.
        
        Args:
            period: Time period to analyze (default: 1 year)
            benchmark: Benchmark to compare against (default: instance benchmark)
            
        Returns:
            Matplotlib Figure object
        """
        if benchmark is None:
            benchmark = self.benchmark
            
        # Get portfolio and benchmark returns
        portfolio_returns = self._get_historical_returns(period)
        benchmark_returns = self._get_benchmark_returns(period)
        
        if portfolio_returns.empty or benchmark_returns.empty:
            raise ValueError("Insufficient data to plot historical returns")
            
        # Calculate cumulative returns
        portfolio_cum_returns = (1 + portfolio_returns).cumprod() - 1
        benchmark_cum_returns = (1 + benchmark_returns).cumprod() - 1
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot portfolio and benchmark cumulative returns
        portfolio_cum_returns.plot(ax=ax, label='Portfolio', linewidth=2)
        benchmark_cum_returns.plot(ax=ax, label=f'Benchmark ({benchmark})', linewidth=2, linestyle='--')
        
        # Format the plot
        ax.set_title(f'Cumulative Returns ({period if isinstance(period, str) else period.value})', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Cumulative Return (%)')
        ax.yaxis.set_major_formatter(plt.FuncFormatter('{0:.0%}'.format))
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
        
    def plot_rolling_volatility(self, window: int = 21, period: Union[str, TimePeriod] = TimePeriod.ONE_YEAR) -> plt.Figure:
        """
        Plot rolling volatility of the portfolio and benchmark.
        
        Args:
            window: Rolling window in days (default: 21 days ~ 1 month)
            period: Time period to analyze (default: 1 year)
            
        Returns:
            Matplotlib Figure object
        """
        # Get portfolio and benchmark returns
        portfolio_returns = self._get_historical_returns(period)
        benchmark_returns = self._get_benchmark_returns(period)
        
        if portfolio_returns.empty or benchmark_returns.empty:
            raise ValueError("Insufficient data to plot rolling volatility")
            
        # Calculate rolling volatility (annualized)
        portfolio_vol = portfolio_returns.rolling(window=window).std() * np.sqrt(252)
        benchmark_vol = benchmark_returns.rolling(window=window).std() * np.sqrt(252)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot rolling volatility
        portfolio_vol.plot(ax=ax, label='Portfolio', linewidth=2)
        benchmark_vol.plot(ax=ax, label=f'Benchmark ({self.benchmark})', linewidth=2, linestyle='--')
        
        # Format the plot
        ax.set_title(f'{window}-Day Rolling Volatility', fontsize=14, fontweight='bold')
        ax.set_ylabel('Annualized Volatility')
        ax.yaxis.set_major_formatter(plt.FuncFormatter('{0:.0%}'.format))
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def plot_drawdown(self, period: Union[str, TimePeriod] = TimePeriod.FIVE_YEARS) -> plt.Figure:
        """
        Plot the drawdown of the portfolio and benchmark.
        
        Args:
            period: Time period to analyze (default: 5 years)
            
        Returns:
            Matplotlib Figure object
        """
        # Get portfolio and benchmark returns
        portfolio_returns = self._get_historical_returns(period)
        benchmark_returns = self._get_benchmark_returns(period)
        
        if portfolio_returns.empty or benchmark_returns.empty:
            raise ValueError("Insufficient data to plot drawdown")
            
        # Calculate cumulative returns
        portfolio_cum_returns = (1 + portfolio_returns).cumprod()
        benchmark_cum_returns = (1 + benchmark_returns).cumprod()
        
        # Calculate running maximum (peak)
        portfolio_peak = portfolio_cum_returns.expanding(min_periods=1).max()
        benchmark_peak = benchmark_cum_returns.expanding(min_periods=1).max()
        
        # Calculate drawdown
        portfolio_drawdown = (portfolio_cum_returns / portfolio_peak) - 1
        benchmark_drawdown = (benchmark_cum_returns / benchmark_peak) - 1
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot drawdown
        portfolio_drawdown.plot(ax=ax, label='Portfolio', linewidth=2)
        benchmark_drawdown.plot(ax=ax, label=f'Benchmark ({self.benchmark})', linewidth=2, linestyle='--')
        
        # Format the plot
        ax.set_title('Underwater Plot (Drawdown)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Drawdown')
        ax.yaxis.set_major_formatter(plt.FuncFormatter('{0:.0%}'.format))
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
        
    def plot_rolling_beta(self, window: int = 63, period: Union[str, TimePeriod] = TimePeriod.THREE_YEARS) -> plt.Figure:
        """
        Plot rolling beta of the portfolio against the benchmark.
        
        Args:
            window: Rolling window in days (default: 63 days ~ 3 months)
            period: Time period to analyze (default: 3 years)
            
        Returns:
            Matplotlib Figure object
        """
        # Get portfolio and benchmark returns
        portfolio_returns = self._get_historical_returns(period)
        benchmark_returns = self._get_benchmark_returns(period)
        
        if portfolio_returns.empty or benchmark_returns.empty:
            raise ValueError("Insufficient data to plot rolling beta")
            
        # Align the series
        returns_df = pd.DataFrame({
            'Portfolio': portfolio_returns,
            'Benchmark': benchmark_returns
        }).dropna()
        
        # Calculate rolling beta
        def calculate_beta(returns):
            cov = returns.cov()
            return cov.loc['Portfolio', 'Benchmark'] / returns['Benchmark'].var()
            
        rolling_beta = returns_df.rolling(window=window).apply(calculate_beta, raw=False)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot rolling beta
        rolling_beta.plot(ax=ax, linewidth=2)
        
        # Add horizontal line at beta=1 for reference
        ax.axhline(y=1, color='red', linestyle='--', linewidth=1)
        
        # Format the plot
        ax.set_title(f'{window}-Day Rolling Beta vs {self.benchmark}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Beta')
        ax.legend([f'Portfolio Beta', 'Market Beta (1.0)'])
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def get_rebalancing_suggestions(self, target_weights: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
        """
        Get suggestions for rebalancing the portfolio.
        
        Args:
            target_weights: Optional dictionary of target weights by ticker.
                          If None, uses the portfolio's current target weights.
                          
        Returns:
            Dictionary with 'current', 'target', and 'suggested_trades' information
        """
        if target_weights is None:
            target_weights = self.portfolio.weights
            
        if not target_weights:
            raise ValueError("No target weights provided and portfolio has no default weights")
            
        # Calculate current values and weights
        total_value = self.portfolio.calculate_total_value()
        current_values = {}
        current_weights = {}
        
        for asset in self.portfolio.assets:
            if asset.ticker in target_weights:
                try:
                    current_values[asset.ticker] = asset.current_value()
                except (ValueError, AttributeError):
                    # Fallback to manual calculation if current_value() fails
                    quantity = getattr(asset, 'quantity', 1)
                    latest_price = getattr(asset, 'current_price', 0) or getattr(asset, 'purchase_price', 0)
                    current_values[asset.ticker] = latest_price * quantity
                current_weights[asset.ticker] = current_values[asset.ticker] / total_value
        
        # Calculate target values and differences
        target_values = {ticker: weight * total_value for ticker, weight in target_weights.items()}
        
        # Generate suggested trades
        suggested_trades = {}
        for ticker in target_weights:
            current = current_values.get(ticker, 0)
            target = target_values[ticker]
            difference = target - current
            
            # Get current price to calculate share quantity
            asset = next((a for a in self.portfolio.assets if a.ticker == ticker), None)
            if asset and hasattr(asset, 'current_price') and asset.current_price > 0:
                share_difference = difference / asset.current_price
                suggested_trades[ticker] = {
                    'current_value': current,
                    'current_weight': current_weights.get(ticker, 0),
                    'target_value': target,
                    'target_weight': target_weights[ticker],
                    'value_to_trade': difference,
                    'shares_to_trade': share_difference,
                    'current_price': asset.current_price,
                    'action': 'BUY' if difference > 0 else 'SELL' if difference < 0 else 'HOLD'
                }
        
        return {
            'current': current_weights,
            'target': target_weights,
            'suggested_trades': suggested_trades,
            'total_value': total_value
        }
    
    def plot_rebalancing_suggestions(self, target_weights: Optional[Dict[str, float]] = None) -> plt.Figure:
        """
        Create a visual representation of rebalancing suggestions.
        
        Args:
            target_weights: Optional dictionary of target weights by ticker
            
        Returns:
            Matplotlib Figure object
        """
        rebalancing = self.get_rebalancing_suggestions(target_weights)
        
        # Prepare data for plotting
        tickers = list(rebalancing['current'].keys())
        current_weights = [rebalancing['current'][t] for t in tickers]
        target_weights = [rebalancing['target'][t] for t in tickers]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot current vs target weights
        x = np.arange(len(tickers))
        width = 0.35
        
        ax1.bar(x - width/2, current_weights, width, label='Current')
        ax1.bar(x + width/2, target_weights, width, label='Target')
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(tickers, rotation=45, ha='right')
        ax1.set_ylabel('Weight')
        ax1.set_title('Current vs Target Weights')
        ax1.legend()
        ax1.yaxis.set_major_formatter(plt.FuncFormatter('{0:.0%}'.format))
        
        # Plot suggested trades
        trades = []
        trade_values = []
        trade_labels = []
        
        for ticker, trade in rebalancing['suggested_trades'].items():
            if trade['action'] != 'HOLD':
                trades.append(trade['value_to_trade'])
                trade_labels.append(f"{ticker} ({trade['action']})")
        
        if trades:  # Only plot if there are trades to make
            colors = ['green' if t > 0 else 'red' for t in trades]
            ax2.bar(trade_labels, trades, color=colors)
            ax2.set_title('Suggested Trades')
            ax2.set_ylabel('Amount to Trade ($)')
            ax2.tick_params(axis='x', rotation=45, ha='right')
            
            # Add value labels on top of bars
            for i, v in enumerate(trades):
                ax2.text(i, v, f"${abs(v):,.0f}", ha='center', va='bottom' if v > 0 else 'top')
        else:
            ax2.text(0.5, 0.5, 'No rebalancing needed', 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax2.transAxes)
            ax2.axis('off')
        
        plt.tight_layout()
        return fig
    
    def optimize_portfolio(self, method: str = 'sharpe', target_return: Optional[float] = None) -> Dict[str, float]:
        """
        Optimize portfolio weights using mean-variance optimization.
        
        Args:
            method: Optimization method ('sharpe', 'min_vol', or 'efficient_risk')
            target_return: Required return for 'efficient_risk' method
            
        Returns:
            Dictionary of optimized weights by ticker
        """
        # Get historical returns for all assets
        tickers = [asset.ticker for asset in self.portfolio.assets]
        if not tickers:
            return {}
            
        # Get historical data
        data = yf.download(tickers, period='3y')['Adj Close']
        returns = data.pct_change().dropna()
        
        if returns.empty or len(returns) < 10:  # Need sufficient data
            return {}
        
        # Calculate expected returns and covariance matrix
        mu = returns.mean() * 252  # Annualized
        Sigma = returns.cov() * 252  # Annualized
        
        # Number of assets
        n = len(mu)
        
        # Define optimization functions
        def portfolio_return(weights):
            return np.sum(weights * mu)
            
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))
            
        def neg_sharpe_ratio(weights):
            r = portfolio_return(weights)
            vol = portfolio_volatility(weights)
            return -(r - self.risk_free_rate) / vol if vol > 0 else 1e6
            
        def min_volatility(weights):
            return portfolio_volatility(weights)
        
        # Constraints and bounds
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Sum of weights = 1
        ]
        
        # Add target return constraint if specified
        if method == 'efficient_risk' and target_return is not None:
            constraints.append({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target_return})
        
        bounds = tuple((0, 1) for _ in range(n))  # No short selling
        
        # Initial guess (equal weights)
        init_guess = np.array([1/n] * n)
        
        # Optimize
        if method == 'sharpe':
            result = minimize(
                neg_sharpe_ratio, 
                init_guess, 
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        elif method == 'min_vol':
            result = minimize(
                min_volatility,
                init_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        elif method == 'efficient_risk' and target_return is not None:
            result = minimize(
                min_volatility,
                init_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        else:
            raise ValueError("Invalid optimization method or missing target return")
        
        # Process results
        if result.success:
            optimized_weights = {ticker: weight for ticker, weight in zip(tickers, result.x)}
            
            # Filter out near-zero weights
            optimized_weights = {k: v for k, v in optimized_weights.items() if v > 1e-4}
            
            # Normalize to sum to 1
            total = sum(optimized_weights.values())
            if total > 0:
                optimized_weights = {k: v/total for k, v in optimized_weights.items()}
            
            return optimized_weights
        else:
            raise RuntimeError("Portfolio optimization failed")

# Example usage
if __name__ == "__main__":
    from portfolio import Portfolio
    from assets import StockAsset, CryptoAsset, ForexAsset, OptionAsset
    from datetime import datetime, timedelta
    
    # Create a sample portfolio
    portfolio = Portfolio("Sample Portfolio")
    
    # Add some sample assets
    aapl = StockAsset(
        ticker="AAPL",
        name="Apple Inc.",
        purchase_price=150.0,
        current_price=175.0,
        quantity=100,
        sector="Technology",
        purchase_date=datetime.now() - timedelta(days=365)
    )
    
    msft = StockAsset(
        ticker="MSFT",
        name="Microsoft Corporation",
        purchase_price=300.0,
        current_price=350.0,
        quantity=50,
        sector="Technology",
        purchase_date=datetime.now() - timedelta(days=180)
    )
    
    portfolio.add_asset(aapl, weight=0.6)
    portfolio.add_asset(msft, weight=0.4)
    
    # Create and use the PortfolioOverview
    overview = PortfolioOverview(portfolio)
    
    # Get and print the summary
    summary = overview.get_summary()
    print("Portfolio Summary:")
    print(f"Total Value: ${summary['portfolio_summary']['total_value']:,.2f}")
    print(f"Total Return: {summary['portfolio_summary']['pct_return']:.2f}%")
    print(f"Annualized Return: {summary['performance_metrics']['annualized_return']:.2f}%")
    print(f"Volatility: {summary['risk_metrics']['volatility']:.2f}%")
    print(f"Sharpe Ratio: {summary['performance_metrics']['sharpe_ratio']:.2f}")
    
    # Generate plots
    overview.plot_asset_allocation()
    overview.plot_sector_allocation()
    overview.plot_performance()
