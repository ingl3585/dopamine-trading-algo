# performance_analyzer.py

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    total_return: float = 0.0
    avg_return: float = 0.0
    return_volatility: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 1.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    performance_score: float = 50.0

@dataclass
class TradeMetrics:
    """Container for trade-based metrics"""
    total_trades: int = 0
    avg_hold_time: float = 0.0
    median_hold_time: float = 0.0
    avg_mae: float = 0.0  # Maximum Adverse Excursion
    avg_mfe: float = 0.0  # Maximum Favorable Excursion
    consecutive_losses: int = 0
    consecutive_wins: int = 0

class PerformanceAnalyzer:
    """
    Analyzes trading performance and calculates comprehensive metrics.
    
    Responsibilities:
    - Calculate returns-based metrics (Sharpe ratio, volatility, etc.)
    - Calculate risk metrics (drawdown, VaR, etc.)
    - Analyze trade patterns and statistics
    - Generate performance scores and ratings
    - Track performance over time
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.daily_returns = deque(maxlen=252)  # One year of daily returns
        self.performance_history = deque(maxlen=500)
        
        # Performance tracking
        self.cumulative_pnl = 0.0
        self.high_water_mark = 0.0
        self.max_drawdown_seen = 0.0
        
        logger.info("Performance analyzer initialized")
    
    def analyze_returns(self, trade_history: List[Any]) -> Dict[str, Any]:
        """
        Analyze returns-based performance metrics
        
        Args:
            trade_history: List of completed trades
            
        Returns:
            Dictionary of returns metrics
        """
        try:
            if not trade_history:
                return {}
            
            # Extract PnL values
            pnls = [getattr(trade, 'pnl', 0.0) for trade in trade_history]
            
            if not pnls:
                return {}
            
            # Calculate basic return metrics
            total_return = sum(pnls)
            avg_return = np.mean(pnls)
            return_volatility = np.std(pnls) if len(pnls) > 1 else 0.0
            
            # Win/loss analysis
            wins = [pnl for pnl in pnls if pnl > 0]
            losses = [pnl for pnl in pnls if pnl < 0]
            
            win_rate = len(wins) / len(pnls) if pnls else 0.0
            avg_win = np.mean(wins) if wins else 0.0
            avg_loss = abs(np.mean(losses)) if losses else 0.0
            
            # Profit factor
            total_wins = sum(wins) if wins else 0.0
            total_losses = abs(sum(losses)) if losses else 0.0
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Sharpe ratio (simplified)
            sharpe_ratio = avg_return / return_volatility if return_volatility > 0 else 0.0
            
            return {
                'total_return': total_return,
                'avg_return': avg_return,
                'return_volatility': return_volatility,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'winning_trades': len(wins),
                'losing_trades': len(losses),
                'total_trades': len(pnls)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing returns: {e}")
            return {}
    
    def analyze_risk(self, trade_history: List[Any]) -> Dict[str, Any]:
        """
        Analyze risk-based performance metrics
        
        Args:
            trade_history: List of completed trades
            
        Returns:
            Dictionary of risk metrics
        """
        try:
            if not trade_history:
                return {}
            
            pnls = [getattr(trade, 'pnl', 0.0) for trade in trade_history]
            
            if not pnls:
                return {}
            
            # Maximum drawdown calculation
            cumulative_pnl = np.cumsum(pnls)
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdown = running_max - cumulative_pnl
            max_drawdown = np.max(drawdown)
            
            # Value at Risk (VaR) - 95% confidence
            var_95 = np.percentile(pnls, 5) if len(pnls) > 20 else min(pnls)
            
            # Conditional VaR (Expected Shortfall)
            cvar_95 = np.mean([pnl for pnl in pnls if pnl <= var_95]) if var_95 < 0 else 0.0
            
            # Downside deviation
            negative_returns = [pnl for pnl in pnls if pnl < 0]
            downside_deviation = np.std(negative_returns) if negative_returns else 0.0
            
            # Risk-adjusted return
            avg_return = np.mean(pnls)
            risk_adjusted_return = avg_return / max_drawdown if max_drawdown > 0 else 0.0
            
            return {
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'downside_deviation': downside_deviation,
                'risk_adjusted_return': risk_adjusted_return,
                'calmar_ratio': avg_return / max_drawdown if max_drawdown > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing risk: {e}")
            return {}
    
    def analyze_trades(self, trade_history: List[Any]) -> Dict[str, Any]:
        """
        Analyze trade-based metrics
        
        Args:
            trade_history: List of completed trades
            
        Returns:
            Dictionary of trade analytics
        """
        try:
            if not trade_history:
                return {}
            
            # Hold time analysis
            hold_times = []
            for trade in trade_history:
                if hasattr(trade, 'entry_time') and hasattr(trade, 'exit_time'):
                    hold_time = trade.exit_time - trade.entry_time
                    hold_times.append(hold_time / 3600)  # Convert to hours
            
            # MAE/MFE analysis (if available)
            maes = []
            mfes = []
            for trade in trade_history:
                if hasattr(trade, 'max_adverse_excursion'):
                    maes.append(getattr(trade, 'max_adverse_excursion', 0.0))
                if hasattr(trade, 'max_favorable_excursion'):
                    mfes.append(getattr(trade, 'max_favorable_excursion', 0.0))
            
            # Consecutive wins/losses
            consecutive_losses = self._calculate_consecutive_losses(trade_history)
            consecutive_wins = self._calculate_consecutive_wins(trade_history)
            
            return {
                'total_trades': len(trade_history),
                'avg_hold_time': np.mean(hold_times) if hold_times else 0.0,
                'median_hold_time': np.median(hold_times) if hold_times else 0.0,
                'max_hold_time': np.max(hold_times) if hold_times else 0.0,
                'min_hold_time': np.min(hold_times) if hold_times else 0.0,
                'avg_mae': np.mean(maes) if maes else 0.0,
                'avg_mfe': np.mean(mfes) if mfes else 0.0,
                'consecutive_losses': consecutive_losses,
                'consecutive_wins': consecutive_wins,
                'trade_frequency': len(trade_history) / max(1, len(hold_times)) if hold_times else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trades: {e}")
            return {}
    
    def _calculate_consecutive_losses(self, trade_history: List[Any]) -> int:
        """Calculate current consecutive losses"""
        if not trade_history:
            return 0
        
        consecutive = 0
        for trade in reversed(trade_history):
            pnl = getattr(trade, 'pnl', 0.0)
            if pnl < 0:
                consecutive += 1
            else:
                break
        
        return consecutive
    
    def _calculate_consecutive_wins(self, trade_history: List[Any]) -> int:
        """Calculate current consecutive wins"""
        if not trade_history:
            return 0
        
        consecutive = 0
        for trade in reversed(trade_history):
            pnl = getattr(trade, 'pnl', 0.0)
            if pnl > 0:
                consecutive += 1
            else:
                break
        
        return consecutive
    
    def calculate_performance_score(self, returns_metrics: Dict[str, Any], 
                                   risk_metrics: Dict[str, Any]) -> float:
        """
        Calculate overall performance score (0-100)
        
        Args:
            returns_metrics: Returns analysis results
            risk_metrics: Risk analysis results
            
        Returns:
            Performance score between 0-100
        """
        try:
            # Extract key metrics
            sharpe_ratio = returns_metrics.get('sharpe_ratio', 0.0)
            win_rate = returns_metrics.get('win_rate', 0.0)
            profit_factor = min(returns_metrics.get('profit_factor', 1.0), 5.0)  # Cap at 5
            max_drawdown = abs(risk_metrics.get('max_drawdown', 0.0))
            
            # Normalize metrics to 0-100 scale
            sharpe_score = min(100, max(0, sharpe_ratio * 50 + 50))
            win_rate_score = win_rate * 100
            profit_factor_score = min(100, profit_factor * 20)
            drawdown_score = max(0, 100 - max_drawdown * 10)
            
            # Weighted average (customizable weights)
            weights = {
                'sharpe': 0.3,
                'win_rate': 0.25,
                'profit_factor': 0.25,
                'drawdown': 0.2
            }
            
            performance_score = (
                sharpe_score * weights['sharpe'] +
                win_rate_score * weights['win_rate'] +
                profit_factor_score * weights['profit_factor'] +
                drawdown_score * weights['drawdown']
            )
            
            return min(100, max(0, performance_score))
            
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 50.0  # Neutral score
    
    def get_comprehensive_analytics(self, trade_history: List[Any], 
                                   position_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get comprehensive performance analytics
        
        Args:
            trade_history: List of completed trades
            position_history: List of position performance records
            
        Returns:
            Complete analytics dictionary
        """
        try:
            # Calculate all metrics
            returns_metrics = self.analyze_returns(trade_history)
            risk_metrics = self.analyze_risk(trade_history)
            trade_metrics = self.analyze_trades(trade_history)
            
            # Calculate performance score
            performance_score = self.calculate_performance_score(returns_metrics, risk_metrics)
            
            # Additional analysis from position history
            position_analytics = self._analyze_position_history(position_history)
            
            return {
                'returns': returns_metrics,
                'risk': risk_metrics,
                'trades': trade_metrics,
                'positions': position_analytics,
                'performance_score': performance_score,
                'summary': {
                    'total_trades': returns_metrics.get('total_trades', 0),
                    'win_rate': returns_metrics.get('win_rate', 0.0),
                    'profit_factor': returns_metrics.get('profit_factor', 1.0),
                    'max_drawdown': risk_metrics.get('max_drawdown', 0.0),
                    'sharpe_ratio': returns_metrics.get('sharpe_ratio', 0.0),
                    'performance_score': performance_score
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting comprehensive analytics: {e}")
            return {}
    
    def _analyze_position_history(self, position_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze position history for additional insights"""
        try:
            if not position_history:
                return {}
            
            # Extract position metrics
            hold_times = [pos.get('hold_time_hours', 0) for pos in position_history]
            pnls = [pos.get('total_pnl', 0) for pos in position_history]
            maes = [pos.get('max_adverse_excursion', 0) for pos in position_history]
            mfes = [pos.get('max_favorable_excursion', 0) for pos in position_history]
            
            return {
                'total_positions': len(position_history),
                'avg_hold_time_hours': np.mean(hold_times) if hold_times else 0.0,
                'avg_pnl': np.mean(pnls) if pnls else 0.0,
                'avg_mae': np.mean(maes) if maes else 0.0,
                'avg_mfe': np.mean(mfes) if mfes else 0.0,
                'longest_hold': max(hold_times) if hold_times else 0.0,
                'shortest_hold': min(hold_times) if hold_times else 0.0,
                'best_position': max(pnls) if pnls else 0.0,
                'worst_position': min(pnls) if pnls else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing position history: {e}")
            return {}
    
    def get_performance_summary(self, trade_history: List[Any]) -> Dict[str, Any]:
        """
        Get a simplified performance summary compatible with existing interfaces
        
        Args:
            trade_history: List of completed trades
            
        Returns:
            Performance summary dictionary
        """
        try:
            analytics = self.get_comprehensive_analytics(trade_history, [])
            returns_metrics = analytics.get('returns', {})
            risk_metrics = analytics.get('risk', {})
            trade_metrics = analytics.get('trades', {})
            
            return {
                'total_trades': returns_metrics.get('total_trades', 0),
                'winning_trades': returns_metrics.get('winning_trades', 0),
                'losing_trades': returns_metrics.get('losing_trades', 0),
                'win_rate': returns_metrics.get('win_rate', 0.0),
                'total_pnl': returns_metrics.get('total_return', 0.0),
                'avg_pnl_per_trade': returns_metrics.get('avg_return', 0.0),
                'consecutive_losses': trade_metrics.get('consecutive_losses', 0),
                'max_drawdown_pct': abs(risk_metrics.get('max_drawdown', 0.0)) * 100,
                'profit_factor': returns_metrics.get('profit_factor', 1.0),
                'sharpe_ratio': returns_metrics.get('sharpe_ratio', 0.0),
                'performance_score': analytics.get('performance_score', 50.0)
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl_per_trade': 0.0,
                'consecutive_losses': 0,
                'max_drawdown_pct': 0.0,
                'profit_factor': 1.0,
                'sharpe_ratio': 0.0,
                'performance_score': 50.0
            }
    
    def update_daily_returns(self, daily_pnl: float):
        """Update daily returns tracking"""
        self.daily_returns.append(daily_pnl)
        
        # Update cumulative tracking
        self.cumulative_pnl += daily_pnl
        self.high_water_mark = max(self.high_water_mark, self.cumulative_pnl)
        
        # Update max drawdown
        current_drawdown = self.high_water_mark - self.cumulative_pnl
        self.max_drawdown_seen = max(self.max_drawdown_seen, current_drawdown)
    
    def get_daily_metrics(self) -> Dict[str, Any]:
        """Get daily performance metrics"""
        if not self.daily_returns:
            return {}
        
        returns_array = np.array(self.daily_returns)
        
        return {
            'daily_avg_return': np.mean(returns_array),
            'daily_volatility': np.std(returns_array),
            'daily_sharpe': np.mean(returns_array) / np.std(returns_array) if np.std(returns_array) > 0 else 0.0,
            'cumulative_pnl': self.cumulative_pnl,
            'high_water_mark': self.high_water_mark,
            'current_drawdown': self.high_water_mark - self.cumulative_pnl,
            'max_drawdown_seen': self.max_drawdown_seen,
            'days_tracked': len(self.daily_returns)
        }