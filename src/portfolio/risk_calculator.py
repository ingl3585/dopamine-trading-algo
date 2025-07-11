# risk_calculator.py

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Container for risk calculation results"""
    total_risk: float = 0.0
    position_risk: float = 0.0
    market_risk: float = 0.0
    correlation_risk: float = 0.0
    liquidity_risk: float = 0.0
    concentration_risk: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    max_drawdown: float = 0.0
    risk_score: float = 50.0

@dataclass
class PositionRisk:
    """Risk metrics for individual position"""
    symbol: str
    position_size: float
    unrealized_pnl: float
    position_value: float
    risk_contribution: float
    var_contribution: float
    max_loss_potential: float
    time_decay_risk: float

class RiskCalculator:
    """
    Calculates comprehensive risk metrics for portfolio and positions.
    
    Responsibilities:
    - Calculate Value at Risk (VaR) and Conditional VaR
    - Assess position-level risk contributions
    - Monitor concentration and correlation risks
    - Track maximum drawdown and risk-adjusted returns
    - Generate risk scores and alerts
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Risk calculation parameters
        self.confidence_level = config.get('var_confidence', 0.95)
        self.lookback_period = config.get('risk_lookback_days', 30)
        self.max_position_risk = config.get('max_position_risk', 0.1)  # 10%
        self.max_portfolio_risk = config.get('max_portfolio_risk', 0.2)  # 20%
        
        # Historical data for risk calculations
        self.return_history = deque(maxlen=252)  # One year of returns
        self.volatility_history = deque(maxlen=30)  # 30 days of volatility
        self.correlation_matrix = {}
        
        # Risk tracking
        self.risk_history = deque(maxlen=100)
        self.high_risk_alerts = []
        
        logger.info("Risk calculator initialized")
    
    def calculate_portfolio_risk(self, positions: Dict[str, Any],
                                market_data: Dict[str, Any],
                                historical_returns: List[float] = None) -> RiskMetrics:
        """
        Calculate comprehensive portfolio risk metrics
        
        Args:
            positions: Current portfolio positions
            market_data: Current market conditions
            historical_returns: Historical return data
            
        Returns:
            RiskMetrics object with calculated risk measures
        """
        try:
            # Calculate individual position risks
            position_risks = self._calculate_position_risks(positions, market_data)
            
            # Calculate portfolio-level risks
            total_risk = self._calculate_total_risk(position_risks, market_data)
            concentration_risk = self._calculate_concentration_risk(positions)
            market_risk = self._calculate_market_risk(market_data)
            
            # Calculate VaR and CVaR
            var_95, cvar_95 = self._calculate_var_cvar(historical_returns or [])
            
            # Calculate maximum drawdown
            max_drawdown = self._calculate_max_drawdown(historical_returns or [])
            
            # Calculate overall risk score
            risk_score = self._calculate_risk_score(
                total_risk, concentration_risk, market_risk, var_95
            )
            
            risk_metrics = RiskMetrics(
                total_risk=total_risk,
                position_risk=sum(pr.risk_contribution for pr in position_risks.values()),
                market_risk=market_risk,
                correlation_risk=0.0,  # For single instrument, correlation is minimal
                liquidity_risk=self._calculate_liquidity_risk(positions, market_data),
                concentration_risk=concentration_risk,
                var_95=var_95,
                cvar_95=cvar_95,
                max_drawdown=max_drawdown,
                risk_score=risk_score
            )
            
            # Store risk history
            self.risk_history.append({
                'timestamp': datetime.now(),
                'metrics': risk_metrics,
                'market_conditions': market_data
            })
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return RiskMetrics()
    
    def _calculate_position_risks(self, positions: Dict[str, Any],
                                 market_data: Dict[str, Any]) -> Dict[str, PositionRisk]:
        """Calculate risk for individual positions"""
        position_risks = {}
        
        try:
            total_portfolio_value = self._calculate_portfolio_value(positions)
            volatility = market_data.get('volatility', 0.02)
            
            for symbol, position in positions.items():
                if not isinstance(position, dict):
                    continue
                
                position_size = abs(position.get('current_size', 0.0))
                unrealized_pnl = position.get('unrealized_pnl', 0.0)
                entry_price = position.get('entry_price', 0.0)
                
                # Calculate position value
                current_price = market_data.get('price', entry_price)
                position_value = position_size * current_price
                
                # Calculate risk contribution
                risk_contribution = self._calculate_position_risk_contribution(
                    position_size, position_value, total_portfolio_value, volatility
                )
                
                # Calculate VaR contribution
                var_contribution = risk_contribution * 1.65  # 95% confidence multiplier
                
                # Calculate maximum loss potential
                max_loss_potential = self._calculate_max_loss_potential(
                    position, market_data
                )
                
                # Time decay risk (for time-sensitive positions)
                time_decay_risk = self._calculate_time_decay_risk(position)
                
                position_risks[symbol] = PositionRisk(
                    symbol=symbol,
                    position_size=position_size,
                    unrealized_pnl=unrealized_pnl,
                    position_value=position_value,
                    risk_contribution=risk_contribution,
                    var_contribution=var_contribution,
                    max_loss_potential=max_loss_potential,
                    time_decay_risk=time_decay_risk
                )
                
        except Exception as e:
            logger.error(f"Error calculating position risks: {e}")
        
        return position_risks
    
    def _calculate_position_risk_contribution(self, position_size: float,
                                            position_value: float,
                                            total_portfolio_value: float,
                                            volatility: float) -> float:
        """Calculate individual position's risk contribution"""
        try:
            if total_portfolio_value <= 0:
                return 0.0
            
            # Weight of position in portfolio
            position_weight = position_value / total_portfolio_value
            
            # Risk contribution = weight * volatility * position_value
            risk_contribution = position_weight * volatility * position_value
            
            return risk_contribution
            
        except Exception as e:
            logger.error(f"Error calculating position risk contribution: {e}")
            return 0.0
    
    def _calculate_total_risk(self, position_risks: Dict[str, PositionRisk],
                             market_data: Dict[str, Any]) -> float:
        """Calculate total portfolio risk"""
        try:
            # Sum of individual position risks
            position_risk_sum = sum(pr.risk_contribution for pr in position_risks.values())
            
            # Market risk adjustment
            market_volatility = market_data.get('volatility', 0.02)
            market_adjustment = 1.0 + market_volatility * 10  # Scale volatility impact
            
            # Total risk with market adjustment
            total_risk = position_risk_sum * market_adjustment
            
            # Normalize to 0-1 scale
            normalized_risk = min(1.0, total_risk / 10000.0)  # Normalize by typical portfolio size
            
            return normalized_risk
            
        except Exception as e:
            logger.error(f"Error calculating total risk: {e}")
            return 0.5
    
    def _calculate_concentration_risk(self, positions: Dict[str, Any]) -> float:
        """Calculate concentration risk of portfolio"""
        try:
            if not positions:
                return 0.0
            
            # Calculate position sizes
            position_sizes = []
            for position in positions.values():
                if isinstance(position, dict):
                    size = abs(position.get('current_size', 0.0))
                    position_sizes.append(size)
            
            if not position_sizes:
                return 0.0
            
            # Calculate Herfindahl-Hirschman Index (HHI)
            total_size = sum(position_sizes)
            if total_size == 0:
                return 0.0
            
            weights = [size / total_size for size in position_sizes]
            hhi = sum(w * w for w in weights)
            
            # Normalize HHI to risk scale (0-1)
            # HHI ranges from 1/n to 1, where n is number of positions
            min_hhi = 1.0 / len(positions)
            concentration_risk = (hhi - min_hhi) / (1.0 - min_hhi)
            
            return concentration_risk
            
        except Exception as e:
            logger.error(f"Error calculating concentration risk: {e}")
            return 0.0
    
    def _calculate_market_risk(self, market_data: Dict[str, Any]) -> float:
        """Calculate market risk based on current conditions"""
        try:
            volatility = market_data.get('volatility', 0.02)
            volume = market_data.get('volume', 1000)
            spread = market_data.get('spread', 0.01)
            
            # Normalize volatility risk (0-1 scale)
            volatility_risk = min(1.0, volatility / 0.1)  # 10% volatility = max risk
            
            # Volume risk (lower volume = higher risk)
            volume_risk = max(0.0, 1.0 - volume / 10000.0)  # Normalize by typical volume
            
            # Spread risk
            spread_risk = min(1.0, spread / 0.05)  # 5% spread = max risk
            
            # Combined market risk
            market_risk = (volatility_risk * 0.6 + volume_risk * 0.2 + spread_risk * 0.2)
            
            return market_risk
            
        except Exception as e:
            logger.error(f"Error calculating market risk: {e}")
            return 0.3  # Default medium risk
    
    def _calculate_liquidity_risk(self, positions: Dict[str, Any],
                                 market_data: Dict[str, Any]) -> float:
        """Calculate liquidity risk"""
        try:
            # For MNQ futures, liquidity is generally high
            # Risk increases with position size relative to market volume
            
            total_position_size = sum(
                abs(pos.get('current_size', 0.0)) for pos in positions.values()
                if isinstance(pos, dict)
            )
            
            market_volume = market_data.get('volume', 1000)
            
            # Liquidity risk = position_size / market_volume
            liquidity_risk = min(1.0, total_position_size / market_volume)
            
            return liquidity_risk
            
        except Exception as e:
            logger.error(f"Error calculating liquidity risk: {e}")
            return 0.1  # Default low risk for liquid futures
    
    def _calculate_var_cvar(self, historical_returns: List[float]) -> Tuple[float, float]:
        """Calculate Value at Risk and Conditional Value at Risk"""
        try:
            if not historical_returns or len(historical_returns) < 20:
                return 0.0, 0.0
            
            returns_array = np.array(historical_returns)
            
            # Calculate VaR at confidence level
            var_95 = np.percentile(returns_array, (1 - self.confidence_level) * 100)
            
            # Calculate CVaR (Expected Shortfall)
            cvar_95 = np.mean(returns_array[returns_array <= var_95]) if var_95 < 0 else 0.0
            
            return abs(var_95), abs(cvar_95)
            
        except Exception as e:
            logger.error(f"Error calculating VaR/CVaR: {e}")
            return 0.0, 0.0
    
    def _calculate_max_drawdown(self, historical_returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        try:
            if not historical_returns:
                return 0.0
            
            cumulative_returns = np.cumsum(historical_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = running_max - cumulative_returns
            max_drawdown = np.max(drawdown)
            
            return max_drawdown
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _calculate_max_loss_potential(self, position: Dict[str, Any],
                                     market_data: Dict[str, Any]) -> float:
        """Calculate maximum potential loss for position"""
        try:
            position_size = abs(position.get('current_size', 0.0))
            entry_price = position.get('entry_price', 0.0)
            current_price = market_data.get('price', entry_price)
            
            # Assume maximum adverse movement based on volatility
            volatility = market_data.get('volatility', 0.02)
            max_adverse_movement = volatility * 3  # 3 standard deviations
            
            # Calculate potential loss
            if position.get('current_size', 0.0) > 0:  # Long position
                worst_price = current_price * (1 - max_adverse_movement)
                max_loss = position_size * (entry_price - worst_price)
            else:  # Short position
                worst_price = current_price * (1 + max_adverse_movement)
                max_loss = position_size * (worst_price - entry_price)
            
            return max(0.0, max_loss)
            
        except Exception as e:
            logger.error(f"Error calculating max loss potential: {e}")
            return 0.0
    
    def _calculate_time_decay_risk(self, position: Dict[str, Any]) -> float:
        """Calculate time decay risk for position"""
        try:
            entry_time = position.get('entry_time')
            if not entry_time:
                return 0.0
            
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time)
            elif not isinstance(entry_time, datetime):
                return 0.0
            
            hold_time = (datetime.now() - entry_time).total_seconds() / 3600  # hours
            max_hold_time = self.config.get('max_hold_time_hours', 24)
            
            # Time decay risk increases as position ages
            time_decay_risk = min(1.0, hold_time / max_hold_time)
            
            return time_decay_risk
            
        except Exception as e:
            logger.error(f"Error calculating time decay risk: {e}")
            return 0.0
    
    def _calculate_portfolio_value(self, positions: Dict[str, Any]) -> float:
        """Calculate total portfolio value"""
        try:
            total_value = 0.0
            
            for position in positions.values():
                if isinstance(position, dict):
                    position_size = abs(position.get('current_size', 0.0))
                    entry_price = position.get('entry_price', 0.0)
                    unrealized_pnl = position.get('unrealized_pnl', 0.0)
                    
                    # Position value = size * price + unrealized P&L
                    position_value = position_size * entry_price + unrealized_pnl
                    total_value += position_value
            
            return total_value
            
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {e}")
            return 0.0
    
    def _calculate_risk_score(self, total_risk: float, concentration_risk: float,
                             market_risk: float, var_95: float) -> float:
        """Calculate overall risk score (0-100, lower is better)"""
        try:
            # Weighted combination of risk factors
            weights = {
                'total_risk': 0.4,
                'concentration_risk': 0.2,
                'market_risk': 0.2,
                'var_risk': 0.2
            }
            
            # Normalize VaR to 0-1 scale
            var_risk = min(1.0, var_95 / 1000.0)  # Normalize by typical loss amount
            
            # Calculate weighted risk score
            risk_score = (
                total_risk * weights['total_risk'] +
                concentration_risk * weights['concentration_risk'] +
                market_risk * weights['market_risk'] +
                var_risk * weights['var_risk']
            )
            
            # Convert to 0-100 scale (higher = more risk)
            return min(100.0, risk_score * 100.0)
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 50.0
    
    def get_risk_summary(self, risk_metrics: RiskMetrics) -> Dict[str, Any]:
        """Get summary of risk metrics"""
        try:
            # Risk level classification
            if risk_metrics.risk_score >= 80:
                risk_level = "CRITICAL"
            elif risk_metrics.risk_score >= 60:
                risk_level = "HIGH"
            elif risk_metrics.risk_score >= 40:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            return {
                'risk_level': risk_level,
                'risk_score': risk_metrics.risk_score,
                'total_risk': risk_metrics.total_risk,
                'concentration_risk': risk_metrics.concentration_risk,
                'market_risk': risk_metrics.market_risk,
                'var_95': risk_metrics.var_95,
                'cvar_95': risk_metrics.cvar_95,
                'max_drawdown': risk_metrics.max_drawdown,
                'risk_alerts': self._generate_risk_alerts(risk_metrics)
            }
            
        except Exception as e:
            logger.error(f"Error getting risk summary: {e}")
            return {}
    
    def _generate_risk_alerts(self, risk_metrics: RiskMetrics) -> List[str]:
        """Generate risk alerts based on metrics"""
        alerts = []
        
        try:
            if risk_metrics.total_risk > 0.8:
                alerts.append("Total portfolio risk exceeds 80%")
            
            if risk_metrics.concentration_risk > 0.7:
                alerts.append("High concentration risk detected")
            
            if risk_metrics.market_risk > 0.6:
                alerts.append("Elevated market risk conditions")
            
            if risk_metrics.var_95 > 1000:  # Adjust threshold as needed
                alerts.append(f"High VaR: ${risk_metrics.var_95:.2f}")
            
            if risk_metrics.max_drawdown > 0.1:  # 10% drawdown
                alerts.append(f"Maximum drawdown: {risk_metrics.max_drawdown:.1%}")
            
        except Exception as e:
            logger.error(f"Error generating risk alerts: {e}")
        
        return alerts
    
    def update_risk_history(self, returns: List[float], volatility: float):
        """Update historical data for risk calculations"""
        try:
            # Update return history
            self.return_history.extend(returns)
            
            # Update volatility history
            self.volatility_history.append(volatility)
            
            logger.debug(f"Updated risk history: {len(self.return_history)} returns, "
                        f"{len(self.volatility_history)} volatility readings")
            
        except Exception as e:
            logger.error(f"Error updating risk history: {e}")
    
    def get_risk_trends(self) -> Dict[str, Any]:
        """Get risk trend analysis"""
        try:
            if len(self.risk_history) < 2:
                return {}
            
            recent_risks = [r['metrics'].risk_score for r in self.risk_history[-10:]]
            
            return {
                'current_risk': recent_risks[-1] if recent_risks else 50.0,
                'avg_risk_10_periods': np.mean(recent_risks),
                'risk_trend': 'increasing' if recent_risks[-1] > recent_risks[0] else 'decreasing',
                'risk_volatility': np.std(recent_risks),
                'periods_tracked': len(self.risk_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting risk trends: {e}")
            return {}