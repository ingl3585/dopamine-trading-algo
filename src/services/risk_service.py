"""
Risk Management Service - Core risk assessment and position sizing
"""

import numpy as np
import logging
from typing import Dict
from datetime import datetime

from src.shared.types import TradeDecision, AccountInfo

logger = logging.getLogger(__name__)

class RiskManagementService:
    """
    Core risk management service implementing Kelly criterion and dynamic risk assessment
    """
    
    def __init__(self, config):
        self.config = config
        self.max_position_size = config.get('max_position_size', 0.1)  # 10% of account
        self.max_daily_loss = config.get('max_daily_loss', 0.02)  # 2% daily loss limit
        self.kelly_lookback = config.get('kelly_lookback', 100)
        self.trade_history = []
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()
        
    def assess_risk(self, decision: TradeDecision, account: AccountInfo) -> float:
        """Assess risk level for a trading decision"""
        try:
            # Check daily loss limits
            current_date = datetime.now().date()
            if current_date != self.last_reset_date:
                self.daily_pnl = 0.0
                self.last_reset_date = current_date
            
            # Calculate various risk factors
            risk_factors = {}
            
            # 1. Position size risk
            position_risk = self._calculate_position_risk(decision.size, account)
            risk_factors['position_risk'] = position_risk
            
            # 2. Confidence risk (inverse of confidence)
            confidence_risk = 1.0 - decision.confidence
            risk_factors['confidence_risk'] = confidence_risk
            
            # 3. Daily loss risk
            daily_loss_risk = self._calculate_daily_loss_risk(account)
            risk_factors['daily_loss_risk'] = daily_loss_risk
            
            # 4. Volatility risk
            volatility_risk = self._calculate_volatility_risk(decision)
            risk_factors['volatility_risk'] = volatility_risk
            
            # 5. Correlation risk (if multiple positions)
            correlation_risk = self._calculate_correlation_risk(account)
            risk_factors['correlation_risk'] = correlation_risk
            
            # Combine risk factors with weights
            weights = {
                'position_risk': 0.3,
                'confidence_risk': 0.25,
                'daily_loss_risk': 0.2,
                'volatility_risk': 0.15,
                'correlation_risk': 0.1
            }
            
            total_risk = sum(risk_factors[factor] * weights[factor] for factor in risk_factors)
            
            # Ensure risk is bounded [0, 1]
            total_risk = max(0.0, min(1.0, total_risk))
            
            logger.debug(f"Risk assessment: {risk_factors}, total_risk: {total_risk:.3f}")
            
            return total_risk
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            return 1.0  # Conservative: high risk if error
    
    def size_position(self, signal_strength: float, account: AccountInfo, 
                     current_risk: float = 0.0) -> float:
        """Calculate optimal position size using Kelly criterion and risk constraints"""
        try:
            # Base Kelly criterion calculation
            kelly_fraction = self._calculate_kelly_fraction()
            
            # Adjust Kelly based on signal strength
            adjusted_kelly = kelly_fraction * abs(signal_strength)
            
            # Apply risk constraints
            risk_adjusted_kelly = adjusted_kelly * (1.0 - current_risk)
            
            # Apply maximum position size constraint
            max_fraction = min(self.max_position_size, risk_adjusted_kelly)
            
            # Calculate actual position size based on buying power
            max_position_value = account.buying_power * max_fraction
            
            # Account for leverage (futures typically 50:1 leverage for MNQ)
            leverage = self.config.get('leverage', 50.0)
            position_size = max_position_value / leverage
            
            # Ensure minimum position size
            min_position = self.config.get('min_position_size', 1.0)
            if position_size > 0 and position_size < min_position:
                position_size = min_position
            
            # Round to valid increment
            position_increment = self.config.get('position_increment', 1.0)
            position_size = round(position_size / position_increment) * position_increment
            
            logger.debug(f"Position sizing: kelly={kelly_fraction:.3f}, signal={signal_strength:.3f}, "
                        f"risk={current_risk:.3f}, final_size={position_size}")
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error in position sizing: {e}")
            return 1.0  # Default minimum position
    
    def update_trade_outcome(self, pnl: float, duration: float):
        """Update trade history for Kelly calculation"""
        try:
            self.trade_history.append({
                'pnl': pnl,
                'duration': duration,
                'timestamp': datetime.now()
            })
            
            # Keep only recent trades for Kelly calculation
            if len(self.trade_history) > self.kelly_lookback:
                self.trade_history = self.trade_history[-self.kelly_lookback:]
            
            # Update daily PnL
            self.daily_pnl += pnl
            
        except Exception as e:
            logger.error(f"Error updating trade outcome: {e}")
    
    def _calculate_kelly_fraction(self) -> float:
        """Calculate Kelly fraction based on trade history"""
        try:
            if len(self.trade_history) < 10:  # Need minimum trade history
                return 0.01  # Conservative default
            
            pnls = [trade['pnl'] for trade in self.trade_history]
            
            # Calculate win rate and average win/loss
            wins = [pnl for pnl in pnls if pnl > 0]
            losses = [pnl for pnl in pnls if pnl < 0]
            
            if not wins or not losses:
                return 0.01  # Conservative if no wins or losses
            
            win_rate = len(wins) / len(pnls)
            avg_win = np.mean(wins)
            avg_loss = abs(np.mean(losses))
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
            if avg_loss == 0:
                return 0.01
            
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Conservative bounds on Kelly
            kelly_fraction = max(0.001, min(0.1, kelly_fraction))
            
            return kelly_fraction
            
        except Exception as e:
            logger.error(f"Error calculating Kelly fraction: {e}")
            return 0.01
    
    def _calculate_position_risk(self, position_size: float, account: AccountInfo) -> float:
        """Calculate risk based on position size relative to account"""
        try:
            if account.buying_power == 0:
                return 1.0
            
            position_value = position_size * self.config.get('contract_value', 2000)  # MNQ contract value
            risk_ratio = position_value / account.buying_power
            
            # Risk increases exponentially with position size
            return min(1.0, risk_ratio / self.max_position_size)
            
        except Exception as e:
            logger.error(f"Error calculating position risk: {e}")
            return 0.5
    
    def _calculate_daily_loss_risk(self, account: AccountInfo) -> float:
        """Calculate risk based on daily losses"""
        try:
            if account.buying_power == 0:
                return 1.0
            
            daily_loss_ratio = abs(self.daily_pnl) / account.buying_power
            
            # Risk increases as we approach daily loss limit
            return min(1.0, daily_loss_ratio / self.max_daily_loss)
            
        except Exception as e:
            logger.error(f"Error calculating daily loss risk: {e}")
            return 0.0
    
    def _calculate_volatility_risk(self, decision: TradeDecision) -> float:
        """Calculate risk based on market volatility"""
        try:
            # Extract volatility from decision context if available
            volatility = decision.reasoning.get('volatility', 0.02) if hasattr(decision, 'reasoning') else 0.02
            
            # Higher volatility = higher risk
            # Normal volatility around 0.02, high volatility > 0.05
            normalized_vol = min(1.0, volatility / 0.05)
            
            return normalized_vol
            
        except Exception as e:
            logger.error(f"Error calculating volatility risk: {e}")
            return 0.5
    
    def _calculate_correlation_risk(self, account: AccountInfo) -> float:
        """Calculate risk based on position correlation"""
        try:
            # For single instrument (MNQ), correlation risk is low
            # In multi-instrument system, this would check correlation between positions
            
            # Risk increases with position concentration
            position_concentration = account.position_size / account.buying_power if account.buying_power > 0 else 0
            
            return min(1.0, position_concentration * 2)  # Risk increases with concentration
            
        except Exception as e:
            logger.error(f"Error calculating correlation risk: {e}")
            return 0.1
    
    def get_risk_metrics(self) -> Dict:
        """Get comprehensive risk metrics"""
        try:
            kelly_fraction = self._calculate_kelly_fraction()
            
            # Calculate additional metrics
            if self.trade_history:
                pnls = [trade['pnl'] for trade in self.trade_history]
                win_rate = len([p for p in pnls if p > 0]) / len(pnls)
                sharpe_ratio = np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0
                max_drawdown = self._calculate_max_drawdown()
            else:
                win_rate = 0.0
                sharpe_ratio = 0.0
                max_drawdown = 0.0
            
            return {
                'kelly_fraction': kelly_fraction,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'daily_pnl': self.daily_pnl,
                'trades_count': len(self.trade_history),
                'max_position_size': self.max_position_size,
                'max_daily_loss': self.max_daily_loss
            }
            
        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
            return {}
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from trade history"""
        try:
            if not self.trade_history:
                return 0.0
            
            cumulative_pnl = np.cumsum([trade['pnl'] for trade in self.trade_history])
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdown = running_max - cumulative_pnl
            
            return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0