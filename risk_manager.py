# risk_manager.py

from dataclasses import dataclass
from typing import Optional, Dict
import logging

from trading_agent import Decision
from data_processor import MarketData
from advanced_risk import AdvancedRiskManager

logger = logging.getLogger(__name__)

@dataclass
class Order:
    action: str
    size: int
    price: float
    stop_price: float = 0.0
    target_price: float = 0.0
    timestamp: float = 0.0
    confidence: float = 0.0
    features: Optional[object] = None
    market_data: Optional[object] = None
    intelligence_data: Optional[Dict] = None
    decision_data: Optional[Dict] = None


class RiskManager:
    def __init__(self, portfolio, meta_learner):
        self.portfolio = portfolio
        self.meta_learner = meta_learner
        self.advanced_risk = AdvancedRiskManager(meta_learner)
        
    def validate_order(self, decision: Decision, market_data: MarketData) -> Optional[Order]:
        if decision.action == 'hold':
            return None
        
        # Update advanced risk metrics
        self.advanced_risk.update_risk_metrics(market_data)
        
        # Check learned loss tolerance using actual account balance
        loss_tolerance = self.meta_learner.get_parameter('loss_tolerance_factor')
        max_loss = market_data.account_balance * loss_tolerance
        if market_data.daily_pnl <= -max_loss:
            logger.info(f"Daily loss limit reached: {market_data.daily_pnl:.2f} <= -{max_loss:.2f}")
            return None
        
        # Check learned consecutive loss tolerance
        consecutive_tolerance = self.meta_learner.get_parameter('consecutive_loss_tolerance')
        if self.portfolio.get_consecutive_losses() >= consecutive_tolerance:
            logger.info(f"Consecutive loss limit reached: {self.portfolio.get_consecutive_losses()}")
            return None
        
        # Enhanced margin utilization check
        if market_data.margin_utilization > 0.8:  # Don't exceed 80% margin usage
            logger.info(f"Margin utilization too high: {market_data.margin_utilization:.1%}")
            return None
        
        # Calculate position size using enhanced account data
        size = self._calculate_adaptive_position_size(decision, market_data)
        if size == 0:
            logger.info("Position size calculated as 0")
            return None
        
        # Apply advanced risk management - Kelly optimization and drawdown prevention
        intelligence_data = getattr(decision, 'intelligence_data', {})
        kelly_optimized_size = self.advanced_risk.optimize_kelly_position_size(
            size, market_data, intelligence_data
        )
        
        # Real-time drawdown prevention
        approved, final_size, reason = self.advanced_risk.check_drawdown_prevention(
            kelly_optimized_size, market_data
        )
        
        if not approved:
            logger.info(f"Order rejected by advanced risk management: {reason}")
            return None
        
        if final_size != kelly_optimized_size:
            logger.info(f"Position size adjusted by advanced risk: {kelly_optimized_size} -> {final_size} ({reason})")
        
        # Apply learned stop/target preferences
        stop_price, target_price = self._calculate_adaptive_levels(decision, market_data)
        
        return Order(
            action=decision.action,
            size=final_size,
            price=market_data.price,
            stop_price=stop_price,
            target_price=target_price,
            timestamp=market_data.timestamp,
            confidence=decision.confidence
        )
    
    def _calculate_adaptive_position_size(self, decision: Decision, market_data: MarketData) -> int:
        # Use enhanced account data from NinjaTrader
        available_margin = market_data.available_margin
        account_balance = market_data.account_balance
        buying_power = market_data.buying_power
        
        # Learned position sizing factors
        position_factor = self.meta_learner.get_parameter('position_size_factor')
        max_position_factor = self.meta_learner.get_parameter('max_position_factor')
        
        # Calculate based on confidence and available capital
        confidence_multiplier = decision.confidence
        
        # Multiple sizing approaches - use the most conservative
        sizing_approaches = []
        
        # 1. Based on available margin
        if available_margin > 0:
            # Margin per contract for MNQ
            estimated_margin_per_contract = 500
            max_by_margin = int((available_margin * position_factor * confidence_multiplier) / estimated_margin_per_contract)
            sizing_approaches.append(max_by_margin)
        
        # 2. Based on account balance percentage
        max_risk_per_trade = account_balance * position_factor * confidence_multiplier
        estimated_margin_per_contract = 500
        max_by_balance = int(max_risk_per_trade / estimated_margin_per_contract)
        sizing_approaches.append(max_by_balance)
        
        # 3. Based on buying power (more conservative)
        if buying_power > 0:
            max_by_buying_power = int((buying_power * position_factor * 0.5 * confidence_multiplier) / estimated_margin_per_contract)
            sizing_approaches.append(max_by_buying_power)
        
        # 4. Apply maximum position factor limit
        max_contracts_absolute = int(account_balance * max_position_factor / estimated_margin_per_contract)
        sizing_approaches.append(max_contracts_absolute)
        
        # 5. Use decision size as upper bound
        sizing_approaches.append(int(decision.size))
        
        # Take the minimum (most conservative)
        final_size = max(0, min(sizing_approaches)) if sizing_approaches else 0
        
        # Additional safety: never risk more than 2% of account per trade
        max_safe_size = max(1, int(account_balance * 0.02 / estimated_margin_per_contract))
        final_size = min(final_size, max_safe_size)
        
        if final_size != int(decision.size):
            logger.info(f"Position size adjusted: {decision.size} -> {final_size} (Account: ${account_balance:.0f}, Available: ${available_margin:.0f})")
        
        return final_size
    
    def _calculate_adaptive_levels(self, decision: Decision, market_data: MarketData) -> tuple:
        # Learned preferences for stops and targets
        stop_preference = self.meta_learner.get_parameter('stop_preference')
        target_preference = self.meta_learner.get_parameter('target_preference')
        
        stop_distance_factor = self.meta_learner.get_parameter('stop_distance_factor')
        target_distance_factor = self.meta_learner.get_parameter('target_distance_factor')
        
        stop_price = 0.0
        target_price = 0.0
        
        # Apply learned stop preference
        if decision.stop_price > 0 and stop_preference > 0.3:
            if decision.action == 'buy' and decision.stop_price < market_data.price:
                stop_price = decision.stop_price
            elif decision.action == 'sell' and decision.stop_price > market_data.price:
                stop_price = decision.stop_price
        
        # Apply learned target preference  
        if decision.target_price > 0 and target_preference > 0.3:
            if decision.action == 'buy' and decision.target_price > market_data.price:
                target_price = decision.target_price
            elif decision.action == 'sell' and decision.target_price < market_data.price:
                target_price = decision.target_price
        
        # If no specific levels provided, use learned distance factors
        if stop_price == 0 and stop_preference > 0.5:
            if decision.action == 'buy':
                stop_price = market_data.price * (1 - stop_distance_factor)
            else:
                stop_price = market_data.price * (1 + stop_distance_factor)
        
        if target_price == 0 and target_preference > 0.5:
            if decision.action == 'buy':
                target_price = market_data.price * (1 + target_distance_factor)
            else:
                target_price = market_data.price * (1 - target_distance_factor)
        
        return stop_price, target_price
    
    def process_trade_outcome(self, trade_outcome: Dict):
        """Process trade outcome for advanced risk learning"""
        self.advanced_risk.update_risk_metrics(None, trade_outcome)
    
    def get_risk_summary(self) -> Dict:
        """Get comprehensive risk summary including advanced metrics"""
        basic_summary = {
            'consecutive_losses': self.portfolio.get_consecutive_losses(),
            'position_count': self.portfolio.get_position_count(),
            'win_rate': self.portfolio.get_win_rate()
        }
        
        advanced_summary = self.advanced_risk.get_risk_summary()
        
        return {**basic_summary, **advanced_summary}
    
    def run_monte_carlo_analysis(self, decision: Decision, market_data: MarketData) -> Dict:
        """Run Monte Carlo analysis for position sizing"""
        intelligence_data = getattr(decision, 'intelligence_data', {})
        scenarios = self.advanced_risk.run_monte_carlo_simulation(
            decision.size, market_data, intelligence_data
        )
        
        return {
            'scenarios': [
                {
                    'scenario_id': s.scenario_id,
                    'probability': s.probability,
                    'expected_pnl': s.expected_pnl,
                    'var_95': s.var_95,
                    'var_99': s.var_99,
                    'stress_factor': s.stress_factor
                }
                for s in scenarios
            ],
            'recommended_action': 'proceed' if scenarios[0].expected_pnl > 0 else 'reduce_size'
        }