# risk_manager.py

from dataclasses import dataclass
from typing import Optional, Dict

from trading_agent import Decision
from data_processor import MarketData

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
    def __init__(self, portfolio, config, meta_learner):
        self.portfolio = portfolio
        self.config = config
        self.meta_learner = meta_learner
        
        # Base margin requirement for MNQ
        self.margin_per_contract = 500
        
    def validate_order(self, decision: Decision, market_data: MarketData) -> Optional[Order]:
        if decision.action == 'hold':
            return None
        
        # All limits now come from meta-learner
        max_daily_loss = self.meta_learner.get_parameter('max_daily_loss')
        if market_data.daily_pnl <= -max_daily_loss:
            return None
        
        consecutive_limit = self.meta_learner.get_parameter('consecutive_loss_limit')
        if self.portfolio.get_consecutive_losses() >= consecutive_limit:
            return None
        
        # Position size calculation using meta-learned parameters
        size = self._calculate_position_size(decision, market_data)
        if size == 0:
            return None
        
        # Validate prices
        stop_price = self._validate_stop_price(decision.stop_price, market_data.price, decision.action)
        target_price = self._validate_target_price(decision.target_price, market_data.price, decision.action)
        
        return Order(
            action=decision.action,
            size=size,
            price=market_data.price,
            stop_price=stop_price,
            target_price=target_price,
            timestamp=market_data.timestamp,
            confidence=decision.confidence
        )
    
    def _calculate_position_size(self, decision: Decision, market_data: MarketData) -> int:
        base_size = decision.size
        
        # Account-based limits
        max_by_margin = int(market_data.buying_power * 0.8 / self.margin_per_contract)
        
        # Meta-learned position limits
        max_position_size = int(self.meta_learner.get_parameter('max_position_count'))
        
        # Risk-based limits using meta-learned risk parameters
        max_risk_per_trade = self.meta_learner.get_parameter('stop_loss_base') * 2  # Estimate based on typical stop
        risk_amount = market_data.account_balance * max_risk_per_trade
        estimated_stop_distance = 20 * 2  # Conservative estimate: 20 points * $2 per point
        max_by_risk = int(risk_amount / estimated_stop_distance)
        
        # Apply all limits
        max_size = min(max_by_margin, max_by_risk, max_position_size)
        final_size = min(int(base_size), max_size)
        
        return max(0, final_size)
    
    def _validate_stop_price(self, stop_price: float, current_price: float, action: str) -> float:
        if stop_price <= 0:
            return 0
        
        # Ensure stop is in correct direction
        if action == 'buy' and stop_price >= current_price:
            return 0
        elif action == 'sell' and stop_price <= current_price:
            return 0
        
        # Use meta-learned bounds for stop distance
        min_stop = self.meta_learner.get_parameter('stop_loss_base') * 0.5
        max_stop = self.meta_learner.get_parameter('stop_loss_base') * 3.0
        
        distance = abs(stop_price - current_price) / current_price
        if distance < min_stop or distance > max_stop:
            return 0
        
        return stop_price
    
    def _validate_target_price(self, target_price: float, current_price: float, action: str) -> float:
        if target_price <= 0:
            return 0
        
        # Ensure target is in correct direction
        if action == 'buy' and target_price <= current_price:
            return 0
        elif action == 'sell' and target_price >= current_price:
            return 0
        
        # Use meta-learned bounds for target distance
        min_target = self.meta_learner.get_parameter('take_profit_base') * 0.5
        max_target = self.meta_learner.get_parameter('take_profit_base') * 4.0
        
        distance = abs(target_price - current_price) / current_price
        if distance < min_target or distance > max_target:
            return 0
        
        return target_price