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
    def __init__(self, portfolio, meta_learner):
        self.portfolio = portfolio
        self.meta_learner = meta_learner
        
    def validate_order(self, decision: Decision, market_data: MarketData) -> Optional[Order]:
        if decision.action == 'hold':
            return None
        
        # All limits from meta-learner - no hardcoded values
        max_daily_loss = self.meta_learner.get_parameter('max_daily_loss')
        if market_data.daily_pnl <= -max_daily_loss:
            return None
        
        consecutive_limit = self.meta_learner.get_parameter('consecutive_loss_limit')
        if self.portfolio.get_consecutive_losses() >= consecutive_limit:
            return None
        
        # Position size calculation using only meta-learned parameters
        size = self._calculate_position_size(decision, market_data)
        if size == 0:
            return None
        
        # Validate prices using meta-learned bounds
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
        
        # Get meta-learned parameters for all calculations
        margin_per_contract = self.meta_learner.get_parameter('margin_per_contract')
        buying_power_usage = self.meta_learner.get_parameter('buying_power_usage')
        risk_per_trade = self.meta_learner.get_parameter('risk_per_trade')
        point_value = self.meta_learner.get_parameter('point_value')
        estimated_stop_points = self.meta_learner.get_parameter('estimated_stop_points')
        max_position_size = self.meta_learner.get_parameter('max_position_count')
        
        # Account-based limits
        max_by_margin = int(market_data.buying_power * buying_power_usage / margin_per_contract)
        
        # Risk-based limits using meta-learned parameters
        risk_amount = market_data.account_balance * risk_per_trade
        estimated_stop_distance = estimated_stop_points * point_value
        max_by_risk = int(risk_amount / estimated_stop_distance) if estimated_stop_distance > 0 else 1
        
        # Apply all limits
        max_size = min(max_by_margin, max_by_risk, int(max_position_size))
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
        stop_loss_base = self.meta_learner.get_parameter('stop_loss_base')
        stop_min_multiplier = self.meta_learner.get_parameter('stop_min_multiplier')
        stop_max_multiplier = self.meta_learner.get_parameter('stop_max_multiplier')
        
        min_stop = stop_loss_base * stop_min_multiplier
        max_stop = stop_loss_base * stop_max_multiplier
        
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
        take_profit_base = self.meta_learner.get_parameter('take_profit_base')
        target_min_multiplier = self.meta_learner.get_parameter('target_min_multiplier')
        target_max_multiplier = self.meta_learner.get_parameter('target_max_multiplier')
        
        min_target = take_profit_base * target_min_multiplier
        max_target = take_profit_base * target_max_multiplier
        
        distance = abs(target_price - current_price) / current_price
        if distance < min_target or distance > max_target:
            return 0
        
        return target_price