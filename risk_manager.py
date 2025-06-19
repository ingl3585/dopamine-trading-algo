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
    def __init__(self, portfolio, config):
        self.portfolio = portfolio
        self.config = config
        
        # Risk parameters
        self.max_position_size = 3
        self.max_daily_loss = 500
        self.max_risk_per_trade = 0.02  # 2% of account
        self.margin_per_contract = 500  # MNQ margin requirement
        
    def validate_order(self, decision: Decision, market_data: MarketData) -> Optional[Order]:
        if decision.action == 'hold':
            return None
            
        # Daily loss check
        if market_data.daily_pnl <= -self.max_daily_loss:
            return None
            
        # Consecutive loss check
        if self.portfolio.get_consecutive_losses() >= 5:
            return None
            
        # Position size calculation
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
        # Start with agent's suggested size
        base_size = decision.size
        
        # Account-based limits
        max_by_margin = int(market_data.buying_power * 0.8 / self.margin_per_contract)
        
        # Risk-based limits (assume 20 point stop for calculation)
        risk_amount = market_data.account_balance * self.max_risk_per_trade
        estimated_stop_distance = 20 * 2  # 20 points * $2 per point
        max_by_risk = int(risk_amount / estimated_stop_distance)
        
        # Apply limits
        max_size = min(max_by_margin, max_by_risk, self.max_position_size)
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
            
        # Ensure reasonable stop distance (0.1% to 5%)
        distance = abs(stop_price - current_price) / current_price
        if distance < 0.001 or distance > 0.05:
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
            
        # Ensure reasonable target distance (0.1% to 10%)
        distance = abs(target_price - current_price) / current_price
        if distance < 0.001 or distance > 0.10:
            return 0
            
        return target_price