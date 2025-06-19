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
        
        # Check learned loss tolerance
        loss_tolerance = self.meta_learner.get_parameter('loss_tolerance_factor')
        max_loss = market_data.account_balance * loss_tolerance
        if market_data.daily_pnl <= -max_loss:
            return None
        
        # Check learned consecutive loss tolerance
        consecutive_tolerance = self.meta_learner.get_parameter('consecutive_loss_tolerance')
        if self.portfolio.get_consecutive_losses() >= consecutive_tolerance:
            return None
        
        # Calculate position size based on available margin and learned preferences
        size = self._calculate_adaptive_position_size(decision, market_data)
        if size == 0:
            return None
        
        # Apply learned stop/target preferences
        stop_price, target_price = self._calculate_adaptive_levels(decision, market_data)
        
        return Order(
            action=decision.action,
            size=size,
            price=market_data.price,
            stop_price=stop_price,
            target_price=target_price,
            timestamp=market_data.timestamp,
            confidence=decision.confidence
        )
    
    def _calculate_adaptive_position_size(self, decision: Decision, market_data: MarketData) -> int:
        # Use actual account data from NinjaTrader
        available_margin = market_data.buying_power
        
        # Learned position sizing factors
        position_factor = self.meta_learner.get_parameter('position_size_factor')
        max_position_factor = self.meta_learner.get_parameter('max_position_factor')
        
        # Calculate based on confidence and available capital
        confidence_multiplier = decision.confidence
        
        # Base size from available margin
        base_size = available_margin * position_factor * confidence_multiplier
        
        # Convert to contracts (assuming $500 margin per contract as typical for MNQ)
        estimated_margin_per_contract = 500
        max_contracts_by_margin = int(base_size / estimated_margin_per_contract)
        
        # Apply maximum position factor
        max_contracts_by_account = int(market_data.account_balance * max_position_factor / estimated_margin_per_contract)
        
        # Final size
        final_size = min(max_contracts_by_margin, max_contracts_by_account, int(decision.size))
        
        return max(0, final_size)
    
    def _calculate_adaptive_levels(self, decision: Decision, market_data: MarketData) -> tuple:
        # Learned preferences for stops and targets
        stop_preference = self.meta_learner.get_parameter('stop_preference')
        target_preference = self.meta_learner.get_parameter('target_preference')
        
        stop_distance_factor = self.meta_learner.get_parameter('stop_distance_factor')
        target_distance_factor = self.meta_learner.get_parameter('target_distance_factor')
        
        stop_price = 0.0
        target_price = 0.0
        
        # Apply learned stop preference
        if decision.stop_price > 0 and stop_preference > 0.3:  # Some threshold for using stops
            if decision.action == 'buy' and decision.stop_price < market_data.price:
                stop_price = decision.stop_price
            elif decision.action == 'sell' and decision.stop_price > market_data.price:
                stop_price = decision.stop_price
        
        # Apply learned target preference  
        if decision.target_price > 0 and target_preference > 0.3:  # Some threshold for using targets
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