# services/portfolio.py

import logging

log = logging.getLogger(__name__)

class Portfolio:
    def __init__(self, max_position):
        self.position = 0
        self.max_position = max_position

    def update_position(self, pos):
        """Update current position from NinjaTrader"""
        self.position = pos

    def get_available_capacity(self, action):
        """Get remaining capacity for new positions"""
        if action == 1:  # BUY
            return max(0, self.max_position - self.position)
        elif action == 2:  # SELL  
            return max(0, self.max_position + self.position)
        return 0

    def calculate_trade_size(self, action, confidence, base_size, min_size):
        if action == 0:  # HOLD
            return 0

        # Get current position metrics
        position_util = self.get_position_utilization()
        available_capacity = self.get_available_capacity(action)
        
        if available_capacity == 0:
            log.debug(f"No capacity for action {action}, position: {self.position}")
            return 0

        # Confidence-based sizing
        # Map confidence (0.15-0.85) to size multiplier (0.5-1.5)
        conf_normalized = max(0.0, min(1.0, (confidence - 0.15) / 0.7))
        size_multiplier = 0.5 + conf_normalized

        # Position-based size reduction
        if position_util < 0.5:
            # Low utilization - allow normal sizing
            max_size = base_size
        elif position_util < 0.8:
            # Medium utilization - reduce size
            max_size = max(min_size, base_size // 2)
            size_multiplier *= 0.7
        else:
            # High utilization - small sizes only
            max_size = min_size
            size_multiplier *= 0.5

        # Calculate final size
        desired_size = round(base_size * size_multiplier)
        final_size = min(desired_size, max_size, available_capacity)
        final_size = max(0, final_size)

        # Basic validation
        if not self._is_valid_trade(action, final_size):
            return 0
        
        return final_size

    def _is_valid_trade(self, action, size):
        """Simple trade validation"""
        if size <= 0:
            return False
            
        # Check if trade would exceed position limits
        projected_pos = self.position + (size if action == 1 else -size)
        return abs(projected_pos) <= self.max_position

    def get_risk_level(self):
        """Get current risk level description"""
        util = self.get_position_utilization()
        if util < 0.3:
            return "LOW"
        elif util < 0.6:
            return "MEDIUM" 
        elif util < 0.8:
            return "HIGH"
        else:
            return "CRITICAL"

    def get_trade_description(self, action, size):
        """Simple trade description for logging"""
        if action == 0 or size == 0:
            return "HOLD"
        
        current_util = self.get_position_utilization()
        projected_pos = self.position + (size if action == 1 else -size)
        projected_util = abs(projected_pos) / self.max_position
        
        action_name = "BUY" if action == 1 else "SELL"
        
        return f"{action_name}_{size} ({current_util:.0%}→{projected_util:.0%})"