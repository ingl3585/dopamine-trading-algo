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