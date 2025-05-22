# services/portfolio.py

import logging

log = logging.getLogger(__name__)

class Portfolio:
    def __init__(self, cfg):
        self.cfg = cfg
        self.position = 0
        self.max_position = cfg.MAX_SIZE

    def update_position(self, pos):
        self.position = pos

    def get_available_capacity(self, action):
        if action == 1:  # BUY
            return max(0, self.max_position - self.position)
        elif action == 2:  # SELL  
            return max(0, self.max_position + self.position)
        return 0
    
    def calculate_position_size(self, action: int, confidence: float) -> int:
        if action == 0:
            return 0
        
        available_capacity = self.get_available_capacity(action)
        
        if available_capacity <= 0:
            return 0
        
        base_size = self.cfg.BASE_SIZE
        
        confidence_multiplier = max(0.5, confidence)
        scaled_size = int(base_size * confidence_multiplier)
        
        scaled_size = max(self.cfg.MIN_SIZE, min(scaled_size, self.cfg.MAX_SIZE))
        
        final_size = min(scaled_size, available_capacity)
        
        return final_size