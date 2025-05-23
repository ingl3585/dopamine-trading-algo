# model/reward.py

import numpy as np

class RewardCalculator:
    def __init__(self):
        self.transaction_cost = 0.001
        
    def compute_reward(self, price_change, atr, action_taken=None):
        base_reward = price_change / max(atr, 1e-6)
        
        if action_taken is not None and action_taken != 0:
            base_reward -= self.transaction_cost
            
        return np.clip(base_reward, -5.0, 5.0)
    
    def compute_position_reward(self, position, price_change, atr):
        if position == 0:
            return 0.0
            
        pnl = position * price_change
        normalized_pnl = pnl / max(atr, 1e-6)
        
        return np.clip(normalized_pnl, -5.0, 5.0)