# model/reward.py

import numpy as np

class RewardCalculator:
    def __init__(self):
        self.transaction_cost = 0.001  # 0.1% transaction cost per trade
        
    def compute_reward(self, price_change, atr, action_taken=None):
        """
        Simple, clean reward function based on normalized P&L
        """
        # Normalize price change by ATR to account for volatility
        base_reward = price_change / max(atr, 1e-6)
        
        # Apply transaction cost if an action was taken
        if action_taken is not None and action_taken != 0:  # 0 = HOLD
            base_reward -= self.transaction_cost
            
        # Clip extreme values to prevent instability
        return np.clip(base_reward, -5.0, 5.0)
    
    def compute_position_reward(self, position, price_change, atr):
        """
        Calculate reward based on current position and price movement
        """
        if position == 0:
            return 0.0
            
        # Positive position profits from price increases
        # Negative position profits from price decreases
        pnl = position * price_change
        normalized_pnl = pnl / max(atr, 1e-6)
        
        return np.clip(normalized_pnl, -5.0, 5.0)