# market_env.py - Pure black box environment (cleaned)

import numpy as np
from typing import List, Tuple

class MarketEnv:
    """
    Pure black box environment that tracks:
    • Recent price movements for pattern recognition
    • Volatility context (without using ATR indicator)
    • Position tracking for reward calculation
    """
    FLAT, LONG, SHORT = 0, 1, 2

    def __init__(self, lookback_window: int = 20):
        self.lookback_window = lookback_window
        self.reset()

    def reset(self):
        self.prices: List[float] = []
        self.position = self.FLAT
        self.entry_price = None
        self.hold_bars = 0
        return self._obs()

    def step(self, price: float, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Pure black box step - no indicators"""
        self.prices.append(price)
        
        # Keep only recent prices for observation
        if len(self.prices) > self.lookback_window:
            self.prices = self.prices[-self.lookback_window:]
        
        # Calculate basic volatility context (not an indicator, just recent range)
        recent_volatility = 0.0
        if len(self.prices) >= 10:
            recent_high = max(self.prices[-10:])
            recent_low = min(self.prices[-10:])
            recent_volatility = recent_high - recent_low
        
        reward = 0.0
        done = False

        # Position tracking for reward calculation
        if self.position != self.FLAT:
            self.hold_bars += 1
            side = 1 if self.position == self.LONG else -1
            # Raw price P&L (no normalization by indicators)
            reward = side * (price - self.entry_price)

        return self._obs(), reward, done, {
            "recent_volatility": recent_volatility,
            "price": price,
            "bars_held": self.hold_bars
        }

    def _obs(self) -> np.ndarray:
        """
        Pure black box observation:
        - Recent price changes (no indicators)
        - Price position relative to recent range
        - Simple momentum measure
        """
        if len(self.prices) < 2:
            return np.zeros(15, dtype=np.float32)
        
        obs = []
        
        # Recent price changes (normalized by current price level)
        current_price = self.prices[-1]
        for i in range(min(10, len(self.prices)-1)):
            if len(self.prices) > i+1:
                change = (self.prices[-(i+1)] - self.prices[-(i+2)]) / current_price
                obs.append(change)
            else:
                obs.append(0.0)
        
        # Price position in recent range
        if len(self.prices) >= 10:
            recent_high = max(self.prices[-10:])
            recent_low = min(self.prices[-10:])
            if recent_high > recent_low:
                position_in_range = (current_price - recent_low) / (recent_high - recent_low)
                obs.append(position_in_range)
            else:
                obs.append(0.5)
        else:
            obs.append(0.5)
        
        # Simple momentum (no indicator, just recent direction)
        if len(self.prices) >= 5:
            momentum = (self.prices[-1] - self.prices[-5]) / current_price
            obs.append(momentum)
        else:
            obs.append(0.0)
        
        # Recent volatility measure (range as % of price)
        if len(self.prices) >= 10:
            recent_range = (max(self.prices[-10:]) - min(self.prices[-10:])) / current_price
            obs.append(recent_range)
        else:
            obs.append(0.0)
        
        # Time-based features (position age, etc.)
        obs.append(float(self.hold_bars) / 100.0)  # Normalize hold time
        obs.append(float(self.position))  # Current position
        
        # Pad to fixed length
        while len(obs) < 15:
            obs.append(0.0)
        
        return np.array(obs[:15], dtype=np.float32)
    
    def get_obs(self) -> np.ndarray:
        """Public method to get current observation"""
        return self._obs()