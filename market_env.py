# market_env.py - TRULY SIMPLIFIED (based on original)

import numpy as np
from typing import List, Tuple

class MarketEnv:
    """
    Simple market environment for the RL agent
    Tracks recent price movements for pattern recognition
    """
    
    def __init__(self, lookback_window: int = 20):
        self.lookback_window = lookback_window
        self.reset()

    def reset(self):
        """Reset environment state"""
        self.prices: List[float] = []
        self.bars_processed = 0
        return self.get_obs()

    def update(self, price: float) -> np.ndarray:
        """Update with new price and return observation"""
        self.prices.append(price)
        self.bars_processed += 1
        
        # Keep only recent prices
        if len(self.prices) > self.lookback_window:
            self.prices = self.prices[-self.lookback_window:]
        
        return self.get_obs()

    def get_obs(self) -> np.ndarray:
        """
        Get market observation for RL agent
        Returns 15 features representing current market state
        """
        if len(self.prices) < 2:
            return np.zeros(15, dtype=np.float32)
        
        obs = []
        current_price = self.prices[-1]
        
        # Recent price changes (normalized by current price)
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
        
        # Simple momentum (recent direction)
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
        
        # Time-based feature
        obs.append(float(self.bars_processed % 24) / 24.0)  # Hour of day normalized
        
        # Market structure (trend direction)
        if len(self.prices) >= 10:
            trend = (self.prices[-1] - self.prices[-10]) / current_price
            obs.append(trend)
        else:
            obs.append(0.0)
        
        # Pad to exactly 15 features
        while len(obs) < 15:
            obs.append(0.0)
        
        return np.array(obs[:15], dtype=np.float32)

    def get_price_context(self):
        """Get current price context"""
        if not self.prices:
            return {'current_price': 0, 'volatility': 0}
        
        current_price = self.prices[-1]
        volatility = 0.0
        
        if len(self.prices) >= 10:
            recent_range = max(self.prices[-10:]) - min(self.prices[-10:])
            volatility = recent_range
        
        return {
            'current_price': current_price,
            'volatility': volatility,
            'bars_processed': self.bars_processed
        }

# Factory function to match our simplified system
def create_market_environment(lookback_window=20):
    return MarketEnv(lookback_window)