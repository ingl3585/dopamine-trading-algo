# indicators/research_indicators.py

import numpy as np
from typing import Tuple

class ResearchIndicators:
    """Research-backed technical indicators - FIXED"""
    
    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> float:
        """RSI - FIXED to handle edge cases properly"""
        if len(prices) < period + 1:
            return 50.0
        
        # Use the last 'period + 1' prices to get 'period' deltas
        price_slice = prices[-(period + 1):]
        deltas = np.diff(price_slice)
        
        if len(deltas) == 0:
            return 50.0
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        # Handle edge cases
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        
        if avg_gain == 0:
            return 0.0
        
        rs = avg_gain / avg_loss
        rsi_value = 100.0 - (100.0 / (1.0 + rs))
        
        # Ensure RSI is in valid range
        return max(0.0, min(100.0, rsi_value))
    
    @staticmethod
    def bollinger_bands(prices: np.ndarray, period: int = 20, 
                       std_dev: float = 2.0) -> Tuple[float, float, float]:
        """Bollinger Bands - IMPROVED"""
        if len(prices) < period:
            if len(prices) > 0:
                mid = prices[-1]
                return mid, mid, mid
            else:
                return 0.0, 0.0, 0.0
        
        recent_prices = prices[-period:]
        mid = np.mean(recent_prices)
        std = np.std(recent_prices, ddof=0)  # Use population std
        
        # Handle zero std (flat prices)
        if std == 0:
            return mid, mid, mid
        
        upper = mid + (std_dev * std)
        lower = mid - (std_dev * std)
        
        return upper, mid, lower
    
    @staticmethod
    def ema(prices: np.ndarray, period: int) -> float:
        """EMA - IMPROVED"""
        if len(prices) < 1:
            return 0.0
        if len(prices) == 1:
            return prices[0]
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2.0 / (period + 1)
        
        # Start with SMA for first value
        ema_val = np.mean(prices[:period])
        
        # Calculate EMA for remaining values
        for price in prices[period:]:
            ema_val = (price * multiplier) + (ema_val * (1 - multiplier))
        
        return ema_val
    
    @staticmethod
    def sma(prices: np.ndarray, period: int) -> float:
        """Simple Moving Average - IMPROVED"""
        if len(prices) < 1:
            return 0.0
        if len(prices) < period:
            return np.mean(prices)
        return np.mean(prices[-period:])
    
    @staticmethod
    def volume_ratio(volumes: np.ndarray, period: int = 20) -> float:
        """Volume ratio - IMPROVED"""
        if len(volumes) < 1:
            return 1.0
        if len(volumes) < period:
            return 1.0
        
        current_vol = volumes[-1]
        avg_vol = np.mean(volumes[-period:])
        
        # Handle zero average volume
        if avg_vol <= 0:
            return 1.0
        
        return max(0.1, current_vol / avg_vol)  # Minimum ratio of 0.1