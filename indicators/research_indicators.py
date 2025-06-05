# indicators/research_indicators

import numpy as np
from typing import Tuple

class ResearchIndicators:
    """Research-backed technical indicators"""
    
    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> float:
        """RSI - highest standalone reliability per research"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))
    
    @staticmethod
    def bollinger_bands(prices: np.ndarray, period: int = 20, 
                       std_dev: float = 2.0) -> Tuple[float, float, float]:
        """Bollinger Bands - highest standalone reliability per research"""
        if len(prices) < period:
            mid = prices[-1] if len(prices) > 0 else 0
            return mid, mid, mid
        
        recent_prices = prices[-period:]
        mid = np.mean(recent_prices)
        std = np.std(recent_prices)
        
        upper = mid + (std_dev * std)
        lower = mid - (std_dev * std)
        
        return upper, mid, lower
    
    @staticmethod
    def ema(prices: np.ndarray, period: int) -> float:
        """EMA - preferred over SMA per research"""
        if len(prices) < 1:
            return 0.0
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2.0 / (period + 1)
        ema_val = prices[-period]
        
        for price in prices[-period+1:]:
            ema_val = (price * multiplier) + (ema_val * (1 - multiplier))
        
        return ema_val
    
    @staticmethod
    def sma(prices: np.ndarray, period: int) -> float:
        """Simple Moving Average"""
        if len(prices) < period:
            return np.mean(prices) if len(prices) > 0 else 0.0
        return np.mean(prices[-period:])
    
    @staticmethod
    def volume_ratio(volumes: np.ndarray, period: int = 20) -> float:
        """Simple volume analysis per research"""
        if len(volumes) < period:
            return 1.0
        
        current_vol = volumes[-1]
        avg_vol = np.mean(volumes[-period:])
        
        return current_vol / avg_vol if avg_vol > 0 else 1.0