# features/feature_extractor.py

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from indicators.research_indicators import ResearchIndicators
from config import ResearchConfig

@dataclass
class ResearchFeatures:
    """Enhanced research-aligned feature vector"""
    
    # 15-minute features
    rsi_15m: float
    bb_position_15m: float
    ema_trend_15m: float
    price_vs_sma_15m: float
    volume_ratio_15m: float
    volume_breakout_15m: bool
    
    # 5-minute features  
    rsi_5m: float
    bb_position_5m: float
    ema_trend_5m: float
    price_vs_sma_5m: float
    volume_ratio_5m: float
    volume_breakout_5m: bool
    
    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Get feature names for model interpretation"""
        return [
            'rsi_15m', 'bb_position_15m', 'ema_trend_15m', 'price_vs_sma_15m', 
            'volume_ratio_15m', 'volume_breakout_15m',
            'rsi_5m', 'bb_position_5m', 'ema_trend_5m', 'price_vs_sma_5m', 
            'volume_ratio_5m', 'volume_breakout_5m'
        ]
    
    def to_array(self) -> np.ndarray:
        """Convert to array for ML model"""
        return np.array([
            self.rsi_15m, self.bb_position_15m, self.ema_trend_15m, 
            self.price_vs_sma_15m, self.volume_ratio_15m, float(self.volume_breakout_15m),
            self.rsi_5m, self.bb_position_5m, self.ema_trend_5m, 
            self.price_vs_sma_5m, self.volume_ratio_5m, float(self.volume_breakout_5m)
        ])

class FeatureExtractor:
    """Extract research-aligned features from market data"""
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.indicators = ResearchIndicators()
    
    def extract_features(self, price_15m: List[float], volume_15m: List[float],
                        price_5m: List[float], volume_5m: List[float]) -> Optional[ResearchFeatures]:
        """Extract features from multi-timeframe market data"""
        
        try:
            # Validate input data
            if not self._validate_data(price_15m, price_5m):
                return None
            
            # Convert to numpy arrays
            prices_15m = np.array(price_15m)
            volumes_15m = np.array(volume_15m) if volume_15m else np.ones_like(prices_15m)
            prices_5m = np.array(price_5m)
            volumes_5m = np.array(volume_5m) if volume_5m else np.ones_like(prices_5m)
            
            # Extract 15-minute features
            features_15m = self._extract_timeframe_features(
                prices_15m, volumes_15m, "15m"
            )
            
            # Extract 5-minute features
            features_5m = self._extract_timeframe_features(
                prices_5m, volumes_5m, "5m"
            )
            
            # Combine into enhanced feature vector
            return ResearchFeatures(
                rsi_15m=features_15m['rsi'],
                bb_position_15m=features_15m['bb_position'],
                ema_trend_15m=features_15m['ema_trend'],
                price_vs_sma_15m=features_15m['price_vs_sma'],
                volume_ratio_15m=features_15m['volume_ratio'],
                volume_breakout_15m=features_15m['volume_breakout'],
                rsi_5m=features_5m['rsi'],
                bb_position_5m=features_5m['bb_position'],
                ema_trend_5m=features_5m['ema_trend'],
                price_vs_sma_5m=features_5m['price_vs_sma'],
                volume_ratio_5m=features_5m['volume_ratio'],
                volume_breakout_5m=features_5m['volume_breakout']
            )
            
        except Exception as e:
            import logging
            logging.error(f"Feature extraction error: {e}")
            return None
    
    def _validate_data(self, price_15m: List[float], price_5m: List[float]) -> bool:
        """Validate input data has sufficient length"""
        min_length = max(self.config.SMA_PERIOD, self.config.BB_PERIOD, self.config.RSI_PERIOD)
        return len(price_15m) >= min_length and len(price_5m) >= min_length
    
    def _extract_timeframe_features(self, prices: np.ndarray, 
                                   volumes: np.ndarray, timeframe: str) -> dict:
        """Extract enhanced features for a specific timeframe"""
        
        # Calculate indicators
        rsi = self.indicators.rsi(prices, self.config.RSI_PERIOD)
        
        bb_upper, bb_mid, bb_lower = self.indicators.bollinger_bands(
            prices, self.config.BB_PERIOD, self.config.BB_STD
        )
        
        # Bollinger Band position calculation
        if bb_upper != bb_lower:
            bb_position = (prices[-1] - bb_lower) / (bb_upper - bb_lower)
        else:
            bb_position = 0.5
        
        # Calculate moving averages
        ema = self.indicators.ema(prices, self.config.EMA_PERIOD)
        sma = self.indicators.sma(prices, self.config.SMA_PERIOD)
        current_price = prices[-1]
        
        # Relationships
        ema_trend = (ema - sma) / sma if sma != 0 else 0.0
        price_vs_sma = (current_price - sma) / sma if sma != 0 else 0.0
        
        # Enhanced volume analysis
        volume_ratio = self.indicators.volume_ratio(volumes, self.config.VOLUME_PERIOD)
        volume_breakout = volume_ratio > 1.5  # Significant volume spike
        
        return {
            'rsi': rsi,
            'bb_position': bb_position,
            'ema_trend': ema_trend,
            'price_vs_sma': price_vs_sma,
            'volume_ratio': volume_ratio,
            'volume_breakout': volume_breakout
        }