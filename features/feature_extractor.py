# features/feature_extractor.py

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from indicators.research_indicators import ResearchIndicators
from config import ResearchConfig

@dataclass
class ResearchFeatures:
    """Research-aligned feature vector"""
    
    # 15-minute features
    rsi_15m: float
    bb_position_15m: float
    ema_15m: float
    sma_15m: float
    volume_ratio_15m: float
    
    # 5-minute features  
    rsi_5m: float
    bb_position_5m: float
    ema_5m: float
    sma_5m: float
    volume_ratio_5m: float
    
    def to_array(self) -> np.ndarray:
        """Convert to array for ML model"""
        return np.array([
            self.rsi_15m, self.bb_position_15m, self.ema_15m, 
            self.sma_15m, self.volume_ratio_15m,
            self.rsi_5m, self.bb_position_5m, self.ema_5m, 
            self.sma_5m, self.volume_ratio_5m
        ])
    
    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Get feature names for model interpretation"""
        return [
            'rsi_15m', 'bb_position_15m', 'ema_15m', 'sma_15m', 'volume_ratio_15m',
            'rsi_5m', 'bb_position_5m', 'ema_5m', 'sma_5m', 'volume_ratio_5m'
        ]

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
            volumes_15m = np.array(volume_15m)
            prices_5m = np.array(price_5m)
            volumes_5m = np.array(volume_5m)
            
            # Extract 15-minute features
            features_15m = self._extract_timeframe_features(
                prices_15m, volumes_15m, "15m"
            )
            
            # Extract 5-minute features
            features_5m = self._extract_timeframe_features(
                prices_5m, volumes_5m, "5m"
            )
            
            # Combine into feature vector
            return ResearchFeatures(
                rsi_15m=features_15m['rsi'],
                bb_position_15m=features_15m['bb_position'],
                ema_15m=features_15m['ema'],
                sma_15m=features_15m['sma'],
                volume_ratio_15m=features_15m['volume_ratio'],
                rsi_5m=features_5m['rsi'],
                bb_position_5m=features_5m['bb_position'],
                ema_5m=features_5m['ema'],
                sma_5m=features_5m['sma'],
                volume_ratio_5m=features_5m['volume_ratio']
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
        """Extract features for a specific timeframe"""
        
        # Calculate indicators
        rsi = self.indicators.rsi(prices, self.config.RSI_PERIOD)
        
        bb_upper, bb_mid, bb_lower = self.indicators.bollinger_bands(
            prices, self.config.BB_PERIOD, self.config.BB_STD
        )
        bb_position = ((prices[-1] - bb_lower) / (bb_upper - bb_lower)) \
                     if bb_upper != bb_lower else 0.5
        
        ema = self.indicators.ema(prices, self.config.EMA_PERIOD)
        sma = self.indicators.sma(prices, self.config.SMA_PERIOD)
        volume_ratio = self.indicators.volume_ratio(volumes, self.config.VOLUME_PERIOD)
        
        return {
            'rsi': rsi,
            'bb_position': bb_position,
            'ema': ema,
            'sma': sma,
            'volume_ratio': volume_ratio
        }