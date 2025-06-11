# features/feature_extractor.py - SIMPLE 1-minute enhancement

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from indicators.research_indicators import ResearchIndicators
from config import ResearchConfig

@dataclass
class ResearchFeatures:
    """Enhanced research-aligned feature vector with 1-minute data"""
    
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
    
    # 1-minute features for entry timing
    rsi_1m: float
    bb_position_1m: float
    ema_trend_1m: float
    price_vs_sma_1m: float
    volume_ratio_1m: float
    volume_spike_1m: bool  # Higher threshold for 1m
    
    # Simple confluence features
    timeframe_alignment: float  # How well all timeframes agree on direction
    entry_timing_quality: float  # Quality of 1m entry timing
    
    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Get feature names for model interpretation"""
        return [
            'rsi_15m', 'bb_position_15m', 'ema_trend_15m', 'price_vs_sma_15m', 
            'volume_ratio_15m', 'volume_breakout_15m',
            'rsi_5m', 'bb_position_5m', 'ema_trend_5m', 'price_vs_sma_5m', 
            'volume_ratio_5m', 'volume_breakout_5m',
            'rsi_1m', 'bb_position_1m', 'ema_trend_1m', 'price_vs_sma_1m',
            'volume_ratio_1m', 'volume_spike_1m',
            'timeframe_alignment', 'entry_timing_quality'
        ]
    
    def to_array(self) -> np.ndarray:
        """Convert to array for ML model"""
        return np.array([
            self.rsi_15m, self.bb_position_15m, self.ema_trend_15m, 
            self.price_vs_sma_15m, self.volume_ratio_15m, float(self.volume_breakout_15m),
            self.rsi_5m, self.bb_position_5m, self.ema_trend_5m, 
            self.price_vs_sma_5m, self.volume_ratio_5m, float(self.volume_breakout_5m),
            self.rsi_1m, self.bb_position_1m, self.ema_trend_1m,
            self.price_vs_sma_1m, self.volume_ratio_1m, float(self.volume_spike_1m),
            self.timeframe_alignment, self.entry_timing_quality
        ])

class FeatureExtractor:
    """Extract research-aligned features from market data - ENHANCED with 1-minute"""
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.indicators = ResearchIndicators()
    
    def extract_features(self, price_15m: List[float], volume_15m: List[float],
                        price_5m: List[float], volume_5m: List[float],
                        price_1m: List[float] = None, volume_1m: List[float] = None) -> Optional[ResearchFeatures]:
        """Extract features from multi-timeframe market data - ENHANCED"""
        
        try:
            # Validate input data
            if not self._validate_data(price_15m, price_5m):
                return None
            
            # Convert to numpy arrays
            prices_15m = np.array(price_15m)
            volumes_15m = np.array(volume_15m) if volume_15m else np.ones_like(prices_15m)
            prices_5m = np.array(price_5m)
            volumes_5m = np.array(volume_5m) if volume_5m else np.ones_like(prices_5m)
            
            # Handle 1-minute data (use 5m as fallback if not available)
            if price_1m and len(price_1m) >= 10:
                prices_1m = np.array(price_1m)
                volumes_1m = np.array(volume_1m) if volume_1m else np.ones_like(prices_1m)
            else:
                # Use recent 5m data as fallback
                prices_1m = prices_5m[-20:] if len(prices_5m) >= 20 else prices_5m
                volumes_1m = volumes_5m[-20:] if len(volumes_5m) >= 20 else volumes_5m
            
            # Extract features for each timeframe
            features_15m = self._extract_timeframe_features(prices_15m, volumes_15m, "15m")
            features_5m = self._extract_timeframe_features(prices_5m, volumes_5m, "5m")
            features_1m = self._extract_timeframe_features(prices_1m, volumes_1m, "1m")
            
            # Calculate simple confluence
            timeframe_alignment = self._calculate_alignment(features_15m, features_5m, features_1m)
            entry_timing_quality = self._calculate_entry_quality(features_1m, features_5m)
            
            # Create enhanced feature vector
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
                volume_breakout_5m=features_5m['volume_breakout'],
                
                # 1-minute features
                rsi_1m=features_1m['rsi'],
                bb_position_1m=features_1m['bb_position'],
                ema_trend_1m=features_1m['ema_trend'],
                price_vs_sma_1m=features_1m['price_vs_sma'],
                volume_ratio_1m=features_1m['volume_ratio'],
                volume_spike_1m=features_1m['volume_spike'],
                
                # Confluence features
                timeframe_alignment=timeframe_alignment,
                entry_timing_quality=entry_timing_quality
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
        
        # Different volume thresholds by timeframe
        if timeframe == "1m":
            volume_spike = volume_ratio > 2.0  # Higher threshold for 1m spikes
            volume_breakout = volume_ratio > 1.3
        else:
            volume_spike = False  # Only relevant for 1m
            volume_breakout = volume_ratio > 1.5
        
        return {
            'rsi': rsi,
            'bb_position': bb_position,
            'ema_trend': ema_trend,
            'price_vs_sma': price_vs_sma,
            'volume_ratio': volume_ratio,
            'volume_breakout': volume_breakout,
            'volume_spike': volume_spike
        }
    
    def _calculate_alignment(self, features_15m: dict, features_5m: dict, features_1m: dict) -> float:
        """ENHANCED: Calculate simple timeframe alignment"""
        
        # Get trend direction for each timeframe
        trend_15m = 1 if features_15m['ema_trend'] > 0.002 else (-1 if features_15m['ema_trend'] < -0.002 else 0)
        trend_5m = 1 if features_5m['ema_trend'] > 0.001 else (-1 if features_5m['ema_trend'] < -0.001 else 0)
        trend_1m = 1 if features_1m['ema_trend'] > 0.0005 else (-1 if features_1m['ema_trend'] < -0.0005 else 0)
        
        trends = [trend_15m, trend_5m, trend_1m]
        non_neutral = [t for t in trends if t != 0]
        
        # Perfect alignment
        if len(set(trends)) == 1 and trends[0] != 0:
            return 1.0
        
        # Good alignment (non-neutral trends agree)
        elif len(non_neutral) >= 2 and len(set(non_neutral)) == 1:
            return 0.7
        
        # Partial alignment
        elif len(non_neutral) >= 1:
            return 0.4
        
        # No clear direction
        else:
            return 0.0
    
    def _calculate_entry_quality(self, features_1m: dict, features_5m: dict) -> float:
        """ENHANCED: Calculate 1-minute entry timing quality"""
        
        quality_factors = []
        
        # 1. RSI in good entry zone (not extreme)
        rsi_1m = features_1m['rsi']
        if 25 <= rsi_1m <= 40 or 60 <= rsi_1m <= 75:
            quality_factors.append(1.0)  # Good entry zones
        elif 40 <= rsi_1m <= 60:
            quality_factors.append(0.5)  # Neutral zone
        else:
            quality_factors.append(0.2)  # Extreme zones
        
        # 2. Bollinger band position for entry timing
        bb_1m = features_1m['bb_position']
        if bb_1m < 0.3 or bb_1m > 0.7:
            quality_factors.append(0.8)  # Near bands = good entry
        else:
            quality_factors.append(0.5)  # Middle zone
        
        # 3. Volume confirmation
        if features_1m['volume_ratio'] > 1.2:
            quality_factors.append(0.8)  # Volume support
        else:
            quality_factors.append(0.4)  # Weak volume
        
        # 4. Trend consistency between 1m and 5m
        trend_consistency = abs(features_1m['ema_trend'] - features_5m['ema_trend'])
        if trend_consistency < 0.005:
            quality_factors.append(0.9)  # Very consistent
        elif trend_consistency < 0.01:
            quality_factors.append(0.6)  # Moderately consistent
        else:
            quality_factors.append(0.3)  # Inconsistent
        
        return np.mean(quality_factors)