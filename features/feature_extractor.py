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
        """IMPROVED: More nuanced timeframe alignment calculation"""
        
        # 1. TREND DIRECTION ANALYSIS (More Realistic Thresholds)
        
        # 15-minute trend (primary trend)
        if features_15m['ema_trend'] > 0.0002:
            trend_15m = 1  # Bullish
        elif features_15m['ema_trend'] < -0.0002:
            trend_15m = -1  # Bearish
        else:
            trend_15m = 0  # Neutral
        
        # 5-minute trend (intermediate trend)
        if features_5m['ema_trend'] > 0.0001:
            trend_5m = 1  # Bullish
        elif features_5m['ema_trend'] < -0.0001:
            trend_5m = -1  # Bearish
        else:
            trend_5m = 0  # Neutral
        
        # 1-minute trend (short-term trend)
        if features_1m['ema_trend'] > 0.00005:
            trend_1m = 1  # Bullish
        elif features_1m['ema_trend'] < -0.00005:
            trend_1m = -1  # Bearish
        else:
            trend_1m = 0  # Neutral
        
        # 2. ALIGNMENT SCORING (Professional Approach)
        
        # Perfect alignment (all same direction) - HIGHEST SCORE
        if trend_15m != 0 and trend_15m == trend_5m == trend_1m:
            return 1.0  # Perfect alignment
        
        # Strong alignment (15m and 5m agree, 1m not opposing)
        if trend_15m != 0 and trend_15m == trend_5m:
            if trend_1m == trend_15m:
                return 0.9  # Very strong alignment
            elif trend_1m == 0:
                return 0.8  # Strong alignment (1m neutral)
            else:
                return 0.4  # Weak alignment (1m opposing)
        
        # Moderate alignment (15m dominant with some support)
        if trend_15m != 0:  # Clear 15m trend
            same_direction_count = 0
            neutral_count = 0
            
            if trend_5m == trend_15m:
                same_direction_count += 1
            elif trend_5m == 0:
                neutral_count += 1
                
            if trend_1m == trend_15m:
                same_direction_count += 1
            elif trend_1m == 0:
                neutral_count += 1
            
            # Scoring based on support
            if same_direction_count >= 1:
                base_score = 0.6 + (same_direction_count * 0.1)
                if neutral_count > 0:
                    base_score += 0.05  # Bonus for neutral (not opposing)
                return min(0.8, base_score)
        
        # 3. MOMENTUM ANALYSIS (Additional Factor)
        
        # Check if shorter timeframes show acceleration in 15m direction
        if trend_15m != 0:
            momentum_boost = 0.0
            
            # 5m momentum in 15m direction
            if trend_5m == trend_15m and abs(features_5m['ema_trend']) > abs(features_15m['ema_trend']) * 0.5:
                momentum_boost += 0.1
            
            # 1m momentum in 15m direction  
            if trend_1m == trend_15m and abs(features_1m['ema_trend']) > abs(features_5m['ema_trend']) * 0.5:
                momentum_boost += 0.1
            
            if momentum_boost > 0:
                return 0.5 + momentum_boost  # Moderate alignment with momentum
        
        # 4. PARTIAL ALIGNMENT (Any two timeframes agree)
        
        trends = [trend_15m, trend_5m, trend_1m]
        non_zero_trends = [t for t in trends if t != 0]
        
        if len(non_zero_trends) >= 2:
            # Check if any two non-zero trends agree
            if len(set(non_zero_trends)) == 1:  # All non-zero trends are same direction
                return 0.5  # Partial alignment
            elif trend_5m == trend_1m and trend_5m != 0:  # Short-term alignment
                return 0.4  # Short-term agreement
        
        # 5. WEAK SIGNALS (At least one clear trend)
        
        if trend_15m != 0 or trend_5m != 0:
            return 0.3  # Some directional bias
        
        # 6. NO CLEAR DIRECTION
        return 0.1  # Minimal alignment


    def _calculate_entry_quality(self, features_1m: dict, features_5m: dict) -> float:
        """IMPROVED: More practical entry timing quality assessment"""
        
        quality_components = []
        
        # 1. RSI ENTRY ZONES (More Practical)
        rsi_1m = features_1m['rsi']
        
        if 20 <= rsi_1m <= 35 or 65 <= rsi_1m <= 80:
            quality_components.append(0.9)  # Excellent entry zones
        elif 35 <= rsi_1m <= 45 or 55 <= rsi_1m <= 65:
            quality_components.append(0.7)  # Good entry zones
        elif 45 <= rsi_1m <= 55:
            quality_components.append(0.5)  # Neutral zone
        else:
            quality_components.append(0.3)  # Challenging zones
        
        # 2. BOLLINGER BAND POSITIONING (Refined)
        bb_1m = features_1m['bb_position']
        
        if bb_1m < 0.2 or bb_1m > 0.8:
            quality_components.append(0.8)  # Near bands - good for entries
        elif bb_1m < 0.35 or bb_1m > 0.65:
            quality_components.append(0.6)  # Approaching bands
        else:
            quality_components.append(0.4)  # Middle area
        
        # 3. VOLUME CONFIRMATION (Simplified)
        volume_ratio_1m = features_1m['volume_ratio']
        
        if volume_ratio_1m > 1.5:
            quality_components.append(0.9)  # Strong volume
        elif volume_ratio_1m > 1.2:
            quality_components.append(0.7)  # Good volume
        elif volume_ratio_1m > 0.8:
            quality_components.append(0.5)  # Adequate volume
        else:
            quality_components.append(0.3)  # Weak volume
        
        # 4. TREND CONSISTENCY (1m vs 5m)
        trend_consistency = abs(features_1m['ema_trend'] - features_5m['ema_trend'])
        
        if trend_consistency < 0.0002:
            quality_components.append(0.8)  # Very consistent
        elif trend_consistency < 0.0005:
            quality_components.append(0.6)  # Moderately consistent
        elif trend_consistency < 0.001:
            quality_components.append(0.4)  # Somewhat consistent
        else:
            quality_components.append(0.2)  # Inconsistent
        
        # 5. PRICE POSITION RELATIVE TO MOVING AVERAGES
        price_vs_sma_1m = features_1m['price_vs_sma']
        
        if abs(price_vs_sma_1m) < 0.002:
            quality_components.append(0.7)  # Near SMA - good for entries
        elif abs(price_vs_sma_1m) < 0.005:
            quality_components.append(0.5)  # Reasonable distance
        else:
            quality_components.append(0.3)  # Extended from SMA
        
        # Return weighted average (emphasize most important factors)
        weights = [0.25, 0.2, 0.2, 0.2, 0.15]  # RSI and BB most important
        weighted_score = sum(score * weight for score, weight in zip(quality_components, weights))
        
        return min(0.95, max(0.1, weighted_score))  # Clamp between 0.1 and 0.95