"""
Order Flow Analysis - Smart money detection and market maker identification
"""

import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class OrderFlowAnalyzer:
    """
    Advanced order flow analysis for smart money vs retail detection
    """
    
    def detect_smart_money_flow(self, prices: List[float], volumes: List[float], 
                               timestamps: List[float]) -> float:
        """Detect smart money vs retail flow patterns"""
        try:
            if len(prices) < 10 or len(volumes) < 10:
                return 0.0
            
            # Calculate price and volume metrics
            price_changes = np.diff(prices[-10:])
            volume_changes = np.diff(volumes[-10:])
            
            # Smart money indicators
            smart_money_score = 0.0
            
            # 1. Volume-price divergence (smart money often trades against retail)
            correlation = np.corrcoef(price_changes, volume_changes[:-1])[0, 1] if len(price_changes) > 1 else 0
            if not np.isnan(correlation):
                # Negative correlation suggests smart money (buying dips, selling rallies)
                divergence_score = -correlation * 0.3
                smart_money_score += divergence_score
            
            # 2. Volume clustering (institutional block trading)
            volume_mean = np.mean(volumes[-10:])
            volume_spikes = sum(1 for v in volumes[-10:] if v > volume_mean * 2)
            if volume_spikes >= 2:  # Multiple large volume events
                smart_money_score += 0.2
            
            # 3. Price action efficiency (smart money moves prices efficiently)
            price_efficiency = self._calculate_price_efficiency(prices[-10:], volumes[-10:])
            smart_money_score += price_efficiency * 0.3
            
            # 4. Timing analysis (smart money trades at optimal times)
            if timestamps:
                timing_score = self._analyze_timing_patterns(timestamps[-10:], volumes[-10:])
                smart_money_score += timing_score * 0.2
            
            # Normalize score
            smart_money_score = np.tanh(smart_money_score)
            
            return smart_money_score
            
        except Exception as e:
            logger.error(f"Error in smart money detection: {e}")
            return 0.0

    def detect_market_maker_activity(self, prices: List[float], volumes: List[float],
                                   mm_patterns: Dict) -> float:
        """Detect market maker accumulation/distribution patterns"""
        try:
            if len(prices) < 5 or len(volumes) < 5:
                return 0.0
            
            mm_score = 0.0
            
            # 1. Bid-ask spread analysis (simulated from price action)
            price_volatility = np.std(prices[-5:]) / np.mean(prices[-5:])
            
            # Tight spreads with consistent volume suggest MM presence
            if price_volatility < 0.002:  # Very tight price action
                avg_volume = np.mean(volumes[-5:])
                volume_consistency = 1.0 - (np.std(volumes[-5:]) / avg_volume if avg_volume > 0 else 1.0)
                if volume_consistency > 0.7:  # Consistent volume
                    mm_score += 0.3
            
            # 2. Mean reversion patterns (MMs provide liquidity)
            price_returns = np.diff(prices[-5:]) / prices[-5:-1]
            mean_reversion = self._calculate_mean_reversion(price_returns)
            mm_score += mean_reversion * 0.4
            
            # 3. Volume distribution patterns
            volume_distribution = self._analyze_volume_distribution(volumes[-10:])
            mm_score += volume_distribution * 0.3
            
            # Store patterns for learning
            if mm_score > 0.5:
                pattern_key = f"vol_{np.mean(volumes[-3:]):.0f}_price_vol_{price_volatility:.4f}"
                if pattern_key not in mm_patterns:
                    mm_patterns[pattern_key] = {'count': 0, 'avg_score': 0.0}
                
                pattern = mm_patterns[pattern_key]
                pattern['avg_score'] = (pattern['avg_score'] * pattern['count'] + mm_score) / (pattern['count'] + 1)
                pattern['count'] += 1
            
            return np.tanh(mm_score)
            
        except Exception as e:
            logger.error(f"Error in market maker detection: {e}")
            return 0.0

    def _calculate_price_efficiency(self, prices: List[float], volumes: List[float]) -> float:
        """Calculate how efficiently prices move with volume"""
        try:
            if len(prices) < 3:
                return 0.0
            
            # Calculate price movement per unit volume
            total_price_change = abs(prices[-1] - prices[0])
            total_volume = sum(volumes)
            
            if total_volume == 0:
                return 0.0
            
            efficiency = total_price_change / total_volume
            
            # Higher efficiency suggests smart money (moving prices with less volume)
            return min(1.0, efficiency * 1000)  # Scale appropriately
            
        except Exception as e:
            return 0.0

    def _analyze_timing_patterns(self, timestamps: List[float], volumes: List[float]) -> float:
        """Analyze timing patterns for smart money detection"""
        try:
            if len(timestamps) < 5:
                return 0.0
            
            # Convert to time of day
            from datetime import datetime
            times_of_day = [datetime.fromtimestamp(ts).hour for ts in timestamps]
            
            # Smart money often trades during less active hours
            off_hours_volume = 0
            total_volume = sum(volumes)
            
            for i, hour in enumerate(times_of_day):
                if hour < 9 or hour > 16:  # Outside regular hours
                    off_hours_volume += volumes[i]
            
            if total_volume > 0:
                off_hours_ratio = off_hours_volume / total_volume
                return min(1.0, off_hours_ratio * 2)  # Scale appropriately
            
            return 0.0
            
        except Exception as e:
            return 0.0

    def _calculate_mean_reversion(self, returns: np.ndarray) -> float:
        """Calculate mean reversion tendency"""
        try:
            if len(returns) < 2:
                return 0.0
            
            # Calculate autocorrelation at lag 1
            mean_return = np.mean(returns)
            variance = np.var(returns)
            
            if variance == 0:
                return 0.0
            
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            
            if np.isnan(autocorr):
                return 0.0
            
            # Negative autocorrelation suggests mean reversion
            return max(0.0, -autocorr)
            
        except Exception as e:
            return 0.0

    def _analyze_volume_distribution(self, volumes: List[float]) -> float:
        """Analyze volume distribution patterns"""
        try:
            if len(volumes) < 5:
                return 0.0
            
            # Calculate volume distribution metrics
            volume_array = np.array(volumes)
            mean_vol = np.mean(volume_array)
            
            # Look for consistent, moderate volumes (MM characteristic)
            consistency = 1.0 - (np.std(volume_array) / mean_vol if mean_vol > 0 else 1.0)
            
            # Moderate volumes (not too high, not too low)
            max_vol = np.max(volume_array)
            min_vol = np.min(volume_array)
            
            if max_vol > 0:
                volume_ratio = min_vol / max_vol
                moderation_score = volume_ratio  # Higher ratio = more consistent
            else:
                moderation_score = 0.0
            
            return (consistency * 0.6 + moderation_score * 0.4)
            
        except Exception as e:
            return 0.0