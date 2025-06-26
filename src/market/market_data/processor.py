"""
Market Data Processing - Clean market data handling and feature extraction
"""

import numpy as np
import logging
from typing import Dict, List
from collections import deque
from datetime import datetime

from src.shared.types import MarketData

logger = logging.getLogger(__name__)

class MarketDataProcessor:
    """
    Core market data processing service
    """
    
    def __init__(self, config):
        self.config = config
        self.price_history = deque(maxlen=1000)
        self.volume_history = deque(maxlen=1000)
        self.timestamp_history = deque(maxlen=1000)
        
    def process_data(self, raw_data: Dict) -> MarketData:
        """Process raw market data into standardized format"""
        try:
            # Extract and validate raw data
            timestamp = raw_data.get('timestamp', datetime.now().timestamp())
            open_price = float(raw_data.get('open', 0.0))
            high_price = float(raw_data.get('high', 0.0))
            low_price = float(raw_data.get('low', 0.0))
            close_price = float(raw_data.get('close', 0.0))
            volume = float(raw_data.get('volume', 0.0))
            
            # Create market data object
            market_data = MarketData(
                timestamp=timestamp,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume
            )
            
            # Store in history
            self.price_history.append(close_price)
            self.volume_history.append(volume)
            self.timestamp_history.append(timestamp)
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            # Return default market data
            return MarketData(
                timestamp=datetime.now().timestamp(),
                open=0.0, high=0.0, low=0.0, close=0.0, volume=0.0
            )
    
    def extract_features(self, market_data: MarketData) -> Dict:
        """Extract trading features from market data"""
        try:
            features = {}
            
            # Basic price features
            if len(self.price_history) >= 2:
                prices = list(self.price_history)
                
                # Price momentum
                if len(prices) >= 5:
                    features['price_momentum'] = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] != 0 else 0.0
                else:
                    features['price_momentum'] = 0.0
                
                # Price position (where current price sits in recent range)
                if len(prices) >= 20:
                    recent_high = max(prices[-20:])
                    recent_low = min(prices[-20:])
                    if recent_high != recent_low:
                        features['price_position'] = (prices[-1] - recent_low) / (recent_high - recent_low)
                    else:
                        features['price_position'] = 0.5
                else:
                    features['price_position'] = 0.5
                
                # Volatility
                if len(prices) >= 10:
                    recent_prices = prices[-10:]
                    features['volatility'] = np.std(recent_prices) / np.mean(recent_prices) if np.mean(recent_prices) != 0 else 0.0
                else:
                    features['volatility'] = 0.01
            
            # Volume features
            if len(self.volume_history) >= 2:
                volumes = list(self.volume_history)
                
                # Volume momentum
                if len(volumes) >= 5:
                    recent_vol = np.mean(volumes[-3:])
                    previous_vol = np.mean(volumes[-8:-3]) if len(volumes) >= 8 else recent_vol
                    features['volume_momentum'] = (recent_vol - previous_vol) / previous_vol if previous_vol != 0 else 0.0
                else:
                    features['volume_momentum'] = 0.0
            
            # Time-based features
            if len(self.timestamp_history) >= 1:
                current_time = datetime.fromtimestamp(self.timestamp_history[-1])
                features['time_of_day'] = (current_time.hour * 60 + current_time.minute) / (24 * 60)
            else:
                features['time_of_day'] = 0.5
            
            # Pattern confidence (placeholder for now)
            features['pattern_score'] = 0.5
            features['confidence'] = 0.7
            
            # Ensure all features have valid values
            for key, value in features.items():
                if np.isnan(value) or np.isinf(value):
                    features[key] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {
                'price_momentum': 0.0,
                'volume_momentum': 0.0,
                'price_position': 0.5,
                'volatility': 0.01,
                'time_of_day': 0.5,
                'pattern_score': 0.5,
                'confidence': 0.5
            }
    
    def get_historical_data(self, lookback_periods: int = 100) -> Dict:
        """Get historical data for analysis"""
        try:
            lookback = min(lookback_periods, len(self.price_history))
            
            if lookback == 0:
                return {'prices': [], 'volumes': [], 'timestamps': []}
            
            return {
                'prices': list(self.price_history)[-lookback:],
                'volumes': list(self.volume_history)[-lookback:],
                'timestamps': list(self.timestamp_history)[-lookback:]
            }
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return {'prices': [], 'volumes': [], 'timestamps': []}
    
    def get_data_quality_metrics(self) -> Dict:
        """Get data quality metrics"""
        try:
            if not self.price_history:
                return {'data_points': 0, 'quality_score': 0.0}
            
            # Calculate basic quality metrics
            prices = list(self.price_history)
            volumes = list(self.volume_history)
            
            # Check for valid prices
            valid_prices = sum(1 for p in prices if p > 0)
            price_quality = valid_prices / len(prices) if prices else 0.0
            
            # Check for valid volumes
            valid_volumes = sum(1 for v in volumes if v > 0)
            volume_quality = valid_volumes / len(volumes) if volumes else 0.0
            
            # Overall quality score
            quality_score = (price_quality + volume_quality) / 2
            
            return {
                'data_points': len(self.price_history),
                'price_quality': price_quality,
                'volume_quality': volume_quality,
                'quality_score': quality_score,
                'latest_price': prices[-1] if prices else 0.0,
                'latest_volume': volumes[-1] if volumes else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating data quality: {e}")
            return {'data_points': 0, 'quality_score': 0.0}