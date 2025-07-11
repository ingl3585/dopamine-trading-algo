"""
Market Data Processing - Clean market data handling and feature extraction
"""

import numpy as np
import logging
from typing import Dict, List
from collections import deque
from datetime import datetime

from src.market_analysis.data_processor import MarketData
from src.shared.constants import DEFAULT_BUFFER_SIZE_LARGE

logger = logging.getLogger(__name__)

class MarketDataProcessor:
    """
    Core market data processing service
    """
    
    def __init__(self, config):
        self.config = config
        self.price_history = deque(maxlen=DEFAULT_BUFFER_SIZE_LARGE)
        self.volume_history = deque(maxlen=DEFAULT_BUFFER_SIZE_LARGE)
        self.timestamp_history = deque(maxlen=DEFAULT_BUFFER_SIZE_LARGE)
        
    def process_data(self, raw_data: Dict, is_historical: bool = False) -> MarketData:
        """
        Process raw market data into standardized format
        
        Args:
            raw_data: Raw data from NinjaTrader
            is_historical: True for historical bootstrap data (no account fields), False for live data
        """
        try:
            # STRICT: All market data must come from NinjaTrader - no defaults allowed
            required_fields = ['timestamp', 'close', 'volume']
            for field in required_fields:
                if field not in raw_data:
                    raise ValueError(f"Missing required market data field '{field}' from NinjaTrader")
            
            # Extract and validate raw data
            timestamp = float(raw_data['timestamp'])
            open_price = float(raw_data.get('open', raw_data['close']))  # Use close if OHLC not available
            high_price = float(raw_data.get('high', raw_data['close']))
            low_price = float(raw_data.get('low', raw_data['close']))
            close_price = float(raw_data['close'])
            volume = float(raw_data['volume'])
            
            # Handle account data based on data type
            if is_historical:
                # Historical data doesn't include account fields - use None values
                account_balance = None
                buying_power = None
                daily_pnl = None
                unrealized_pnl = None
                net_liquidation = None
                margin_used = None
                available_margin = None
                open_positions = None
                total_position_size = None
                margin_utilization = None
                buying_power_ratio = None
                daily_pnl_pct = None
            else:
                # STRICT: Live data must include all account fields from NinjaTrader
                required_account_fields = [
                    'account_balance', 'buying_power', 'daily_pnl', 'unrealized_pnl',
                    'net_liquidation', 'margin_used', 'available_margin', 
                    'open_positions', 'total_position_size'
                ]
                
                missing_account_fields = []
                for field in required_account_fields:
                    if field not in raw_data:
                        missing_account_fields.append(field)
                
                if missing_account_fields:
                    raise ValueError(f"Missing required account data fields from NinjaTrader: {missing_account_fields}")
                
                # Extract all account data from NinjaTrader (no defaults!)
                account_balance = float(raw_data['account_balance'])
                buying_power = float(raw_data['buying_power'])
                daily_pnl = float(raw_data['daily_pnl'])
                unrealized_pnl = float(raw_data['unrealized_pnl'])
                net_liquidation = float(raw_data['net_liquidation'])
                margin_used = float(raw_data['margin_used'])
                available_margin = float(raw_data['available_margin'])
                open_positions = int(raw_data['open_positions'])
                total_position_size = int(raw_data['total_position_size'])
                
                # Calculate computed ratios from real NinjaTrader data
                margin_utilization = (margin_used / net_liquidation) if net_liquidation > 0 else 0.0
                buying_power_ratio = (buying_power / account_balance) if account_balance > 0 else 0.0
                daily_pnl_pct = (daily_pnl / account_balance) if account_balance > 0 else 0.0
            
            # Create market data object
            market_data = MarketData(
                # Core market data (required from NinjaTrader)
                timestamp=timestamp,
                price=close_price,
                volume=volume,
                # Account data (ALL from NinjaTrader - no defaults)
                account_balance=account_balance,
                buying_power=buying_power,
                daily_pnl=daily_pnl,
                unrealized_pnl=unrealized_pnl,
                net_liquidation=net_liquidation,
                margin_used=margin_used,
                available_margin=available_margin,
                open_positions=open_positions,
                total_position_size=total_position_size,
                # Computed ratios (calculated from real data)
                margin_utilization=margin_utilization,
                buying_power_ratio=buying_power_ratio,
                daily_pnl_pct=daily_pnl_pct,
                # OHLC data from NinjaTrader
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price
            )
            
            # Store in history
            self.price_history.append(close_price)
            self.volume_history.append(volume)
            self.timestamp_history.append(timestamp)
            
            return market_data
            
        except Exception as e:
            logger.error(f"CRITICAL ERROR: Failed to process market data from NinjaTrader: {e}")
            logger.error("All portfolio data must come from NinjaTrader - no defaults allowed!")
            logger.error("System will stop - fix NinjaTrader data feed.")
            # Re-raise the exception to stop the system
            raise RuntimeError(f"Market data processing failed - missing required NinjaTrader data: {e}") from e
    
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
                    try:
                        recent_prices = prices[-10:]
                        # Ensure all prices are valid numbers
                        valid_prices = [p for p in recent_prices if isinstance(p, (int, float)) and not np.isnan(p) and not np.isinf(p)]
                        if len(valid_prices) >= 2:
                            mean_price = np.mean(valid_prices)
                            if mean_price != 0:
                                features['volatility'] = np.std(valid_prices) / mean_price
                            else:
                                features['volatility'] = 0.01
                        else:
                            features['volatility'] = 0.01
                    except Exception as e:
                        logger.warning(f"Error calculating volatility: {e}")
                        features['volatility'] = 0.01
                else:
                    features['volatility'] = 0.01
            
            # Volume features
            if len(self.volume_history) >= 2:
                volumes = list(self.volume_history)
                
                # Volume momentum
                if len(volumes) >= 5:
                    try:
                        # Validate volume data
                        valid_volumes = [v for v in volumes if isinstance(v, (int, float)) and not np.isnan(v) and not np.isinf(v) and v >= 0]
                        if len(valid_volumes) >= 5:
                            recent_vol = np.mean(valid_volumes[-3:])
                            previous_vol = np.mean(valid_volumes[-8:-3]) if len(valid_volumes) >= 8 else recent_vol
                            features['volume_momentum'] = (recent_vol - previous_vol) / previous_vol if previous_vol != 0 else 0.0
                        else:
                            features['volume_momentum'] = 0.0
                    except Exception as e:
                        logger.warning(f"Error calculating volume momentum: {e}")
                        features['volume_momentum'] = 0.0
                else:
                    features['volume_momentum'] = 0.0
            
            # Time-based features
            if len(self.timestamp_history) >= 1:
                try:
                    timestamp = self.timestamp_history[-1]
                    # Validate timestamp is reasonable (not negative, not too far in future)
                    if 0 < timestamp < 2147483647:  # Valid Unix timestamp range
                        current_time = datetime.fromtimestamp(timestamp)
                        features['time_of_day'] = (current_time.hour * 60 + current_time.minute) / (24 * 60)
                    else:
                        # Use current time as fallback
                        current_time = datetime.now()
                        features['time_of_day'] = (current_time.hour * 60 + current_time.minute) / (24 * 60)
                except (ValueError, OSError) as e:
                    logger.warning(f"Invalid timestamp {self.timestamp_history[-1]}: {e}")
                    # Use current time as fallback
                    current_time = datetime.now()
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