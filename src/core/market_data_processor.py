"""
Unified Market Data Processing System

This module consolidates market data processing functionality from the previous
separate processors, providing comprehensive data handling while following 
clean architecture principles.

Responsibilities:
- Process live and historical market data from NinjaTrader
- Validate data quality and completeness
- Extract trading features for AI subsystems
- Manage multiple timeframe data buffers
- Track bar completion events
- Provide metrics and quality assessment
- Handle data normalization and standardization
"""

import time
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Union
from collections import deque
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Comprehensive market data structure"""
    # Core market data (required)
    timestamp: float
    price: float
    volume: float
    
    # Enhanced account data (required for live data, None for historical)
    account_balance: Optional[float] = None
    buying_power: Optional[float] = None
    daily_pnl: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    net_liquidation: Optional[float] = None
    margin_used: Optional[float] = None
    available_margin: Optional[float] = None
    open_positions: Optional[int] = None
    total_position_size: Optional[int] = None
    
    # Computed ratios from TCP bridge (required for live data, None for historical)
    margin_utilization: Optional[float] = None
    buying_power_ratio: Optional[float] = None
    daily_pnl_pct: Optional[float] = None
    
    # OHLC data from NinjaTrader (optional)
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    
    # Historical price data (optional)
    prices_1m: Optional[List[float]] = None
    prices_5m: Optional[List[float]] = None
    prices_15m: Optional[List[float]] = None
    prices_1h: Optional[List[float]] = None
    prices_4h: Optional[List[float]] = None
    volumes_1m: Optional[List[float]] = None
    volumes_5m: Optional[List[float]] = None
    volumes_15m: Optional[List[float]] = None
    volumes_1h: Optional[List[float]] = None
    volumes_4h: Optional[List[float]] = None

@dataclass
class DataProcessingMetrics:
    """Comprehensive metrics for data processing performance and quality"""
    total_updates: int = 0
    failed_updates: int = 0
    historical_processed: bool = False
    last_15m_bar: float = 0.0
    last_1h_bar: float = 0.0
    last_4h_bar: float = 0.0
    data_quality_score: float = 0.0
    feature_extraction_errors: int = 0

class IMarketDataProcessor:
    """Interface for market data processing following interface segregation principle"""
    
    def process_live_data(self, raw_data: Dict[str, Any]) -> Optional[MarketData]:
        """Process live market data"""
        raise NotImplementedError
    
    def process_historical_data(self, historical_data: Dict[str, Any]) -> bool:
        """Process historical data for bootstrapping"""
        raise NotImplementedError
    
    def extract_features(self, market_data: MarketData) -> Dict[str, Any]:
        """Extract trading features from market data"""
        raise NotImplementedError

class IDataValidator:
    """Interface for data validation"""
    
    def validate_live_data(self, raw_data: Dict[str, Any]) -> bool:
        """Validate live data"""
        raise NotImplementedError
    
    def validate_historical_data(self, historical_data: Dict[str, Any]) -> bool:
        """Validate historical data"""
        raise NotImplementedError

class ITimeframeManager:
    """Interface for timeframe management"""
    
    def check_new_bar(self, timeframe: str) -> bool:
        """Check if new bar is available for timeframe"""
        raise NotImplementedError
    
    def build_higher_timeframes(self, raw_data: Dict[str, Any]) -> None:
        """Build higher timeframe data from lower timeframes"""
        raise NotImplementedError

class MarketDataProcessor(IMarketDataProcessor, IDataValidator, ITimeframeManager):
    """
    Unified market data processing system following clean architecture principles.
    
    Features:
    - Comprehensive live and historical data processing
    - Multi-timeframe data management with automatic bar detection
    - Feature extraction for AI subsystems
    - Data quality validation and metrics
    - Error handling and recovery
    - Performance monitoring and optimization
    """
    
    def __init__(self, config: Dict[str, Any], max_history: int = 1000):
        """
        Initialize market data processor
        
        Args:
            config: Configuration dictionary
            max_history: Maximum number of data points to keep in memory
        """
        self.config = config
        self.max_history = max_history
        
        # Initialize data buffers for multiple timeframes
        self._initialize_data_buffers()
        
        # Initialize timeframe tracking
        self._initialize_timeframe_tracking()
        
        # Initialize metrics and validation
        self.metrics = DataProcessingMetrics()
        self.quality_threshold = 0.9
        self.min_bars_per_timeframe = 5
        
        logger.info(f"Market data processor initialized with max_history={max_history}")
    
    def _initialize_data_buffers(self) -> None:
        """Initialize deque buffers for different timeframes"""
        # Price buffers
        self.prices_1m = deque(maxlen=self.max_history)
        self.prices_5m = deque(maxlen=200)
        self.prices_15m = deque(maxlen=100)
        self.prices_1h = deque(maxlen=50)
        self.prices_4h = deque(maxlen=30)
        
        # Volume buffers
        self.volumes_1m = deque(maxlen=self.max_history)
        self.volumes_5m = deque(maxlen=200)
        self.volumes_15m = deque(maxlen=100)
        self.volumes_1h = deque(maxlen=50)
        self.volumes_4h = deque(maxlen=30)
        
        # Additional tracking buffers
        self.price_history = deque(maxlen=self.max_history)
        self.volume_history = deque(maxlen=self.max_history)
        self.timestamp_history = deque(maxlen=self.max_history)
    
    def _initialize_timeframe_tracking(self) -> None:
        """Initialize timeframe interval tracking"""
        self.last_15m_interval = 0
        self.last_1h_interval = 0
        self.last_4h_interval = 0
        
        # Bar availability flags
        self.new_15m_bar_available = False
        self.new_1h_bar_available = False
        self.new_4h_bar_available = False
    
    # IMarketDataProcessor implementation
    def process_live_data(self, raw_data: Dict[str, Any]) -> Optional[MarketData]:
        """
        Process live market data update with comprehensive validation
        
        Args:
            raw_data: Raw market data from TCP server
            
        Returns:
            MarketData: Processed market data or None if processing failed
        """
        try:
            self.metrics.total_updates += 1
            
            # Log progress for monitoring
            if self.metrics.total_updates % 20 == 0:
                logger.info(f"Processing market data update #{self.metrics.total_updates}")
            
            # Validate input data
            if not self.validate_live_data(raw_data):
                logger.warning("Live data validation failed")
                self.metrics.failed_updates += 1
                return None
            
            # Process data using internal processing logic
            market_data = self._process_data_internal(raw_data, is_historical=False)
            
            if not market_data:
                logger.warning("Market data processing returned None")
                self.metrics.failed_updates += 1
                return None
            
            # Check and update timeframe bars
            self._update_timeframe_tracking(raw_data)
            
            # Build higher timeframes
            self.build_higher_timeframes(raw_data)
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error processing live market data: {e}")
            self.metrics.failed_updates += 1
            return None
    
    def process_historical_data(self, historical_data: Dict[str, Any]) -> bool:
        """
        Process historical data for pattern bootstrapping
        
        Args:
            historical_data: Historical market data
            
        Returns:
            bool: True if processing succeeded
        """
        try:
            logger.info("Processing historical data for pattern learning...")
            
            # Validate historical data quality
            if not self.validate_historical_data(historical_data):
                logger.error("Historical data validation failed")
                return False
            
            # Prime all data buffers with historical data
            self._prime_with_historical_data(historical_data)
            
            # Build initial higher timeframes
            self._build_initial_higher_timeframes(historical_data)
            
            self.metrics.historical_processed = True
            logger.info("Historical data processing completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error processing historical data: {e}")
            return False
    
    def extract_features(self, market_data: MarketData) -> Dict[str, Any]:
        """
        Extract comprehensive trading features from market data
        
        Args:
            market_data: Processed market data
            
        Returns:
            Dict containing extracted features
        """
        try:
            features = {}
            
            # Extract price-based features
            features.update(self._extract_price_features())
            
            # Extract volume-based features
            features.update(self._extract_volume_features())
            
            # Extract time-based features
            features.update(self._extract_time_features())
            
            # Extract account-based features (for live data)
            if market_data.account_balance is not None:
                features.update(self._extract_account_features(market_data))
            
            # Ensure all features have valid values
            features = self._sanitize_features(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            self.metrics.feature_extraction_errors += 1
            return self._get_default_features()
    
    # IDataValidator implementation
    def validate_live_data(self, raw_data: Dict[str, Any]) -> bool:
        """Validate live data from NinjaTrader"""
        try:
            if not isinstance(raw_data, dict):
                logger.error("Raw data must be a dictionary")
                return False
            
            # Check for basic required fields from NinjaTrader live data
            required_fields = ['timestamp', 'current_price']
            for field in required_fields:
                if field not in raw_data:
                    logger.error(f"Missing required field '{field}' in live data")
                    return False
            
            # Validate data types and ranges
            timestamp = raw_data.get('timestamp')
            if not isinstance(timestamp, (int, float)) or timestamp <= 0:
                logger.error(f"Invalid timestamp: {timestamp}")
                return False
            
            price = raw_data.get('current_price')
            if not isinstance(price, (int, float)) or price <= 0:
                logger.error(f"Invalid current_price: {price}")
                return False
            
            # Validate that we have at least one timeframe of price data
            price_fields = ['price_1m', 'price_5m', 'price_15m', 'price_1h', 'price_4h']
            has_price_data = any(field in raw_data and isinstance(raw_data[field], list) and len(raw_data[field]) > 0 
                               for field in price_fields)
            if not has_price_data:
                logger.error("No valid price arrays found in live data")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating live data: {e}")
            return False
    
    def validate_historical_data(self, historical_data: Dict[str, Any]) -> bool:
        """Validate historical data quality and completeness"""
        try:
            if not historical_data:
                logger.error("Historical data is empty")
                return False
            
            # Check for required data structures
            required_fields = ['bars_4h', 'bars_1h', 'bars_15m', 'bars_5m', 'bars_1m']
            for field in required_fields:
                if field not in historical_data:
                    logger.error(f"Missing required field: {field}")
                    return False
                
                bars = historical_data[field]
                if not isinstance(bars, list) or len(bars) < self.min_bars_per_timeframe:
                    logger.error(f"Insufficient data in {field}: {len(bars) if isinstance(bars, list) else 'invalid'} bars")
                    return False
            
            # Validate data quality for each timeframe
            for timeframe, bars in [(k, v) for k, v in historical_data.items() if k.startswith('bars_')]:
                if not self._validate_bars_quality(bars, timeframe):
                    return False
            
            logger.info(f"Historical data validation passed: "
                       f"4h={len(historical_data.get('bars_4h', []))}, "
                       f"1h={len(historical_data.get('bars_1h', []))}, "
                       f"15m={len(historical_data.get('bars_15m', []))}, "
                       f"5m={len(historical_data.get('bars_5m', []))}, "
                       f"1m={len(historical_data.get('bars_1m', []))} bars")
            return True
            
        except Exception as e:
            logger.error(f"Error validating historical data: {e}")
            return False
    
    # ITimeframeManager implementation
    def check_new_bar(self, timeframe: str) -> bool:
        """Check if new bar is available for specified timeframe"""
        if timeframe == '15m':
            return self.check_and_reset_15m_bar_flag()
        elif timeframe == '1h':
            return self.check_and_reset_1h_bar_flag()
        elif timeframe == '4h':
            return self.check_and_reset_4h_bar_flag()
        else:
            logger.warning(f"Unknown timeframe: {timeframe}")
            return False
    
    def build_higher_timeframes(self, raw_data: Dict[str, Any]) -> None:
        """Build higher timeframe data from lower timeframes"""
        current_timestamp = raw_data.get('timestamp', time.time())
        
        # Convert .NET Ticks to Unix timestamp if needed
        if current_timestamp > 1e15:
            unix_timestamp = (current_timestamp - 621355968000000000) / 10000000
        else:
            unix_timestamp = current_timestamp
        
        # Check for new 1H bar (3600 seconds = 1 hour)
        current_1h_interval = int(unix_timestamp // 3600) * 3600
        if current_1h_interval > self.last_1h_interval:
            self.last_1h_interval = current_1h_interval
            self.new_1h_bar_available = True
            
            # Build new 1H bar from last 4 x 15m bars
            if len(self.prices_15m) >= 4:
                recent_15m_prices = list(self.prices_15m)[-4:]
                recent_15m_volumes = list(self.volumes_15m)[-4:]
                
                h1_close = recent_15m_prices[-1]
                h1_volume = sum(recent_15m_volumes)
                
                self.prices_1h.append(h1_close)
                self.volumes_1h.append(h1_volume)
                self.metrics.last_1h_bar = h1_close
                
                logger.debug(f"New 1H bar built: {h1_close:.2f} (Vol: {h1_volume})")
        
        # Check for new 4H bar (14400 seconds = 4 hours)
        current_4h_interval = int(unix_timestamp // 14400) * 14400
        if current_4h_interval > self.last_4h_interval:
            self.last_4h_interval = current_4h_interval
            self.new_4h_bar_available = True
            
            # Build new 4H bar from last 4 x 1H bars
            if len(self.prices_1h) >= 4:
                recent_1h_prices = list(self.prices_1h)[-4:]
                recent_1h_volumes = list(self.volumes_1h)[-4:]
                
                h4_close = recent_1h_prices[-1]
                h4_volume = sum(recent_1h_volumes)
                
                self.prices_4h.append(h4_close)
                self.volumes_4h.append(h4_volume)
                self.metrics.last_4h_bar = h4_close
                
                logger.info(f"New 4H bar built: {h4_close:.2f} (Vol: {h4_volume})")
    
    # Internal processing methods
    def _process_data_internal(self, raw_data: Dict[str, Any], is_historical: bool = False) -> Optional[MarketData]:
        """Internal data processing logic"""
        try:
            # Extract and validate core data
            timestamp = float(raw_data['timestamp'])
            close_price = float(raw_data.get('current_price', raw_data.get('close', raw_data.get('price', 0))))
            volume = float(raw_data.get('volume', 1000))
            
            # OHLC data
            open_price = float(raw_data.get('open', close_price))
            high_price = float(raw_data.get('high', close_price))
            low_price = float(raw_data.get('low', close_price))
            
            # Handle account data based on data type
            if is_historical:
                # Historical data doesn't include account fields
                account_data = {
                    'account_balance': None, 'buying_power': None, 'daily_pnl': None,
                    'unrealized_pnl': None, 'net_liquidation': None, 'margin_used': None,
                    'available_margin': None, 'open_positions': None, 'total_position_size': None,
                    'margin_utilization': None, 'buying_power_ratio': None, 'daily_pnl_pct': None
                }
            else:
                # Extract account data for live trading
                account_data = self._extract_account_data(raw_data)
            
            # Update price/volume buffers based on available data
            self._update_data_buffers(raw_data)
            
            # Create market data object
            market_data = MarketData(
                # Core market data
                timestamp=timestamp,
                price=close_price,
                volume=volume,
                # Account data
                **account_data,
                # OHLC data
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                # Historical price arrays
                prices_1m=list(self.prices_1m) if self.prices_1m else None,
                prices_5m=list(self.prices_5m) if self.prices_5m else None,
                prices_15m=list(self.prices_15m) if self.prices_15m else None,
                prices_1h=list(self.prices_1h) if self.prices_1h else None,
                prices_4h=list(self.prices_4h) if self.prices_4h else None,
                volumes_1m=list(self.volumes_1m) if self.volumes_1m else None,
                volumes_5m=list(self.volumes_5m) if self.volumes_5m else None,
                volumes_15m=list(self.volumes_15m) if self.volumes_15m else None,
                volumes_1h=list(self.volumes_1h) if self.volumes_1h else None,
                volumes_4h=list(self.volumes_4h) if self.volumes_4h else None
            )
            
            # Store in master history buffers
            self.price_history.append(close_price)
            self.volume_history.append(volume)
            self.timestamp_history.append(timestamp)
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error in internal data processing: {e}")
            return None
    
    def _extract_account_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract account data from raw data"""
        try:
            # Basic account fields
            account_balance = raw_data.get('account_balance', 25000.0)
            buying_power = raw_data.get('buying_power', 25000.0)
            daily_pnl = raw_data.get('daily_pnl', 0.0)
            unrealized_pnl = raw_data.get('unrealized_pnl', 0.0)
            net_liquidation = raw_data.get('net_liquidation', 25000.0)
            margin_used = raw_data.get('margin_used', 0.0)
            available_margin = raw_data.get('available_margin', 25000.0)
            open_positions = raw_data.get('open_positions', 0)
            total_position_size = raw_data.get('total_position_size', 0)
            
            # Calculate computed ratios
            margin_utilization = (margin_used / net_liquidation) if net_liquidation > 0 else 0.0
            buying_power_ratio = (buying_power / account_balance) if account_balance > 0 else 0.0
            daily_pnl_pct = (daily_pnl / account_balance) if account_balance > 0 else 0.0
            
            return {
                'account_balance': float(account_balance),
                'buying_power': float(buying_power),
                'daily_pnl': float(daily_pnl),
                'unrealized_pnl': float(unrealized_pnl),
                'net_liquidation': float(net_liquidation),
                'margin_used': float(margin_used),
                'available_margin': float(available_margin),
                'open_positions': int(open_positions),
                'total_position_size': int(total_position_size),
                'margin_utilization': margin_utilization,
                'buying_power_ratio': buying_power_ratio,
                'daily_pnl_pct': daily_pnl_pct
            }
            
        except Exception as e:
            logger.warning(f"Error extracting account data: {e}")
            # Return default values for missing account data
            return {
                'account_balance': 25000.0, 'buying_power': 25000.0, 'daily_pnl': 0.0,
                'unrealized_pnl': 0.0, 'net_liquidation': 25000.0, 'margin_used': 0.0,
                'available_margin': 25000.0, 'open_positions': 0, 'total_position_size': 0,
                'margin_utilization': 0.0, 'buying_power_ratio': 1.0, 'daily_pnl_pct': 0.0
            }
    
    def _update_data_buffers(self, raw_data: Dict[str, Any]) -> None:
        """Update data buffers based on available timeframe data"""
        # Update 1m data
        if 'price_1m' in raw_data and raw_data['price_1m']:
            self.prices_1m.extend(raw_data['price_1m'])
        elif 'close' in raw_data:
            self.prices_1m.append(float(raw_data['close']))
            
        if 'volume_1m' in raw_data and raw_data['volume_1m']:
            self.volumes_1m.extend(raw_data['volume_1m'])
        elif 'volume' in raw_data:
            self.volumes_1m.append(float(raw_data['volume']))
        
        # Update other timeframes if available
        for tf in ['5m', '15m']:
            if f'price_{tf}' in raw_data and raw_data[f'price_{tf}']:
                getattr(self, f'prices_{tf}').extend(raw_data[f'price_{tf}'])
            if f'volume_{tf}' in raw_data and raw_data[f'volume_{tf}']:
                getattr(self, f'volumes_{tf}').extend(raw_data[f'volume_{tf}'])
    
    def _update_timeframe_tracking(self, raw_data: Dict[str, Any]) -> None:
        """Update timeframe interval tracking"""
        current_timestamp = raw_data.get('timestamp', time.time())
        
        # Convert .NET Ticks to Unix timestamp if needed
        if current_timestamp > 1e15:
            unix_timestamp = (current_timestamp - 621355968000000000) / 10000000
        else:
            unix_timestamp = current_timestamp
        
        # Check for new 15-minute interval
        current_15m_interval = int(unix_timestamp // 900) * 900
        if current_15m_interval > self.last_15m_interval:
            self.last_15m_interval = current_15m_interval
            self.new_15m_bar_available = True
            self.metrics.last_15m_bar = raw_data.get('close', 0)
            
            interval_time = datetime.fromtimestamp(current_15m_interval)
            logger.info(f"New 15-minute interval started: {interval_time.strftime('%H:%M:%S')}")
    
    def _prime_with_historical_data(self, historical_data: Dict[str, Any]) -> None:
        """Prime all data buffers with historical data"""
        # Prime individual timeframe buffers from NinjaTrader historical data
        if 'bars_1m' in historical_data:
            bars = historical_data['bars_1m']
            self.prices_1m.extend([bar['close'] for bar in bars])
            self.volumes_1m.extend([bar['volume'] for bar in bars])

        if 'bars_5m' in historical_data:
            bars = historical_data['bars_5m']
            self.prices_5m.extend([bar['close'] for bar in bars])
            self.volumes_5m.extend([bar['volume'] for bar in bars])

        if 'bars_15m' in historical_data:
            bars = historical_data['bars_15m']
            self.prices_15m.extend([bar['close'] for bar in bars])
            self.volumes_15m.extend([bar['volume'] for bar in bars])

        # Process pre-built 1h and 4h bars from NinjaTrader
        if 'bars_1h' in historical_data:
            bars = historical_data['bars_1h']
            self.prices_1h.extend([bar['close'] for bar in bars])
            self.volumes_1h.extend([bar['volume'] for bar in bars])

        if 'bars_4h' in historical_data:
            bars = historical_data['bars_4h']
            self.prices_4h.extend([bar['close'] for bar in bars])
            self.volumes_4h.extend([bar['volume'] for bar in bars])
        
        logger.info(f"Data processor primed with historical data: "
                    f"1m={len(self.prices_1m)}, 5m={len(self.prices_5m)}, 15m={len(self.prices_15m)}, "
                    f"1h={len(self.prices_1h)}, 4h={len(self.prices_4h)} bars")
    
    def _build_initial_higher_timeframes(self, historical_data: Dict[str, Any]) -> None:
        """Build initial 1H and 4H timeframes from historical data as fallback"""
        initial_1h_count = len(self.prices_1h)
        initial_4h_count = len(self.prices_4h)
        
        # Only build 1H from 15m data if we don't already have 1H data from NinjaTrader
        if len(self.prices_1h) == 0 and 'bars_15m' in historical_data and len(historical_data['bars_15m']) >= 4:
            bars_15m = historical_data['bars_15m']
            
            # Group every 4 bars into 1H
            for i in range(0, len(bars_15m) - 3, 4):
                hour_bars = bars_15m[i:i+4]
                if len(hour_bars) == 4:
                    hour_close = hour_bars[-1]['close']
                    hour_volume = sum(bar['volume'] for bar in hour_bars)
                    
                    self.prices_1h.append(hour_close)
                    self.volumes_1h.append(hour_volume)
            
            if len(self.prices_1h) > 0:
                logger.info(f"Built {len(self.prices_1h)} 1H bars from 15m data (fallback method)")
        
        # Only build 4H from 1H data if we don't already have 4H data from NinjaTrader
        if len(self.prices_4h) == 0 and len(self.prices_1h) >= 4:
            for i in range(0, len(self.prices_1h) - 3, 4):
                if i + 4 <= len(self.prices_1h):
                    hour_prices = list(self.prices_1h)[i:i+4]
                    hour_volumes = list(self.volumes_1h)[i:i+4]
                    
                    h4_close = hour_prices[-1]
                    h4_volume = sum(hour_volumes)
                    
                    self.prices_4h.append(h4_close)
                    self.volumes_4h.append(h4_volume)
            
            if len(self.prices_4h) > 0:
                logger.info(f"Built {len(self.prices_4h)} 4H bars from 1H data (fallback method)")
        
        # Log what method was used
        if initial_1h_count > 0:
            logger.info(f"Using {initial_1h_count} pre-built 1H bars from NinjaTrader")
        if initial_4h_count > 0:
            logger.info(f"Using {initial_4h_count} pre-built 4H bars from NinjaTrader")
    
    def _validate_bars_quality(self, bars: List[Dict[str, Any]], timeframe: str) -> bool:
        """Validate individual bars for data quality"""
        try:
            if not bars or len(bars) < self.min_bars_per_timeframe:
                logger.error(f"Insufficient bars for {timeframe}: {len(bars)}")
                return False
            
            valid_bars = 0
            for bar in bars:
                if isinstance(bar, dict):
                    required_fields = ['open', 'high', 'low', 'close', 'volume']
                    if all(field in bar and isinstance(bar[field], (int, float)) and bar[field] > 0 
                           for field in required_fields[:4]):  # OHLC must be positive
                        if bar['high'] >= bar['low'] and bar['high'] >= max(bar['open'], bar['close']):
                            valid_bars += 1
            
            quality_ratio = valid_bars / len(bars)
            if quality_ratio < self.quality_threshold:
                logger.error(f"Poor data quality for {timeframe}: {quality_ratio:.1%} valid bars")
                return False
            
            logger.info(f"{timeframe} data quality: {quality_ratio:.1%} ({valid_bars}/{len(bars)} valid bars)")
            return True
            
        except Exception as e:
            logger.error(f"Error validating bars quality for {timeframe}: {e}")
            return False
    
    # Feature extraction methods
    def _extract_price_features(self) -> Dict[str, float]:
        """Extract price-based features"""
        features = {}
        
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
                    valid_prices = [p for p in recent_prices if isinstance(p, (int, float)) and not np.isnan(p) and not np.isinf(p)]
                    if len(valid_prices) >= 2:
                        mean_price = np.mean(valid_prices)
                        features['volatility'] = (np.std(valid_prices) / mean_price) if mean_price != 0 else 0.01
                    else:
                        features['volatility'] = 0.01
                except Exception as e:
                    logger.warning(f"Error calculating volatility: {e}")
                    features['volatility'] = 0.01
            else:
                features['volatility'] = 0.01
        
        return features
    
    def _extract_volume_features(self) -> Dict[str, float]:
        """Extract volume-based features"""
        features = {}
        
        if len(self.volume_history) >= 5:
            try:
                volumes = list(self.volume_history)
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
        
        return features
    
    def _extract_time_features(self) -> Dict[str, float]:
        """Extract time-based features"""
        features = {}
        
        if len(self.timestamp_history) >= 1:
            try:
                timestamp = self.timestamp_history[-1]
                if 0 < timestamp < 2147483647:  # Valid Unix timestamp range
                    current_time = datetime.fromtimestamp(timestamp)
                    features['time_of_day'] = (current_time.hour * 60 + current_time.minute) / (24 * 60)
                else:
                    current_time = datetime.now()
                    features['time_of_day'] = (current_time.hour * 60 + current_time.minute) / (24 * 60)
            except (ValueError, OSError) as e:
                logger.warning(f"Invalid timestamp {self.timestamp_history[-1]}: {e}")
                current_time = datetime.now()
                features['time_of_day'] = (current_time.hour * 60 + current_time.minute) / (24 * 60)
        else:
            features['time_of_day'] = 0.5
        
        return features
    
    def _extract_account_features(self, market_data: MarketData) -> Dict[str, float]:
        """Extract account-based features for live data"""
        features = {}
        
        if market_data.margin_utilization is not None:
            features['margin_utilization'] = market_data.margin_utilization
        if market_data.buying_power_ratio is not None:
            features['buying_power_ratio'] = market_data.buying_power_ratio
        if market_data.daily_pnl_pct is not None:
            features['daily_pnl_pct'] = market_data.daily_pnl_pct
        
        return features
    
    def _sanitize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all features have valid values"""
        sanitized = {}
        for key, value in features.items():
            if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                sanitized[key] = float(value)
            else:
                sanitized[key] = 0.0
        
        # Add default features if missing
        default_features = {
            'price_momentum': 0.0, 'volume_momentum': 0.0, 'price_position': 0.5,
            'volatility': 0.01, 'time_of_day': 0.5, 'pattern_score': 0.5, 'confidence': 0.7
        }
        for key, default_value in default_features.items():
            if key not in sanitized:
                sanitized[key] = default_value
        
        return sanitized
    
    def _get_default_features(self) -> Dict[str, float]:
        """Get default feature set for error cases"""
        return {
            'price_momentum': 0.0, 'volume_momentum': 0.0, 'price_position': 0.5,
            'volatility': 0.01, 'time_of_day': 0.5, 'pattern_score': 0.5, 'confidence': 0.5
        }
    
    # Bar tracking methods
    def check_and_reset_15m_bar_flag(self) -> bool:
        """Check if a new 15-minute bar is available and reset the flag"""
        if self.new_15m_bar_available:
            self.new_15m_bar_available = False
            return True
        return False
    
    def check_and_reset_1h_bar_flag(self) -> bool:
        """Check if a new 1-hour bar is available and reset the flag"""
        if self.new_1h_bar_available:
            self.new_1h_bar_available = False
            return True
        return False
    
    def check_and_reset_4h_bar_flag(self) -> bool:
        """Check if a new 4-hour bar is available and reset the flag"""
        if self.new_4h_bar_available:
            self.new_4h_bar_available = False
            return True
        return False
    
    # Backward compatibility methods
    def process(self, raw_data: Dict[str, Any]) -> Optional[MarketData]:
        """
        Generic process method for backward compatibility.
        Routes to appropriate processing method based on data type.
        """
        # Determine if this is live or historical data
        if 'bars_1m' in raw_data or 'bars_5m' in raw_data:
            # This is historical data
            success = self.process_historical_data(raw_data)
            return None if not success else self._create_sample_market_data(raw_data)
        else:
            # This is live data
            return self.process_live_data(raw_data)
    
    def prime_with_historical_data(self, historical_data: Dict[str, Any]) -> bool:
        """
        Alias for process_historical_data for backward compatibility.
        """
        return self.process_historical_data(historical_data)
    
    def _create_sample_market_data(self, raw_data: Dict[str, Any]) -> MarketData:
        """Create a sample MarketData object from historical data for compatibility"""
        # Get the most recent price from available data
        current_price = 0.0
        current_volume = 1000.0
        current_timestamp = raw_data.get('timestamp', time.time())
        
        # Extract latest price from any available timeframe
        for timeframe in ['bars_1m', 'bars_5m', 'bars_15m', 'bars_1h', 'bars_4h']:
            if timeframe in raw_data and raw_data[timeframe]:
                bars = raw_data[timeframe]
                if bars and isinstance(bars, list) and len(bars) > 0:
                    latest_bar = bars[-1]
                    current_price = latest_bar.get('close', current_price)
                    current_volume = latest_bar.get('volume', current_volume)
                    current_timestamp = latest_bar.get('timestamp', current_timestamp)
                    break
        
        return MarketData(
            timestamp=float(current_timestamp),
            price=float(current_price),
            volume=float(current_volume)
        )

    # System management and monitoring methods
    def get_processing_metrics(self) -> Dict[str, Any]:
        """Get comprehensive processing metrics"""
        success_rate = 1.0 - (self.metrics.failed_updates / max(1, self.metrics.total_updates))
        
        return {
            'total_updates': self.metrics.total_updates,
            'failed_updates': self.metrics.failed_updates,
            'success_rate': success_rate,
            'historical_processed': self.metrics.historical_processed,
            'last_15m_bar': self.metrics.last_15m_bar,
            'last_1h_bar': self.metrics.last_1h_bar,
            'last_4h_bar': self.metrics.last_4h_bar,
            'data_quality_score': self.metrics.data_quality_score,
            'feature_extraction_errors': self.metrics.feature_extraction_errors
        }
    
    def get_data_quality_metrics(self) -> Dict[str, Any]:
        """Get comprehensive data quality metrics"""
        try:
            if not self.price_history:
                return {'data_points': 0, 'quality_score': 0.0}
            
            prices = list(self.price_history)
            volumes = list(self.volume_history)
            
            # Check for valid prices and volumes
            valid_prices = sum(1 for p in prices if p > 0)
            valid_volumes = sum(1 for v in volumes if v > 0)
            
            price_quality = valid_prices / len(prices) if prices else 0.0
            volume_quality = valid_volumes / len(volumes) if volumes else 0.0
            overall_quality = (price_quality + volume_quality) / 2
            
            # Update metrics
            self.metrics.data_quality_score = overall_quality
            
            return {
                'data_points': len(self.price_history),
                'price_quality': price_quality,
                'volume_quality': volume_quality,
                'quality_score': overall_quality,
                'latest_price': prices[-1] if prices else 0.0,
                'latest_volume': volumes[-1] if volumes else 0.0,
                'buffer_utilization': {
                    '1m': len(self.prices_1m),
                    '5m': len(self.prices_5m),
                    '15m': len(self.prices_15m),
                    '1h': len(self.prices_1h),
                    '4h': len(self.prices_4h)
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating data quality: {e}")
            return {'data_points': 0, 'quality_score': 0.0}
    
    def get_historical_data(self, lookback_periods: int = 100) -> Dict[str, List[float]]:
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
    
    def reset_metrics(self) -> None:
        """Reset processing metrics"""
        self.metrics = DataProcessingMetrics()
        logger.info("Data processing metrics reset")
    
    def is_ready_for_trading(self) -> bool:
        """Check if data processor is ready for live trading"""
        return (
            self.metrics.historical_processed and 
            self.metrics.total_updates > 0 and
            len(self.price_history) > 0 and
            self.metrics.data_quality_score > 0.8
        )