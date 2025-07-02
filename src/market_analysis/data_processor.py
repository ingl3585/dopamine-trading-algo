# data_processor.py

import time
import logging

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    timestamp: float
    price: float
    volume: float
    prices_1m: List[float]
    prices_5m: List[float] 
    prices_15m: List[float]
    prices_1h: List[float]
    prices_4h: List[float]
    volumes_1m: List[float]
    volumes_5m: List[float]
    volumes_15m: List[float]
    volumes_1h: List[float]
    volumes_4h: List[float]
    # Enhanced account data
    account_balance: float
    buying_power: float
    daily_pnl: float
    unrealized_pnl: float
    net_liquidation: float
    margin_used: float
    available_margin: float
    open_positions: int
    total_position_size: int
    # Computed ratios from TCP bridge
    margin_utilization: float
    buying_power_ratio: float
    daily_pnl_pct: float


class DataProcessor:
    def __init__(self, max_history=1000):
        self.prices_1m = deque(maxlen=max_history)
        self.prices_5m = deque(maxlen=200)
        self.prices_15m = deque(maxlen=100)
        self.prices_1h = deque(maxlen=50)   # 2+ days of hourly data
        self.prices_4h = deque(maxlen=30)   # 5 days of 4-hour data
        self.volumes_1m = deque(maxlen=max_history)
        self.volumes_5m = deque(maxlen=200)
        self.volumes_15m = deque(maxlen=100)
        self.volumes_1h = deque(maxlen=50)
        self.volumes_4h = deque(maxlen=30)
        
        # Track bar changes for commentary triggering and timeframe building
        self.last_15m_interval = 0  # Track 15-minute timestamp intervals
        self.last_1h_interval = 0   # Track 1-hour timestamp intervals
        self.last_4h_interval = 0   # Track 4-hour timestamp intervals
        self.new_15m_bar_available = False
        self.new_1h_bar_available = False
        self.new_4h_bar_available = False

    def process(self, raw_data: Dict) -> Optional[MarketData]:
        if not self._is_valid_data(raw_data):
            return None

        # Check for new 15-minute interval first (before processing any data)
        current_timestamp = raw_data.get('timestamp', time.time())
        
        # Convert .NET Ticks to Unix timestamp if needed
        if current_timestamp > 1e15:  # Definitely .NET ticks (very large number)
            unix_timestamp = (current_timestamp - 621355968000000000) / 10000000
        else:
            unix_timestamp = current_timestamp  # Already Unix timestamp
        
        # Round timestamp to 15-minute intervals (900 seconds = 15 minutes)
        current_15m_interval = int(unix_timestamp // 900) * 900
        
        # Check if we've moved to a new 15-minute interval
        if current_15m_interval > self.last_15m_interval:
            self.last_15m_interval = current_15m_interval
            self.new_15m_bar_available = True
            from datetime import datetime
            interval_time = datetime.fromtimestamp(current_15m_interval)
            logger.info(f"New 15-minute interval started: {interval_time.strftime('%H:%M:%S')}")

        # Update price/volume buffers
        if 'price_1m' in raw_data and raw_data['price_1m']:
            self.prices_1m.extend(raw_data['price_1m'])
            
        if 'volume_1m' in raw_data and raw_data['volume_1m']:
            self.volumes_1m.extend(raw_data['volume_1m'])
            
        if 'price_5m' in raw_data and raw_data['price_5m']:
            self.prices_5m.extend(raw_data['price_5m'])
            
        if 'volume_5m' in raw_data and raw_data['volume_5m']:
            self.volumes_5m.extend(raw_data['volume_5m'])
            
        if 'price_15m' in raw_data and raw_data['price_15m']:
            self.prices_15m.extend(raw_data['price_15m'])
            
        if 'volume_15m' in raw_data and raw_data['volume_15m']:
            self.volumes_15m.extend(raw_data['volume_15m'])

        if not self.prices_1m:
            return None

        # Build higher timeframes before returning
        self._build_higher_timeframes(raw_data)
        
        return MarketData(
            timestamp=raw_data.get('timestamp', time.time()),
            price=self.prices_1m[-1],
            volume=self.volumes_1m[-1] if self.volumes_1m else 1000,
            prices_1m=list(self.prices_1m),
            prices_5m=list(self.prices_5m),
            prices_15m=list(self.prices_15m),
            prices_1h=list(self.prices_1h) if self.prices_1h else [],
            prices_4h=list(self.prices_4h) if self.prices_4h else [],
            volumes_1m=list(self.volumes_1m),
            volumes_5m=list(self.volumes_5m),
            volumes_15m=list(self.volumes_15m),
            volumes_1h=list(self.volumes_1h) if self.volumes_1h else [],
            volumes_4h=list(self.volumes_4h) if self.volumes_4h else [],
            # Enhanced account data from NinjaTrader
            account_balance=raw_data.get('account_balance', 25000),
            buying_power=raw_data.get('buying_power', 25000),
            daily_pnl=raw_data.get('daily_pnl', 0),
            unrealized_pnl=raw_data.get('unrealized_pnl', 0),
            net_liquidation=raw_data.get('net_liquidation', 25000),
            margin_used=raw_data.get('margin_used', 0),
            available_margin=raw_data.get('available_margin', 25000),
            open_positions=raw_data.get('open_positions', 0),
            total_position_size=raw_data.get('total_position_size', 0.0),
            # Computed ratios from TCP bridge
            margin_utilization=raw_data.get('margin_utilization', 0.0),
            buying_power_ratio=raw_data.get('buying_power_ratio', 1.0),
            daily_pnl_pct=raw_data.get('daily_pnl_pct', 0.0)
        )
    
    def check_and_reset_15m_bar_flag(self) -> bool:
        """Check if a new 15-minute bar is available and reset the flag"""
        if self.new_15m_bar_available:
            self.new_15m_bar_available = False
            return True
        return False

    def _is_valid_data(self, data: Dict) -> bool:
        return (isinstance(data, dict) and 
                ('price_1m' in data or 'price_5m' in data or 'price_15m' in data))

    def prime_with_historical_data(self, historical_data: Dict):
        """Pre-populates the data buffers with historical data."""
        if 'bars_1m' in historical_data:
            self.prices_1m.extend([bar['close'] for bar in historical_data['bars_1m']])
            self.volumes_1m.extend([bar['volume'] for bar in historical_data['bars_1m']])

        if 'bars_5m' in historical_data:
            self.prices_5m.extend([bar['close'] for bar in historical_data['bars_5m']])
            self.volumes_5m.extend([bar['volume'] for bar in historical_data['bars_5m']])

        if 'bars_15m' in historical_data:
            self.prices_15m.extend([bar['close'] for bar in historical_data['bars_15m']])
            self.volumes_15m.extend([bar['volume'] for bar in historical_data['bars_15m']])
        
        # Build initial higher timeframes from historical data
        self._build_initial_higher_timeframes(historical_data)
        
        logger.info(f"Data processor primed with historical data: "
                    f"1m={len(self.prices_1m)}, 5m={len(self.prices_5m)}, 15m={len(self.prices_15m)}, "
                    f"1h={len(self.prices_1h)}, 4h={len(self.prices_4h)} bars")
    
    def _build_initial_higher_timeframes(self, historical_data: Dict):
        """Build initial 1H and 4H timeframes from historical data"""
        # Build 1H from 15m data (4 x 15m = 1H)
        if 'bars_15m' in historical_data and len(historical_data['bars_15m']) >= 4:
            bars_15m = historical_data['bars_15m']
            
            # Group every 4 bars into 1H
            for i in range(0, len(bars_15m) - 3, 4):
                hour_bars = bars_15m[i:i+4]
                if len(hour_bars) == 4:
                    # OHLC for the hour
                    hour_close = hour_bars[-1]['close']
                    hour_volume = sum(bar['volume'] for bar in hour_bars)
                    
                    self.prices_1h.append(hour_close)
                    self.volumes_1h.append(hour_volume)
        
        # Build 4H from 1H data (4 x 1H = 4H)
        if len(self.prices_1h) >= 4:
            # Group every 4 hours into 4H
            for i in range(0, len(self.prices_1h) - 3, 4):
                if i + 4 <= len(self.prices_1h):
                    hour_prices = list(self.prices_1h)[i:i+4]
                    hour_volumes = list(self.volumes_1h)[i:i+4]
                    
                    # 4H close and volume
                    h4_close = hour_prices[-1]
                    h4_volume = sum(hour_volumes)
                    
                    self.prices_4h.append(h4_close)
                    self.volumes_4h.append(h4_volume)

    def _build_higher_timeframes(self, raw_data: Dict):
        """Build 1H and 4H timeframes from real-time 15m data"""
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
                
                logger.info(f"New 4H bar built: {h4_close:.2f} (Vol: {h4_volume})")
    
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