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
    volumes_1m: List[float]
    volumes_5m: List[float]
    volumes_15m: List[float]
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
        self.volumes_1m = deque(maxlen=max_history)
        self.volumes_5m = deque(maxlen=200)
        self.volumes_15m = deque(maxlen=100)
        
        # Track bar changes for commentary triggering
        self.last_15m_interval = 0  # Track 15-minute timestamp intervals
        self.new_15m_bar_available = False

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

        return MarketData(
            timestamp=raw_data.get('timestamp', time.time()),
            price=self.prices_1m[-1],
            volume=self.volumes_1m[-1] if self.volumes_1m else 1000,
            prices_1m=list(self.prices_1m),
            prices_5m=list(self.prices_5m),
            prices_15m=list(self.prices_15m),
            volumes_1m=list(self.volumes_1m),
            volumes_5m=list(self.volumes_5m),
            volumes_15m=list(self.volumes_15m),
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