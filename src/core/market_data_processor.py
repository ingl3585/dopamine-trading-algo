# market_data_processor.py

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from src.market_analysis.data_processor import DataProcessor
from src.data_models.trading_domain_models import MarketData

logger = logging.getLogger(__name__)

@dataclass
class DataProcessingMetrics:
    """Metrics for data processing performance"""
    total_updates: int = 0
    failed_updates: int = 0
    last_15m_bar: float = 0.0
    last_1h_bar: float = 0.0
    last_4h_bar: float = 0.0
    historical_processed: bool = False

class MarketDataProcessor:
    """
    Handles all market data processing responsibilities.
    
    Responsibilities:
    - Process live market data updates
    - Validate data quality and completeness
    - Track timeframe bar completions
    - Handle historical data bootstrapping
    - Monitor data processing metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_processor = DataProcessor()
        self.metrics = DataProcessingMetrics()
        
        # Data validation parameters
        self.quality_threshold = 0.9  # 90% quality threshold
        self.min_bars_per_timeframe = 5
        
        logger.info("Market data processor initialized")
    
    def process_live_data(self, raw_data: Dict[str, Any]) -> Optional[MarketData]:
        """
        Process live market data update
        
        Args:
            raw_data: Raw market data from TCP server
            
        Returns:
            MarketData: Processed market data or None if processing failed
        """
        try:
            self.metrics.total_updates += 1
            
            # Enhanced logging for debugging
            if self.metrics.total_updates % 20 == 0:  # Log every 20 data updates
                logger.info(f"Processing market data update #{self.metrics.total_updates}")
            
            market_data = self.data_processor.process(raw_data)
            
            if not market_data:
                logger.warning("Market data processing returned None")
                self.metrics.failed_updates += 1
                return None
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error processing live market data: {e}")
            self.metrics.failed_updates += 1
            return None
    
    def check_new_15m_bar(self) -> bool:
        """Check if a new 15-minute bar was completed"""
        return self.data_processor.check_and_reset_15m_bar_flag()
    
    def check_new_1h_bar(self) -> bool:
        """Check if a new 1-hour bar was completed"""
        return self.data_processor.check_and_reset_1h_bar_flag()
    
    def check_new_4h_bar(self) -> bool:
        """Check if a new 4-hour bar was completed"""
        return self.data_processor.check_and_reset_4h_bar_flag()
    
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
            
            # Prime the data processor with historical data
            self.data_processor.prime_with_historical_data(historical_data)
            logger.info("Data processor primed with historical data")
            
            self.metrics.historical_processed = True
            return True
            
        except Exception as e:
            logger.error(f"Error processing historical data: {e}")
            return False
    
    def validate_historical_data(self, historical_data: Dict[str, Any]) -> bool:
        """
        Validate historical data quality and completeness
        
        Args:
            historical_data: Historical market data to validate
            
        Returns:
            bool: True if data is valid
        """
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
                if not self.validate_bars_quality(bars, timeframe):
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
    
    def validate_bars_quality(self, bars: List[Dict[str, Any]], timeframe: str) -> bool:
        """
        Validate individual bars for data quality
        
        Args:
            bars: List of bar data
            timeframe: Timeframe identifier
            
        Returns:
            bool: True if bars meet quality standards
        """
        try:
            if not bars or len(bars) < self.min_bars_per_timeframe:
                logger.error(f"Insufficient bars for {timeframe}: {len(bars)}")
                return False
            
            # Check for valid price data
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
    
    def get_processing_metrics(self) -> Dict[str, Any]:
        """Get current data processing metrics"""
        return {
            'total_updates': self.metrics.total_updates,
            'failed_updates': self.metrics.failed_updates,
            'success_rate': 1 - (self.metrics.failed_updates / max(1, self.metrics.total_updates)),
            'historical_processed': self.metrics.historical_processed,
            'last_15m_bar': self.metrics.last_15m_bar,
            'last_1h_bar': self.metrics.last_1h_bar,
            'last_4h_bar': self.metrics.last_4h_bar
        }
    
    def reset_metrics(self):
        """Reset processing metrics"""
        self.metrics = DataProcessingMetrics()
        logger.info("Data processing metrics reset")
    
    def is_ready_for_trading(self) -> bool:
        """Check if data processor is ready for live trading"""
        return self.metrics.historical_processed and self.metrics.total_updates > 0