# core/trading_system.py

import logging
import time
from typing import Dict
from config import ResearchConfig
from features.feature_extractor import FeatureExtractor
from models.logistic_model import LogisticSignalModel
from communication.tcp_bridge import TCPBridge

log = logging.getLogger(__name__)

class TradingSystem:
    """Main research-aligned trading system"""
    
    def __init__(self):
        self.config = ResearchConfig()
        self.feature_extractor = FeatureExtractor(self.config)
        self.model = LogisticSignalModel(self.config)
        self.tcp_bridge = TCPBridge(self.config)
        
        # Price history for training feedback
        self.price_history = []
        
        # Statistics
        self.stats = {
            'signals_generated': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'hold_signals': 0,
            'avg_confidence': 0.0
        }
        
        self._setup_logging()
        log.info("Research-aligned trading system initialized")
    
    def start(self):
        """Start the trading system"""
        try:
            # Setup TCP callback
            self.tcp_bridge.on_market_data = self.process_market_data
            
            # Start TCP bridge
            self.tcp_bridge.start()
            
            log.info("Trading system started - waiting for market data")
            
            # Keep running
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            log.info("Shutdown requested")
        except Exception as e:
            log.error(f"System error: {e}")
        finally:
            self.stop()
    
    def process_market_data(self, data: Dict):
        """Process incoming market data and generate signals"""
        try:
            # Extract multi-timeframe data
            price_15m = data.get("price_15m", [])
            volume_15m = data.get("volume_15m", [])
            price_5m = data.get("price_5m", [])
            volume_5m = data.get("volume_5m", [])
            
            # Extract features
            features = self.feature_extractor.extract_features(
                price_15m, volume_15m, price_5m, volume_5m
            )
            
            if features is None:
                return
            
            # Generate signal
            action, confidence, quality = self.model.predict(features)
            
            # Update statistics
            self._update_stats(action, confidence)
            
            # Send signal if confidence meets threshold
            if confidence >= self.config.CONFIDENCE_THRESHOLD:
                self.tcp_bridge.send_signal(action, confidence, quality)
            
            # Update training data
            self._update_training_data(features, price_5m)
            
            # Log periodic statistics
            if self.stats['signals_generated'] % 100 == 0:
                self._log_statistics()
                
        except Exception as e:
            log.error(f"Market data processing error: {e}")
    
    def stop(self):
        """Stop the trading system"""
        log.info("Stopping trading system...")
        self.tcp_bridge.stop()
        self.model.save_model()
        log.info("System stopped")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _update_stats(self, action: int, confidence: float):
        """Update system statistics"""
        self.stats['signals_generated'] += 1
        
        if action == 1:
            self.stats['buy_signals'] += 1
        elif action == 2:
            self.stats['sell_signals'] += 1
        else:
            self.stats['hold_signals'] += 1
        
        # Update average confidence
        total = self.stats['signals_generated']
        self.stats['avg_confidence'] = (
            (self.stats['avg_confidence'] * (total - 1) + confidence) / total
        )
    
    def _update_training_data(self, features, price_5m):
        """Update model training data"""
        # Store current price for next iteration
        if len(price_5m) > 0:
            current_price = price_5m[-1]
            self.price_history.append(current_price)
            
            # Calculate price change for training
            if len(self.price_history) >= 2:
                previous_price = self.price_history[-2]
                price_change = (current_price - previous_price) / previous_price
                self.model.add_training_sample(features, price_change)
            
            # Keep limited history
            if len(self.price_history) > 100:
                self.price_history = self.price_history[-50:]
    
    def _log_statistics(self):
        """Log system statistics"""
        log.info("=== System Statistics ===")
        log.info(f"Total Signals: {self.stats['signals_generated']}")
        log.info(f"Buy: {self.stats['buy_signals']}, "
                f"Sell: {self.stats['sell_signals']}, "
                f"Hold: {self.stats['hold_signals']}")
        log.info(f"Average Confidence: {self.stats['avg_confidence']:.3f}")
        log.info(f"Model Trained: {self.model.is_trained}")