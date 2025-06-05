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
            log.info("=== PROCESSING MARKET DATA ===")
            
            # Extract multi-timeframe data
            price_15m = data.get("price_15m", [])
            volume_15m = data.get("volume_15m", [])
            price_5m = data.get("price_5m", [])
            volume_5m = data.get("volume_5m", [])
            
            log.info(f"Received data - 15m: {len(price_15m)} bars, 5m: {len(price_5m)} bars")
            
            if len(price_15m) > 0:
                log.info(f"15m price range: {min(price_15m):.2f} - {max(price_15m):.2f}")
            if len(price_5m) > 0:
                log.info(f"5m price range: {min(price_5m):.2f} - {max(price_5m):.2f}")
            
            # Extract features
            log.info("Extracting features...")
            features = self.feature_extractor.extract_features(
                price_15m, volume_15m, price_5m, volume_5m
            )
            
            if features is None:
                log.warning("Feature extraction returned None")
                return
            
            log.info("Features extracted successfully")
            
            # Generate signal
            log.info("Generating ML signal...")
            action, confidence, quality = self.model.predict(features)
            log.info(f"Generated signal: Action={action}, Confidence={confidence:.3f}, Quality={quality}")
            
            # Update statistics
            self._update_stats(action, confidence)
            
            # Send signal if confidence meets threshold
            if confidence >= self.config.CONFIDENCE_THRESHOLD:
                log.info(f"Sending signal (confidence {confidence:.3f} >= threshold {self.config.CONFIDENCE_THRESHOLD})")
                self.tcp_bridge.send_signal(action, confidence, quality)
            else:
                log.info(f"Signal below threshold - not sending (confidence {confidence:.3f} < {self.config.CONFIDENCE_THRESHOLD})")
            
            # Update training data
            log.info("Updating training data...")
            self._update_training_data(features, price_5m)
            
            # Log periodic statistics
            if self.stats['signals_generated'] % 10 == 0:  # Every 10 signals instead of 100
                self._log_statistics()
                
            log.info("=== MARKET DATA PROCESSING COMPLETE ===")
                
        except Exception as e:
            log.error(f"Market data processing error: {e}")
            import traceback
            log.error(f"Traceback: {traceback.format_exc()}")
    
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
        if len(price_5m) < 2:
            return
            
        # If this is a large batch (historical data)
        if len(price_5m) > 10:
            log.info(f"Processing {len(price_5m)} historical bars for training")
            
            # Create training samples from consecutive price changes
            for i in range(1, len(price_5m)):
                previous_price = price_5m[i-1]
                current_price = price_5m[i]
                price_change = (current_price - previous_price) / previous_price
                
                # Use existing method - it handles everything
                self.model.add_training_sample(features, price_change)
            
            # Set up for future real-time updates
            self.price_history = [price_5m[-1]]
            
        else:
            # Your existing real-time logic
            if len(price_5m) > 0:
                current_price = price_5m[-1]
                self.price_history.append(current_price)
                
                if len(self.price_history) >= 2:
                    previous_price = self.price_history[-2]
                    price_change = (current_price - previous_price) / previous_price
                    self.model.add_training_sample(features, price_change)
                
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