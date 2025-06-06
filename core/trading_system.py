# core/trading_system.py

import logging
import numpy as np
from typing import Dict
from config import ResearchConfig
from features.feature_extractor import FeatureExtractor
from models.logistic_model import LogisticSignalModel
from communication.tcp_bridge import TCPBridge

log = logging.getLogger(__name__)

class TradingSystem:
    """Main research-aligned trading system"""
    
    def __init__(self):
        # Configure logging first
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        self.config = ResearchConfig()
        self.feature_extractor = FeatureExtractor(self.config)
        self.model = LogisticSignalModel(self.config)
        
        # TCP bridge connects immediately like the working version
        self.tcp_bridge = TCPBridge(self.config)
        
        # Simple price tracking
        self.last_price = None
        self.trained_on_historical = False
        
        # Statistics
        self.stats = {
            'signals_generated': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'hold_signals': 0,
            'avg_confidence': 0.0
        }
        
        log.info("Research-aligned trading system initialized")
    
    def start(self):
        """Start the trading system"""
        try:
            # Setup TCP callback
            self.tcp_bridge.on_market_data = self.process_market_data
            
            # Start TCP bridge (connections already established)
            self.tcp_bridge.start()
            
            log.info("Trading system started - waiting for market data")
            log.info("Press Ctrl+C to stop the system")
            
            # Wait for shutdown signal
            try:
                import signal
                signal.pause()  # Unix only
            except AttributeError:
                # Windows fallback
                import threading
                self._shutdown_event = threading.Event()
                
                # Wait with timeout so we can check for shutdown periodically
                while not self._shutdown_event.is_set():
                    self._shutdown_event.wait(timeout=1)
                        
        except KeyboardInterrupt:
            log.info("Shutdown requested via Ctrl+C")
        except Exception as e:
            log.error(f"System error: {e}")
        finally:
            self.stop()

    def shutdown(self):
        """Trigger shutdown"""
        log.info("Shutdown requested...")
        if hasattr(self, '_shutdown_event'):
            self._shutdown_event.set()
        self.stop()
    
    def stop(self):
        """Stop the trading system"""
        log.info("Stopping trading system...")
        if hasattr(self, 'tcp_bridge'):
            self.tcp_bridge.stop()
        self.model.save_model()
        log.info("System stopped")
    
    def process_market_data(self, data: Dict):
        """Process incoming market data and generate signals"""
        try:
            # Extract data
            price_15m = data.get("price_15m", [])
            volume_15m = data.get("volume_15m", [])
            price_5m = data.get("price_5m", [])
            volume_5m = data.get("volume_5m", [])
            
            if not price_5m:
                return
            
            current_price = price_5m[-1]
            
            # Extract features
            features = self.feature_extractor.extract_features(
                price_15m, volume_15m, price_5m, volume_5m
            )
            
            if features is None:
                return
            
            # Train on historical data once
            if not self.trained_on_historical:
                log.info(f"Training on {len(price_5m)} historical bars")
                self._train_on_historical_data(data)  # Pass full data
                self.trained_on_historical = True
            
            # Generate signal
            action, confidence, quality = self.model.predict(features)
            log.info(f"Signal: Action={action}, Confidence={confidence:.3f}, Quality={quality}")
            
            # Update statistics
            self._update_stats(action, confidence)
            
            # Send signal if confidence meets threshold
            if confidence >= self.config.CONFIDENCE_THRESHOLD:
                self.tcp_bridge.send_signal(action, confidence, quality)
            
            # Update model with real-time data
            if self.last_price is not None:
                price_change = (current_price - self.last_price) / self.last_price
                self.model.add_training_sample(features, price_change)
            
            self.last_price = current_price
                
        except Exception as e:
            log.error(f"Market data processing error: {e}")
    
    def _train_on_historical_data(self, data):
        """Train using research-aligned feature-based signals - IMPROVED"""
        price_15m = data.get("price_15m", [])
        volume_15m = data.get("volume_15m", [])
        price_5m = data.get("price_5m", [])
        volume_5m = data.get("volume_5m", [])
        
        min_samples = max(50, self.config.SMA_PERIOD)
        training_samples = 0
        
        # Process MORE historical data for better training
        for i in range(min_samples, len(price_5m) - 5):  # Leave buffer for forward-looking
            # Get data up to point i
            hist_price_15m = price_15m[:i+1]
            hist_vol_15m = volume_15m[:i+1] 
            hist_price_5m = price_5m[:i+1]
            hist_vol_5m = volume_5m[:i+1]
            
            # Extract features for this specific time point
            features = self.feature_extractor.extract_features(
                hist_price_15m, hist_vol_15m, hist_price_5m, hist_vol_5m
            )
            
            if features is None:
                continue
            
            # Look ahead multiple periods for better signal labeling
            future_prices = price_5m[i+1:i+6]  # Next 5 periods
            if len(future_prices) < 5:
                continue
                
            # Calculate forward-looking price change (more robust)
            current_price = price_5m[i]
            max_future = max(future_prices)
            min_future = min(future_prices)
            
            # Determine signal based on significant future price movement
            price_change_up = (max_future - current_price) / current_price
            price_change_down = (current_price - min_future) / current_price
            
            # More realistic signal thresholds
            if price_change_up > 0.002:  # 0.2% upward movement
                signal = 1  # BUY
            elif price_change_down > 0.002:  # 0.2% downward movement  
                signal = 2  # SELL
            else:
                signal = 0  # HOLD
            
            # Store for training
            self.model.feature_history.append(features)
            self.model.signal_history.append(signal)
            training_samples += 1

        # Print signal distribution for debugging
        unique_signals, counts = np.unique(self.model.signal_history, return_counts=True)
        signal_dist = dict(zip(unique_signals, counts))
        log.info(f"Training samples: {training_samples}")
        log.info(f"Signal distribution: {signal_dist}")
        
        # Force training if we have enough samples
        if training_samples >= self.config.MIN_TRAINING_SAMPLES:
            self.model.train()
            log.info("Initial model training completed")
        else:
            log.warning(f"Insufficient training samples: {training_samples}/{self.config.MIN_TRAINING_SAMPLES}")
    
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