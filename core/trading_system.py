# core/trading_system.py

import threading
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
        
        # TCP bridge
        self.tcp_bridge = TCPBridge(self.config)

        # Handle shutdown
        self.shutdown_event = threading.Event()
        
        # Price tracking
        self.last_price = None
        self.trained_on_historical = False
        
        # Statistics
        self.stats = {
            'signals_generated': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'hold_signals': 0,
            'avg_confidence': 0.0,
            'high_confidence_signals': 0,
            'moderate_confidence_signals': 0,
            'low_confidence_signals': 0
        }
        
        log.info("Research-aligned trading system initialized")
    
    def start(self):
        """Start the trading system"""
        try:
            # Set TCP callback
            self.tcp_bridge.on_market_data = self.process_market_data
            
            # Start TCP bridge
            self.tcp_bridge.start()
            
            log.info("Trading system started - waiting for market data")
            
            while not self.shutdown_event.is_set():
                self.shutdown_event.wait(timeout=1)
            
        except KeyboardInterrupt:
            log.info("Shutdown requested via Ctrl+C")
        except Exception as e:
            log.error(f"System error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the trading system"""
        log.info("Stopping trading system...")
        
        # Stop TCP bridge safely
        try:
            if hasattr(self, 'tcp_bridge') and self.tcp_bridge:
                self.tcp_bridge.stop()
        except Exception as e:
            log.warning(f"Error stopping TCP bridge: {e}")
        
        # Save model safely
        try:
            if hasattr(self, 'model') and self.model:
                self.model.save_model()
        except Exception as e:
            log.warning(f"Error saving model: {e}")
        
        # Print final statistics
        self._print_final_stats()
        
        log.info("System stopped")
    
    def process_market_data(self, data: Dict):
        """Process incoming market data and generate signals"""
        try:
            # Extract data
            price_15m = data.get("price_15m", [])
            volume_15m = data.get("volume_15m", [])
            price_5m = data.get("price_5m", [])
            volume_5m = data.get("volume_5m", [])
            
            # Extract features
            features = self.feature_extractor.extract_features(
                price_15m, volume_15m, price_5m, volume_5m
            )

            # Train on historical data
            if not self.trained_on_historical:
                log.info(f"Training on {len(price_5m)} historical bars")
                self._train_on_historical_data(data)
                self.trained_on_historical = True
            
            # Generate signal
            action, confidence, quality = self.model.predict(features)
            
            # Determine signal strength and whether to send
            signal_strength = self._assess_signal_strength(confidence)
            should_send = confidence >= self.config.CONFIDENCE_THRESHOLD
            
            log.info(f"Signal: Action={self._get_action_name(action)}, "
                    f"Confidence={confidence:.3f}, Quality={quality}, "
                    f"Strength={signal_strength}, Send={should_send}")
            
            # Update statistics
            self._update_stats(action, confidence, signal_strength)
            
            # Send signal if confidence meets threshold
            if should_send:
                self.tcp_bridge.send_signal(action, confidence, quality)
            else:
                log.info(f"Signal below threshold ({self.config.CONFIDENCE_THRESHOLD:.1f}) - not sent")

            # Add to training samples
            self.model.add_training_sample(features)
                
        except Exception as e:
            log.error(f"Market data processing error: {e}")
    
    def _assess_signal_strength(self, confidence: float) -> str:
        """Assess signal strength based on confidence levels"""
        if confidence >= 0.8:
            return "excellent"
        elif confidence >= 0.7:
            return "good"
        elif confidence >= 0.6:
            return "fair"
        else:
            return "poor"
    
    def _get_action_name(self, action: int) -> str:
        """Convert action number to readable name"""
        return {0: "HOLD", 1: "BUY", 2: "SELL"}.get(action, "UNKNOWN")
    
    def _train_on_historical_data(self, data):
        """Train using research-aligned feature-based signals"""
        price_15m = data.get("price_15m", [])
        volume_15m = data.get("volume_15m", [])
        price_5m = data.get("price_5m", [])
        volume_5m = data.get("volume_5m", [])
        
        min_samples = max(50, self.config.SMA_PERIOD)
        training_samples = 0
        
        # Process historical data for training
        for i in range(min_samples, len(price_5m) - 5):
            # Get data up to point i
            hist_price_15m = price_15m[:i+1]
            hist_vol_15m = volume_15m[:i+1] 
            hist_price_5m = price_5m[:i+1]
            hist_vol_5m = volume_5m[:i+1]
            
            # Extract features for this specific time point
            features = self.feature_extractor.extract_features(
                hist_price_15m, hist_vol_15m, hist_price_5m, hist_vol_5m
            )

            if features is not None:
                signal = self.model._generate_signal_from_features(features)
                
                # Store for training
                self.model.feature_history.append(features)
                self.model.signal_history.append(signal)
                training_samples += 1

        # Signal distribution
        unique_signals, counts = np.unique(self.model.signal_history, return_counts=True)
        signal_dist = dict(zip(unique_signals, counts))
        log.info(f"Training samples: {training_samples}")
        log.info(f"Signal distribution: {signal_dist}")
        
        # Train if we have enough samples
        if training_samples >= self.config.MIN_TRAINING_SAMPLES:
            self.model.train()
            log.info("Initial model training completed")
        else:
            log.warning(f"Insufficient training samples: {training_samples}/{self.config.MIN_TRAINING_SAMPLES}")
    
    def _update_stats(self, action: int, confidence: float, signal_strength: str):
        """Update system statistics"""
        self.stats['signals_generated'] += 1
        
        # Count by action
        if action == 1:
            self.stats['buy_signals'] += 1
        elif action == 2:
            self.stats['sell_signals'] += 1
        else:
            self.stats['hold_signals'] += 1
        
        # Count by signal strength
        if signal_strength == "excellent":
            self.stats['high_confidence_signals'] += 1
        elif signal_strength == "good":
            self.stats['moderate_confidence_signals'] += 1
        elif signal_strength == "fair":
            self.stats['low_confidence_signals'] += 1
        
        # Update average confidence
        total = self.stats['signals_generated']
        self.stats['avg_confidence'] = (
            (self.stats['avg_confidence'] * (total - 1) + confidence) / total
        )
    
    def _print_final_stats(self):
        """Print comprehensive final statistics"""
        total = self.stats['signals_generated']
        if total > 0:
            log.info("=== FINAL SYSTEM STATISTICS ===")
            log.info(f"Total Signals Generated: {total}")
            log.info(f"Signal Distribution:")
            log.info(f"  HOLD: {self.stats['hold_signals']} ({self.stats['hold_signals']/total*100:.1f}%)")
            log.info(f"  BUY:  {self.stats['buy_signals']} ({self.stats['buy_signals']/total*100:.1f}%)")
            log.info(f"  SELL: {self.stats['sell_signals']} ({self.stats['sell_signals']/total*100:.1f}%)")
            log.info(f"Average Confidence: {self.stats['avg_confidence']:.3f}")
            log.info(f"Signal Quality Distribution:")
            log.info(f"  Excellent: {self.stats['high_confidence_signals']}")
            log.info(f"  Good: {self.stats['moderate_confidence_signals']}")
            log.info(f"  Fair: {self.stats['low_confidence_signals']}")
            log.info("================================")
        else:
            log.info("No signals generated during session")